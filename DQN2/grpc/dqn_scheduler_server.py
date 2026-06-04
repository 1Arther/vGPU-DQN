import argparse
import copy
import os
import sys
from concurrent import futures
from types import SimpleNamespace
from typing import Dict, List, Tuple

import grpc
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import dqn_scheduler_pb2 as pb2
import dqn_scheduler_pb2_grpc as pb2_grpc

from DQN2.algorithm.vgpu_dqn_sim import (
    allocate_pod_to_gpus,
    build_vgpu_graph,
    create_agent,
    device as SIM_DEVICE,
    get_feasible_gpus_for_pod,
    select_dqn_action,
    select_gpus_for_pod_by_anchor,
)


def grpc_gpu_to_sim(gpu_state) -> Dict:
    total_mem = float(gpu_state.total_mem)
    used_mem = float(gpu_state.used_mem)
    used_core = float(gpu_state.used_core)

    return {
        "gpu_id": int(gpu_state.index),
        "index": int(gpu_state.index),
        "uuid": gpu_state.uuid,
        "memory_total": total_mem,
        "memory_free": max(0.0, total_mem - used_mem),
        "core_total": 100.0,
        "core_free": max(0.0, 100.0 - used_core),
        "pod_count": int(gpu_state.used_num),
        "util": max(0.0, min(100.0, used_core)),
        "number": int(gpu_state.number),
        "device_type": gpu_state.device_type,
    }


def grpc_pod_to_sim(pod_request, fallback_index: int = 0) -> Dict:
    pod_namespace = pod_request.pod_namespace or "default"
    pod_name = pod_request.pod_name or f"pod-{fallback_index}"

    return {
        "task_id": f"{pod_namespace}/{pod_name}",
        "pod_namespace": pod_namespace,
        "pod_name": pod_name,
        "vgpu_number": max(1, int(pod_request.nums)),
        "memory_demand": float(pod_request.mem_req),
        "core_demand": float(pod_request.core_req),
    }


def make_runtime_args(args):
    return SimpleNamespace(
        hidden_dim=args.hidden_dim,
        lr=2e-4,
        gamma=0.9,
        batch_size=128,
        buffer_size=60000,
        priority_beta=0.4,
        epsilon_start=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        success_weight=2.0,
        balance_weight=1.0,
        failure_weight=2.0,
        inter_balance_weight=1.0,
        intra_balance_weight=1.0,
        delta_balance_weight=1.0,
        delta_inter_balance_weight=2.0,
        delta_intra_balance_weight=3.0,
        checkpoint_lowmid_load_threshold=1.0,
        disable_job_features=False,
        action_rerank_topk=args.action_rerank_topk,
        action_rerank_q_weight=args.action_rerank_q_weight,
        action_rerank_balance_weight=args.action_rerank_balance_weight,
        action_rerank_inter_weight=0.0,
        action_rerank_intra_weight=0.0,
        train_action_rerank=False,
    )


def fallback_order(gpus: List[Dict], pod: Dict, mode: str = "binpack") -> List[int]:
    feasible = []
    infeasible = []

    for gpu in gpus:
        if (
            gpu["memory_free"] >= pod["memory_demand"]
            and gpu["core_free"] >= pod["core_demand"]
        ):
            feasible.append(gpu)
        else:
            infeasible.append(gpu)

    if mode == "spread":
        feasible.sort(
            key=lambda g: (
                g["pod_count"],
                g["memory_total"] - g["memory_free"],
                g["core_total"] - g["core_free"],
                -g["gpu_id"],
            )
        )
    elif mode == "original":
        feasible.sort(key=lambda g: -g["gpu_id"])
    else:
        feasible.sort(
            key=lambda g: (
                -(g["memory_total"] - g["memory_free"]),
                -(g["core_total"] - g["core_free"]),
                -g["pod_count"],
                -g["gpu_id"],
            )
        )

    infeasible.sort(key=lambda g: -g["gpu_id"])
    return [int(g["gpu_id"]) for g in feasible + infeasible]


class DQNSchedulerService(pb2_grpc.DQNSchedulerServicer):
    def __init__(
        self,
        model_path: str,
        hidden_dim: int = 512,
        device_name: str = "cpu",
        fallback: str = "binpack",
        action_rerank_topk: int = 32,
        action_rerank_balance_weight: float = 1.0,
        action_rerank_q_weight: float = 1.0,
    ):
        self.model_path = model_path
        self.fallback = fallback
        self.device = SIM_DEVICE
        self.args = make_runtime_args(
            SimpleNamespace(
                hidden_dim=hidden_dim,
                action_rerank_topk=action_rerank_topk,
                action_rerank_balance_weight=action_rerank_balance_weight,
                action_rerank_q_weight=action_rerank_q_weight,
            )
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DQN checkpoint not found: {model_path}")

        self.agent = create_agent(self.args)
        self.agent.load(model_path)
        self.agent.epsilon = 0.0
        self.agent.gnn_encoder.to(self.device)
        self.agent.q_net.to(self.device)
        self.agent.gnn_encoder.eval()
        self.agent.q_net.eval()

        print(f"[DQN-gRPC] model loaded: {model_path}")
        print(
            "[DQN-gRPC] "
            f"hidden_dim={hidden_dim}, device={self.device}, "
            f"rerank_topk={action_rerank_topk}, fallback={fallback}"
        )

    def _schedule(self, gpus: List[Dict], pods: List[Dict]) -> Tuple[List[Dict], bool, str]:
        gpus = sorted(copy.deepcopy(gpus), key=lambda x: int(x["gpu_id"]))
        pods = copy.deepcopy(pods)
        allocations: Dict[str, List[int]] = {}
        rows = []
        fallback_used = False

        while len(allocations) < len(pods):
            graph_data = build_vgpu_graph(gpus, pods, allocations, args=self.args)
            valid_mask = torch.tensor(
                graph_data[2],
                dtype=torch.float32,
                device=self.device,
            ) > 0

            if not valid_mask.any():
                break

            gpu_idx, pod_idx = select_dqn_action(
                agent=self.agent,
                graph_data=graph_data,
                valid_mask=valid_mask,
                gpus=gpus,
                pods=pods,
                args=self.args,
                train=False,
            )

            pod = pods[pod_idx]
            selected = select_gpus_for_pod_by_anchor(
                gpus=gpus,
                pod=pod,
                anchor_gpu_idx=gpu_idx,
                args=self.args,
            )

            if selected is None:
                fallback_used = True
                selected = self._fallback_select(gpus, pod)

            if selected is None:
                break

            allocate_pod_to_gpus(gpus, pod, selected)
            allocations[pod["task_id"]] = selected
            rows.append(
                {
                    "pod_namespace": pod["pod_namespace"],
                    "pod_name": pod["pod_name"],
                    "selected_indexes": [int(gpus[i]["gpu_id"]) for i in selected],
                    "success": True,
                    "reason": "dqn" if not fallback_used else "fallback",
                }
            )

        allocated = set(allocations)
        for pod in pods:
            if pod["task_id"] in allocated:
                continue
            rows.append(
                {
                    "pod_namespace": pod["pod_namespace"],
                    "pod_name": pod["pod_name"],
                    "selected_indexes": [],
                    "success": False,
                    "reason": "no feasible allocation",
                }
            )

        return rows, fallback_used, "ok"

    def _fallback_select(self, gpus: List[Dict], pod: Dict):
        required = max(1, int(pod.get("vgpu_number", 1)))
        by_id = {int(g["gpu_id"]): idx for idx, g in enumerate(gpus)}
        selected = []

        for gpu_id in fallback_order(gpus, pod, mode=self.fallback):
            idx = by_id[gpu_id]
            if idx not in get_feasible_gpus_for_pod(gpus, pod):
                continue
            selected.append(idx)
            if len(selected) >= required:
                return selected

        return None

    def Predict(self, request, context):
        gpus = [grpc_gpu_to_sim(g) for g in request.gpus]
        pod = grpc_pod_to_sim(
            SimpleNamespace(
                pod_namespace=request.pod_namespace,
                pod_name=request.pod_name,
                mem_req=request.mem_req,
                core_req=request.core_req,
                nums=request.nums,
            )
        )

        if not gpus:
            return pb2.PredictResponse(
                ordered_indexes=[],
                selected_index=-1,
                fallback=True,
                reason="empty gpu list",
            )

        rows, fallback_used, reason = self._schedule(gpus, [pod])
        success_rows = [r for r in rows if r["success"]]
        selected = success_rows[0]["selected_indexes"] if success_rows else []
        ordered = selected + [i for i in fallback_order(gpus, pod, self.fallback) if i not in selected]

        scores = [
            pb2.GPUScore(
                index=int(g["gpu_id"]),
                score=0.0,
                fit=(
                    g["memory_free"] >= pod["memory_demand"]
                    and g["core_free"] >= pod["core_demand"]
                ),
            )
            for g in sorted(gpus, key=lambda x: int(x["gpu_id"]))
        ]

        return pb2.PredictResponse(
            ordered_indexes=ordered,
            selected_index=selected[0] if selected else -1,
            scores=scores,
            fallback=fallback_used or not bool(selected),
            reason=reason if selected else "no feasible allocation",
        )

    def ScheduleJob(self, request, context):
        gpus = [grpc_gpu_to_sim(g) for g in request.gpus]
        pods = [
            grpc_pod_to_sim(p, fallback_index=i)
            for i, p in enumerate(request.pods)
        ]

        if not gpus:
            return pb2.ScheduleJobResponse(
                allocations=[],
                fallback=True,
                reason="empty gpu list",
            )

        rows, fallback_used, reason = self._schedule(gpus, pods)
        return pb2.ScheduleJobResponse(
            allocations=[
                pb2.PodAllocation(
                    pod_namespace=r["pod_namespace"],
                    pod_name=r["pod_name"],
                    selected_indexes=r["selected_indexes"],
                    success=r["success"],
                    reason=r["reason"],
                )
                for r in rows
            ],
            fallback=fallback_used,
            reason=reason,
        )


def serve(args):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    service = DQNSchedulerService(
        model_path=args.model,
        hidden_dim=args.hidden_dim,
        device_name=args.device,
        fallback=args.fallback,
        action_rerank_topk=args.action_rerank_topk,
        action_rerank_balance_weight=args.action_rerank_balance_weight,
        action_rerank_q_weight=args.action_rerank_q_weight,
    )
    pb2_grpc.add_DQNSchedulerServicer_to_server(service, server)

    listen_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(listen_addr)
    print(f"[DQN-gRPC] listening on {listen_addr}")
    server.start()
    server.wait_for_termination()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=(
            "DQN2/outputs_mixed_load_preload_v16_jobfeatures_multiscore_trainrerank/"
            "vgpu_dqn_mixed_best.pth"
        ),
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--action-rerank-topk", type=int, default=32)
    parser.add_argument("--action-rerank-balance-weight", type=float, default=1.0)
    parser.add_argument("--action-rerank-q-weight", type=float, default=1.0)
    parser.add_argument(
        "--fallback",
        type=str,
        default="binpack",
        choices=["binpack", "spread", "original"],
    )
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    serve(parse_args())
