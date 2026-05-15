import argparse
import os
import sys
from concurrent import futures
from types import SimpleNamespace
from typing import Dict, List, Tuple

import grpc
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import dqn_scheduler_pb2 as pb2
import dqn_scheduler_pb2_grpc as pb2_grpc

from DQN2.algorithm.vgpu_dqn_sim import create_agent


def compute_fit(gpu: Dict, mem_req: int, core_req: int) -> bool:
    total_mem = int(gpu.get("total_mem", 0))
    used_mem = int(gpu.get("used_mem", 0))
    used_core = int(gpu.get("used_core", 0))
    used_num = int(gpu.get("used_num", 0))
    number = int(gpu.get("number", 0))

    if number > 0 and used_num >= number:
        return False

    if total_mem - used_mem < int(mem_req):
        return False

    if used_core + int(core_req) > 100:
        return False

    if int(core_req) == 100 and used_num > 0:
        return False

    if used_core == 100 and int(core_req) == 0:
        return False

    return True


def fallback_order(
    gpus: List[Dict],
    mem_req: int,
    core_req: int,
    mode: str = "binpack",
) -> List[int]:
    feasible = []
    infeasible = []

    for g in gpus:
        if compute_fit(g, mem_req, core_req):
            feasible.append(g)
        else:
            infeasible.append(g)

    if mode == "spread":
        feasible.sort(
            key=lambda g: (
                int(g.get("used_num", 0)),
                int(g.get("used_mem", 0)),
                int(g.get("used_core", 0)),
                -int(g.get("index", 0)),
            )
        )
    elif mode == "original":
        feasible.sort(key=lambda g: -int(g.get("index", 0)))
    else:
        feasible.sort(
            key=lambda g: (
                -int(g.get("used_mem", 0)),
                -int(g.get("used_core", 0)),
                -int(g.get("used_num", 0)),
                -int(g.get("index", 0)),
            )
        )

    infeasible.sort(key=lambda g: -int(g.get("index", 0)))

    return [int(g["index"]) for g in feasible + infeasible]


def build_single_pod_graph(
    gpus: List[Dict],
    mem_req: int,
    core_req: int,
    nums: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为真实 Volcano 调度场景构造 v5 GNN-DQN 输入。

    v5 模型输入：
    - gpu_feats: [num_gpus, 6]
    - pod_feats: [num_pods, 4]
    - adj:       [num_gpus, num_pods]

    真实部署时，一次只调度当前 Pod，所以 num_pods=1。
    DQN 原动作是 (gpu_idx, pod_idx)，这里 pod_idx 恒为 0。
    """

    gpus = sorted(gpus, key=lambda x: int(x["index"]))
    num_gpus = len(gpus)

    max_mem = max([int(g["total_mem"]) for g in gpus] + [int(mem_req), 1])
    max_core = 100.0
    max_used_num = max([int(g.get("number", 10)) or 10 for g in gpus] + [1])

    gpu_feats = []
    adj = []

    for g in gpus:
        total_mem = max(int(g.get("total_mem", 1)), 1)
        used_mem = int(g.get("used_mem", 0))
        free_mem = max(total_mem - used_mem, 0)

        used_core = int(g.get("used_core", 0))
        free_core = max(100 - used_core, 0)

        used_num = int(g.get("used_num", 0))
        number = int(g.get("number", 10)) or 10

        fit = compute_fit(g, mem_req, core_req)

        # 6维 GPU 特征，对齐 v5 create_agent(gpu_feat_dim=6)
        gpu_feats.append(
            [
                total_mem / max_mem,
                free_mem / max(total_mem, 1),
                used_mem / max(total_mem, 1),
                free_core / max_core,
                used_core / max_core,
                used_num / max(number, max_used_num, 1),
            ]
        )

        adj.append([1.0 if fit else 0.0])

    # 4维 Pod 特征，对齐 v5 create_agent(pod_feat_dim=4)
    pod_feats = np.array(
        [
            [
                int(mem_req) / max_mem,
                int(core_req) / max_core,
                max(int(nums), 1) / max(num_gpus, 1),
                1.0,
            ]
        ],
        dtype=np.float32,
    )

    gpu_feats = np.array(gpu_feats, dtype=np.float32)
    adj = np.array(adj, dtype=np.float32)

    return gpu_feats, pod_feats, adj


class DQNSchedulerService(pb2_grpc.DQNSchedulerServicer):
    def __init__(
        self,
        model_path: str,
        hidden_dim: int = 256,
        device_name: str = "cpu",
        fallback: str = "binpack",
    ):
        self.model_path = model_path
        self.fallback = fallback

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DQN checkpoint not found: {model_path}")

        # 必须和 v5 训练/测试参数结构兼容。
        args = SimpleNamespace(
            hidden_dim=hidden_dim,
            lr=3e-4,
            gamma=0.9,
            batch_size=64,
            buffer_size=20000,
            priority_beta=0.4,
            epsilon_start=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
        )

        print(f"[DQN-gRPC] loading v5 GNN-DQN model: {model_path}")
        self.agent = create_agent(args)
        self.agent.load(model_path)
        self.agent.epsilon = 0.0

        self.torch_device = torch.device(device_name)
        self.agent.gnn_encoder.to(self.torch_device)
        self.agent.q_net.to(self.torch_device)
        self.agent.gnn_encoder.eval()
        self.agent.q_net.eval()

        print("[DQN-gRPC] model loaded")
        print(f"[DQN-gRPC] hidden_dim={hidden_dim}, device={self.torch_device}, fallback={fallback}")

    def Predict(self, request, context):
        gpus = []

        for g in request.gpus:
            gpus.append(
                {
                    "index": int(g.index),
                    "uuid": g.uuid,
                    "total_mem": int(g.total_mem),
                    "used_mem": int(g.used_mem),
                    "used_core": int(g.used_core),
                    "used_num": int(g.used_num),
                    "number": int(g.number),
                    "device_type": g.device_type,
                }
            )

        gpus = sorted(gpus, key=lambda x: int(x["index"]))

        mem_req = int(request.mem_req)
        core_req = int(request.core_req)
        nums = int(request.nums)

        if not gpus:
            return pb2.PredictResponse(
                ordered_indexes=[],
                selected_index=-1,
                fallback=True,
                reason="empty gpu list",
            )

        try:
            gpu_feats, pod_feats, adj = build_single_pod_graph(
                gpus=gpus,
                mem_req=mem_req,
                core_req=core_req,
                nums=nums,
            )

            valid_mask_np = adj > 0

            if not valid_mask_np.any():
                ordered = fallback_order(gpus, mem_req, core_req, mode=self.fallback)
                return pb2.PredictResponse(
                    ordered_indexes=ordered,
                    selected_index=ordered[0] if ordered else -1,
                    fallback=True,
                    reason="no feasible gpu, fallback",
                )

            gpu_feats_t = torch.tensor(
                gpu_feats,
                dtype=torch.float32,
                device=self.torch_device,
            )
            pod_feats_t = torch.tensor(
                pod_feats,
                dtype=torch.float32,
                device=self.torch_device,
            )
            adj_t = torch.tensor(
                adj,
                dtype=torch.float32,
                device=self.torch_device,
            )

            valid_mask = torch.tensor(
                valid_mask_np,
                dtype=torch.bool,
                device=self.torch_device,
            )

            with torch.no_grad():
                gpu_emb, pod_emb = self.agent.gnn_encoder(gpu_feats_t, pod_feats_t, adj_t)
                q_mat = self.agent.q_net(gpu_emb, pod_emb)

                # 当前真实调度只有一个 pod，所以 q_mat shape = [num_gpus, 1]
                q_mat[~valid_mask] = -1e9
                q_scores = q_mat[:, 0].detach().cpu().numpy().tolist()

            items = []
            for local_pos, g in enumerate(gpus):
                idx = int(g["index"])
                fit = compute_fit(g, mem_req, core_req)
                score = float(q_scores[local_pos])

                items.append(
                    {
                        "index": idx,
                        "score": score,
                        "fit": fit,
                    }
                )

            feasible = [x for x in items if x["fit"]]
            infeasible = [x for x in items if not x["fit"]]

            feasible.sort(key=lambda x: (-x["score"], -x["index"]))
            infeasible.sort(key=lambda x: (-x["score"], -x["index"]))

            ordered_items = feasible + infeasible
            ordered_indexes = [x["index"] for x in ordered_items]
            selected_index = ordered_indexes[0] if ordered_indexes else -1

            score_msgs = [
                pb2.GPUScore(
                    index=x["index"],
                    score=x["score"],
                    fit=x["fit"],
                )
                for x in ordered_items
            ]

            return pb2.PredictResponse(
                ordered_indexes=ordered_indexes,
                selected_index=selected_index,
                scores=score_msgs,
                fallback=False,
                reason="dqn",
            )

        except Exception as e:
            ordered = fallback_order(gpus, mem_req, core_req, mode=self.fallback)
            score_msgs = [
                pb2.GPUScore(
                    index=int(g["index"]),
                    score=0.0,
                    fit=compute_fit(g, mem_req, core_req),
                )
                for g in gpus
            ]

            return pb2.PredictResponse(
                ordered_indexes=ordered,
                selected_index=ordered[0] if ordered else -1,
                scores=score_msgs,
                fallback=True,
                reason=f"dqn error, fallback={self.fallback}, error={repr(e)}",
            )


def serve(args):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))

    service = DQNSchedulerService(
        model_path=args.model,
        hidden_dim=args.hidden_dim,
        device_name=args.device,
        fallback=args.fallback,
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
        default="DQN2/outputs_mixed_load_fixed_v5/vgpu_dqn_mixed_best.pth",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
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