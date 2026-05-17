#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接加载 vGPU-DQN 模型，测试真实部署中的关键 case：

4 张 GPU，8 个 Pod；
每个 Pod 只申请 1 个 vGPU；
正常期望分布：2 2 2 2。

运行示例：
python DQN2/algorithm/test_exact_4gpu_8pod_direct.py \
  --model DQN2/outputs_mixed_load_fixed_v5/vgpu_dqn_mixed_best.pth

如果你想模拟“只申请 vgpu-number，不申请显存/算力”的极端 YAML：
python DQN2/algorithm/test_exact_4gpu_8pod_direct.py \
  --model DQN2/outputs_mixed_load_fixed_v5/vgpu_dqn_mixed_best.pth \
  --memory-demand 0 \
  --core-demand 0
"""

import argparse
import copy
import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from DQN2.algorithm.vgpu_dqn_sim import (
    create_agent,
    device,
    build_vgpu_graph,
    select_gpus_for_pod_by_anchor,
    allocate_pod_to_gpus,
)


def build_exact_scenario(
    num_gpus: int = 4,
    num_pods: int = 8,
    memory_total: float = 48.0,
    core_total: float = 100.0,
    memory_demand: float = 0.0,
    core_demand: float = 0.0,
):
    gpus = []

    for i in range(num_gpus):
        gpus.append(
            {
                "gpu_id": i,
                "memory_total": float(memory_total),
                "memory_free": float(memory_total),
                "core_total": float(core_total),
                "core_free": float(core_total),
                "pod_count": 0,
                "util": 0.0,
            }
        )

    pods = []

    for i in range(num_pods):
        pods.append(
            {
                "task_id": f"pod-{i}",
                "vgpu_number": 1,
                "memory_demand": float(memory_demand),
                "core_demand": float(core_demand),
            }
        )

    return {
        "scenario_id": "exact-4gpu-8pod-single-vgpu",
        "target_load": 0.0,
        "gpus": gpus,
        "pods": pods,
        "num_gpus": num_gpus,
        "num_pods": num_pods,
    }


def load_agent(model_path: str, hidden_dim: int):
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

    agent = create_agent(args)
    agent.load(model_path)
    agent.epsilon = 0.0

    agent.gnn_encoder.to(device)
    agent.q_net.to(device)
    agent.gnn_encoder.eval()
    agent.q_net.eval()

    return agent


def print_gpu_state(gpus):
    states = []

    for g in gpus:
        used_mem = g["memory_total"] - g["memory_free"]
        used_core = g["core_total"] - g["core_free"]

        states.append(
            {
                "gpu": g["gpu_id"],
                "pod_count": g["pod_count"],
                "used_mem": round(used_mem, 4),
                "free_mem": round(g["memory_free"], 4),
                "used_core": round(used_core, 4),
                "free_core": round(g["core_free"], 4),
            }
        )

    for s in states:
        print(
            f"  GPU {s['gpu']}: "
            f"pod_count={s['pod_count']}, "
            f"used_mem={s['used_mem']}, free_mem={s['free_mem']}, "
            f"used_core={s['used_core']}, free_core={s['free_core']}"
        )


def get_q_matrix(agent, graph_data):
    gpu_feats, pod_feats, adj = graph_data

    gpu_feats_t = torch.tensor(gpu_feats, dtype=torch.float32, device=device)
    pod_feats_t = torch.tensor(pod_feats, dtype=torch.float32, device=device)
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)

    valid_mask = adj_t > 0

    with torch.no_grad():
        gpu_emb, pod_emb = agent.gnn_encoder(gpu_feats_t, pod_feats_t, adj_t)
        q_mat = agent.q_net(gpu_emb, pod_emb)
        q_mat[~valid_mask] = -1e9

    return q_mat.detach().cpu().numpy(), valid_mask.detach().cpu().numpy()


def run_direct_test(args):
    scenario = build_exact_scenario(
        num_gpus=args.num_gpus,
        num_pods=args.num_pods,
        memory_total=args.memory_total,
        core_total=args.core_total,
        memory_demand=args.memory_demand,
        core_demand=args.core_demand,
    )

    agent = load_agent(
        model_path=args.model,
        hidden_dim=args.hidden_dim,
    )

    gpus = copy.deepcopy(scenario["gpus"])
    pods = copy.deepcopy(scenario["pods"])
    allocations = {}

    print("========== Direct Model Test ==========")
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"num_gpus: {args.num_gpus}")
    print(f"num_pods: {args.num_pods}")
    print(f"pod vgpu_number: 1")
    print(f"pod memory_demand: {args.memory_demand}")
    print(f"pod core_demand: {args.core_demand}")
    print()
    print("初始 GPU 状态：")
    print_gpu_state(gpus)

    step = 0

    while len(allocations) < len(pods):
        graph_data = build_vgpu_graph(
            gpus=gpus,
            pods=pods,
            allocations=allocations,
        )

        _, _, adj = graph_data
        valid_mask = torch.tensor(adj, dtype=torch.float32, device=device) > 0

        if not valid_mask.any():
            print("\n没有可行动作，提前结束。")
            break

        q_mat, valid_mask_np = get_q_matrix(agent, graph_data)

        gpu_idx, pod_idx = agent.act(graph_data, valid_mask)

        pod = pods[pod_idx]

        selected_gpu_indices = select_gpus_for_pod_by_anchor(
            gpus=gpus,
            pod=pod,
            anchor_gpu_idx=gpu_idx,
        )

        print("\n========== Step", step, "==========")
        print(f"模型选择: gpu_idx={gpu_idx}, pod_idx={pod_idx}, pod_id={pod['task_id']}")

        print("当前每个 GPU 对该 pod 的 Q 值：")
        for gi in range(args.num_gpus):
            q_value = q_mat[gi][pod_idx]
            fit = valid_mask_np[gi][pod_idx]
            print(f"  GPU {gi}: q={q_value:.6f}, fit={fit}")

        if selected_gpu_indices is None:
            print("选择失败：selected_gpu_indices=None")
            break

        allocate_pod_to_gpus(
            gpus=gpus,
            pod=pod,
            selected_gpu_indices=selected_gpu_indices,
        )

        allocations[pod["task_id"]] = selected_gpu_indices

        print(f"实际分配 GPU: {selected_gpu_indices}")
        print("分配后 GPU 状态：")
        print_gpu_state(gpus)

        step += 1

    counts = Counter()

    for pod_id, gpu_list in allocations.items():
        for gid in gpu_list:
            counts[gid] += 1

    final_dist = [counts.get(i, 0) for i in range(args.num_gpus)]

    print("\n========== Final Result ==========")
    print(f"allocations: {allocations}")
    print(f"final distribution: {final_dist}")
    print(f"expected distribution: {[args.num_pods // args.num_gpus] * args.num_gpus}")

    max_load = max(final_dist) if final_dist else 0
    min_load = min(final_dist) if final_dist else 0

    print(f"max_load={max_load}, min_load={min_load}, imbalance={max_load - min_load}")

    if final_dist == [2, 2, 2, 2]:
        print("结论：正常，模型在该 case 下分配均衡。")
    else:
        print("结论：异常，模型在 4GPU/8Pod/单 vGPU 场景下分配不均衡。")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="DQN2/outputs_mixed_load_fixed_v5/vgpu_dqn_mixed_best.pth",
    )

    parser.add_argument("--hidden-dim", type=int, default=256)

    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-pods", type=int, default=8)

    parser.add_argument("--memory-total", type=float, default=48.0)
    parser.add_argument("--core-total", type=float, default=100.0)

    # 重点：
    # 如果真实 YAML 只申请 volcano.sh/vgpu-number: 1，没有 memory/core，
    # 那就保持 0。
    #
    # 如果真实 YAML 也申请 vgpu-memory / vgpu-cores，
    # 就改成真实值，比如 memory-demand=8, core-demand=25。
    parser.add_argument("--memory-demand", type=float, default=0.0)
    parser.add_argument("--core-demand", type=float, default=0.0)

    return parser.parse_args()


if __name__ == "__main__":
    run_direct_test(parse_args())