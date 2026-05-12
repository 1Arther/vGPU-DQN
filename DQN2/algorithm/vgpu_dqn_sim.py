"""
hami-core vGPU GNN-DQN simulation.

新改动：
1. 支持动态 GPU batch：
   - 每个训练 batch 的 GPU 数量和 GPU 资源可以不同；
   - 每个测试 batch 的 GPU 数量和 GPU 资源也可以不同。

2. 支持动态 Pod batch：
   - 每个训练 batch 的 Pod 数量和资源需求可以不同；
   - 每个测试 batch 的 Pod 数量和资源需求也可以不同。

3. 对比方法扩展为：
   - DQN
   - Volcano-binpack
   - Volcano-spread
   - Simple-spread
   - Random

4. 增加综合目标函数：
   objective = success_weight * success_rate
             - balance_weight * balance_score
             - failure_weight * failure_rate

5. best checkpoint 按 DQN 在固定测试集上的 objective 选择。
   baseline 只用于最终比较，不参与 best model 选择。

6. 增加 early stopping。

7. 保存测试明细：
   - test_comparison_detail.csv
   - test_comparison_summary.csv
"""

import argparse
import copy
import csv
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from DQN2.vgpu_gpu_generator import reset_gpus_from_template
    from DQN2.vgpu_scenario_generator import (
        generate_scenarios,
        load_scenarios,
        save_scenarios,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    from DQN2.vgpu_gpu_generator import reset_gpus_from_template
    from DQN2.vgpu_scenario_generator import (
        generate_scenarios,
        load_scenarios,
        save_scenarios,
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 0. 工具函数
# ============================================================
def parse_int_list(text: str) -> List[int]:
    if text is None or text.strip() == "":
        return []

    return [int(x.strip()) for x in text.split(",") if x.strip()]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. 经验回放
# ============================================================
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def __len__(self):
        return len(self.buffer)

    def add(self, error: float, sample):
        max_priority = max(self.priorities.max() if self.buffer else 1.0, 1e-5)

        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample

        self.priorities[self.pos] = max(max_priority, abs(error), 1e-5)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4, random_sample_ratio: float = 0.2):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: len(self.buffer)]

        if len(priorities) == 0:
            raise RuntimeError("empty replay buffer")

        if priorities.sum() <= 0:
            probabilities = np.ones_like(priorities) / len(priorities)
        else:
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()

        random_sample_size = int(batch_size * random_sample_ratio)
        priority_sample_size = batch_size - random_sample_size

        priority_indices = np.random.choice(
            len(self.buffer),
            priority_sample_size,
            p=probabilities,
        )

        random_indices = np.random.choice(
            len(self.buffer),
            random_sample_size,
        )

        indices = np.concatenate([priority_indices, random_indices])
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_errors):
        for idx, error in zip(batch_indices, batch_errors):
            self.priorities[idx] = max(abs(error), 1e-5)


# ============================================================
# 2. GNN 编码器
# ============================================================
class BipartiteGNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        task_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.node_linears = nn.ModuleList()
        self.task_linears = nn.ModuleList()

        self.node_linears.append(nn.Linear(node_in_dim + task_in_dim, hidden_dim))
        self.task_linears.append(nn.Linear(task_in_dim + node_in_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.node_linears.append(nn.Linear(2 * hidden_dim, hidden_dim))
            self.task_linears.append(nn.Linear(2 * hidden_dim, hidden_dim))

        self.act = nn.ReLU()
        self.out_dim = hidden_dim

    def forward(
        self,
        node_feats: torch.Tensor,
        task_feats: torch.Tensor,
        adj: torch.Tensor,
    ):
        node_emb = node_feats
        task_emb = task_feats

        for layer_idx in range(self.num_layers):
            node_deg = torch.clamp(adj.sum(dim=1, keepdim=True), min=1e-6)
            task_deg = torch.clamp(adj.sum(dim=0, keepdim=True), min=1e-6)

            node_agg = (
                adj.unsqueeze(-1) * task_emb.unsqueeze(0)
            ).sum(dim=1) / node_deg

            task_agg = (
                adj.transpose(0, 1).unsqueeze(-1) * node_emb.unsqueeze(0)
            ).sum(dim=1) / task_deg.transpose(0, 1)

            node_in = torch.cat([node_emb, node_agg], dim=-1)
            task_in = torch.cat([task_emb, task_agg], dim=-1)

            node_emb = self.act(self.node_linears[layer_idx](node_in))
            task_emb = self.act(self.task_linears[layer_idx](task_in))

        return node_emb, task_emb


class QValueMLP(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_emb: torch.Tensor, task_emb: torch.Tensor):
        n, embed_dim = node_emb.size()
        m, task_embed_dim = task_emb.size()

        assert embed_dim == task_embed_dim

        node_expand = node_emb.unsqueeze(1).expand(n, m, embed_dim)
        task_expand = task_emb.unsqueeze(0).expand(n, m, embed_dim)

        combo = torch.cat([node_expand, task_expand], dim=-1)
        q_flat = self.mlp(combo.reshape(n * m, 2 * embed_dim))

        return q_flat.reshape(n, m)


# ============================================================
# 3. DQN Agent
# ============================================================
class GNNBasedDQNAgent:
    def __init__(
        self,
        node_in_dim: int,
        task_in_dim: int,
        gnn_hidden_dim: int = 128,
        q_hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 5000,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.gnn_encoder = BipartiteGNN(
            node_in_dim,
            task_in_dim,
            gnn_hidden_dim,
        ).to(device)

        self.q_net = QValueMLP(
            gnn_hidden_dim,
            q_hidden_dim,
        ).to(device)

        self.target_gnn = copy.deepcopy(self.gnn_encoder).to(device)
        self.target_q_net = copy.deepcopy(self.q_net).to(device)

        self.memory = PrioritizedReplayBuffer(buffer_size)

        self.optimizer = optim.Adam(
            list(self.gnn_encoder.parameters()) + list(self.q_net.parameters()),
            lr=lr,
        )

    def update_target_model(self):
        self.target_gnn.load_state_dict(self.gnn_encoder.state_dict())
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def act(self, graph_data, valid_mask: torch.Tensor) -> Tuple[int, int]:
        if random.random() < self.epsilon:
            valid_idx = torch.nonzero(valid_mask, as_tuple=False)

            if valid_idx.size(0) == 0:
                return 0, 0

            rand_i = random.randint(0, valid_idx.size(0) - 1)
            gpu_idx, pod_idx = valid_idx[rand_i].tolist()

            return gpu_idx, pod_idx

        node_feats, task_feats, adj = graph_data

        node_feats_t = torch.tensor(node_feats, dtype=torch.float32, device=device)
        task_feats_t = torch.tensor(task_feats, dtype=torch.float32, device=device)
        adj_t = torch.tensor(adj, dtype=torch.float32, device=device)

        with torch.no_grad():
            node_emb, task_emb = self.gnn_encoder(node_feats_t, task_feats_t, adj_t)
            q_mat = self.q_net(node_emb, task_emb)
            q_mat[~valid_mask] = -1e9

            max_flat_idx = torch.argmax(q_mat)
            gpu_idx = max_flat_idx // q_mat.size(1)
            pod_idx = max_flat_idx % q_mat.size(1)

            return gpu_idx.item(), pod_idx.item()

    def remember(self, graph_data, action, reward: float, next_graph_data, done: bool):
        node_feats, task_feats, adj = graph_data

        node_feats_t = torch.tensor(node_feats, dtype=torch.float32, device=device)
        task_feats_t = torch.tensor(task_feats, dtype=torch.float32, device=device)
        adj_t = torch.tensor(adj, dtype=torch.float32, device=device)

        with torch.no_grad():
            node_emb, task_emb = self.gnn_encoder(node_feats_t, task_feats_t, adj_t)
            q_mat = self.q_net(node_emb, task_emb)
            current_q = q_mat[action[0], action[1]].item()

            if done:
                max_next_q = 0.0
            else:
                next_node_feats, next_task_feats, next_adj = next_graph_data

                next_node_feats_t = torch.tensor(next_node_feats, dtype=torch.float32, device=device)
                next_task_feats_t = torch.tensor(next_task_feats, dtype=torch.float32, device=device)
                next_adj_t = torch.tensor(next_adj, dtype=torch.float32, device=device)
                next_valid_mask = next_adj_t > 0

                if not next_valid_mask.any():
                    max_next_q = 0.0
                else:
                    next_node_emb, next_task_emb = self.target_gnn(
                        next_node_feats_t,
                        next_task_feats_t,
                        next_adj_t,
                    )
                    next_q_mat = self.target_q_net(next_node_emb, next_task_emb)
                    next_q_mat[~next_valid_mask] = -1e9
                    max_next_q = next_q_mat.max().item()

            target_q = reward + self.gamma * max_next_q
            td_error = abs(current_q - target_q)

        self.memory.add(
            td_error,
            (
                graph_data,
                action,
                reward,
                next_graph_data,
                done,
            ),
        )

    def replay(self, batch_size: int, beta: float = 0.4) -> float:
        if len(self.memory) < batch_size:
            return 0.0

        samples, indices, weights = self.memory.sample(batch_size, beta=beta)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

        losses = []
        td_errors = []

        self.optimizer.zero_grad()

        for i, (graph_data, action, reward, next_graph_data, done) in enumerate(samples):
            node_feats, task_feats, adj = graph_data

            node_feats_t = torch.tensor(node_feats, dtype=torch.float32, device=device)
            task_feats_t = torch.tensor(task_feats, dtype=torch.float32, device=device)
            adj_t = torch.tensor(adj, dtype=torch.float32, device=device)

            node_emb, task_emb = self.gnn_encoder(node_feats_t, task_feats_t, adj_t)
            q_mat = self.q_net(node_emb, task_emb)
            current_q = q_mat[action[0], action[1]]

            if done:
                target_q = torch.tensor(reward, dtype=torch.float32, device=device)
            else:
                next_node_feats, next_task_feats, next_adj = next_graph_data

                next_node_feats_t = torch.tensor(next_node_feats, dtype=torch.float32, device=device)
                next_task_feats_t = torch.tensor(next_task_feats, dtype=torch.float32, device=device)
                next_adj_t = torch.tensor(next_adj, dtype=torch.float32, device=device)
                next_valid_mask = next_adj_t > 0

                with torch.no_grad():
                    if not next_valid_mask.any():
                        max_next_q = torch.tensor(0.0, dtype=torch.float32, device=device)
                    else:
                        next_node_emb, next_task_emb = self.target_gnn(
                            next_node_feats_t,
                            next_task_feats_t,
                            next_adj_t,
                        )
                        next_q_mat = self.target_q_net(next_node_emb, next_task_emb)
                        next_q_mat[~next_valid_mask] = -1e9
                        max_next_q = next_q_mat.max()

                target_q = reward + self.gamma * max_next_q

            td_error = current_q - target_q
            loss = (td_error ** 2) * weights_t[i]

            losses.append(loss)
            td_errors.append(abs(td_error.item()))

        total_loss = torch.mean(torch.stack(losses))
        total_loss.backward()

        nn.utils.clip_grad_norm_(
            list(self.gnn_encoder.parameters()) + list(self.q_net.parameters()),
            max_norm=1.0,
        )

        self.optimizer.step()
        self.memory.update_priorities(indices, td_errors)

        return total_loss.item()

    def update_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )


# ============================================================
# 4. 数据集加载 / 生成
# ============================================================
def load_or_generate_vgpu_dataset(args):
    train_path = os.path.join(args.data_dir, "train_scenarios.json")
    test_path = os.path.join(args.data_dir, "test_scenarios.json")

    need_generate = (
        args.regenerate_data
        or not os.path.exists(train_path)
        or not os.path.exists(test_path)
    )

    gpu_memory_choices = parse_int_list(args.gpu_memory_choices)
    gpu_core_choices = parse_int_list(args.gpu_core_choices)
    pod_memory_choices = parse_int_list(args.pod_memory_choices)
    pod_core_choices = parse_int_list(args.pod_core_choices)

    if need_generate:
        train_scenarios = generate_scenarios(
            num_batches=args.train_batches,
            min_gpus=args.min_gpus,
            max_gpus=args.max_gpus,
            min_pods=args.min_pods,
            max_pods=args.max_pods,
            gpu_memory_choices=gpu_memory_choices,
            gpu_core_choices=gpu_core_choices,
            pod_memory_choices=pod_memory_choices,
            pod_core_choices=pod_core_choices,
        )

        test_scenarios = generate_scenarios(
            num_batches=args.test_batches,
            min_gpus=args.min_gpus,
            max_gpus=args.max_gpus,
            min_pods=args.min_pods,
            max_pods=args.max_pods,
            gpu_memory_choices=gpu_memory_choices,
            gpu_core_choices=gpu_core_choices,
            pod_memory_choices=pod_memory_choices,
            pod_core_choices=pod_core_choices,
        )

        save_scenarios(train_scenarios, train_path)
        save_scenarios(test_scenarios, test_path)

        print(f"generated scenarios saved to: {args.data_dir}")

    else:
        train_scenarios = load_scenarios(train_path)
        test_scenarios = load_scenarios(test_path)

        print(f"loaded scenarios from: {args.data_dir}")

    return train_scenarios, test_scenarios


# ============================================================
# 5. 图构建、资源更新、指标和 reward
# ============================================================
def build_vgpu_graph(
    gpus_info: List[Dict],
    pods_info: List[Dict],
    allocations: Dict[str, int],
):
    max_memory = max(g["memory_total"] for g in gpus_info)
    max_core = max(g["core_total"] for g in gpus_info)
    max_pods = max(1, len(pods_info))

    node_feats = []

    for gpu in gpus_info:
        memory_used_ratio = 1.0 - gpu["memory_free"] / gpu["memory_total"]
        core_used_ratio = 1.0 - gpu["core_free"] / gpu["core_total"]
        pod_count_ratio = gpu["pod_count"] / max_pods
        util_ratio = gpu["util"] / 100.0
        memory_free_ratio = gpu["memory_free"] / max_memory
        core_free_ratio = gpu["core_free"] / max_core

        node_feats.append(
            [
                memory_used_ratio,
                core_used_ratio,
                pod_count_ratio,
                util_ratio,
                memory_free_ratio,
                core_free_ratio,
            ]
        )

    task_feats = []

    for pod in pods_info:
        allocated_flag = 1.0 if pod["task_id"] in allocations else 0.0

        task_feats.append(
            [
                pod["memory_demand"] / max_memory,
                pod["core_demand"] / max_core,
                float(pod["vgpu_number"]),
                allocated_flag,
            ]
        )

    adj = np.zeros((len(gpus_info), len(pods_info)), dtype=np.float32)

    for i, gpu in enumerate(gpus_info):
        for j, pod in enumerate(pods_info):
            if pod["task_id"] in allocations:
                continue

            if is_pod_allocatable(gpu, pod):
                adj[i, j] = 1.0

    return (
        np.array(node_feats, dtype=np.float32),
        np.array(task_feats, dtype=np.float32),
        adj,
    )


def is_pod_allocatable(gpu: Dict, pod: Dict) -> bool:
    return (
        gpu["memory_free"] >= pod["memory_demand"]
        and gpu["core_free"] >= pod["core_demand"]
    )


def update_gpu_resources(gpu: Dict, pod: Dict):
    gpu["memory_free"] -= pod["memory_demand"]
    gpu["core_free"] -= pod["core_demand"]
    gpu["pod_count"] += 1

    core_used_ratio = 1.0 - gpu["core_free"] / gpu["core_total"]
    gpu["util"] = max(0.0, min(100.0, core_used_ratio * 100.0))


def balance_score(gpus_info: List[Dict]) -> float:
    """
    负载均衡指标，越低越好。

    当前只看：
        std(memory_usage) + std(core_usage)
    不看 pod_count。
    """
    memory_usages = [
        1.0 - g["memory_free"] / g["memory_total"]
        for g in gpus_info
    ]

    core_usages = [
        1.0 - g["core_free"] / g["core_total"]
        for g in gpus_info
    ]

    return float(np.std(memory_usages) + np.std(core_usages))


def calculate_objective(
    balance: float,
    success_rate: float,
    success_weight: float = 2.0,
    balance_weight: float = 1.0,
    failure_weight: float = 2.0,
) -> float:
    """
    综合评价目标，越大越好。

    新改动：
        把成功率、负载均衡、失败率统一成一个 objective。

    objective = α * success_rate - β * balance_score - γ * failure_rate
    """
    failure_rate = 1.0 - success_rate

    return (
        success_weight * success_rate
        - balance_weight * balance
        - failure_weight * failure_rate
    )


def calculate_step_reward(
    gpus_info: List[Dict],
    allocated_count: int,
    total_pods: int,
) -> float:
    """
    每一步奖励。

    保留较温和的 step reward，主要让模型知道：
        1. 成功分配是好事；
        2. 负载越均衡越好。
    """
    current_balance = balance_score(gpus_info)
    current_success_rate = allocated_count / total_pods

    return 1.0 - current_balance + 0.2 * current_success_rate


def calculate_terminal_reward(
    gpus_info: List[Dict],
    allocated_count: int,
    total_pods: int,
    success_weight: float,
    balance_weight: float,
    failure_weight: float,
) -> float:
    final_balance = balance_score(gpus_info)
    final_success_rate = allocated_count / total_pods

    return calculate_objective(
        balance=final_balance,
        success_rate=final_success_rate,
        success_weight=success_weight,
        balance_weight=balance_weight,
        failure_weight=failure_weight,
    )


def format_gpu_loads(gpus_info: List[Dict]) -> str:
    parts = []

    for gpu in gpus_info:
        mem_used = 1.0 - gpu["memory_free"] / gpu["memory_total"]
        core_used = 1.0 - gpu["core_free"] / gpu["core_total"]

        parts.append(
            f"GPU{gpu['gpu_id']}"
            f"(mem={mem_used:.2f}, core={core_used:.2f}, pods={gpu['pod_count']})"
        )

    return "; ".join(parts)


# ============================================================
# 6. DQN episode
# ============================================================
def run_one_episode(
    agent: GNNBasedDQNAgent,
    scenario: Dict,
    train: bool = True,
    success_weight: float = 2.0,
    balance_weight: float = 1.0,
    failure_weight: float = 2.0,
    failed_pod_penalty: float = 0.2,
):
    """
    执行一个 scenario 的完整调度。

    新改动：
        输入不再是固定 gpu_templates + pods；
        而是一个包含随机 GPU 和随机 Pod 的 scenario。
    """
    gpus = reset_gpus_from_template(scenario["gpus"])
    pods = copy.deepcopy(scenario["pods"])

    allocations: Dict[str, int] = {}
    allocated_pod_indices = set()

    graph_data = build_vgpu_graph(gpus, pods, allocations)

    total_reward = 0.0
    step_count = 0

    while len(allocated_pod_indices) < len(pods):
        _, _, adj = graph_data

        valid_mask = torch.tensor(adj, dtype=torch.float32, device=device) > 0

        if not valid_mask.any():
            allocated_count = len(allocations)
            failed_count = len(pods) - allocated_count

            reward = (
                -failed_pod_penalty * failed_count
                + calculate_terminal_reward(
                    gpus,
                    allocated_count,
                    len(pods),
                    success_weight=success_weight,
                    balance_weight=balance_weight,
                    failure_weight=failure_weight,
                )
            )

            total_reward += reward
            break

        action = agent.act(graph_data, valid_mask)
        gpu_idx, pod_idx = action
        pod = pods[pod_idx]

        if not valid_mask[gpu_idx, pod_idx]:
            reward = -5.0
            next_graph_data = graph_data
            done = True

        else:
            update_gpu_resources(gpus[gpu_idx], pod)

            allocations[pod["task_id"]] = gpu_idx
            allocated_pod_indices.add(pod_idx)

            allocated_count = len(allocations)

            reward = calculate_step_reward(
                gpus,
                allocated_count,
                len(pods),
            )

            next_graph_data = build_vgpu_graph(gpus, pods, allocations)

            done = (
                len(allocated_pod_indices) == len(pods)
                or not np.any(next_graph_data[2] > 0)
            )

            if done:
                failed_count = len(pods) - len(allocations)

                reward += (
                    -failed_pod_penalty * failed_count
                    + calculate_terminal_reward(
                        gpus,
                        len(allocations),
                        len(pods),
                        success_weight=success_weight,
                        balance_weight=balance_weight,
                        failure_weight=failure_weight,
                    )
                )

        total_reward += reward
        step_count += 1

        if train:
            agent.remember(
                graph_data,
                action,
                reward,
                next_graph_data,
                done,
            )

        graph_data = next_graph_data

        if done:
            break

    allocated_count = len(allocations)
    success_rate = allocated_count / len(pods)
    failure_rate = 1.0 - success_rate
    score = balance_score(gpus)

    objective = calculate_objective(
        balance=score,
        success_rate=success_rate,
        success_weight=success_weight,
        balance_weight=balance_weight,
        failure_weight=failure_weight,
    )

    return {
        "method": "dqn",
        "reward": total_reward,
        "balance_score": score,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "allocated_count": allocated_count,
        "failure_count": len(pods) - allocated_count,
        "total_pods": len(pods),
        "num_gpus": len(gpus),
        "steps": step_count,
        "objective": objective,
        "allocations": allocations,
        "gpus": gpus,
        "pods": pods,
    }


# ============================================================
# 7. baseline
# ============================================================
def run_heuristic_baseline(
    scenario: Dict,
    policy: str,
    success_weight: float = 2.0,
    balance_weight: float = 1.0,
    failure_weight: float = 2.0,
    random_seed: int = None,
):
    """
    baseline 统一入口。

    支持：
        volcano-binpack:
            选择放置后负载最高的 GPU，尽量压实资源。

        volcano-spread:
            选择放置后负载最低的 GPU，尽量分散资源。

        simple-spread:
            选择当前负载最低的 GPU，等价于原 Least-loaded 思路。

        random:
            在可行 GPU 中随机选择。
    """
    rng = random.Random(random_seed) if random_seed is not None else random

    gpus = reset_gpus_from_template(scenario["gpus"])
    pods = copy.deepcopy(scenario["pods"])
    allocations = {}

    for pod in pods:
        candidates = []

        for idx, gpu in enumerate(gpus):
            if not is_pod_allocatable(gpu, pod):
                continue

            current_mem_used = 1.0 - gpu["memory_free"] / gpu["memory_total"]
            current_core_used = 1.0 - gpu["core_free"] / gpu["core_total"]
            current_load = current_mem_used + current_core_used

            post_mem_used = 1.0 - (
                gpu["memory_free"] - pod["memory_demand"]
            ) / gpu["memory_total"]

            post_core_used = 1.0 - (
                gpu["core_free"] - pod["core_demand"]
            ) / gpu["core_total"]

            post_load = post_mem_used + post_core_used

            if policy == "volcano-binpack":
                score = post_load

            elif policy == "volcano-spread":
                score = -post_load

            elif policy == "simple-spread":
                score = -current_load

            elif policy == "random":
                score = 0.0

            else:
                raise ValueError(f"unknown policy: {policy}")

            candidates.append((score, idx))

        if not candidates:
            continue

        if policy == "random":
            _, best_gpu_idx = rng.choice(candidates)
        else:
            _, best_gpu_idx = max(candidates, key=lambda x: x[0])

        update_gpu_resources(gpus[best_gpu_idx], pod)
        allocations[pod["task_id"]] = best_gpu_idx

    allocated_count = len(allocations)
    total_pods = len(pods)
    success_rate = allocated_count / total_pods
    failure_rate = 1.0 - success_rate
    score = balance_score(gpus)

    objective = calculate_objective(
        balance=score,
        success_rate=success_rate,
        success_weight=success_weight,
        balance_weight=balance_weight,
        failure_weight=failure_weight,
    )

    return {
        "method": policy,
        "reward": "",
        "balance_score": score,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "allocated_count": allocated_count,
        "failure_count": total_pods - allocated_count,
        "total_pods": total_pods,
        "num_gpus": len(gpus),
        "steps": allocated_count,
        "objective": objective,
        "allocations": allocations,
        "gpus": gpus,
        "pods": pods,
    }


# ============================================================
# 8. 测试评估
# ============================================================
def result_to_row(batch_id: int, result: Dict) -> Dict:
    return {
        "batch_id": batch_id,
        "method": result["method"],
        "num_gpus": result["num_gpus"],
        "num_pods": result["total_pods"],
        "allocated_count": result["allocated_count"],
        "failure_count": result["failure_count"],
        "balance_score": result["balance_score"],
        "success_rate": result["success_rate"],
        "failure_rate": result["failure_rate"],
        "objective": result["objective"],
        "steps": result["steps"],
    }


def evaluate_on_test_scenarios(
    agent: GNNBasedDQNAgent,
    test_scenarios: List[Dict],
    args,
):
    """
    在固定测试 scenarios 上评估 DQN 和所有 baseline。

    新改动：
        返回每个 batch 的明细 detail_df；
        返回每个方法的平均 summary_df。
    """
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    detail_rows = []

    baseline_methods = [
        "volcano-binpack",
        "volcano-spread",
        "simple-spread",
        "random",
    ]

    for scenario in test_scenarios:
        batch_id = scenario["batch_id"]

        dqn_result = run_one_episode(
            agent,
            scenario,
            train=False,
            success_weight=args.success_weight,
            balance_weight=args.balance_weight,
            failure_weight=args.failure_weight,
            failed_pod_penalty=args.failed_pod_penalty,
        )

        detail_rows.append(result_to_row(batch_id, dqn_result))

        for method in baseline_methods:
            random_seed = args.seed + 100000 + batch_id if method == "random" else None

            baseline_result = run_heuristic_baseline(
                scenario,
                policy=method,
                success_weight=args.success_weight,
                balance_weight=args.balance_weight,
                failure_weight=args.failure_weight,
                random_seed=random_seed,
            )

            detail_rows.append(result_to_row(batch_id, baseline_result))

    agent.epsilon = old_epsilon

    detail_df = pd.DataFrame(detail_rows)

    summary_df = (
        detail_df.groupby("method")
        .agg(
            avg_balance_score=("balance_score", "mean"),
            std_balance_score=("balance_score", "std"),
            avg_success_rate=("success_rate", "mean"),
            std_success_rate=("success_rate", "std"),
            avg_failure_rate=("failure_rate", "mean"),
            avg_allocated_count=("allocated_count", "mean"),
            avg_failure_count=("failure_count", "mean"),
            avg_objective=("objective", "mean"),
            std_objective=("objective", "std"),
            avg_num_gpus=("num_gpus", "mean"),
            avg_num_pods=("num_pods", "mean"),
        )
        .reset_index()
    )

    return detail_df, summary_df


def get_dqn_metrics(summary_df: pd.DataFrame):
    dqn_row = summary_df[summary_df["method"] == "dqn"]

    if dqn_row.empty:
        raise RuntimeError("DQN row not found in summary_df")

    row = dqn_row.iloc[0]

    return {
        "objective": float(row["avg_objective"]),
        "balance_score": float(row["avg_balance_score"]),
        "success_rate": float(row["avg_success_rate"]),
        "failure_rate": float(row["avg_failure_rate"]),
    }


# ============================================================
# 9. 训练
# ============================================================
def train(args):
    seed_everything(args.seed)

    print(f"Using device: {device}")

    train_scenarios, test_scenarios = load_or_generate_vgpu_dataset(args)

    agent = GNNBasedDQNAgent(
        node_in_dim=6,
        task_in_dim=4,
        gnn_hidden_dim=args.hidden_dim,
        q_hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=1.0,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "vgpu_sim_training_log.csv")
    final_model_path = os.path.join(args.output_dir, "vgpu_dqn_sim_final.pth")
    best_model_path = os.path.join(args.output_dir, "vgpu_dqn_sim_best.pth")

    best_eval_objective = -float("inf")
    best_episode = -1
    patience_counter = 0
    stopped_early = False

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "episode",
                "train_batch_id",
                "num_gpus",
                "num_pods",
                "reward",
                "balance_score",
                "success_rate",
                "failure_rate",
                "allocated_count",
                "failure_count",
                "objective",
                "loss",
                "epsilon",
                "steps",
                "eval_objective",
                "eval_balance_score",
                "eval_success_rate",
                "eval_failure_rate",
                "best_eval_objective",
                "best_episode",
                "patience_counter",
            ]
        )

        for episode in range(1, args.episodes + 1):
            batch_id = (episode - 1) % len(train_scenarios)
            scenario = train_scenarios[batch_id]

            result = run_one_episode(
                agent,
                scenario,
                train=True,
                success_weight=args.success_weight,
                balance_weight=args.balance_weight,
                failure_weight=args.failure_weight,
                failed_pod_penalty=args.failed_pod_penalty,
            )

            loss = agent.replay(args.batch_size)
            agent.update_epsilon()

            if episode % args.target_update == 0:
                agent.update_target_model()

            eval_objective = ""
            eval_balance_score = ""
            eval_success_rate = ""
            eval_failure_rate = ""

            if episode % args.eval_interval == 0:
                _, eval_summary_df = evaluate_on_test_scenarios(
                    agent,
                    test_scenarios,
                    args,
                )

                dqn_metrics = get_dqn_metrics(eval_summary_df)

                eval_objective = dqn_metrics["objective"]
                eval_balance_score = dqn_metrics["balance_score"]
                eval_success_rate = dqn_metrics["success_rate"]
                eval_failure_rate = dqn_metrics["failure_rate"]

                if eval_objective > best_eval_objective + args.early_stop_min_delta:
                    best_eval_objective = eval_objective
                    best_episode = episode
                    patience_counter = 0

                    torch.save(
                        {
                            "gnn_encoder": agent.gnn_encoder.state_dict(),
                            "q_net": agent.q_net.state_dict(),
                            "args": vars(args),
                            "best_episode": best_episode,
                            "best_eval_objective": best_eval_objective,
                            "best_eval_balance_score": eval_balance_score,
                            "best_eval_success_rate": eval_success_rate,
                            "best_eval_failure_rate": eval_failure_rate,
                            "selection_rule": (
                                "max weighted objective: "
                                "success_weight * success_rate "
                                "- balance_weight * balance_score "
                                "- failure_weight * failure_rate"
                            ),
                        },
                        best_model_path,
                    )

                    print(
                        f"new best model saved: episode={episode}, "
                        f"eval_objective={eval_objective:.4f}, "
                        f"eval_balance={eval_balance_score:.4f}, "
                        f"eval_success={eval_success_rate:.4f}, "
                        f"eval_failure={eval_failure_rate:.4f}"
                    )

                else:
                    patience_counter += 1

                if (
                    args.early_stop_patience > 0
                    and patience_counter >= args.early_stop_patience
                ):
                    print(
                        f"early stopping at episode={episode}, "
                        f"best_episode={best_episode}, "
                        f"best_eval_objective={best_eval_objective:.4f}"
                    )
                    stopped_early = True

            writer.writerow(
                [
                    episode,
                    batch_id,
                    result["num_gpus"],
                    result["total_pods"],
                    result["reward"],
                    result["balance_score"],
                    result["success_rate"],
                    result["failure_rate"],
                    result["allocated_count"],
                    result["failure_count"],
                    result["objective"],
                    loss,
                    agent.epsilon,
                    result["steps"],
                    eval_objective,
                    eval_balance_score,
                    eval_success_rate,
                    eval_failure_rate,
                    best_eval_objective if best_episode > 0 else "",
                    best_episode if best_episode > 0 else "",
                    patience_counter,
                ]
            )

            if episode % args.log_interval == 0:
                msg = (
                    f"episode={episode:04d} "
                    f"reward={result['reward']:.4f} "
                    f"objective={result['objective']:.4f} "
                    f"balance_score={result['balance_score']:.4f} "
                    f"success_rate={result['success_rate']:.4f} "
                    f"failure_rate={result['failure_rate']:.4f} "
                    f"loss={loss:.6f} "
                    f"epsilon={agent.epsilon:.4f}"
                )

                if eval_objective != "":
                    msg += (
                        f" eval_objective={eval_objective:.4f} "
                        f"eval_balance={eval_balance_score:.4f} "
                        f"eval_success={eval_success_rate:.4f}"
                    )

                if best_episode > 0:
                    msg += (
                        f" best_episode={best_episode} "
                        f"best_eval_objective={best_eval_objective:.4f}"
                    )

                print(msg)

            if stopped_early:
                break

    torch.save(
        {
            "gnn_encoder": agent.gnn_encoder.state_dict(),
            "q_net": agent.q_net.state_dict(),
            "args": vars(args),
        },
        final_model_path,
    )

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        agent.gnn_encoder.load_state_dict(checkpoint["gnn_encoder"])
        agent.q_net.load_state_dict(checkpoint["q_net"])
        agent.update_target_model()

        print(
            f"\nloaded best model from episode={checkpoint['best_episode']}, "
            f"best_eval_objective={checkpoint['best_eval_objective']:.4f}, "
            f"best_eval_balance={checkpoint['best_eval_balance_score']:.4f}, "
            f"best_eval_success={checkpoint['best_eval_success_rate']:.4f}"
        )
    else:
        print("\nwarning: best model not found, using final model")

    detail_df, summary_df = evaluate_on_test_scenarios(
        agent,
        test_scenarios,
        args,
    )

    detail_path = os.path.join(args.output_dir, "test_comparison_detail.csv")
    summary_path = os.path.join(args.output_dir, "test_comparison_summary.csv")

    detail_df.to_csv(detail_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n=== Final test comparison summary ===")
    print(summary_df.to_string(index=False))

    print(f"\ntraining log saved to : {log_path}")
    print(f"final model saved to  : {final_model_path}")
    print(f"best model saved to   : {best_model_path}")
    print(f"test detail saved to  : {detail_path}")
    print(f"test summary saved to : {summary_path}")
    print(f"data dir              : {args.data_dir}")


# ============================================================
# 10. CLI
# ============================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="hami-core vGPU GNN-DQN dynamic simulation"
    )

    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--train-batches", type=int, default=500)
    parser.add_argument("--test-batches", type=int, default=50)

    parser.add_argument("--min-gpus", type=int, default=3)
    parser.add_argument("--max-gpus", type=int, default=8)
    parser.add_argument("--min-pods", type=int, default=20)
    parser.add_argument("--max-pods", type=int, default=80)

    # 兼容旧参数：如果传了 --gpus / --pods，则固定规模
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--pods", type=int, default=None)

    parser.add_argument(
        "--gpu-memory-choices",
        type=str,
        default="16384,24576,32768",
        help="GPU memory choices in MB",
    )

    parser.add_argument(
        "--gpu-core-choices",
        type=str,
        default="80,100,120",
        help="GPU core choices",
    )

    parser.add_argument(
        "--pod-memory-choices",
        type=str,
        default="1024,2048,4096,6144,8192",
        help="Pod vgpu-memory choices in MB",
    )

    parser.add_argument(
        "--pod-core-choices",
        type=str,
        default="5,10,15,20,25",
        help="Pod vgpu-cores choices",
    )

    parser.add_argument("--data-dir", type=str, default="DQN2/data/hami_dynamic")
    parser.add_argument("--output-dir", type=str, default="DQN2/outputs_hami_dynamic")
    parser.add_argument("--regenerate-data", action="store_true")

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=5000)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)

    parser.add_argument("--target-update", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=20)

    parser.add_argument("--success-weight", type=float, default=2.0)
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--failure-weight", type=float, default=2.0)
    parser.add_argument("--failed-pod-penalty", type=float, default=0.2)

    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="number of eval rounds without improvement before early stopping; 0 disables",
    )

    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser


def normalize_args(args):
    if args.gpus is not None:
        args.min_gpus = args.gpus
        args.max_gpus = args.gpus

    if args.pods is not None:
        args.min_pods = args.pods
        args.max_pods = args.pods

    if args.min_gpus <= 0 or args.max_gpus < args.min_gpus:
        raise ValueError("invalid GPU range")

    if args.min_pods <= 0 or args.max_pods < args.min_pods:
        raise ValueError("invalid Pod range")

    return args


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_args(args)
    train(args)