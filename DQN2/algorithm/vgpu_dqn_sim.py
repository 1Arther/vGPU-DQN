"""
vGPU 单节点负载均衡仿真实验脚本。

本脚本只负责：
1. 加载 / 生成 vGPU 仿真数据；
2. 构建 GPU-Pod 二分图；
3. 使用 GNN-DQN 进行调度决策；
4. 与 Least-loaded、Random baseline 进行对比；
5. 输出 reward、balance_score、loss、success_rate 等实验结果。

数据生成逻辑已拆分到：
    vgpu_gpu_generator.py
    vgpu_pod_generator.py

核心抽象：
    GPU  = 单节点内的一张物理 GPU
    Pod  = 一个申请 vGPU 资源的任务
    action = (gpu_idx, pod_idx)，表示把某个 Pod 分配到某张 GPU 上

建模原则：
1. DQN 和 baseline 使用同一批测试 Pod；
2. GPU 和 Pod 任务数据保存为 JSON，保证实验可复现；
3. 最终测试时关闭 epsilon 探索；
4. 默认 Pod 数改为 20，6 个 Pod 只适合调试；
5. balance_score 是最终评价指标，reward 是训练信号，loss 只反映训练稳定性；
6. pod_count 只作为状态特征，不进入 reward / balance_score。
"""

import argparse
import copy
import csv
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from DQN2.vgpu_gpu_generator import (
        generate_gpus,
        load_gpus,
        reset_gpus_from_template,
        save_gpus,
    )
    from DQN2.vgpu_pod_generator import (
        generate_pod_batches,
        load_pod_batches,
        save_pod_batches,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    from DQN2.vgpu_gpu_generator import (
        generate_gpus,
        load_gpus,
        reset_gpus_from_template,
        save_gpus,
    )
    from DQN2.vgpu_pod_generator import (
        generate_pod_batches,
        load_pod_batches,
        save_pod_batches,
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# 2. GNN 编码器：GPU-Pod 二分图
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
# 3. GNN-DQN Agent
# ============================================================
class GNNBasedDQNAgent:
    def __init__(
        self,
        node_in_dim: int,
        task_in_dim: int,
        gnn_hidden_dim: int = 128,
        q_hidden_dim: int = 128,
        lr: float = 1e-3,
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

        node_feats_t = torch.tensor(
            node_feats,
            dtype=torch.float32,
            device=device,
        )
        task_feats_t = torch.tensor(
            task_feats,
            dtype=torch.float32,
            device=device,
        )
        adj_t = torch.tensor(
            adj,
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            node_emb, task_emb = self.gnn_encoder(
                node_feats_t,
                task_feats_t,
                adj_t,
            )
            q_mat = self.q_net(node_emb, task_emb)
            q_mat[~valid_mask] = -1e9

            max_flat_idx = torch.argmax(q_mat)
            gpu_idx = max_flat_idx // q_mat.size(1)
            pod_idx = max_flat_idx % q_mat.size(1)

            return gpu_idx.item(), pod_idx.item()

    def remember(self, graph_data, action, reward: float, next_graph_data, done: bool):
        node_feats, task_feats, adj = graph_data

        node_feats_t = torch.tensor(
            node_feats,
            dtype=torch.float32,
            device=device,
        )
        task_feats_t = torch.tensor(
            task_feats,
            dtype=torch.float32,
            device=device,
        )
        adj_t = torch.tensor(
            adj,
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            node_emb, task_emb = self.gnn_encoder(
                node_feats_t,
                task_feats_t,
                adj_t,
            )
            q_mat = self.q_net(node_emb, task_emb)
            current_q = q_mat[action[0], action[1]].item()

            if done:
                max_next_q = 0.0
            else:
                next_node_feats, next_task_feats, next_adj = next_graph_data

                next_node_feats_t = torch.tensor(
                    next_node_feats,
                    dtype=torch.float32,
                    device=device,
                )
                next_task_feats_t = torch.tensor(
                    next_task_feats,
                    dtype=torch.float32,
                    device=device,
                )
                next_adj_t = torch.tensor(
                    next_adj,
                    dtype=torch.float32,
                    device=device,
                )
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
        weights_t = torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        )

        losses = []
        td_errors = []

        self.optimizer.zero_grad()

        for i, (graph_data, action, reward, next_graph_data, done) in enumerate(samples):
            node_feats, task_feats, adj = graph_data

            node_feats_t = torch.tensor(
                node_feats,
                dtype=torch.float32,
                device=device,
            )
            task_feats_t = torch.tensor(
                task_feats,
                dtype=torch.float32,
                device=device,
            )
            adj_t = torch.tensor(
                adj,
                dtype=torch.float32,
                device=device,
            )

            node_emb, task_emb = self.gnn_encoder(
                node_feats_t,
                task_feats_t,
                adj_t,
            )
            q_mat = self.q_net(node_emb, task_emb)
            current_q = q_mat[action[0], action[1]]

            if done:
                target_q = torch.tensor(
                    reward,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                next_node_feats, next_task_feats, next_adj = next_graph_data

                next_node_feats_t = torch.tensor(
                    next_node_feats,
                    dtype=torch.float32,
                    device=device,
                )
                next_task_feats_t = torch.tensor(
                    next_task_feats,
                    dtype=torch.float32,
                    device=device,
                )
                next_adj_t = torch.tensor(
                    next_adj,
                    dtype=torch.float32,
                    device=device,
                )

                next_valid_mask = next_adj_t > 0

                with torch.no_grad():
                    if not next_valid_mask.any():
                        max_next_q = torch.tensor(
                            0.0,
                            dtype=torch.float32,
                            device=device,
                        )
                    else:
                        next_node_emb, next_task_emb = self.target_gnn(
                            next_node_feats_t,
                            next_task_feats_t,
                            next_adj_t,
                        )
                        next_q_mat = self.target_q_net(
                            next_node_emb,
                            next_task_emb,
                        )
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
# 4. 数据加载 / 生成
# ============================================================
def load_or_generate_vgpu_dataset(args):
    gpus_path = os.path.join(args.data_dir, "gpus_info.json")
    train_path = os.path.join(args.data_dir, "train_pods.json")
    test_path = os.path.join(args.data_dir, "test_pods.json")

    need_generate = (
        args.regenerate_data
        or not os.path.exists(gpus_path)
        or not os.path.exists(train_path)
        or not os.path.exists(test_path)
    )

    if need_generate:
        gpus_info = generate_gpus(
            num_gpus=args.gpus,
            memory_total=args.gpu_memory,
            core_total=args.gpu_cores,
        )

        train_batches = generate_pod_batches(
            num_batches=args.train_batches,
            num_pods_per_batch=args.train_pods,
        )

        test_batches = generate_pod_batches(
            num_batches=args.test_batches,
            num_pods_per_batch=args.test_pods,
        )

        save_gpus(gpus_info, gpus_path)
        save_pod_batches(train_batches, train_path)
        save_pod_batches(test_batches, test_path)

        print(f"generated data saved to: {args.data_dir}")

    else:
        gpus_info = load_gpus(gpus_path)
        train_batches = load_pod_batches(train_path)
        test_batches = load_pod_batches(test_path)

        print(f"loaded data from: {args.data_dir}")

    return gpus_info, train_batches, test_batches


# ============================================================
# 5. vGPU 图构建、资源更新、reward 和评价指标
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

    adj = np.zeros(
        (len(gpus_info), len(pods_info)),
        dtype=np.float32,
    )

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
    gpu["util"] = max(
        0.0,
        min(100.0, core_used_ratio * 100.0),
    )


def balance_score(gpus_info: List[Dict]) -> float:
    """
    最终评价指标，越小越好。

    只看资源压力：
        std(memory_usage) + std(core_usage)

    不使用 pod_count 标准差。
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


def calculate_vgpu_reward(gpus_info: List[Dict]) -> float:
    """
    reward 和 balance_score 对齐。
    balance_score 越小，reward 越大。
    """
    return 1.0 - balance_score(gpus_info)


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
# 6. 训练、测试和 baseline
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_episode(
    agent: GNNBasedDQNAgent,
    gpu_templates: List[Dict],
    pods_info: List[Dict],
    train: bool = True,
):
    gpus = reset_gpus_from_template(gpu_templates)
    pods = copy.deepcopy(pods_info)

    allocations: Dict[str, int] = {}
    allocated_pod_indices = set()

    graph_data = build_vgpu_graph(
        gpus,
        pods,
        allocations,
    )

    total_reward = 0.0
    step_count = 0

    while len(allocated_pod_indices) < len(pods):
        _, _, adj = graph_data

        adj_t = torch.tensor(
            adj,
            dtype=torch.float32,
            device=device,
        )
        valid_mask = adj_t > 0

        if not valid_mask.any():
            total_reward -= 5.0
            break

        action = agent.act(
            graph_data,
            valid_mask,
        )

        gpu_idx, pod_idx = action
        pod = pods[pod_idx]

        if not valid_mask[gpu_idx, pod_idx]:
            reward = -5.0
            next_graph_data = graph_data
            done = True
        else:
            update_gpu_resources(
                gpus[gpu_idx],
                pod,
            )

            allocations[pod["task_id"]] = gpu_idx
            allocated_pod_indices.add(pod_idx)

            reward = calculate_vgpu_reward(gpus)

            next_graph_data = build_vgpu_graph(
                gpus,
                pods,
                allocations,
            )

            done = (
                len(allocated_pod_indices) == len(pods)
                or not np.any(next_graph_data[2] > 0)
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

    return (
        total_reward,
        balance_score(gpus),
        allocations,
        gpus,
        pods,
        step_count,
        allocated_count,
        success_rate,
    )


def least_loaded_baseline(
    gpu_templates: List[Dict],
    pods: List[Dict],
):
    gpus = reset_gpus_from_template(gpu_templates)
    allocations = {}

    for pod in copy.deepcopy(pods):
        candidates = []

        for idx, gpu in enumerate(gpus):
            if is_pod_allocatable(gpu, pod):
                memory_used = 1.0 - gpu["memory_free"] / gpu["memory_total"]
                core_used = 1.0 - gpu["core_free"] / gpu["core_total"]
                score = memory_used + core_used
                candidates.append((score, idx))

        if not candidates:
            continue

        _, best_gpu_idx = min(candidates)

        update_gpu_resources(
            gpus[best_gpu_idx],
            pod,
        )

        allocations[pod["task_id"]] = best_gpu_idx

    allocated_count = len(allocations)
    success_rate = allocated_count / len(pods)

    return (
        balance_score(gpus),
        allocations,
        gpus,
        allocated_count,
        success_rate,
    )


def random_baseline(
    gpu_templates: List[Dict],
    pods: List[Dict],
):
    gpus = reset_gpus_from_template(gpu_templates)
    allocations = {}

    for pod in copy.deepcopy(pods):
        candidates = [
            idx
            for idx, gpu in enumerate(gpus)
            if is_pod_allocatable(gpu, pod)
        ]

        if not candidates:
            continue

        gpu_idx = random.choice(candidates)

        update_gpu_resources(
            gpus[gpu_idx],
            pod,
        )

        allocations[pod["task_id"]] = gpu_idx

    allocated_count = len(allocations)
    success_rate = allocated_count / len(pods)

    return (
        balance_score(gpus),
        allocations,
        gpus,
        allocated_count,
        success_rate,
    )


def evaluate_on_test_batches(
    agent: GNNBasedDQNAgent,
    gpu_templates: List[Dict],
    test_batches: List[List[Dict]],
):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    dqn_scores = []
    least_scores = []
    random_scores = []

    dqn_success_rates = []
    least_success_rates = []
    random_success_rates = []

    last_result = None

    for pods in test_batches:
        (
            dqn_reward,
            dqn_score,
            dqn_alloc,
            dqn_gpus,
            _,
            _,
            dqn_allocated_count,
            dqn_success_rate,
        ) = run_one_episode(
            agent,
            gpu_templates,
            pods,
            train=False,
        )

        (
            least_score,
            least_alloc,
            least_gpus,
            least_allocated_count,
            least_success_rate,
        ) = least_loaded_baseline(
            gpu_templates,
            copy.deepcopy(pods),
        )

        (
            random_score,
            random_alloc,
            random_gpus,
            random_allocated_count,
            random_success_rate,
        ) = random_baseline(
            gpu_templates,
            copy.deepcopy(pods),
        )

        dqn_scores.append(dqn_score)
        least_scores.append(least_score)
        random_scores.append(random_score)

        dqn_success_rates.append(dqn_success_rate)
        least_success_rates.append(least_success_rate)
        random_success_rates.append(random_success_rate)

        last_result = (
            dqn_reward,
            dqn_score,
            dqn_alloc,
            dqn_gpus,
            dqn_allocated_count,
            dqn_success_rate,
            least_score,
            least_alloc,
            least_gpus,
            least_allocated_count,
            least_success_rate,
            random_score,
            random_alloc,
            random_gpus,
            random_allocated_count,
            random_success_rate,
        )

    agent.epsilon = old_epsilon

    return {
        "dqn_avg_score": float(np.mean(dqn_scores)),
        "least_avg_score": float(np.mean(least_scores)),
        "random_avg_score": float(np.mean(random_scores)),
        "dqn_avg_success": float(np.mean(dqn_success_rates)),
        "least_avg_success": float(np.mean(least_success_rates)),
        "random_avg_success": float(np.mean(random_success_rates)),
        "last_result": last_result,
    }


def train(args):
    seed_everything(args.seed)

    print(f"Using device: {device}")

    gpu_templates, train_batches, test_batches = load_or_generate_vgpu_dataset(args)

    agent = GNNBasedDQNAgent(
        node_in_dim=6,
        task_in_dim=4,
        gnn_hidden_dim=args.hidden_dim,
        q_hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_size=5000,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(
        args.output_dir,
        "vgpu_sim_training_log.csv",
    )

    with open(
        log_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "episode",
                "reward",
                "balance_score",
                "loss",
                "epsilon",
                "steps",
                "allocated_count",
                "success_rate",
                "train_batch_id",
            ]
        )

        for episode in range(1, args.episodes + 1):
            batch_id = (episode - 1) % len(train_batches)
            pods = train_batches[batch_id]

            (
                reward,
                score,
                _,
                _,
                _,
                steps,
                allocated_count,
                success_rate,
            ) = run_one_episode(
                agent,
                gpu_templates,
                pods,
                train=True,
            )

            loss = agent.replay(args.batch_size)
            agent.update_epsilon()

            if episode % args.target_update == 0:
                agent.update_target_model()

            writer.writerow(
                [
                    episode,
                    reward,
                    score,
                    loss,
                    agent.epsilon,
                    steps,
                    allocated_count,
                    success_rate,
                    batch_id,
                ]
            )

            if episode % args.log_interval == 0:
                print(
                    f"episode={episode:04d} "
                    f"reward={reward:.4f} "
                    f"balance_score={score:.4f} "
                    f"success_rate={success_rate:.2f} "
                    f"loss={loss:.6f} "
                    f"epsilon={agent.epsilon:.4f}"
                )

    model_path = os.path.join(
        args.output_dir,
        "vgpu_dqn_sim.pth",
    )

    torch.save(
        {
            "gnn_encoder": agent.gnn_encoder.state_dict(),
            "q_net": agent.q_net.state_dict(),
            "args": vars(args),
        },
        model_path,
    )

    eval_result = evaluate_on_test_batches(
        agent,
        gpu_templates,
        test_batches,
    )

    (
        dqn_reward,
        dqn_score,
        dqn_alloc,
        dqn_gpus,
        dqn_allocated_count,
        dqn_success_rate,
        least_score,
        least_alloc,
        least_gpus,
        least_allocated_count,
        least_success_rate,
        random_score,
        random_alloc,
        random_gpus,
        random_allocated_count,
        random_success_rate,
    ) = eval_result["last_result"]

    print("\n=== Final test comparison ===")
    print("balance_score 越低越好；success_rate 越高越好。")
    print()

    print(f"DQN avg balance_score         : {eval_result['dqn_avg_score']:.4f}")
    print(f"Least-loaded avg balance_score: {eval_result['least_avg_score']:.4f}")
    print(f"Random avg balance_score      : {eval_result['random_avg_score']:.4f}")

    print()

    print(f"DQN avg success_rate          : {eval_result['dqn_avg_success']:.4f}")
    print(f"Least-loaded avg success_rate : {eval_result['least_avg_success']:.4f}")
    print(f"Random avg success_rate       : {eval_result['random_avg_success']:.4f}")

    print("\n--- Last test batch detail ---")

    print(
        f"DQN simulation     : reward={dqn_reward:.4f}, "
        f"balance_score={dqn_score:.4f}, "
        f"allocated={dqn_allocated_count}, "
        f"success_rate={dqn_success_rate:.2f}, "
        f"allocation={dqn_alloc}"
    )
    print(f"                   loads={format_gpu_loads(dqn_gpus)}")

    print(
        f"Least-loaded rule  : balance_score={least_score:.4f}, "
        f"allocated={least_allocated_count}, "
        f"success_rate={least_success_rate:.2f}, "
        f"allocation={least_alloc}"
    )
    print(f"                   loads={format_gpu_loads(least_gpus)}")

    print(
        f"Random rule        : balance_score={random_score:.4f}, "
        f"allocated={random_allocated_count}, "
        f"success_rate={random_success_rate:.2f}, "
        f"allocation={random_alloc}"
    )
    print(f"                   loads={format_gpu_loads(random_gpus)}")

    print(f"\ntraining log saved to: {log_path}")
    print(f"model saved to       : {model_path}")
    print(f"data dir             : {args.data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vGPU GNN-DQN load balancing simulation"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="训练轮数；正式实验建议 2000 以上",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=3,
        help="单节点物理 GPU 数量",
    )

    parser.add_argument(
        "--gpu-memory",
        type=int,
        default=24576,
        help="单张 GPU 显存容量，单位 MB",
    )

    parser.add_argument(
        "--gpu-cores",
        type=int,
        default=100,
        help="单张 GPU 抽象 core 总量",
    )

    parser.add_argument(
        "--pods",
        type=int,
        default=20,
        help="兼容旧参数；默认等同于 --train-pods 和 --test-pods",
    )

    parser.add_argument(
        "--train-pods",
        type=int,
        default=None,
        help="每个训练批次的 Pod 数量",
    )

    parser.add_argument(
        "--test-pods",
        type=int,
        default=None,
        help="每个测试批次的 Pod 数量",
    )

    parser.add_argument(
        "--train-batches",
        type=int,
        default=200,
        help="训练 Pod 批次数",
    )

    parser.add_argument(
        "--test-batches",
        type=int,
        default=20,
        help="测试 Pod 批次数",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="DQN2/data/vgpu_sim",
        help="GPU 和 Pod 数据保存目录",
    )

    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="重新生成并覆盖数据集",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="GNN 和 Q 网络隐藏层维度",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="经验回放 batch size",
    )

    parser.add_argument(
        "--target-update",
        type=int,
        default=20,
        help="target 网络更新间隔",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="日志打印间隔",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="DQN2/outputs",
        help="输出目录",
    )

    args = parser.parse_args()

    if args.train_pods is None:
        args.train_pods = args.pods

    if args.test_pods is None:
        args.test_pods = args.pods

    train(args)