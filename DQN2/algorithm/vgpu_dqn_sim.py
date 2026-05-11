"""
vGPU 单节点负载均衡仿真实验脚本。

设计目标：
1. 沿用仓库中 DQN_LB2.py 的核心思想：GNN 编码二分图 + DQN 选择分配动作。
2. 将原来的“服务器节点 node - 任务 task”分配问题，改成“物理 GPU - Pod 任务”分配问题。
3. 先做仿真，不直接改 Kubernetes / Volcano 调度器，方便快速验证算法效果。

运行示例：
    python DQN2/algorithm/vgpu_dqn_sim.py --episodes 300 --pods 6

核心抽象：
    GPU  = 单节点内的一张物理 GPU
    Pod  = 一个申请 vGPU 资源的任务
    action = (gpu_idx, pod_idx)，表示把某个 Pod 分配到某张 GPU 上
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. 经验回放：基本沿用原 DQN_LB2.py 的 PrioritizedReplayBuffer 思路
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

        priority_indices = np.random.choice(len(self.buffer), priority_sample_size, p=probabilities)
        random_indices = np.random.choice(len(self.buffer), random_sample_size)
        indices = np.concatenate([priority_indices, random_indices])
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_errors):
        for idx, error in zip(batch_indices, batch_errors):
            self.priorities[idx] = max(abs(error), 1e-5)


# ============================================================
# 2. GNN 编码器：沿用“节点-任务二分图”的结构
# ============================================================
class BipartiteGNN(nn.Module):
    def __init__(self, node_in_dim: int, task_in_dim: int, hidden_dim: int = 128, num_layers: int = 2):
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

    def forward(self, node_feats: torch.Tensor, task_feats: torch.Tensor, adj: torch.Tensor):
        node_emb = node_feats
        task_emb = task_feats

        for layer_idx in range(self.num_layers):
            node_deg = torch.clamp(adj.sum(dim=1, keepdim=True), min=1e-6)
            task_deg = torch.clamp(adj.sum(dim=0, keepdim=True), min=1e-6)

            node_agg = (adj.unsqueeze(-1) * task_emb.unsqueeze(0)).sum(dim=1) / node_deg
            task_agg = (adj.transpose(0, 1).unsqueeze(-1) * node_emb.unsqueeze(0)).sum(dim=1) / task_deg.transpose(0, 1)

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
# 3. GNN-DQN Agent：动作仍然是 (gpu_idx, pod_idx)
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

        self.gnn_encoder = BipartiteGNN(node_in_dim, task_in_dim, gnn_hidden_dim).to(device)
        self.q_net = QValueMLP(gnn_hidden_dim, q_hidden_dim).to(device)
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
        # epsilon-greedy：训练早期随机探索，后期更多按 Q 值选择
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
                    next_node_emb, next_task_emb = self.target_gnn(next_node_feats_t, next_task_feats_t, next_adj_t)
                    next_q_mat = self.target_q_net(next_node_emb, next_task_emb)
                    next_q_mat[~next_valid_mask] = -1e9
                    max_next_q = next_q_mat.max().item()

            target_q = reward + self.gamma * max_next_q
            td_error = abs(current_q - target_q)

        self.memory.add(td_error, (graph_data, action, reward, next_graph_data, done))

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
                        next_node_emb, next_task_emb = self.target_gnn(next_node_feats_t, next_task_feats_t, next_adj_t)
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
        nn.utils.clip_grad_norm_(list(self.gnn_encoder.parameters()) + list(self.q_net.parameters()), max_norm=1.0)
        self.optimizer.step()
        self.memory.update_priorities(indices, td_errors)
        return total_loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================
# 4. vGPU 场景修改点：把原来的 node/task 改成 GPU/Pod
# ============================================================
def make_simulated_gpus(num_gpus: int = 3, memory_total: int = 24576, core_total: int = 100) -> List[Dict]:
    """
    [vGPU修改点-1]
    原算法中的 node 表示集群服务器节点；这里改成单节点内的物理 GPU。
    每张 GPU 维护剩余显存、剩余 core、当前 Pod 数等信息。
    """
    return [
        {
            "gpu_id": i,
            "memory_total": memory_total,
            "memory_free": memory_total,
            "core_total": core_total,
            "core_free": core_total,
            "pod_count": 0,
            "util": 0.0,
        }
        for i in range(num_gpus)
    ]


def make_pod_batch(num_pods: int = 6) -> List[Dict]:
    """
    [vGPU修改点-2]
    原算法中的 task 表示普通资源任务；这里改成申请 vGPU 的 Pod。
    一个 Pod 就是一个 vGPU 任务，适合仿真 Deployment replicas=N 的场景。
    """
    memory_choices = [2048, 4096, 6144, 8192]
    core_choices = [5, 10, 15, 20]
    pods = []
    for i in range(num_pods):
        mem = random.choice(memory_choices)
        core = random.choice(core_choices)
        pods.append(
            {
                "task_id": f"pod-{i}",
                "vgpu_number": 1,
                "memory_demand": mem,
                "core_demand": core,
            }
        )
    return pods


def build_vgpu_graph(gpus_info: List[Dict], pods_info: List[Dict], allocations: Dict[str, int]):
    """
    [vGPU修改点-3]
    构建 GPU-Pod 二分图。

    node_feats 表示每张 GPU 的状态：
        1. memory_used_ratio
        2. core_used_ratio
        3. pod_count_ratio
        4. util_ratio
        5. memory_free_ratio
        6. core_free_ratio

    task_feats 表示每个 Pod 的资源需求：
        1. memory_demand_ratio
        2. core_demand_ratio
        3. vgpu_number
        4. allocated_flag

    adj[gpu_idx, pod_idx] = 1 表示该 GPU 当前资源足够，可以承载该 Pod。
    """
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
            [memory_used_ratio, core_used_ratio, pod_count_ratio, util_ratio, memory_free_ratio, core_free_ratio]
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

    return np.array(node_feats, dtype=np.float32), np.array(task_feats, dtype=np.float32), adj


def is_pod_allocatable(gpu: Dict, pod: Dict) -> bool:
    """[vGPU修改点-4] 判断某张 GPU 是否能承载某个 Pod 的 vGPU 资源请求。"""
    return gpu["memory_free"] >= pod["memory_demand"] and gpu["core_free"] >= pod["core_demand"]


def update_gpu_resources(gpu: Dict, pod: Dict, revert: bool = False):
    """
    [vGPU修改点-5]
    原算法扣减 node 的 cpu/memory/gpu；这里扣减物理 GPU 的 vGPU-memory 和 vGPU-cores。
    """
    factor = -1 if revert else 1
    gpu["memory_free"] -= factor * pod["memory_demand"]
    gpu["core_free"] -= factor * pod["core_demand"]
    gpu["pod_count"] += -1 if revert else 1

    # 仿真中的 util 不是 nvidia-smi 实测值，这里用 core 使用量近似估计。
    core_used_ratio = 1.0 - gpu["core_free"] / gpu["core_total"]
    gpu["util"] = max(0.0, min(100.0, core_used_ratio * 100.0))


def calculate_vgpu_reward(gpus_info: List[Dict]) -> float:
    """
    [vGPU修改点-6]
    负载均衡奖励函数。

    思路：
        GPU 之间显存使用率越接近越好；
        GPU 之间 core 使用率越接近越好；
        GPU 之间 Pod 数量越接近越好。

    标准差越小表示越均衡，因此 reward = 1 - 加权标准差。
    """
    memory_usages = []
    core_usages = []
    pod_counts = []

    max_pod_count = max(1, max(g["pod_count"] for g in gpus_info))
    for gpu in gpus_info:
        memory_usages.append(1.0 - gpu["memory_free"] / gpu["memory_total"])
        core_usages.append(1.0 - gpu["core_free"] / gpu["core_total"])
        pod_counts.append(gpu["pod_count"] / max_pod_count)

    memory_std = float(np.std(memory_usages))
    core_std = float(np.std(core_usages))
    pod_std = float(np.std(pod_counts))

    balance_penalty = memory_std + core_std + 0.2 * pod_std
    return 1.0 - balance_penalty


def balance_score(gpus_info: List[Dict]) -> float:
    """用于评估最终分配结果，数值越小表示越均衡。"""
    memory_usages = [1.0 - g["memory_free"] / g["memory_total"] for g in gpus_info]
    core_usages = [1.0 - g["core_free"] / g["core_total"] for g in gpus_info]
    pod_counts = [g["pod_count"] for g in gpus_info]
    return float(np.std(memory_usages) + np.std(core_usages) + 0.2 * np.std(pod_counts))


# ============================================================
# 5. 训练与基线策略
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_episode(agent: GNNBasedDQNAgent, num_gpus: int, num_pods: int, train: bool = True):
    gpus = make_simulated_gpus(num_gpus=num_gpus)
    pods = make_pod_batch(num_pods=num_pods)
    allocations: Dict[str, int] = {}
    allocated_pod_indices = set()
    graph_data = build_vgpu_graph(gpus, pods, allocations)
    total_reward = 0.0
    step_count = 0

    while len(allocated_pod_indices) < len(pods):
        node_feats, task_feats, adj = graph_data
        adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
        valid_mask = adj_t > 0

        if not valid_mask.any():
            total_reward -= 5.0
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
            reward = calculate_vgpu_reward(gpus)
            next_graph_data = build_vgpu_graph(gpus, pods, allocations)
            next_adj = next_graph_data[2]
            done = len(allocated_pod_indices) == len(pods) or not np.any(next_adj > 0)

        total_reward += reward
        step_count += 1

        if train:
            agent.remember(graph_data, action, reward, next_graph_data, done)

        graph_data = next_graph_data
        if done:
            break

    return total_reward, balance_score(gpus), allocations, gpus, pods, step_count


def least_loaded_baseline(num_gpus: int, pods: List[Dict]):
    """传统基线：每次选择当前综合负载最低且资源足够的 GPU。"""
    gpus = make_simulated_gpus(num_gpus=num_gpus)
    allocations = {}
    for pod in pods:
        candidates = []
        for idx, gpu in enumerate(gpus):
            if is_pod_allocatable(gpu, pod):
                memory_used = 1.0 - gpu["memory_free"] / gpu["memory_total"]
                core_used = 1.0 - gpu["core_free"] / gpu["core_total"]
                score = memory_used + core_used + 0.1 * gpu["pod_count"]
                candidates.append((score, idx))
        if not candidates:
            continue
        _, best_gpu_idx = min(candidates)
        update_gpu_resources(gpus[best_gpu_idx], pod)
        allocations[pod["task_id"]] = best_gpu_idx
    return balance_score(gpus), allocations, gpus


def random_baseline(num_gpus: int, pods: List[Dict]):
    """传统基线：随机选择一个资源足够的 GPU。"""
    gpus = make_simulated_gpus(num_gpus=num_gpus)
    allocations = {}
    for pod in pods:
        candidates = [idx for idx, gpu in enumerate(gpus) if is_pod_allocatable(gpu, pod)]
        if not candidates:
            continue
        gpu_idx = random.choice(candidates)
        update_gpu_resources(gpus[gpu_idx], pod)
        allocations[pod["task_id"]] = gpu_idx
    return balance_score(gpus), allocations, gpus


def train(args):
    seed_everything(args.seed)
    print(f"Using device: {device}")

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
    log_path = os.path.join(args.output_dir, "vgpu_sim_training_log.csv")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "balance_score", "loss", "epsilon", "steps"])

        for episode in range(1, args.episodes + 1):
            reward, score, _, _, _, steps = run_one_episode(
                agent,
                num_gpus=args.gpus,
                num_pods=args.pods,
                train=True,
            )

            loss = agent.replay(args.batch_size)
            agent.update_epsilon()
            if episode % args.target_update == 0:
                agent.update_target_model()

            writer.writerow([episode, reward, score, loss, agent.epsilon, steps])

            if episode % args.log_interval == 0:
                print(
                    f"episode={episode:04d} reward={reward:.4f} "
                    f"balance_score={score:.4f} loss={loss:.6f} epsilon={agent.epsilon:.4f}"
                )

    model_path = os.path.join(args.output_dir, "vgpu_dqn_sim.pth")
    torch.save(
        {
            "gnn_encoder": agent.gnn_encoder.state_dict(),
            "q_net": agent.q_net.state_dict(),
            "args": vars(args),
        },
        model_path,
    )

    # 固定一批 Pod，和传统策略做一次简单对比。
    test_pods = make_pod_batch(num_pods=args.pods)
    dqn_reward, dqn_score, dqn_alloc, dqn_gpus, _, _ = run_one_episode(
        agent,
        num_gpus=args.gpus,
        num_pods=args.pods,
        train=False,
    )
    least_score, least_alloc, least_gpus = least_loaded_baseline(args.gpus, copy.deepcopy(test_pods))
    random_score, random_alloc, random_gpus = random_baseline(args.gpus, copy.deepcopy(test_pods))

    print("\n=== Final quick comparison, lower balance_score is better ===")
    print(f"DQN simulation     : reward={dqn_reward:.4f}, balance_score={dqn_score:.4f}, allocation={dqn_alloc}")
    print(f"Least-loaded rule  : balance_score={least_score:.4f}, allocation={least_alloc}")
    print(f"Random rule        : balance_score={random_score:.4f}, allocation={random_alloc}")
    print(f"\ntraining log saved to: {log_path}")
    print(f"model saved to       : {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vGPU GNN-DQN load balancing simulation")
    parser.add_argument("--episodes", type=int, default=300, help="训练轮数")
    parser.add_argument("--gpus", type=int, default=3, help="单节点物理 GPU 数量")
    parser.add_argument("--pods", type=int, default=6, help="每轮仿真的 Pod 数量")
    parser.add_argument("--hidden-dim", type=int, default=128, help="GNN 和 Q 网络隐藏层维度")
    parser.add_argument("--batch-size", type=int, default=32, help="经验回放 batch size")
    parser.add_argument("--target-update", type=int, default=20, help="target 网络更新间隔")
    parser.add_argument("--log-interval", type=int, default=20, help="日志打印间隔")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="DQN2/outputs", help="输出目录")
    train(parser.parse_args())
