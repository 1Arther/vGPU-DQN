"""
Common module for vGPU-DQN mixed-load experiments.

v5 features:
1. Support one Pod requesting multiple vGPUs.
2. One Pod can be allocated to multiple GPUs inside the same single node.
3. Cross-node scheduling is not considered.
4. Multi-vGPU allocation is all-or-nothing.
5. Volcano vGPU baselines follow source-level device priority:
   - volcano-vgpu-binpack: UsedMem larger first, then GPU index smaller first.
   - volcano-vgpu-spread : UsedNum smaller first, then GPU index smaller first.
6. DQN action remains (gpu_idx, pod_idx):
   - gpu_idx is treated as anchor GPU.
   - If pod.vgpu_number > 1, the simulator completes the remaining GPU set.
"""

import copy
import csv
import json
import os
import random
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. Common utils
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def std(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(values))


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(rows: List[Dict], path: str):
    ensure_dir(os.path.dirname(path))

    if not rows:
        with open(path, "w", newline="", encoding="utf-8"):
            pass
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: List[Dict], path: str):
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


# ============================================================
# 2. Scenario generation
# ============================================================

def generate_gpus(
    num_gpus: int,
    memory_choices: List[float],
    core_total: float,
    heterogeneous: bool = True,
) -> List[Dict]:
    """
    Generate GPU list.

    memory_total unit:
        GB in this simulator.

    core_total:
        100 means full GPU compute percentage.
    """
    gpus = []

    for gpu_id in range(num_gpus):
        if heterogeneous:
            memory_total = float(random.choice(memory_choices)) * random.uniform(0.92, 1.08)
            gpu_core_total = float(core_total) * random.uniform(0.95, 1.05)
        else:
            memory_total = float(memory_choices[0])
            gpu_core_total = float(core_total)

        memory_total = round(memory_total, 4)
        gpu_core_total = round(gpu_core_total, 4)

        gpus.append(
            {
                "gpu_id": gpu_id,
                "memory_total": memory_total,
                "memory_free": memory_total,
                "core_total": gpu_core_total,
                "core_free": gpu_core_total,
                "pod_count": 0,
                "util": 0.0,
            }
        )

    return gpus


def sample_vgpu_numbers(num_pods: int, max_vgpu_per_pod: int) -> List[int]:
    """
    Generate vgpu_number for each Pod.

    Most Pods request 1 vGPU, and a smaller portion request multiple vGPUs.

    Default approximate distribution:
        1 vGPU: 70%
        2 vGPU: 20%
        3 vGPU:  7%
        4 vGPU:  3%
    """
    max_vgpu_per_pod = max(1, int(max_vgpu_per_pod))

    choices = [1]
    weights = [0.70]

    if max_vgpu_per_pod >= 2:
        choices.append(2)
        weights.append(0.20)

    if max_vgpu_per_pod >= 3:
        choices.append(3)
        weights.append(0.07)

    if max_vgpu_per_pod >= 4:
        choices.append(4)
        weights.append(0.03)

    return random.choices(choices, weights=weights, k=num_pods)


def _scaled_weighted_demands(
    total_target: float,
    weights: np.ndarray,
    min_value: float,
    max_value: float,
) -> np.ndarray:
    """
    Generate per-vGPU demand values.

    The weighted sum is close to total_target:

        sum(weights[i] * values[i]) ~= total_target

    Here weights[i] is pod.vgpu_number.
    """
    count = len(weights)

    if count <= 0:
        return np.array([], dtype=np.float32)

    weights = weights.astype(np.float32)

    feasible_min_total = float(np.sum(weights * min_value))
    feasible_max_total = float(np.sum(weights * max_value))
    total_target = max(feasible_min_total, min(float(total_target), feasible_max_total))

    values = np.random.gamma(shape=2.0, scale=1.0, size=count).astype(np.float32)
    values = min_value + (values / max(values.max(), 1e-8)) * (max_value - min_value)

    current_total = float(np.sum(weights * values))

    if current_total > 1e-8:
        values = values * (total_target / current_total)

    for _ in range(20):
        values = np.clip(values, min_value, max_value)
        current_total = float(np.sum(weights * values))
        diff = total_target - current_total

        if abs(diff) < 1e-5:
            break

        if diff > 0:
            mask = values < max_value
        else:
            mask = values > min_value

        if not np.any(mask):
            break

        weight_sum = float(np.sum(weights[mask]))

        if weight_sum <= 1e-8:
            break

        values[mask] += diff / weight_sum

    return np.clip(values, min_value, max_value).astype(np.float32)


def generate_pods_for_load(
    gpus: List[Dict],
    target_load: float,
    min_pods: int,
    max_pods: int,
    scenario_id: str,
) -> List[Dict]:
    """
    Generate Pods according to target_load.

    v5 semantics:
        vgpu_number   = number of vGPUs requested by this Pod.
        memory_demand = memory demand per vGPU.
        core_demand   = core demand per vGPU.

    Total Pod demand:
        total_memory = vgpu_number * memory_demand
        total_core   = vgpu_number * core_demand
    """
    num_pods = random.randint(min_pods, max_pods)

    total_gpu_memory = sum(g["memory_total"] for g in gpus)
    total_gpu_core = sum(g["core_total"] for g in gpus)

    max_gpu_memory = max(g["memory_total"] for g in gpus)
    max_gpu_core = max(g["core_total"] for g in gpus)

    max_vgpu_per_pod = min(4, len(gpus))
    vgpu_numbers = sample_vgpu_numbers(num_pods, max_vgpu_per_pod=max_vgpu_per_pod)
    vgpu_weights = np.array(vgpu_numbers, dtype=np.float32)

    memory_target = total_gpu_memory * target_load * random.uniform(0.97, 1.05)
    core_target = total_gpu_core * target_load * random.uniform(0.92, 1.04)

    memory_demands = _scaled_weighted_demands(
        total_target=memory_target,
        weights=vgpu_weights,
        min_value=0.20,
        max_value=max_gpu_memory * 0.45,
    )

    core_demands = _scaled_weighted_demands(
        total_target=core_target,
        weights=vgpu_weights,
        min_value=0.80,
        max_value=max_gpu_core * 0.45,
    )

    pods = []

    for i in range(num_pods):
        pods.append(
            {
                "task_id": f"{scenario_id}-pod-{i}",
                "vgpu_number": int(vgpu_numbers[i]),
                "memory_demand": round(float(memory_demands[i]), 4),
                "core_demand": round(float(core_demands[i]), 4),
            }
        )

    random.shuffle(pods)
    return pods


def generate_scenario(
    target_load: float,
    min_gpus: int,
    max_gpus: int,
    min_pods: int,
    max_pods: int,
    gpu_memory_choices: List[float],
    gpu_core_total: float,
    scenario_id: str,
    heterogeneous_gpus: bool = True,
) -> Dict:
    num_gpus = random.randint(min_gpus, max_gpus)

    gpus = generate_gpus(
        num_gpus=num_gpus,
        memory_choices=gpu_memory_choices,
        core_total=gpu_core_total,
        heterogeneous=heterogeneous_gpus,
    )

    pods = generate_pods_for_load(
        gpus=gpus,
        target_load=target_load,
        min_pods=min_pods,
        max_pods=max_pods,
        scenario_id=scenario_id,
    )

    scenario = {
        "scenario_id": scenario_id,
        "target_load": target_load,
        "gpus": gpus,
        "pods": pods,
    }

    actual_load, memory_load, core_load = compute_scenario_load(scenario)

    scenario.update(
        {
            "actual_load": actual_load,
            "memory_load": memory_load,
            "core_load": core_load,
            "num_gpus": len(gpus),
            "num_pods": len(pods),
        }
    )

    return scenario


def compute_scenario_load(scenario: Dict) -> Tuple[float, float, float]:
    """
    v5 load calculation uses total vGPU demand.

    total_pod_memory = sum(vgpu_number * memory_demand)
    total_pod_core   = sum(vgpu_number * core_demand)
    """
    gpus = scenario["gpus"]
    pods = scenario["pods"]

    total_gpu_memory = sum(g["memory_total"] for g in gpus)
    total_gpu_core = sum(g["core_total"] for g in gpus)

    total_pod_memory = sum(
        p.get("vgpu_number", 1) * p["memory_demand"]
        for p in pods
    )

    total_pod_core = sum(
        p.get("vgpu_number", 1) * p["core_demand"]
        for p in pods
    )

    memory_load = total_pod_memory / max(total_gpu_memory, 1e-8)
    core_load = total_pod_core / max(total_gpu_core, 1e-8)
    actual_load = max(memory_load, core_load)

    return float(actual_load), float(memory_load), float(core_load)


def reset_gpus(gpu_templates: List[Dict]) -> List[Dict]:
    gpus = copy.deepcopy(gpu_templates)

    for gpu in gpus:
        gpu["memory_free"] = gpu["memory_total"]
        gpu["core_free"] = gpu["core_total"]
        gpu["pod_count"] = 0
        gpu["util"] = 0.0

    return gpus


# ============================================================
# 3. Replay buffer
# ============================================================

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def add(self, error: float, sample):
        priority = max(abs(error), 1e-5)

        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
        random_sample_ratio: float = 0.2,
    ):
        buffer_len = len(self.buffer)

        if buffer_len == 0:
            raise RuntimeError("empty replay buffer")

        if buffer_len < self.capacity:
            priorities = self.priorities[:buffer_len]
        else:
            priorities = self.priorities

        if priorities.sum() <= 0:
            probs = np.ones_like(priorities) / len(priorities)
        else:
            probs = priorities ** self.alpha
            probs = probs / probs.sum()

        random_size = int(batch_size * random_sample_ratio)
        priority_size = batch_size - random_size

        priority_indices = np.random.choice(
            buffer_len,
            priority_size,
            p=probs,
        )

        if random_size > 0:
            random_indices = np.random.choice(buffer_len, random_size)
            indices = np.concatenate([priority_indices, random_indices])
        else:
            indices = priority_indices

        samples = [self.buffer[i] for i in indices]

        weights = (buffer_len * probs[indices]) ** (-beta)
        weights = weights / max(weights.max(), 1e-8)

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = max(abs(error), 1e-5)


# ============================================================
# 4. GNN-DQN model
# ============================================================

class BipartiteGNN(nn.Module):
    def __init__(
        self,
        gpu_feat_dim: int,
        pod_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.gpu_linears = nn.ModuleList()
        self.pod_linears = nn.ModuleList()

        self.gpu_linears.append(nn.Linear(gpu_feat_dim + pod_feat_dim, hidden_dim))
        self.pod_linears.append(nn.Linear(pod_feat_dim + gpu_feat_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.gpu_linears.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.pod_linears.append(nn.Linear(hidden_dim * 2, hidden_dim))

        self.act = nn.ReLU()
        self.out_dim = hidden_dim

    def forward(
        self,
        gpu_feats: torch.Tensor,
        pod_feats: torch.Tensor,
        adj: torch.Tensor,
    ):
        gpu_emb = gpu_feats
        pod_emb = pod_feats

        for layer_idx in range(self.num_layers):
            gpu_deg = torch.clamp(adj.sum(dim=1, keepdim=True), min=1e-6)
            pod_deg = torch.clamp(adj.sum(dim=0, keepdim=True), min=1e-6)

            gpu_agg = (
                adj.unsqueeze(-1) * pod_emb.unsqueeze(0)
            ).sum(dim=1) / gpu_deg

            pod_agg = (
                adj.transpose(0, 1).unsqueeze(-1) * gpu_emb.unsqueeze(0)
            ).sum(dim=1) / pod_deg.transpose(0, 1)

            gpu_in = torch.cat([gpu_emb, gpu_agg], dim=-1)
            pod_in = torch.cat([pod_emb, pod_agg], dim=-1)

            gpu_emb = self.act(self.gpu_linears[layer_idx](gpu_in))
            pod_emb = self.act(self.pod_linears[layer_idx](pod_in))

        return gpu_emb, pod_emb


class QValueMLP(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gpu_emb: torch.Tensor, pod_emb: torch.Tensor):
        num_gpus, dim = gpu_emb.size()
        num_pods, pod_dim = pod_emb.size()

        assert dim == pod_dim

        gpu_expand = gpu_emb.unsqueeze(1).expand(num_gpus, num_pods, dim)
        pod_expand = pod_emb.unsqueeze(0).expand(num_gpus, num_pods, dim)

        pair_emb = torch.cat([gpu_expand, pod_expand], dim=-1)
        q_values = self.mlp(pair_emb.reshape(num_gpus * num_pods, dim * 2))

        return q_values.reshape(num_gpus, num_pods)


class GNNBasedDQNAgent:
    def __init__(
        self,
        gpu_feat_dim: int,
        pod_feat_dim: int,
        hidden_dim: int,
        lr: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        buffer_size: int,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.gnn_encoder = BipartiteGNN(
            gpu_feat_dim=gpu_feat_dim,
            pod_feat_dim=pod_feat_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        self.q_net = QValueMLP(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
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

    def update_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def act(self, graph_data, valid_mask: torch.Tensor) -> Tuple[int, int]:
        if random.random() < self.epsilon:
            valid_indices = torch.nonzero(valid_mask, as_tuple=False)

            if valid_indices.size(0) == 0:
                return 0, 0

            rand_idx = random.randint(0, valid_indices.size(0) - 1)
            gpu_idx, pod_idx = valid_indices[rand_idx].tolist()
            return int(gpu_idx), int(pod_idx)

        gpu_feats, pod_feats, adj = graph_data

        gpu_feats_t = torch.tensor(gpu_feats, dtype=torch.float32, device=device)
        pod_feats_t = torch.tensor(pod_feats, dtype=torch.float32, device=device)
        adj_t = torch.tensor(adj, dtype=torch.float32, device=device)

        with torch.no_grad():
            gpu_emb, pod_emb = self.gnn_encoder(gpu_feats_t, pod_feats_t, adj_t)
            q_mat = self.q_net(gpu_emb, pod_emb)
            q_mat[~valid_mask] = -1e9

            flat_idx = torch.argmax(q_mat)
            gpu_idx = flat_idx // q_mat.size(1)
            pod_idx = flat_idx % q_mat.size(1)

        return int(gpu_idx.item()), int(pod_idx.item())

    def remember(self, graph_data, action, reward: float, next_graph_data, done: bool):
        gpu_feats, pod_feats, adj = graph_data

        gpu_feats_t = torch.tensor(gpu_feats, dtype=torch.float32, device=device)
        pod_feats_t = torch.tensor(pod_feats, dtype=torch.float32, device=device)
        adj_t = torch.tensor(adj, dtype=torch.float32, device=device)

        with torch.no_grad():
            gpu_emb, pod_emb = self.gnn_encoder(gpu_feats_t, pod_feats_t, adj_t)
            q_mat = self.q_net(gpu_emb, pod_emb)
            current_q = q_mat[action[0], action[1]].item()

            if done:
                max_next_q = 0.0
            else:
                next_gpu_feats, next_pod_feats, next_adj = next_graph_data

                next_gpu_feats_t = torch.tensor(
                    next_gpu_feats,
                    dtype=torch.float32,
                    device=device,
                )
                next_pod_feats_t = torch.tensor(
                    next_pod_feats,
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
                    next_gpu_emb, next_pod_emb = self.target_gnn(
                        next_gpu_feats_t,
                        next_pod_feats_t,
                        next_adj_t,
                    )
                    next_q_mat = self.target_q_net(next_gpu_emb, next_pod_emb)
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
            gpu_feats, pod_feats, adj = graph_data

            gpu_feats_t = torch.tensor(gpu_feats, dtype=torch.float32, device=device)
            pod_feats_t = torch.tensor(pod_feats, dtype=torch.float32, device=device)
            adj_t = torch.tensor(adj, dtype=torch.float32, device=device)

            gpu_emb, pod_emb = self.gnn_encoder(gpu_feats_t, pod_feats_t, adj_t)
            q_mat = self.q_net(gpu_emb, pod_emb)
            current_q = q_mat[action[0], action[1]]

            if done:
                target_q = torch.tensor(reward, dtype=torch.float32, device=device)
            else:
                next_gpu_feats, next_pod_feats, next_adj = next_graph_data

                next_gpu_feats_t = torch.tensor(
                    next_gpu_feats,
                    dtype=torch.float32,
                    device=device,
                )
                next_pod_feats_t = torch.tensor(
                    next_pod_feats,
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
                        max_next_q = torch.tensor(0.0, dtype=torch.float32, device=device)
                    else:
                        next_gpu_emb, next_pod_emb = self.target_gnn(
                            next_gpu_feats_t,
                            next_pod_feats_t,
                            next_adj_t,
                        )
                        next_q_mat = self.target_q_net(next_gpu_emb, next_pod_emb)
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

        return float(total_loss.item())

    def save(self, path: str, args=None):
        ensure_dir(os.path.dirname(path))

        torch.save(
            {
                "gnn_encoder": self.gnn_encoder.state_dict(),
                "q_net": self.q_net.state_dict(),
                "target_gnn": self.target_gnn.state_dict(),
                "target_q_net": self.target_q_net.state_dict(),
                "epsilon": self.epsilon,
                "args": vars(args) if args is not None else None,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=device)

        self.gnn_encoder.load_state_dict(checkpoint["gnn_encoder"])
        self.q_net.load_state_dict(checkpoint["q_net"])

        if "target_gnn" in checkpoint:
            self.target_gnn.load_state_dict(checkpoint["target_gnn"])
        else:
            self.target_gnn.load_state_dict(self.gnn_encoder.state_dict())

        if "target_q_net" in checkpoint:
            self.target_q_net.load_state_dict(checkpoint["target_q_net"])
        else:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = checkpoint.get("epsilon", self.epsilon)


def create_agent(args) -> GNNBasedDQNAgent:
    return GNNBasedDQNAgent(
        gpu_feat_dim=6,
        pod_feat_dim=4,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
    )


# ============================================================
# 5. Multi-vGPU environment logic
# ============================================================

def get_required_vgpu_number(pod: Dict) -> int:
    return max(1, int(pod.get("vgpu_number", 1)))


def is_single_vgpu_fit(gpu: Dict, pod: Dict) -> bool:
    return (
        gpu["memory_free"] >= pod["memory_demand"]
        and gpu["core_free"] >= pod["core_demand"]
    )


def get_feasible_gpus_for_pod(gpus: List[Dict], pod: Dict) -> List[int]:
    return [
        idx
        for idx, gpu in enumerate(gpus)
        if is_single_vgpu_fit(gpu, pod)
    ]


def is_pod_allocatable(gpu: Dict, pod: Dict) -> bool:
    """
    Backward-compatible single-GPU fit check.
    """
    return is_single_vgpu_fit(gpu, pod)


def can_allocate_pod_multi_vgpu(
    gpus: List[Dict],
    pod: Dict,
    anchor_gpu_idx: Optional[int] = None,
) -> bool:
    required = get_required_vgpu_number(pod)
    feasible = get_feasible_gpus_for_pod(gpus, pod)

    if anchor_gpu_idx is not None:
        return anchor_gpu_idx in feasible and len(feasible) >= required

    return len(feasible) >= required


def update_gpu_resources(gpu: Dict, pod: Dict):
    """
    Allocate one vGPU slice of this Pod to one GPU.
    """
    gpu["memory_free"] -= pod["memory_demand"]
    gpu["core_free"] -= pod["core_demand"]
    gpu["pod_count"] += 1

    core_used_ratio = 1.0 - gpu["core_free"] / max(gpu["core_total"], 1e-8)
    gpu["util"] = max(0.0, min(100.0, core_used_ratio * 100.0))


def allocate_pod_to_gpus(
    gpus: List[Dict],
    pod: Dict,
    selected_gpu_indices: List[int],
):
    """
    Allocate one Pod to multiple GPUs.

    selected_gpu_indices length must equal pod.vgpu_number.
    Each selected GPU receives one vGPU slice.

    All-or-nothing:
        all fit checks are completed before resource update.
    """
    required = get_required_vgpu_number(pod)

    if len(selected_gpu_indices) != required:
        raise ValueError(
            f"selected GPU count {len(selected_gpu_indices)} "
            f"does not match vgpu_number {required}"
        )

    if len(set(selected_gpu_indices)) != len(selected_gpu_indices):
        raise ValueError(
            "one Pod cannot allocate multiple vGPU slices to the same GPU in this simulator"
        )

    for idx in selected_gpu_indices:
        if not is_single_vgpu_fit(gpus[idx], pod):
            raise ValueError(f"GPU {idx} cannot fit pod {pod['task_id']}")

    for idx in selected_gpu_indices:
        update_gpu_resources(gpus[idx], pod)


def balance_score(gpus: List[Dict]) -> float:
    memory_usages = [
        1.0 - g["memory_free"] / max(g["memory_total"], 1e-8)
        for g in gpus
    ]

    core_usages = [
        1.0 - g["core_free"] / max(g["core_total"], 1e-8)
        for g in gpus
    ]

    return float(np.std(memory_usages) + np.std(core_usages))


def select_gpus_for_pod_by_anchor(
    gpus: List[Dict],
    pod: Dict,
    anchor_gpu_idx: int,
) -> Optional[List[int]]:
    """
    DQN multi-vGPU allocation helper.

    DQN chooses:
        action = (anchor_gpu_idx, pod_idx)

    If this Pod needs multiple vGPUs:
        - The anchor GPU must be included.
        - The remaining GPUs are selected to minimize balance_score after allocation.
        - All-or-nothing.
    """
    required = get_required_vgpu_number(pod)
    feasible = get_feasible_gpus_for_pod(gpus, pod)

    if anchor_gpu_idx not in feasible:
        return None

    if len(feasible) < required:
        return None

    if required == 1:
        return [anchor_gpu_idx]

    remaining = [idx for idx in feasible if idx != anchor_gpu_idx]
    need = required - 1

    best_selected = None
    best_score = float("inf")
    best_tie_key = None

    for combo in combinations(remaining, need):
        selected = [anchor_gpu_idx] + list(combo)

        trial_gpus = copy.deepcopy(gpus)

        try:
            allocate_pod_to_gpus(trial_gpus, pod, selected)
        except ValueError:
            continue

        score = balance_score(trial_gpus)
        tie_key = tuple(sorted(selected))

        if best_selected is None:
            best_selected = selected
            best_score = score
            best_tie_key = tie_key
        elif score < best_score - 1e-12:
            best_selected = selected
            best_score = score
            best_tie_key = tie_key
        elif abs(score - best_score) <= 1e-12 and tie_key < best_tie_key:
            best_selected = selected
            best_score = score
            best_tie_key = tie_key

    return best_selected


def build_vgpu_graph(
    gpus: List[Dict],
    pods: List[Dict],
    allocations: Dict[str, List[int]],
):
    max_memory = max(g["memory_total"] for g in gpus)
    max_core = max(g["core_total"] for g in gpus)
    max_pods = max(1, len(pods))

    gpu_feats = []

    for gpu in gpus:
        memory_used_ratio = 1.0 - gpu["memory_free"] / max(gpu["memory_total"], 1e-8)
        core_used_ratio = 1.0 - gpu["core_free"] / max(gpu["core_total"], 1e-8)
        pod_count_ratio = gpu["pod_count"] / max_pods
        util_ratio = gpu["util"] / 100.0
        memory_free_ratio = gpu["memory_free"] / max_memory
        core_free_ratio = gpu["core_free"] / max_core

        gpu_feats.append(
            [
                memory_used_ratio,
                core_used_ratio,
                pod_count_ratio,
                util_ratio,
                memory_free_ratio,
                core_free_ratio,
            ]
        )

    pod_feats = []

    for pod in pods:
        allocated_flag = 1.0 if pod["task_id"] in allocations else 0.0

        pod_feats.append(
            [
                pod["memory_demand"] / max_memory,
                pod["core_demand"] / max_core,
                get_required_vgpu_number(pod) / max(1.0, len(gpus)),
                allocated_flag,
            ]
        )

    adj = np.zeros((len(gpus), len(pods)), dtype=np.float32)

    for i, gpu in enumerate(gpus):
        for j, pod in enumerate(pods):
            if pod["task_id"] in allocations:
                continue

            if is_single_vgpu_fit(gpu, pod) and can_allocate_pod_multi_vgpu(
                gpus,
                pod,
                anchor_gpu_idx=i,
            ):
                adj[i, j] = 1.0

    return (
        np.array(gpu_feats, dtype=np.float32),
        np.array(pod_feats, dtype=np.float32),
        adj,
    )


def calculate_objective(
    success_rate: float,
    balance: float,
    failure_rate: float,
    args,
) -> float:
    return float(
        args.success_weight * success_rate
        - args.balance_weight * balance
        - args.failure_weight * failure_rate
    )


def calculate_step_reward(
    gpus: List[Dict],
    allocated_count: int,
    total_pods: int,
    args,
) -> float:
    current_success = allocated_count / max(total_pods, 1)
    current_failure = 1.0 - current_success
    current_balance = balance_score(gpus)

    return float(
        1.0
        + 0.5 * args.success_weight * current_success
        - 0.2 * args.balance_weight * current_balance
        - 0.1 * args.failure_weight * current_failure
    )


def calculate_terminal_reward(
    gpus: List[Dict],
    allocated_count: int,
    total_pods: int,
    args,
) -> float:
    success = allocated_count / max(total_pods, 1)
    failure = 1.0 - success
    bal = balance_score(gpus)

    return float(
        3.0 * calculate_objective(success, bal, failure, args)
    )


def finalize_metrics(
    method: str,
    scenario: Dict,
    gpus: List[Dict],
    allocations: Dict[str, List[int]],
    args,
    total_reward: float = 0.0,
    steps: int = 0,
) -> Dict:
    pods = scenario["pods"]
    total_pods = len(pods)
    allocated_count = len(allocations)
    failure_count = total_pods - allocated_count

    total_vgpu_count = sum(get_required_vgpu_number(p) for p in pods)
    allocated_vgpu_count = sum(len(v) for v in allocations.values())
    failure_vgpu_count = total_vgpu_count - allocated_vgpu_count

    success_rate = allocated_count / max(total_pods, 1)
    failure_rate = failure_count / max(total_pods, 1)

    vgpu_success_rate = allocated_vgpu_count / max(total_vgpu_count, 1)
    vgpu_failure_rate = failure_vgpu_count / max(total_vgpu_count, 1)

    bal = balance_score(gpus)
    objective = calculate_objective(success_rate, bal, failure_rate, args)

    actual_load, memory_load, core_load = compute_scenario_load(scenario)

    return {
        "method": method,
        "scenario_id": scenario["scenario_id"],
        "target_load": scenario["target_load"],
        "actual_load": actual_load,
        "memory_load": memory_load,
        "core_load": core_load,
        "balance_score": bal,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "vgpu_success_rate": vgpu_success_rate,
        "vgpu_failure_rate": vgpu_failure_rate,
        "allocated_count": allocated_count,
        "failure_count": failure_count,
        "allocated_vgpu_count": allocated_vgpu_count,
        "failure_vgpu_count": failure_vgpu_count,
        "total_vgpu_count": total_vgpu_count,
        "objective": objective,
        "num_gpus": len(scenario["gpus"]),
        "num_pods": total_pods,
        "steps": steps,
        "reward": total_reward,
    }


def run_one_episode(
    agent: GNNBasedDQNAgent,
    scenario: Dict,
    args,
    train: bool = True,
) -> Dict:
    gpus = reset_gpus(scenario["gpus"])
    pods = copy.deepcopy(scenario["pods"])

    allocations: Dict[str, List[int]] = {}
    total_reward = 0.0
    steps = 0

    graph_data = build_vgpu_graph(gpus, pods, allocations)

    while len(allocations) < len(pods):
        _, _, adj = graph_data

        valid_mask = torch.tensor(
            adj,
            dtype=torch.float32,
            device=device,
        ) > 0

        if not valid_mask.any():
            terminal_reward = calculate_terminal_reward(
                gpus=gpus,
                allocated_count=len(allocations),
                total_pods=len(pods),
                args=args,
            )
            total_reward += terminal_reward
            break

        action = agent.act(graph_data, valid_mask)
        gpu_idx, pod_idx = action
        pod = pods[pod_idx]

        selected_gpu_indices = select_gpus_for_pod_by_anchor(
            gpus=gpus,
            pod=pod,
            anchor_gpu_idx=gpu_idx,
        )

        if selected_gpu_indices is None:
            reward = -5.0
            next_graph_data = graph_data
            done = True
        else:
            allocate_pod_to_gpus(gpus, pod, selected_gpu_indices)
            allocations[pod["task_id"]] = selected_gpu_indices

            reward = calculate_step_reward(
                gpus=gpus,
                allocated_count=len(allocations),
                total_pods=len(pods),
                args=args,
            )

            next_graph_data = build_vgpu_graph(gpus, pods, allocations)

            done = (
                len(allocations) == len(pods)
                or not np.any(next_graph_data[2] > 0)
            )

            if done:
                reward += calculate_terminal_reward(
                    gpus=gpus,
                    allocated_count=len(allocations),
                    total_pods=len(pods),
                    args=args,
                )

        total_reward += reward
        steps += 1

        if train:
            agent.remember(
                graph_data=graph_data,
                action=action,
                reward=reward,
                next_graph_data=next_graph_data,
                done=done,
            )

        graph_data = next_graph_data

        if done:
            break

    return finalize_metrics(
        method="dqn",
        scenario=scenario,
        gpus=gpus,
        allocations=allocations,
        args=args,
        total_reward=total_reward,
        steps=steps,
    )


# ============================================================
# 6. Volcano vGPU source-aligned baselines
# ============================================================

BASELINE_METHODS = [
    "volcano-vgpu-binpack",
    "volcano-vgpu-spread",
    "random",
]


def sorted_gpu_indices_by_policy(
    method: str,
    gpus: List[Dict],
) -> List[int]:
    indices = list(range(len(gpus)))

    if method == "volcano-vgpu-binpack":
        # UsedMem larger first; if equal, lower GPU index first.
        indices.sort(
            key=lambda idx: (
                -(gpus[idx]["memory_total"] - gpus[idx]["memory_free"]),
                idx,
            )
        )
        return indices

    if method == "volcano-vgpu-spread":
        # UsedNum smaller first; if equal, lower GPU index first.
        indices.sort(
            key=lambda idx: (
                gpus[idx]["pod_count"],
                idx,
            )
        )
        return indices

    if method == "random":
        random.shuffle(indices)
        return indices

    raise ValueError(f"unknown baseline method: {method}")


def select_gpus_for_pod_by_policy(
    method: str,
    gpus: List[Dict],
    pod: Dict,
) -> Optional[List[int]]:
    """
    Select a GPU combination for one Pod.

    Volcano vGPU style:
        1. Sort devices according to SchedulePolicy.
        2. Scan from high priority to low priority.
        3. If a device can fit one vGPU slice, add it.
        4. Stop when selected count reaches pod.vgpu_number.
        5. If not enough devices are found, allocation fails.

    This is priority-ordered scanning with fit checks.
    It is not simply taking top-N blindly.
    """
    required = get_required_vgpu_number(pod)
    selected = []

    for idx in sorted_gpu_indices_by_policy(method, gpus):
        if len(selected) >= required:
            break

        if is_single_vgpu_fit(gpus[idx], pod):
            selected.append(idx)

    if len(selected) < required:
        return None

    return selected


def select_gpu_by_baseline(
    method: str,
    gpus: List[Dict],
    pod: Dict,
) -> Optional[int]:
    """
    Backward-compatible wrapper.
    New multi-vGPU code should use select_gpus_for_pod_by_policy().
    """
    selected = select_gpus_for_pod_by_policy(method, gpus, pod)

    if not selected:
        return None

    return selected[0]


def run_baseline(
    method: str,
    scenario: Dict,
    args,
) -> Dict:
    gpus = reset_gpus(scenario["gpus"])
    pods = copy.deepcopy(scenario["pods"])

    allocations: Dict[str, List[int]] = {}

    for pod in pods:
        selected_gpu_indices = select_gpus_for_pod_by_policy(
            method=method,
            gpus=gpus,
            pod=pod,
        )

        if selected_gpu_indices is None:
            continue

        allocate_pod_to_gpus(gpus, pod, selected_gpu_indices)
        allocations[pod["task_id"]] = selected_gpu_indices

    return finalize_metrics(
        method=method,
        scenario=scenario,
        gpus=gpus,
        allocations=allocations,
        args=args,
        total_reward=0.0,
        steps=len(allocations),
    )


# ============================================================
# 7. Evaluation helpers
# ============================================================

def evaluate_dqn_on_scenarios(
    agent: GNNBasedDQNAgent,
    scenarios: List[Dict],
    args,
) -> Dict:
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    rows = []

    for scenario in scenarios:
        row = run_one_episode(
            agent=agent,
            scenario=scenario,
            args=args,
            train=False,
        )
        rows.append(row)

    agent.epsilon = old_epsilon

    return {
        "avg_objective": mean([r["objective"] for r in rows]),
        "avg_balance_score": mean([r["balance_score"] for r in rows]),
        "avg_success_rate": mean([r["success_rate"] for r in rows]),
        "avg_failure_rate": mean([r["failure_rate"] for r in rows]),
        "avg_vgpu_success_rate": mean([r["vgpu_success_rate"] for r in rows]),
        "avg_vgpu_failure_rate": mean([r["vgpu_failure_rate"] for r in rows]),
    }


def summarize_detail_rows(rows: List[Dict], target_load: float) -> List[Dict]:
    methods = ["dqn"] + BASELINE_METHODS
    summary_rows = []

    for method in methods:
        group = [r for r in rows if r["method"] == method]

        if not group:
            continue

        summary_rows.append(
            {
                "method": method,
                "target_load": target_load,
                "avg_actual_load": mean([r["actual_load"] for r in group]),
                "avg_memory_load": mean([r["memory_load"] for r in group]),
                "avg_core_load": mean([r["core_load"] for r in group]),
                "avg_balance_score": mean([r["balance_score"] for r in group]),
                "std_balance_score": std([r["balance_score"] for r in group]),
                "avg_success_rate": mean([r["success_rate"] for r in group]),
                "std_success_rate": std([r["success_rate"] for r in group]),
                "avg_failure_rate": mean([r["failure_rate"] for r in group]),
                "std_failure_rate": std([r["failure_rate"] for r in group]),
                "avg_vgpu_success_rate": mean([r["vgpu_success_rate"] for r in group]),
                "avg_vgpu_failure_rate": mean([r["vgpu_failure_rate"] for r in group]),
                "avg_allocated_count": mean([r["allocated_count"] for r in group]),
                "avg_failure_count": mean([r["failure_count"] for r in group]),
                "avg_allocated_vgpu_count": mean([r["allocated_vgpu_count"] for r in group]),
                "avg_failure_vgpu_count": mean([r["failure_vgpu_count"] for r in group]),
                "avg_total_vgpu_count": mean([r["total_vgpu_count"] for r in group]),
                "avg_objective": mean([r["objective"] for r in group]),
                "std_objective": std([r["objective"] for r in group]),
                "avg_num_gpus": mean([r["num_gpus"] for r in group]),
                "avg_num_pods": mean([r["num_pods"] for r in group]),
            }
        )

    return summary_rows


def print_summary_table(summary_rows: List[Dict]):
    columns = [
        "method",
        "target_load",
        "avg_actual_load",
        "avg_memory_load",
        "avg_core_load",
        "avg_balance_score",
        "std_balance_score",
        "avg_success_rate",
        "std_success_rate",
        "avg_failure_rate",
        "avg_vgpu_success_rate",
        "avg_vgpu_failure_rate",
        "avg_allocated_count",
        "avg_failure_count",
        "avg_allocated_vgpu_count",
        "avg_failure_vgpu_count",
        "avg_objective",
        "std_objective",
        "avg_num_gpus",
        "avg_num_pods",
    ]

    print(" ".join([f"{c:>24}" for c in columns]))

    for row in summary_rows:
        values = []

        for c in columns:
            v = row.get(c, "")

            if isinstance(v, float):
                values.append(f"{v:>24.6f}")
            else:
                values.append(f"{str(v):>24}")

        print(" ".join(values))
