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
    heterogeneous: bool = False,
) -> List[Dict]:
    """
    Generate GPU list.

    默认模拟真实单节点：
    - 同一个节点内 GPU 同规格；
    - 初始占用全部为 0。

    不同 scenario 可以通过 memory_choices 随机选择不同节点型号。
    """
    gpus = []

    if heterogeneous:
        for gpu_id in range(num_gpus):
            memory_total = float(random.choice(memory_choices)) * random.uniform(0.92, 1.08)
            gpu_core_total = float(core_total) * random.uniform(0.95, 1.05)

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

    node_memory_total = float(random.choice(memory_choices))
    node_core_total = float(core_total)

    node_memory_total = round(node_memory_total, 4)
    node_core_total = round(node_core_total, 4)

    for gpu_id in range(num_gpus):
        gpus.append(
            {
                "gpu_id": gpu_id,
                "memory_total": node_memory_total,
                "memory_free": node_memory_total,
                "core_total": node_core_total,
                "core_free": node_core_total,
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
    Generate realistic Volcano vGPU job pods.

    设计目标：
    1. 一个 scenario 表示一个 Job；
    2. 一个 Job 内 Pod 来自少量 workload templates；
    3. 同一个 Job 内允许不同大小 Pod 混合；
    4. 加强 spread 的弱点：
       - spread 只看 pod_count；
       - 不看 memory/core 组合；
       - 在高负载、大小 Pod 混合、多 vGPU Pod 下容易失败；
    5. DQN 有机会通过 memory/core/vgpu_number 图特征学出更优策略。
    """

    num_gpus = len(gpus)

    total_gpu_memory = sum(g["memory_total"] for g in gpus)
    total_gpu_core = sum(g["core_total"] for g in gpus)

    max_gpu_memory = max(g["memory_total"] for g in gpus)
    max_gpu_core = max(g["core_total"] for g in gpus)

    # 根据负载控制 Job 规模
    if target_load <= 0.7:
        num_pods = random.randint(min_pods, max(min_pods, (min_pods + max_pods) // 2))
    elif target_load <= 1.0:
        num_pods = random.randint(min_pods, max_pods)
    else:
        num_pods = random.randint(max(min_pods, (min_pods + max_pods) // 2), max_pods)

    # memory 单位自适应：
    # 如果 GPU memory_total 很大，说明用的是 MB；
    # 否则说明用的是 GB。
    if max_gpu_memory > 1024:
        memory_levels = [
            2048,
            4096,
            8192,
            12288,
            16384,
            24576,
            32768,
            40960,
        ]
    else:
        memory_levels = [
            2,
            4,
            8,
            12,
            16,
            24,
            32,
            40,
        ]

    # 不生成超过单卡 80% 的单 slice demand
    memory_levels = [
        m for m in memory_levels
        if m <= max_gpu_memory * 0.80
    ]

    if not memory_levels:
        memory_levels = [max_gpu_memory * 0.25]

    core_levels = [10, 20, 25, 40, 50, 60, 75, 100]
    core_levels = [
        c for c in core_levels
        if c <= max_gpu_core
    ]

    if not core_levels:
        core_levels = [max_gpu_core * 0.25]

    # Job 类型。
    # 重点增加 memory/core 冲突和混合任务，让 spread 不再天然占优。
    job_type = random.choices(
        population=[
            "small_inference",
            "medium_inference",
            "memory_heavy",
            "core_heavy",
            "training",
            "distributed_training",
            "mixed_conflict",
            "near_infeasible",
        ],
        weights=[
            0.15,
            0.15,
            0.15,
            0.15,
            0.15,
            0.10,
            0.20,
            0.15,
        ],
        k=1,
    )[0]

    def choose_from(candidates, fallback):
        if candidates:
            return random.choice(candidates)
        return random.choice(fallback)

    def make_template(kind: str) -> Dict:
        """
        生成一种 Pod 模板。
        memory_demand/core_demand 是 per-vGPU slice 的需求。
        """

        if kind == "small_inference":
            vgpu_number = 1

            mem_candidates = [
                m for m in memory_levels
                if m <= max_gpu_memory * 0.20
            ]

            core_candidates = [
                c for c in core_levels
                if c <= 25
            ]

            memory_demand = choose_from(mem_candidates, memory_levels)
            core_demand = choose_from(core_candidates, core_levels)

        elif kind == "medium_inference":
            vgpu_number = 1

            mem_candidates = [
                m for m in memory_levels
                if max_gpu_memory * 0.10 <= m <= max_gpu_memory * 0.40
            ]

            core_candidates = [
                c for c in core_levels
                if 20 <= c <= 50
            ]

            memory_demand = choose_from(mem_candidates, memory_levels)
            core_demand = choose_from(core_candidates, core_levels)

        elif kind == "memory_heavy":
            # 高显存，低/中 core。
            # spread 只看 pod_count 时，容易把高显存 Pod 放到不合适位置。
            vgpu_number = random.choices(
                [1, 2],
                weights=[0.80, 0.20],
                k=1,
            )[0]
            vgpu_number = min(vgpu_number, num_gpus)

            mem_candidates = [
                m for m in memory_levels
                if m >= max_gpu_memory * 0.35
            ]

            core_candidates = [
                c for c in core_levels
                if c <= 40
            ]

            memory_demand = choose_from(mem_candidates, memory_levels)
            core_demand = choose_from(core_candidates, core_levels)

        elif kind == "core_heavy":
            # 高 core，低/中显存。
            vgpu_number = random.choices(
                [1, 2],
                weights=[0.85, 0.15],
                k=1,
            )[0]
            vgpu_number = min(vgpu_number, num_gpus)

            mem_candidates = [
                m for m in memory_levels
                if m <= max_gpu_memory * 0.30
            ]

            core_candidates = [
                c for c in core_levels
                if c >= 50
            ]

            memory_demand = choose_from(mem_candidates, memory_levels)
            core_demand = choose_from(core_candidates, core_levels)

        elif kind == "training":
            vgpu_number = random.choices(
                [1, 2],
                weights=[0.65, 0.35],
                k=1,
            )[0]
            vgpu_number = min(vgpu_number, num_gpus)

            mem_candidates = [
                m for m in memory_levels
                if max_gpu_memory * 0.20 <= m <= max_gpu_memory * 0.60
            ]

            core_candidates = [
                c for c in core_levels
                if 40 <= c <= 100
            ]

            memory_demand = choose_from(mem_candidates, memory_levels)
            core_demand = choose_from(core_candidates, core_levels)

        elif kind == "distributed_training":
            vgpu_number = random.choices(
                [2, 4],
                weights=[0.70, 0.30],
                k=1,
            )[0]
            vgpu_number = min(vgpu_number, num_gpus)

            mem_candidates = [
                m for m in memory_levels
                if max_gpu_memory * 0.15 <= m <= max_gpu_memory * 0.50
            ]

            core_candidates = [
                c for c in core_levels
                if 25 <= c <= 75
            ]

            memory_demand = choose_from(mem_candidates, memory_levels)
            core_demand = choose_from(core_candidates, core_levels)

        elif kind == "near_infeasible":
            # 接近不可满足：单 slice 需求偏大。
            # 用于训练模型在高负载时保留关键资源。
            vgpu_number = random.choices(
                [1, 2],
                weights=[0.75, 0.25],
                k=1,
            )[0]
            vgpu_number = min(vgpu_number, num_gpus)

            mem_candidates = [
                m for m in memory_levels
                if m >= max_gpu_memory * 0.45
            ]

            core_candidates = [
                c for c in core_levels
                if c >= 60
            ]

            memory_demand = choose_from(mem_candidates, memory_levels)
            core_demand = choose_from(core_candidates, core_levels)

        else:
            # mixed_conflict：
            # 这里会在外面混合 memory_heavy/core_heavy/training 等模板。
            vgpu_number = random.choices(
                [1, 2, 4],
                weights=[0.65, 0.25, 0.10],
                k=1,
            )[0]
            vgpu_number = min(vgpu_number, num_gpus)

            memory_demand = random.choice(memory_levels)
            core_demand = random.choice(core_levels)

        return {
            "vgpu_number": int(vgpu_number),
            "memory_demand": float(memory_demand),
            "core_demand": float(core_demand),
        }

    # 一个 Job 里通常不是每个 Pod 都独立随机，而是来自少量模板。
    if job_type == "mixed_conflict":
        template_kinds = random.sample(
            ["memory_heavy", "core_heavy", "training", "medium_inference"],
            k=random.choice([2, 3]),
        )
    elif job_type == "near_infeasible":
        # 接近不可满足场景中，也混入一些小 Pod，避免全是巨型 Pod。
        template_kinds = ["near_infeasible", random.choice(["small_inference", "medium_inference"])]
    else:
        template_kinds = [job_type]

        if random.random() < 0.35:
            template_kinds.append(
                random.choice(["small_inference", "medium_inference", "memory_heavy", "core_heavy"])
            )

    templates = [
        make_template(kind)
        for kind in template_kinds
    ]

    pods = []

    for i in range(num_pods):
        tpl = random.choice(templates)

        pods.append(
            {
                "task_id": f"{scenario_id}-pod-{i}",
                "vgpu_number": int(tpl["vgpu_number"]),
                "memory_demand": round(float(tpl["memory_demand"]), 4),
                "core_demand": round(float(tpl["core_demand"]), 4),
            }
        )

    def calc_load(pod_list):
        total_mem = sum(
            p.get("vgpu_number", 1) * p["memory_demand"]
            for p in pod_list
        )

        total_core = sum(
            p.get("vgpu_number", 1) * p["core_demand"]
            for p in pod_list
        )

        memory_load = total_mem / max(total_gpu_memory, 1e-8)
        core_load = total_core / max(total_gpu_core, 1e-8)
        actual_load = max(memory_load, core_load)

        return actual_load, memory_load, core_load

    # 控制负载落在目标附近。
    # 高负载允许更宽，因为真实高负载 Job 本来就更容易失败。
    if target_load <= 1.0:
        lower = target_load * 0.80
        upper = target_load * 1.20
    else:
        lower = target_load * 0.85
        upper = target_load * 1.25

    # 调整 Pod 数量，让总需求接近 target_load。
    for _ in range(80):
        actual_load, _, _ = calc_load(pods)

        if lower <= actual_load <= upper:
            break

        if actual_load > upper and len(pods) > min_pods:
            pods.pop(random.randrange(len(pods)))
            continue

        if actual_load < lower and len(pods) < max_pods:
            tpl = random.choice(templates)
            idx = len(pods)

            pods.append(
                {
                    "task_id": f"{scenario_id}-pod-{idx}",
                    "vgpu_number": int(tpl["vgpu_number"]),
                    "memory_demand": round(float(tpl["memory_demand"]), 4),
                    "core_demand": round(float(tpl["core_demand"]), 4),
                }
            )
            continue

        break

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
    heterogeneous_gpus: bool = False,
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
        gpu.setdefault("memory_free", gpu["memory_total"])
        gpu.setdefault("core_free", gpu["core_total"])
        gpu.setdefault("pod_count", 0)

        core_used_ratio = 1.0 - gpu["core_free"] / max(gpu["core_total"], 1e-8)
        gpu["util"] = gpu.get("util", max(0.0, min(100.0, core_used_ratio * 100.0)))

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

    def q_matrix(self, graph_data, valid_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        gpu_feats, pod_feats, adj = graph_data

        gpu_feats_t = torch.tensor(gpu_feats, dtype=torch.float32, device=device)
        pod_feats_t = torch.tensor(pod_feats, dtype=torch.float32, device=device)
        adj_t = torch.tensor(adj, dtype=torch.float32, device=device)

        with torch.no_grad():
            gpu_emb, pod_emb = self.gnn_encoder(gpu_feats_t, pod_feats_t, adj_t)
            q_mat = self.q_net(gpu_emb, pod_emb)

            if valid_mask is not None:
                q_mat = q_mat.clone()
                q_mat[~valid_mask] = -1e9

        return q_mat.detach().cpu().numpy()

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
    use_job_features = not bool(getattr(args, "disable_job_features", False))

    return GNNBasedDQNAgent(
        gpu_feat_dim=17 if use_job_features else 8,
        pod_feat_dim=15 if use_job_features else 6,
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


def _gpu_usage_ratios(gpus: List[Dict]) -> Tuple[List[float], List[float]]:
    memory_usages = [
        1.0 - g["memory_free"] / max(g["memory_total"], 1e-8)
        for g in gpus
    ]

    core_usages = [
        1.0 - g["core_free"] / max(g["core_total"], 1e-8)
        for g in gpus
    ]

    return memory_usages, core_usages


def inter_gpu_balance_score(gpus: List[Dict]) -> float:
    """
    Balance between GPUs.

    Lower is better: all GPUs have similar memory/core usage ratios.
    """
    memory_usages, core_usages = _gpu_usage_ratios(gpus)
    return float(np.std(memory_usages) + np.std(core_usages))


def intra_gpu_balance_score(gpus: List[Dict]) -> float:
    """
    Balance inside each GPU between memory and core pressure.

    Lower is better: a GPU using 80% memory and 80% core is more balanced than
    a GPU using 80% memory and 10% core.
    """
    memory_usages, core_usages = _gpu_usage_ratios(gpus)

    if not memory_usages:
        return 0.0

    diffs = [
        abs(memory_usage - core_usage)
        for memory_usage, core_usage in zip(memory_usages, core_usages)
    ]

    return float(mean(diffs))


def balance_score(gpus: List[Dict], args=None) -> float:
    inter_weight = 1.0
    intra_weight = 1.0

    if args is not None:
        inter_weight = float(getattr(args, "inter_balance_weight", inter_weight))
        intra_weight = float(getattr(args, "intra_balance_weight", intra_weight))

    return float(
        inter_weight * inter_gpu_balance_score(gpus)
        + intra_weight * intra_gpu_balance_score(gpus)
    )


def select_gpus_for_pod_by_anchor(
    gpus: List[Dict],
    pod: Dict,
    anchor_gpu_idx: int,
    args=None,
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

        score = balance_score(trial_gpus, args=args)
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


def select_dqn_action(
    agent: GNNBasedDQNAgent,
    graph_data,
    valid_mask: torch.Tensor,
    gpus: List[Dict],
    pods: List[Dict],
    args,
    train: bool,
) -> Tuple[int, int]:
    topk = int(getattr(args, "action_rerank_topk", 0))
    use_train_rerank = bool(getattr(args, "train_action_rerank", False))

    if topk <= 0 or (train and not use_train_rerank):
        return agent.act(graph_data, valid_mask)

    if train and random.random() < agent.epsilon:
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)

        if valid_indices.size(0) == 0:
            return 0, 0

        rand_idx = random.randint(0, valid_indices.size(0) - 1)
        gpu_idx, pod_idx = valid_indices[rand_idx].tolist()
        return int(gpu_idx), int(pod_idx)

    q_mat = agent.q_matrix(graph_data, valid_mask=valid_mask)
    valid_np = valid_mask.detach().cpu().numpy().astype(bool)

    valid_flat = np.flatnonzero(valid_np.reshape(-1))
    if len(valid_flat) == 0:
        return 0, 0

    flat_q = q_mat.reshape(-1)
    candidate_count = min(topk, len(valid_flat))
    candidate_flat = valid_flat[
        np.argpartition(-flat_q[valid_flat], candidate_count - 1)[:candidate_count]
    ]

    q_values = flat_q[candidate_flat]
    q_min = float(np.min(q_values))
    q_max = float(np.max(q_values))
    q_den = max(q_max - q_min, 1e-8)

    q_weight = float(getattr(args, "action_rerank_q_weight", 1.0))
    balance_weight = float(getattr(args, "action_rerank_balance_weight", 1.0))
    intra_weight = float(getattr(args, "action_rerank_intra_weight", 0.0))
    inter_weight = float(getattr(args, "action_rerank_inter_weight", 0.0))

    best_action = None
    best_score = -float("inf")
    best_tie_key = None
    num_pods = q_mat.shape[1]

    for flat_idx in candidate_flat:
        gpu_idx = int(flat_idx // num_pods)
        pod_idx = int(flat_idx % num_pods)
        pod = pods[pod_idx]

        selected_gpu_indices = select_gpus_for_pod_by_anchor(
            gpus=gpus,
            pod=pod,
            anchor_gpu_idx=gpu_idx,
            args=args,
        )

        if selected_gpu_indices is None:
            continue

        trial_gpus = copy.deepcopy(gpus)

        try:
            allocate_pod_to_gpus(trial_gpus, pod, selected_gpu_indices)
        except ValueError:
            continue

        q_norm = (float(flat_q[flat_idx]) - q_min) / q_den
        post_balance = balance_score(trial_gpus, args=args)
        post_inter = inter_gpu_balance_score(trial_gpus)
        post_intra = intra_gpu_balance_score(trial_gpus)

        score = (
            q_weight * q_norm
            - balance_weight * post_balance
            - inter_weight * post_inter
            - intra_weight * post_intra
        )
        tie_key = (post_balance, post_intra, post_inter, gpu_idx, pod_idx)

        if best_action is None:
            best_action = (gpu_idx, pod_idx)
            best_score = score
            best_tie_key = tie_key
        elif score > best_score + 1e-12:
            best_action = (gpu_idx, pod_idx)
            best_score = score
            best_tie_key = tie_key
        elif abs(score - best_score) <= 1e-12 and tie_key < best_tie_key:
            best_action = (gpu_idx, pod_idx)
            best_score = score
            best_tie_key = tie_key

    if best_action is None:
        return agent.act(graph_data, valid_mask)

    return best_action


def build_vgpu_graph(
    gpus: List[Dict],
    pods: List[Dict],
    allocations: Dict[str, List[int]],
    args=None,
):
    max_memory = max(g["memory_total"] for g in gpus)
    max_core = max(g["core_total"] for g in gpus)
    max_pods = max(1, len(pods))
    unallocated_pods = [
        pod for pod in pods
        if pod["task_id"] not in allocations
    ]

    use_job_features = not bool(getattr(args, "disable_job_features", False))

    if use_job_features and unallocated_pods:
        pod_memory_ratios = np.array(
            [p["memory_demand"] / max_memory for p in unallocated_pods],
            dtype=np.float32,
        )
        pod_core_ratios = np.array(
            [p["core_demand"] / max_core for p in unallocated_pods],
            dtype=np.float32,
        )
        pod_gaps = np.abs(pod_memory_ratios - pod_core_ratios)
        pod_signed_gaps = pod_memory_ratios - pod_core_ratios

        job_feats = [
            float(np.mean(pod_memory_ratios)),
            float(np.mean(pod_core_ratios)),
            float(np.std(pod_memory_ratios)),
            float(np.std(pod_core_ratios)),
            float(np.mean(pod_gaps)),
            float(np.max(pod_gaps)),
            float(np.mean(pod_signed_gaps > 0.15)),
            float(np.mean(pod_signed_gaps < -0.15)),
            float(len(unallocated_pods) / max_pods),
        ]
    else:
        job_feats = [0.0] * 9

    gpu_feats = []

    for gpu in gpus:
        memory_used_ratio = 1.0 - gpu["memory_free"] / max(gpu["memory_total"], 1e-8)
        core_used_ratio = 1.0 - gpu["core_free"] / max(gpu["core_total"], 1e-8)
        pod_count_ratio = gpu["pod_count"] / max_pods
        util_ratio = gpu["util"] / 100.0
        memory_free_ratio = gpu["memory_free"] / max_memory
        core_free_ratio = gpu["core_free"] / max_core
        memory_core_gap = abs(memory_used_ratio - core_used_ratio)
        memory_minus_core = memory_used_ratio - core_used_ratio

        feat = [
                memory_used_ratio,
                core_used_ratio,
                pod_count_ratio,
                util_ratio,
                memory_free_ratio,
                core_free_ratio,
                memory_core_gap,
                memory_minus_core,
        ]

        if use_job_features:
            feat.extend(job_feats)

        gpu_feats.append(feat)

    pod_feats = []

    for pod in pods:
        allocated_flag = 1.0 if pod["task_id"] in allocations else 0.0
        pod_memory_ratio = pod["memory_demand"] / max_memory
        pod_core_ratio = pod["core_demand"] / max_core
        pod_memory_core_gap = abs(pod_memory_ratio - pod_core_ratio)
        pod_memory_minus_core = pod_memory_ratio - pod_core_ratio

        feat = [
                pod_memory_ratio,
                pod_core_ratio,
                get_required_vgpu_number(pod) / max(1.0, len(gpus)),
                allocated_flag,
                pod_memory_core_gap,
                pod_memory_minus_core,
        ]

        if use_job_features:
            feat.extend(job_feats)

        pod_feats.append(feat)

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
    previous_balance: Optional[float] = None,
    previous_inter_balance: Optional[float] = None,
    previous_intra_balance: Optional[float] = None,
) -> float:
    current_success = allocated_count / max(total_pods, 1)
    current_failure = 1.0 - current_success
    current_balance = balance_score(gpus, args=args)
    current_inter_balance = inter_gpu_balance_score(gpus)
    current_intra_balance = intra_gpu_balance_score(gpus)
    balance_delta = 0.0
    inter_balance_delta = 0.0
    intra_balance_delta = 0.0

    if previous_balance is not None:
        balance_delta = previous_balance - current_balance

    if previous_inter_balance is not None:
        inter_balance_delta = previous_inter_balance - current_inter_balance

    if previous_intra_balance is not None:
        intra_balance_delta = previous_intra_balance - current_intra_balance

    delta_inter_weight = getattr(args, "delta_inter_balance_weight", None)
    delta_intra_weight = getattr(args, "delta_intra_balance_weight", None)

    if delta_inter_weight is None and delta_intra_weight is None:
        delta_reward = getattr(args, "delta_balance_weight", 1.0) * balance_delta
    else:
        if delta_inter_weight is None:
            delta_inter_weight = getattr(args, "delta_balance_weight", 1.0)
        if delta_intra_weight is None:
            delta_intra_weight = getattr(args, "delta_balance_weight", 1.0)
        delta_reward = (
            float(delta_inter_weight) * inter_balance_delta
            + float(delta_intra_weight) * intra_balance_delta
        )

    return float(
        1.0
        + 0.5 * args.success_weight * current_success
        + delta_reward
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
    bal = balance_score(gpus, args=args)

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

    inter_bal = inter_gpu_balance_score(gpus)
    intra_bal = intra_gpu_balance_score(gpus)
    bal = balance_score(gpus, args=args)
    objective = calculate_objective(success_rate, bal, failure_rate, args)

    actual_load, memory_load, core_load = compute_scenario_load(scenario)

    return {
        "method": method,
        "scenario_id": scenario["scenario_id"],
        "workload_type": scenario.get("workload_type", "random"),
        "target_load": scenario["target_load"],
        "actual_load": actual_load,
        "memory_load": memory_load,
        "core_load": core_load,
        "balance_score": bal,
        "inter_gpu_balance_score": inter_bal,
        "intra_gpu_balance_score": intra_bal,
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

    graph_data = build_vgpu_graph(gpus, pods, allocations, args=args)

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

        action = select_dqn_action(
            agent=agent,
            graph_data=graph_data,
            valid_mask=valid_mask,
            gpus=gpus,
            pods=pods,
            args=args,
            train=train,
        )
        gpu_idx, pod_idx = action
        pod = pods[pod_idx]

        selected_gpu_indices = select_gpus_for_pod_by_anchor(
            gpus=gpus,
            pod=pod,
            anchor_gpu_idx=gpu_idx,
            args=args,
        )

        if selected_gpu_indices is None:
            reward = -5.0
            next_graph_data = graph_data
            done = True
        else:
            previous_balance = balance_score(gpus, args=args)
            previous_inter_balance = inter_gpu_balance_score(gpus)
            previous_intra_balance = intra_gpu_balance_score(gpus)

            allocate_pod_to_gpus(gpus, pod, selected_gpu_indices)
            allocations[pod["task_id"]] = selected_gpu_indices

            reward = calculate_step_reward(
                gpus=gpus,
                allocated_count=len(allocations),
                total_pods=len(pods),
                args=args,
                previous_balance=previous_balance,
                previous_inter_balance=previous_inter_balance,
                previous_intra_balance=previous_intra_balance,
            )

            next_graph_data = build_vgpu_graph(gpus, pods, allocations, args=args)

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
    "used-mem-desc",
    "used-mem-asc",
    "index-desc",
    "random",
    "greedy-balance",
    "greedy-objective",
]


def sorted_gpu_indices_by_policy(
    method: str,
    gpus: List[Dict],
) -> List[int]:
    indices = list(range(len(gpus)))

    if method == "used-mem-desc":
        # 类似 binpack:
        # 已使用显存越多，优先级越高。
        # 目标是尽量把任务继续塞到已经使用较多的 GPU 上。
        indices.sort(
            key=lambda idx: (
                -(gpus[idx]["memory_total"] - gpus[idx]["memory_free"]),
                idx,
            )
        )
        return indices

    if method == "used-mem-asc":
        # 类似 spread:
        # 已使用显存越少，优先级越高。
        # 目标是优先选择更空闲的 GPU。
        indices.sort(
            key=lambda idx: (
                gpus[idx]["memory_total"] - gpus[idx]["memory_free"],
                idx,
            )
        )
        return indices

    if method == "index-desc":
        # 模拟 Volcano vGPU 源码中节点内设备扫描顺序：
        # 从高索引 GPU 向低索引 GPU 依次尝试。
        indices.sort(reverse=True)
        return indices

    if method == "index-asc":
        # 可选：从低索引到高索引扫描。
        indices.sort()
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

    if method in {"greedy-balance", "greedy-objective"}:
        return run_greedy_baseline(
            method=method,
            scenario=scenario,
            gpus=gpus,
            pods=pods,
            allocations=allocations,
            args=args,
        )

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


def run_greedy_baseline(
    method: str,
    scenario: Dict,
    gpus: List[Dict],
    pods: List[Dict],
    allocations: Dict[str, List[int]],
    args,
) -> Dict:
    total_pods = len(pods)
    steps = 0

    while len(allocations) < total_pods:
        best_candidate = None
        best_score = -float("inf")
        best_tie_key = None

        for pod_idx, pod in enumerate(pods):
            if pod["task_id"] in allocations:
                continue

            feasible = get_feasible_gpus_for_pod(gpus, pod)

            for anchor_gpu_idx in feasible:
                selected_gpu_indices = select_gpus_for_pod_by_anchor(
                    gpus=gpus,
                    pod=pod,
                    anchor_gpu_idx=anchor_gpu_idx,
                    args=args,
                )

                if selected_gpu_indices is None:
                    continue

                trial_gpus = copy.deepcopy(gpus)

                try:
                    allocate_pod_to_gpus(trial_gpus, pod, selected_gpu_indices)
                except ValueError:
                    continue

                post_balance = balance_score(trial_gpus, args=args)
                post_inter = inter_gpu_balance_score(trial_gpus)
                post_intra = intra_gpu_balance_score(trial_gpus)

                if method == "greedy-balance":
                    score = -post_balance
                elif method == "greedy-objective":
                    success = (len(allocations) + 1) / max(total_pods, 1)
                    failure = 1.0 - success
                    remaining_pods = [
                        p for p in pods
                        if p["task_id"] not in allocations
                        and p["task_id"] != pod["task_id"]
                    ]
                    future_feasible = 0

                    for remaining_pod in remaining_pods:
                        if any(
                            select_gpus_for_pod_by_anchor(
                                gpus=trial_gpus,
                                pod=remaining_pod,
                                anchor_gpu_idx=idx,
                                args=args,
                            ) is not None
                            for idx in get_feasible_gpus_for_pod(trial_gpus, remaining_pod)
                        ):
                            future_feasible += 1

                    future_feasible_ratio = future_feasible / max(len(remaining_pods), 1)
                    score = (
                        calculate_objective(success, post_balance, failure, args)
                        + 0.25 * future_feasible_ratio
                    )
                else:
                    raise ValueError(f"unknown greedy baseline method: {method}")

                tie_key = (
                    post_balance,
                    post_intra,
                    post_inter,
                    pod_idx,
                    tuple(selected_gpu_indices),
                )

                if best_candidate is None:
                    best_candidate = (pod, selected_gpu_indices)
                    best_score = score
                    best_tie_key = tie_key
                elif score > best_score + 1e-12:
                    best_candidate = (pod, selected_gpu_indices)
                    best_score = score
                    best_tie_key = tie_key
                elif abs(score - best_score) <= 1e-12 and tie_key < best_tie_key:
                    best_candidate = (pod, selected_gpu_indices)
                    best_score = score
                    best_tie_key = tie_key

        if best_candidate is None:
            break

        pod, selected_gpu_indices = best_candidate
        allocate_pod_to_gpus(gpus, pod, selected_gpu_indices)
        allocations[pod["task_id"]] = selected_gpu_indices
        steps += 1

    return finalize_metrics(
        method=method,
        scenario=scenario,
        gpus=gpus,
        allocations=allocations,
        args=args,
        total_reward=0.0,
        steps=steps,
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

    def add_group_metrics(result: Dict, prefix: str, group_rows: List[Dict]):
        if not group_rows:
            group_rows = rows

        result[f"{prefix}avg_objective"] = mean([r["objective"] for r in group_rows])
        result[f"{prefix}avg_balance_score"] = mean([r["balance_score"] for r in group_rows])
        result[f"{prefix}avg_inter_gpu_balance_score"] = mean(
            [r["inter_gpu_balance_score"] for r in group_rows]
        )
        result[f"{prefix}avg_intra_gpu_balance_score"] = mean(
            [r["intra_gpu_balance_score"] for r in group_rows]
        )
        result[f"{prefix}avg_success_rate"] = mean([r["success_rate"] for r in group_rows])
        result[f"{prefix}avg_failure_rate"] = mean([r["failure_rate"] for r in group_rows])
        result[f"{prefix}avg_vgpu_success_rate"] = mean(
            [r["vgpu_success_rate"] for r in group_rows]
        )
        result[f"{prefix}avg_vgpu_failure_rate"] = mean(
            [r["vgpu_failure_rate"] for r in group_rows]
        )

    result = {}
    add_group_metrics(result, "", rows)

    conflict_rows = [
        r for r in rows
        if r.get("workload_type") == "mixed_conflict"
    ]
    lowmid_threshold = float(getattr(args, "checkpoint_lowmid_load_threshold", 1.0))
    lowmid_rows = [
        r for r in rows
        if float(r.get("actual_load", 0.0)) <= lowmid_threshold
    ]

    add_group_metrics(result, "conflict_", conflict_rows)
    add_group_metrics(result, "lowmid_", lowmid_rows)

    result["conflict_count"] = len(conflict_rows)
    result["lowmid_count"] = len(lowmid_rows)
    result["eval_count"] = len(rows)

    return result


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
