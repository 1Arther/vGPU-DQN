"""
hami-core vGPU scenario generator.

一个 scenario 表示一个完整调度场景：

{
    "batch_id": 0,
    "target_load": 1.0,
    "actual_load": 1.03,
    "memory_load": 0.87,
    "core_load": 1.03,
    "num_gpus": 3,
    "num_pods": 25,
    "gpus": [...],
    "pods": [...]
}

新增改动：
1. 将 GPU batch 和 Pod batch 统一组合成 scenario；
2. 支持 target_load 控制实验；
3. 保存 actual_load / memory_load / core_load，方便实验分析。
"""

import json
import os
from typing import Dict, List, Optional

from DQN2.vgpu_gpu_generator import generate_gpu_batch
from DQN2.vgpu_pod_generator import (
    generate_pod_batch,
    generate_pod_batch_by_target_load,
)


def compute_load_info(gpus: List[Dict], pods: List[Dict]) -> Dict:
    total_gpu_memory = sum(gpu["memory_total"] for gpu in gpus)
    total_gpu_core = sum(gpu["core_total"] for gpu in gpus)

    total_pod_memory = sum(pod["memory_demand"] for pod in pods)
    total_pod_core = sum(pod["core_demand"] for pod in pods)

    memory_load = total_pod_memory / total_gpu_memory if total_gpu_memory > 0 else 0.0
    core_load = total_pod_core / total_gpu_core if total_gpu_core > 0 else 0.0
    actual_load = max(memory_load, core_load)

    return {
        "memory_load": memory_load,
        "core_load": core_load,
        "actual_load": actual_load,
    }


def generate_scenario_batch(
    batch_id: int,
    min_gpus: int,
    max_gpus: int,
    min_pods: int,
    max_pods: int,
    gpu_memory_choices: Optional[List[int]] = None,
    gpu_core_choices: Optional[List[int]] = None,
    pod_memory_choices: Optional[List[int]] = None,
    pod_core_choices: Optional[List[int]] = None,
    target_load: float = 0.0,
) -> Dict:
    gpus = generate_gpu_batch(
        batch_id=batch_id,
        min_gpus=min_gpus,
        max_gpus=max_gpus,
        memory_choices=gpu_memory_choices,
        core_choices=gpu_core_choices,
    )

    if target_load and target_load > 0:
        pods = generate_pod_batch_by_target_load(
            batch_id=batch_id,
            gpus=gpus,
            target_load=target_load,
            min_pods=min_pods,
            max_pods=max_pods,
            memory_choices=pod_memory_choices,
            core_choices=pod_core_choices,
        )
    else:
        pods = generate_pod_batch(
            batch_id=batch_id,
            min_pods=min_pods,
            max_pods=max_pods,
            memory_choices=pod_memory_choices,
            core_choices=pod_core_choices,
        )

    load_info = compute_load_info(gpus, pods)

    return {
        "batch_id": batch_id,
        "mode": "hami-core",
        "target_load": target_load,
        "actual_load": load_info["actual_load"],
        "memory_load": load_info["memory_load"],
        "core_load": load_info["core_load"],
        "num_gpus": len(gpus),
        "num_pods": len(pods),
        "gpus": gpus,
        "pods": pods,
    }


def generate_scenarios(
    num_batches: int,
    min_gpus: int,
    max_gpus: int,
    min_pods: int,
    max_pods: int,
    gpu_memory_choices: Optional[List[int]] = None,
    gpu_core_choices: Optional[List[int]] = None,
    pod_memory_choices: Optional[List[int]] = None,
    pod_core_choices: Optional[List[int]] = None,
    target_load: float = 0.0,
) -> List[Dict]:
    return [
        generate_scenario_batch(
            batch_id=batch_id,
            min_gpus=min_gpus,
            max_gpus=max_gpus,
            min_pods=min_pods,
            max_pods=max_pods,
            gpu_memory_choices=gpu_memory_choices,
            gpu_core_choices=gpu_core_choices,
            pod_memory_choices=pod_memory_choices,
            pod_core_choices=pod_core_choices,
            target_load=target_load,
        )
        for batch_id in range(num_batches)
    ]


def save_scenarios(scenarios: List[Dict], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)


def load_scenarios(load_path: str) -> List[Dict]:
    with open(load_path, "r", encoding="utf-8") as f:
        return json.load(f)