"""
hami-core vGPU Pod generator.

用于生成 Pod 任务请求。

新增改动：
1. 支持普通随机 Pod batch；
2. 支持按 target_load 生成 Pod batch；
3. target_load 用于实验 3：负载强度控制实验。
"""

import json
import os
import random
from typing import Dict, List, Optional


def generate_pod_batch(
    batch_id: int,
    min_pods: int = 20,
    max_pods: int = 80,
    memory_choices: Optional[List[int]] = None,
    core_choices: Optional[List[int]] = None,
) -> List[Dict]:
    """
    普通随机 Pod batch。
    """
    if memory_choices is None or len(memory_choices) == 0:
        memory_choices = [1024, 2048, 4096, 6144, 8192]

    if core_choices is None or len(core_choices) == 0:
        core_choices = [5, 10, 15, 20, 25]

    if min_pods <= 0 or max_pods < min_pods:
        raise ValueError("invalid Pod range")

    num_pods = random.randint(min_pods, max_pods)
    pods = []

    for i in range(num_pods):
        pods.append(
            {
                "task_id": f"batch-{batch_id}-pod-{i}",
                "batch_id": batch_id,
                "mode": "hami-core",
                "vgpu_number": 1,
                "memory_demand": random.choice(memory_choices),
                "core_demand": random.choice(core_choices),
            }
        )

    return pods


def generate_pod_batch_by_target_load(
    batch_id: int,
    gpus: List[Dict],
    target_load: float,
    min_pods: int = 1,
    max_pods: int = 200,
    memory_choices: Optional[List[int]] = None,
    core_choices: Optional[List[int]] = None,
) -> List[Dict]:
    """
    按目标负载强度生成 Pod。

    target_load 定义：
        max(total_pod_memory / total_gpu_memory,
            total_pod_core / total_gpu_core)

    例如：
        target_load = 0.8 表示 Pod 总需求大约达到 GPU 总资源的 80%
        target_load = 1.2 表示轻度超载
        target_load = 1.5 表示重度超载

    注意：
        因为 Pod 是离散生成的，最终 actual_load 会略高于 target_load。
    """
    if target_load <= 0:
        return generate_pod_batch(
            batch_id=batch_id,
            min_pods=min_pods,
            max_pods=max_pods,
            memory_choices=memory_choices,
            core_choices=core_choices,
        )

    if memory_choices is None or len(memory_choices) == 0:
        memory_choices = [1024, 2048, 4096, 6144, 8192]

    if core_choices is None or len(core_choices) == 0:
        core_choices = [5, 10, 15, 20, 25]

    if min_pods <= 0 or max_pods < min_pods:
        raise ValueError("invalid Pod range")

    total_gpu_memory = sum(gpu["memory_total"] for gpu in gpus)
    total_gpu_core = sum(gpu["core_total"] for gpu in gpus)

    pods = []
    total_pod_memory = 0
    total_pod_core = 0

    while len(pods) < max_pods:
        memory_demand = random.choice(memory_choices)
        core_demand = random.choice(core_choices)

        pod = {
            "task_id": f"batch-{batch_id}-pod-{len(pods)}",
            "batch_id": batch_id,
            "mode": "hami-core",
            "vgpu_number": 1,
            "memory_demand": memory_demand,
            "core_demand": core_demand,
        }

        pods.append(pod)
        total_pod_memory += memory_demand
        total_pod_core += core_demand

        memory_load = total_pod_memory / total_gpu_memory
        core_load = total_pod_core / total_gpu_core
        dominant_load = max(memory_load, core_load)

        if len(pods) >= min_pods and dominant_load >= target_load:
            break

    return pods


def save_pod_batches(pod_batches: List[List[Dict]], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pod_batches, f, ensure_ascii=False, indent=2)


def load_pod_batches(load_path: str) -> List[List[Dict]]:
    with open(load_path, "r", encoding="utf-8") as f:
        return json.load(f)