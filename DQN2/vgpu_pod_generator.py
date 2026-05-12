"""
hami-core vGPU Pod generator.

这个文件负责生成 Pod 请求。

新改动：
1. 每个 batch 的 Pod 数量可以随机；
2. 每个 Pod 的 vgpu-memory / vgpu-cores 随机；
3. task_id 带 batch_id，避免不同 batch 的 pod 名字重复；
4. 当前默认 vgpu_number=1，表示单 Pod 使用一个 vGPU 切片。
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
    fixed_num_pods: Optional[int] = None,
) -> List[Dict]:
    """
    生成一个 Pod batch。

    如果 fixed_num_pods 不为空，则生成固定数量 Pod；
    否则在 [min_pods, max_pods] 内随机。
    """
    if memory_choices is None:
        memory_choices = [1024, 2048, 4096, 6144, 8192]

    if core_choices is None:
        core_choices = [5, 10, 15, 20, 25]

    if fixed_num_pods is not None:
        num_pods = fixed_num_pods
    else:
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


def generate_pod_batches(
    num_batches: int,
    min_pods: int = 20,
    max_pods: int = 80,
    memory_choices: Optional[List[int]] = None,
    core_choices: Optional[List[int]] = None,
    num_pods_per_batch: Optional[int] = None,
) -> List[List[Dict]]:
    """
    生成多个 Pod batch。

    兼容旧参数 num_pods_per_batch：
        如果传入，则每批固定 Pod 数量。
    """
    return [
        generate_pod_batch(
            batch_id=batch_id,
            min_pods=min_pods,
            max_pods=max_pods,
            memory_choices=memory_choices,
            core_choices=core_choices,
            fixed_num_pods=num_pods_per_batch,
        )
        for batch_id in range(num_batches)
    ]


def save_pod_batches(pod_batches: List[List[Dict]], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pod_batches, f, ensure_ascii=False, indent=2)


def load_pod_batches(load_path: str) -> List[List[Dict]]:
    with open(load_path, "r", encoding="utf-8") as f:
        return json.load(f)