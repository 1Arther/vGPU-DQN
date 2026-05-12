"""
hami-core vGPU scenario generator.

新文件。

作用：
    把 GPU batch 和 Pod batch 组合成 scenario。

一个 scenario 表示一个完整调度测试场景：
    {
        batch_id,
        num_gpus,
        num_pods,
        gpus,
        pods
    }

新改动：
1. 训练集和测试集都保存为 scenario；
2. 每个 scenario 的 GPU 和 Pod 都可以随机；
3. 测试 50 个 batch 时，每个 batch 都有自己的 GPU 和 Pod。
"""

import json
import os
from typing import Dict, List, Optional

from DQN2.vgpu_gpu_generator import generate_gpu_batch
from DQN2.vgpu_pod_generator import generate_pod_batch


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
) -> Dict:
    gpus = generate_gpu_batch(
        batch_id=batch_id,
        min_gpus=min_gpus,
        max_gpus=max_gpus,
        memory_choices=gpu_memory_choices,
        core_choices=gpu_core_choices,
    )

    pods = generate_pod_batch(
        batch_id=batch_id,
        min_pods=min_pods,
        max_pods=max_pods,
        memory_choices=pod_memory_choices,
        core_choices=pod_core_choices,
    )

    return {
        "batch_id": batch_id,
        "mode": "hami-core",
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