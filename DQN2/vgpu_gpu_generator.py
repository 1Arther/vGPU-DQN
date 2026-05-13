"""
hami-core vGPU GPU generator.

用于生成单节点内的 GPU 资源配置。

新增改动：
1. 支持每个 batch 随机生成不同数量 GPU；
2. 支持 GPU 显存和 core 异构；
3. 提供 reset_gpus_from_template()，保证 DQN 和 baseline 使用同一初始 GPU 状态。
"""

import copy
import json
import os
import random
from typing import Dict, List, Optional


def generate_gpus(
    num_gpus: int = 3,
    memory_total: int = 24576,
    core_total: int = 100,
) -> List[Dict]:
    """
    兼容旧接口：生成固定数量、同构 GPU。
    """
    gpus = []

    for i in range(num_gpus):
        gpus.append(
            {
                "gpu_id": i,
                "memory_total": memory_total,
                "memory_free": memory_total,
                "core_total": core_total,
                "core_free": core_total,
                "pod_count": 0,
                "util": 0.0,
            }
        )

    return gpus


def generate_gpu_batch(
    batch_id: int,
    min_gpus: int = 3,
    max_gpus: int = 8,
    memory_choices: Optional[List[int]] = None,
    core_choices: Optional[List[int]] = None,
) -> List[Dict]:
    """
    生成一个随机 GPU batch。

    新改动：
        每个 batch 的 GPU 数量可以不同；
        每张 GPU 的 memory/core 也可以不同。
    """
    if memory_choices is None or len(memory_choices) == 0:
        memory_choices = [16384, 24576, 32768]

    if core_choices is None or len(core_choices) == 0:
        core_choices = [80, 100, 120]

    if min_gpus <= 0 or max_gpus < min_gpus:
        raise ValueError("invalid GPU range")

    num_gpus = random.randint(min_gpus, max_gpus)
    gpus = []

    for i in range(num_gpus):
        memory_total = random.choice(memory_choices)
        core_total = random.choice(core_choices)

        gpus.append(
            {
                "gpu_id": i,
                "batch_id": batch_id,
                "memory_total": memory_total,
                "memory_free": memory_total,
                "core_total": core_total,
                "core_free": core_total,
                "pod_count": 0,
                "util": 0.0,
            }
        )

    return gpus


def reset_gpus_from_template(gpu_templates: List[Dict]) -> List[Dict]:
    """
    从 GPU 模板恢复初始状态。

    每个 episode / 每个 baseline 测试前都必须 reset，
    保证不同方法使用同一个初始场景。
    """
    gpus = []

    for gpu in gpu_templates:
        new_gpu = copy.deepcopy(gpu)
        new_gpu["memory_free"] = new_gpu["memory_total"]
        new_gpu["core_free"] = new_gpu["core_total"]
        new_gpu["pod_count"] = 0
        new_gpu["util"] = 0.0
        gpus.append(new_gpu)

    return gpus


def save_gpus(gpus: List[Dict], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(gpus, f, ensure_ascii=False, indent=2)


def load_gpus(load_path: str) -> List[Dict]:
    with open(load_path, "r", encoding="utf-8") as f:
        return json.load(f)