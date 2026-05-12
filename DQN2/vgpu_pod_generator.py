"""
vGPU Pod task generator.

作用：
1. 生成申请 vGPU 资源的 Pod 任务；
2. 保存训练任务集和测试任务集；
3. 从 JSON 加载任务集。

对应原项目中的 task_generator.py 思路：
    原 task_generator.py 生成普通任务；
    这里生成申请 vGPU-memory 和 vGPU-cores 的 Pod。
"""

import json
import os
import random
from typing import Dict, List, Optional


def generate_pod_batch(
    num_pods: int = 20,
    memory_choices: Optional[List[int]] = None,
    core_choices: Optional[List[int]] = None,
    batch_id: Optional[int] = None,
) -> List[Dict]:
    """
    生成一批 Pod 任务。

    每个 Pod 默认申请：
        volcano.sh/vgpu-number: 1
        volcano.sh/vgpu-memory: memory_demand
        volcano.sh/vgpu-cores: core_demand
    """
    if memory_choices is None:
        memory_choices = [1024, 2048, 3072, 4096, 5120]

    if core_choices is None:
        core_choices = [5, 10, 15]

    pods = []

    for i in range(num_pods):
        pod = {
            "task_id": f"pod-{i}",
            "vgpu_number": 1,
            "memory_demand": random.choice(memory_choices),
            "core_demand": random.choice(core_choices),
        }

        if batch_id is not None:
            pod["batch_id"] = batch_id

        pods.append(pod)

    return pods


def generate_pod_batches(
    num_batches: int = 200,
    num_pods_per_batch: int = 20,
    memory_choices: Optional[List[int]] = None,
    core_choices: Optional[List[int]] = None,
) -> List[List[Dict]]:
    """
    生成多批 Pod。

    用途：
        train_pods.json: 多批训练任务
        test_pods.json : 多批测试任务
    """
    batches = []

    for batch_id in range(num_batches):
        batch = generate_pod_batch(
            num_pods=num_pods_per_batch,
            memory_choices=memory_choices,
            core_choices=core_choices,
            batch_id=batch_id,
        )
        batches.append(batch)

    return batches


def save_pod_batches(pod_batches: List[List[Dict]], save_path: str) -> None:
    """
    保存 Pod 批次到 JSON 文件。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pod_batches, f, ensure_ascii=False, indent=2)


def load_pod_batches(load_path: str) -> List[List[Dict]]:
    """
    从 JSON 文件加载 Pod 批次。
    """
    with open(load_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    train_batches = generate_pod_batches(
        num_batches=200,
        num_pods_per_batch=20,
    )

    test_batches = generate_pod_batches(
        num_batches=20,
        num_pods_per_batch=20,
    )

    train_path = "DQN2/data/vgpu_sim/train_pods.json"
    test_path = "DQN2/data/vgpu_sim/test_pods.json"

    save_pod_batches(train_batches, train_path)
    save_pod_batches(test_batches, test_path)

    print(f"train pods saved to: {train_path}")
    print(f"test pods saved to : {test_path}")