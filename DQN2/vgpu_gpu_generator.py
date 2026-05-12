"""
vGPU GPU generator.

作用：
1. 生成单节点内的物理 GPU 资源信息；
2. 保存 GPU 配置到 JSON；
3. 从 JSON 加载 GPU 配置；
4. 每轮仿真时根据模板重置 GPU 状态。

对应原项目中的 node_generator.py 思路：
    原 node_generator.py 生成服务器节点；
    这里生成单节点内的物理 GPU。
"""

import json
import os
from typing import Dict, List


def generate_gpus(
    num_gpus: int = 3,
    memory_total: int = 24576,
    core_total: int = 100,
) -> List[Dict]:
    """
    生成单节点内的物理 GPU 列表。

    memory_total:
        单张 GPU 的总显存，单位 MB。
        例如 24576 表示 24GB。

    core_total:
        抽象化的 GPU core 总量。
        这里用 100 表示 100% 算力额度。
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


def save_gpus(gpus: List[Dict], save_path: str) -> None:
    """
    保存 GPU 配置到 JSON 文件。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(gpus, f, ensure_ascii=False, indent=2)


def load_gpus(load_path: str) -> List[Dict]:
    """
    从 JSON 文件加载 GPU 配置。
    """
    with open(load_path, "r", encoding="utf-8") as f:
        return json.load(f)


def reset_gpus_from_template(gpu_templates: List[Dict]) -> List[Dict]:
    """
    根据 GPU 模板重置 GPU 状态。

    训练和测试时，每个 episode 都要从干净状态开始，
    否则上一轮的资源扣减会影响下一轮。
    """
    gpus = []

    for gpu in gpu_templates:
        memory_total = gpu["memory_total"]
        core_total = gpu["core_total"]

        gpus.append(
            {
                "gpu_id": gpu["gpu_id"],
                "memory_total": memory_total,
                "memory_free": memory_total,
                "core_total": core_total,
                "core_free": core_total,
                "pod_count": 0,
                "util": 0.0,
            }
        )

    return gpus


if __name__ == "__main__":
    gpus = generate_gpus(num_gpus=3, memory_total=24576, core_total=100)
    save_path = "DQN2/data/vgpu_sim/gpus_info.json"
    save_gpus(gpus, save_path)
    print(f"GPU data saved to: {save_path}")