#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate fixed mixed-load scenarios with homogeneous GPUs per node.

语义：
    一个 scenario = 一个空节点 + 一个 Volcano Job 的 Pod 批次。

约束：
    1. 同一个 scenario 内 GPU 同规格。
    2. GPU 初始占用全部为 0。
    3. Pods 属于同一个 Job。
    4. 每个 Pod 可以申请 1~4 个 vGPU。
    5. 每个 vGPU slice 有 memory_demand / core_demand。

输出：
    train_scenarios.jsonl
    val_scenarios.jsonl
    test_scenarios.jsonl

运行示例：
python DQN2/algorithm/generate_mixed_load_fixed_homogeneous.py \
  --output-dir DQN2/data_mixed_load_fixed_homo \
  --train-per-load 800 \
  --val-per-load 120 \
  --test-per-load 120
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict, List

try:
    from DQN2.algorithm.vgpu_dqn_sim import (
        ensure_dir,
        generate_scenario,
        seed_everything,
        write_jsonl,
    )
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    from DQN2.algorithm.vgpu_dqn_sim import (
        ensure_dir,
        generate_scenario,
        seed_everything,
        write_jsonl,
    )


def parse_loads(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def build_exact_job_scenario(
    scenario_id: str,
    num_gpus: int,
    num_pods: int,
    memory_total: float,
    core_total: float,
    memory_demand: float,
    core_demand: float,
    vgpu_number: int = 1,
) -> Dict:
    """
    固定 hard case：
        同规格空节点 + 一个 Job 的 Pod 批次。
    """
    gpus = []

    for i in range(num_gpus):
        gpus.append(
            {
                "gpu_id": i,
                "memory_total": float(memory_total),
                "memory_free": float(memory_total),
                "core_total": float(core_total),
                "core_free": float(core_total),
                "pod_count": 0,
                "util": 0.0,
            }
        )

    pods = []

    for i in range(num_pods):
        pods.append(
            {
                "task_id": f"{scenario_id}-pod-{i}",
                "vgpu_number": int(vgpu_number),
                "memory_demand": float(memory_demand),
                "core_demand": float(core_demand),
            }
        )

    total_gpu_memory = num_gpus * memory_total
    total_gpu_core = num_gpus * core_total

    total_pod_memory = num_pods * vgpu_number * memory_demand
    total_pod_core = num_pods * vgpu_number * core_demand

    memory_load = total_pod_memory / max(total_gpu_memory, 1e-8)
    core_load = total_pod_core / max(total_gpu_core, 1e-8)
    actual_load = max(memory_load, core_load)

    return {
        "scenario_id": scenario_id,
        "target_load": float(actual_load),
        "actual_load": float(actual_load),
        "memory_load": float(memory_load),
        "core_load": float(core_load),
        "num_gpus": int(num_gpus),
        "num_pods": int(num_pods),
        "gpus": gpus,
        "pods": pods,
    }


def build_hard_cases(args, split: str) -> List[Dict]:
    """
    真实部署关键 case。
    这些 case 放进 train/val/test，用来防止模型只在随机场景平均效果好。
    """

    cases = []

    cases.append(
        build_exact_job_scenario(
            scenario_id=f"{split}-hard-4gpu-8pod-1vgpu",
            num_gpus=4,
            num_pods=8,
            memory_total=args.hard_memory_total,
            core_total=args.hard_core_total,
            memory_demand=args.hard_memory_demand,
            core_demand=args.hard_core_demand,
            vgpu_number=1,
        )
    )

    cases.append(
        build_exact_job_scenario(
            scenario_id=f"{split}-hard-4gpu-4pod-1vgpu",
            num_gpus=4,
            num_pods=4,
            memory_total=args.hard_memory_total,
            core_total=args.hard_core_total,
            memory_demand=args.hard_memory_demand,
            core_demand=args.hard_core_demand,
            vgpu_number=1,
        )
    )

    cases.append(
        build_exact_job_scenario(
            scenario_id=f"{split}-hard-4gpu-12pod-small",
            num_gpus=4,
            num_pods=12,
            memory_total=args.hard_memory_total,
            core_total=args.hard_core_total,
            memory_demand=args.hard_memory_demand / 2.0,
            core_demand=max(args.hard_core_demand / 2.0, 1.0),
            vgpu_number=1,
        )
    )

    cases.append(
        build_exact_job_scenario(
            scenario_id=f"{split}-hard-8gpu-16pod-1vgpu",
            num_gpus=8,
            num_pods=16,
            memory_total=args.hard_memory_total,
            core_total=args.hard_core_total,
            memory_demand=args.hard_memory_demand,
            core_demand=args.hard_core_demand,
            vgpu_number=1,
        )
    )

    cases.append(
        build_exact_job_scenario(
            scenario_id=f"{split}-hard-4gpu-4pod-2vgpu",
            num_gpus=4,
            num_pods=4,
            memory_total=args.hard_memory_total,
            core_total=args.hard_core_total,
            memory_demand=args.hard_memory_demand / 2.0,
            core_demand=max(args.hard_core_demand / 2.0, 1.0),
            vgpu_number=2,
        )
    )

    return cases


def generate_split(
    split: str,
    num_per_load: int,
    loads: List[float],
    args,
) -> List[Dict]:
    rows = []

    for load in loads:
        for i in range(num_per_load):
            scenario_id = f"{split}-load-{load:.1f}-scenario-{i}"

            scenario = generate_scenario(
                target_load=load,
                min_gpus=args.min_gpus,
                max_gpus=args.max_gpus,
                min_pods=args.min_pods,
                max_pods=args.max_pods,
                gpu_memory_choices=args.gpu_memory_choices,
                gpu_core_total=args.gpu_core_total,
                scenario_id=scenario_id,
                heterogeneous_gpus=False,
            )

            rows.append(scenario)

    if not args.disable_hard_cases:
        hard_cases = build_hard_cases(args, split=split)

        if split == "train":
            repeat = args.train_hard_repeat
        else:
            repeat = args.eval_hard_repeat

        for _ in range(repeat):
            rows.extend(copy.deepcopy(hard_cases))

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate homogeneous-GPU mixed-load scenarios"
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="DQN2/data_mixed_load_fixed_homo",
    )

    parser.add_argument(
        "--loads",
        type=str,
        default="0.6,0.8,1.0,1.2,1.5",
    )

    parser.add_argument("--train-per-load", type=int, default=800)
    parser.add_argument("--val-per-load", type=int, default=120)
    parser.add_argument("--test-per-load", type=int, default=120)

    parser.add_argument("--min-gpus", type=int, default=4)
    parser.add_argument("--max-gpus", type=int, default=8)

    parser.add_argument("--min-pods", type=int, default=8)
    parser.add_argument("--max-pods", type=int, default=24)

    parser.add_argument(
        "--gpu-memory-choices",
        type=float,
        nargs="+",
        default=[24.0, 40.0, 48.0, 80.0],
    )

    parser.add_argument("--gpu-core-total", type=float, default=100.0)

    parser.add_argument("--disable-hard-cases", action="store_true")
    parser.add_argument("--train-hard-repeat", type=int, default=30)
    parser.add_argument("--eval-hard-repeat", type=int, default=1)

    # 真实测试对应的 hard case，单位保持和 simulator 一致即可。
    # 这里用 MB，是为了和真实 YAML 的 vgpu-memory=8192 对齐。
    parser.add_argument("--hard-memory-total", type=float, default=49152.0)
    parser.add_argument("--hard-memory-demand", type=float, default=8192.0)
    parser.add_argument("--hard-core-total", type=float, default=100.0)
    parser.add_argument("--hard-core-demand", type=float, default=25.0)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    ensure_dir(args.output_dir)

    loads = parse_loads(args.loads)

    print("========== Generate Homogeneous Scenarios ==========")
    print(f"output_dir         : {args.output_dir}")
    print(f"loads              : {loads}")
    print(f"train_per_load     : {args.train_per_load}")
    print(f"val_per_load       : {args.val_per_load}")
    print(f"test_per_load      : {args.test_per_load}")
    print(f"num_gpus range     : {args.min_gpus} ~ {args.max_gpus}")
    print(f"num_pods range     : {args.min_pods} ~ {args.max_pods}")
    print(f"gpu_memory_choices : {args.gpu_memory_choices}")
    print(f"gpu_core_total     : {args.gpu_core_total}")
    print(f"hard_cases         : {not args.disable_hard_cases}")

    train_rows = generate_split(
        split="train",
        num_per_load=args.train_per_load,
        loads=loads,
        args=args,
    )

    val_rows = generate_split(
        split="val",
        num_per_load=args.val_per_load,
        loads=loads,
        args=args,
    )

    test_rows = generate_split(
        split="test",
        num_per_load=args.test_per_load,
        loads=loads,
        args=args,
    )

    train_path = os.path.join(args.output_dir, "train_scenarios.jsonl")
    val_path = os.path.join(args.output_dir, "val_scenarios.jsonl")
    test_path = os.path.join(args.output_dir, "test_scenarios.jsonl")

    write_jsonl(train_rows, train_path)
    write_jsonl(val_rows, val_path)
    write_jsonl(test_rows, test_path)

    print("\n========== Done ==========")
    print(f"train scenarios: {len(train_rows)} -> {train_path}")
    print(f"val scenarios  : {len(val_rows)} -> {val_path}")
    print(f"test scenarios : {len(test_rows)} -> {test_path}")


if __name__ == "__main__":
    main()
