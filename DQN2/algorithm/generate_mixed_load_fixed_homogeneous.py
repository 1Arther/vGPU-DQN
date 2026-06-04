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
import random
import sys
from pathlib import Path
from typing import Dict, List

try:
    from DQN2.algorithm.vgpu_dqn_sim import (
        compute_scenario_load,
        ensure_dir,
        generate_scenario,
        seed_everything,
        write_jsonl,
    )
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    from DQN2.algorithm.vgpu_dqn_sim import (
        compute_scenario_load,
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


def apply_existing_load(scenario: Dict, args) -> Dict:
    """
    Add pre-existing GPU load to simulate non-empty nodes and fragmentation.

    The Job demand loads remain stored as memory_load/core_load/actual_load.
    Existing load is stored separately so target load still describes the
    incoming Job pressure.
    """
    if not args.enable_existing_load:
        scenario["existing_memory_load"] = 0.0
        scenario["existing_core_load"] = 0.0
        return scenario

    gpus = scenario["gpus"]
    memory_used_ratios = []
    core_used_ratios = []

    for gpu in gpus:
        profile = random.choices(
            ["balanced", "memory_heavy", "core_heavy", "light"],
            weights=[
                args.existing_balanced_ratio,
                args.existing_memory_heavy_ratio,
                args.existing_core_heavy_ratio,
                args.existing_light_ratio,
            ],
            k=1,
        )[0]

        base = random.uniform(args.existing_load_min, args.existing_load_max)

        if profile == "memory_heavy":
            memory_used = min(args.existing_load_max, base * random.uniform(1.20, 1.55))
            core_used = max(args.existing_load_min * 0.5, base * random.uniform(0.25, 0.65))
        elif profile == "core_heavy":
            memory_used = max(args.existing_load_min * 0.5, base * random.uniform(0.25, 0.65))
            core_used = min(args.existing_load_max, base * random.uniform(1.20, 1.55))
        elif profile == "light":
            memory_used = base * random.uniform(0.20, 0.55)
            core_used = base * random.uniform(0.20, 0.55)
        else:
            memory_used = base * random.uniform(0.80, 1.20)
            core_used = base * random.uniform(0.80, 1.20)

        memory_used = max(0.0, min(args.existing_load_max, memory_used))
        core_used = max(0.0, min(args.existing_load_max, core_used))

        gpu["memory_free"] = max(
            0.0,
            gpu["memory_total"] * (1.0 - memory_used),
        )
        gpu["core_free"] = max(
            0.0,
            gpu["core_total"] * (1.0 - core_used),
        )
        gpu["pod_count"] = random.randint(
            args.existing_pod_count_min,
            args.existing_pod_count_max,
        )
        gpu["util"] = max(0.0, min(100.0, core_used * 100.0))

        memory_used_ratios.append(memory_used)
        core_used_ratios.append(core_used)

    scenario["existing_memory_load"] = float(sum(memory_used_ratios) / max(len(gpus), 1))
    scenario["existing_core_load"] = float(sum(core_used_ratios) / max(len(gpus), 1))
    scenario["existing_load_enabled"] = True

    return scenario


def build_conflict_scenario(
    scenario_id: str,
    target_load: float,
    args,
) -> Dict:
    """
    Build one Job containing memory-heavy, core-heavy, and balanced Pods.

    This makes intra-GPU balance learnable: a useful policy should pair
    complementary Pod shapes instead of filling GPUs with one skewed shape.
    """
    num_gpus = random.randint(args.min_gpus, args.max_gpus)
    memory_total = float(random.choice(args.gpu_memory_choices))
    core_total = float(args.gpu_core_total)

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

    def make_pod(kind: str, idx: int) -> Dict:
        if kind == "memory_heavy":
            memory_ratio = random.uniform(0.34, 0.58)
            core_ratio = random.uniform(0.10, 0.34)
        elif kind == "core_heavy":
            memory_ratio = random.uniform(0.08, 0.28)
            core_ratio = random.uniform(0.52, 0.86)
        else:
            memory_ratio = random.uniform(0.18, 0.42)
            core_ratio = random.uniform(0.22, 0.55)

        vgpu_number = random.choices(
            [1, 2],
            weights=[0.88, 0.12],
            k=1,
        )[0]
        vgpu_number = min(vgpu_number, num_gpus)

        return {
            "task_id": f"{scenario_id}-pod-{idx}",
            "vgpu_number": int(vgpu_number),
            "memory_demand": round(memory_total * memory_ratio, 4),
            "core_demand": round(core_total * core_ratio, 4),
            "profile": kind,
        }

    pods = [
        make_pod("memory_heavy", 0),
        make_pod("core_heavy", 1),
        make_pod("balanced", 2),
    ]

    lower = target_load * 0.85 if target_load <= 1.0 else target_load * 0.90
    upper = target_load * 1.15 if target_load <= 1.0 else target_load * 1.20

    for _ in range(120):
        scenario = {
            "scenario_id": scenario_id,
            "target_load": float(target_load),
            "gpus": gpus,
            "pods": pods,
        }
        actual_load, _, _ = compute_scenario_load(scenario)

        if lower <= actual_load <= upper:
            break

        if actual_load < lower and len(pods) < args.max_pods:
            kind = random.choices(
                ["memory_heavy", "core_heavy", "balanced"],
                weights=[0.38, 0.38, 0.24],
                k=1,
            )[0]
            pods.append(make_pod(kind, len(pods)))
            continue

        if actual_load > upper and len(pods) > max(args.min_pods, 3):
            removable = list(range(3, len(pods)))
            if removable:
                pods.pop(random.choice(removable))
                continue

        break

    random.shuffle(pods)

    scenario = {
        "scenario_id": scenario_id,
        "target_load": float(target_load),
        "gpus": gpus,
        "pods": pods,
        "workload_type": "mixed_conflict",
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
    rejected_by_load = 0
    conflict_ratio = (
        args.train_conflict_ratio
        if split == "train"
        else args.eval_conflict_ratio
    )

    for load in loads:
        for i in range(num_per_load):
            scenario_id = f"{split}-load-{load:.1f}-scenario-{i}"
            force_conflict = random.random() < conflict_ratio

            scenario = None

            for attempt in range(args.max_generate_attempts):
                candidate_id = scenario_id

                if attempt > 0:
                    candidate_id = f"{scenario_id}-retry-{attempt}"

                if force_conflict:
                    candidate = build_conflict_scenario(
                        scenario_id=candidate_id,
                        target_load=load,
                        args=args,
                    )
                else:
                    candidate = generate_scenario(
                        target_load=load,
                        min_gpus=args.min_gpus,
                        max_gpus=args.max_gpus,
                        min_pods=args.min_pods,
                        max_pods=args.max_pods,
                        gpu_memory_choices=args.gpu_memory_choices,
                        gpu_core_total=args.gpu_core_total,
                        scenario_id=candidate_id,
                        heterogeneous_gpus=False,
                    )

                candidate = apply_existing_load(candidate, args)

                if not args.strict_actual_load:
                    scenario = candidate
                    break

                actual_load = float(candidate.get("actual_load", 0.0))

                if abs(actual_load - load) <= args.actual_load_tolerance:
                    scenario = candidate
                    break

                rejected_by_load += 1

            if scenario is None:
                raise RuntimeError(
                    f"failed to generate {scenario_id} with actual_load within "
                    f"{args.actual_load_tolerance} of target {load}; "
                    f"increase --max-generate-attempts or relax "
                    f"--actual-load-tolerance"
                )

            rows.append(scenario)

    if not args.disable_hard_cases and split in args.hard_case_splits:
        hard_cases = build_hard_cases(args, split=split)

        if split == "train":
            repeat = args.train_hard_repeat
        else:
            repeat = args.eval_hard_repeat

        for _ in range(repeat):
            rows.extend(copy.deepcopy(hard_cases))

    if rejected_by_load:
        print(f"{split}: rejected scenarios outside load tolerance: {rejected_by_load}")

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
    parser.add_argument(
        "--train-loads",
        type=str,
        default=None,
        help="Optional training-only load list. Use repeated low/mid loads to oversample them.",
    )

    parser.add_argument("--train-per-load", type=int, default=800)
    parser.add_argument("--val-per-load", type=int, default=120)
    parser.add_argument("--test-per-load", type=int, default=120)
    parser.add_argument("--train-conflict-ratio", type=float, default=0.0)
    parser.add_argument("--eval-conflict-ratio", type=float, default=0.0)

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

    parser.add_argument("--enable-existing-load", action="store_true")
    parser.add_argument("--existing-load-min", type=float, default=0.05)
    parser.add_argument("--existing-load-max", type=float, default=0.45)
    parser.add_argument("--existing-pod-count-min", type=int, default=0)
    parser.add_argument("--existing-pod-count-max", type=int, default=6)
    parser.add_argument("--existing-balanced-ratio", type=float, default=0.35)
    parser.add_argument("--existing-memory-heavy-ratio", type=float, default=0.25)
    parser.add_argument("--existing-core-heavy-ratio", type=float, default=0.25)
    parser.add_argument("--existing-light-ratio", type=float, default=0.15)

    parser.add_argument("--disable-hard-cases", action="store_true")
    parser.add_argument("--train-hard-repeat", type=int, default=30)
    parser.add_argument("--eval-hard-repeat", type=int, default=1)
    parser.add_argument(
        "--hard-case-splits",
        type=str,
        nargs="+",
        default=["train"],
        choices=["train", "val", "test"],
        help="Splits that receive fixed hard cases. Default keeps test clean.",
    )
    parser.add_argument(
        "--strict-actual-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry random scenarios until actual_load is close to target_load.",
    )
    parser.add_argument(
        "--actual-load-tolerance",
        type=float,
        default=0.12,
        help="Maximum absolute difference between actual_load and target_load.",
    )
    parser.add_argument("--max-generate-attempts", type=int, default=500)

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
    train_loads = parse_loads(args.train_loads) if args.train_loads else loads

    print("========== Generate Homogeneous Scenarios ==========")
    print(f"output_dir         : {args.output_dir}")
    print(f"loads              : {loads}")
    print(f"train_loads        : {train_loads}")
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
        loads=train_loads,
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
