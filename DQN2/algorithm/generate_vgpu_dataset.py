"""
Generate fixed vGPU mixed-load dataset.

This script saves:
    dataset_meta.json
    train_scenarios.jsonl
    val_scenarios.jsonl
    test_load_0.6.jsonl
    test_load_0.8.jsonl
    test_load_1.0.jsonl
    test_load_1.2.jsonl
    test_load_1.5.jsonl

Why:
    To make the experiment reproducible.

Usage:
    python DQN2/algorithm/generate_vgpu_dataset.py \
      --seed 42 \
      --output-dir DQN2/data_mixed_load_fixed \
      --train-batches 3000 \
      --val-batches 20 \
      --test-batches 50
"""

import argparse
import os
import random

try:
    from DQN2.algorithm.vgpu_dqn_sim import (
        ensure_dir,
        generate_scenario,
        parse_float_list,
        save_json,
        seed_everything,
        write_jsonl,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    from DQN2.algorithm.vgpu_dqn_sim import (
        ensure_dir,
        generate_scenario,
        parse_float_list,
        save_json,
        seed_everything,
        write_jsonl,
    )


def generate_train_scenarios(args):
    train_loads = parse_float_list(args.train_target_loads)
    scenarios = []

    for i in range(args.train_batches):
        target_load = random.choice(train_loads)

        scenario = generate_scenario(
            target_load=target_load,
            min_gpus=args.train_min_gpus,
            max_gpus=args.train_max_gpus,
            min_pods=args.train_min_pods,
            max_pods=args.train_max_pods,
            gpu_memory_choices=args.gpu_memory_choices,
            gpu_core_total=args.gpu_cores,
            scenario_id=f"train-{i}",
            heterogeneous_gpus=not args.homogeneous_gpus,
        )

        scenarios.append(scenario)

    return scenarios


def generate_val_scenarios(args):
    eval_loads = parse_float_list(args.eval_loads)
    scenarios = []

    for load in eval_loads:
        for i in range(args.val_batches):
            scenario = generate_scenario(
                target_load=load,
                min_gpus=args.train_min_gpus,
                max_gpus=args.train_max_gpus,
                min_pods=args.train_min_pods,
                max_pods=args.train_max_pods,
                gpu_memory_choices=args.gpu_memory_choices,
                gpu_core_total=args.gpu_cores,
                scenario_id=f"val-load-{load:.1f}-{i}",
                heterogeneous_gpus=not args.homogeneous_gpus,
            )

            scenarios.append(scenario)

    return scenarios


def generate_test_scenarios(args, target_load: float):
    scenarios = []

    for i in range(args.test_batches):
        scenario = generate_scenario(
            target_load=target_load,
            min_gpus=args.test_min_gpus,
            max_gpus=args.test_max_gpus,
            min_pods=args.test_min_pods,
            max_pods=args.test_max_pods,
            gpu_memory_choices=args.gpu_memory_choices,
            gpu_core_total=args.gpu_cores,
            scenario_id=f"test-load-{target_load:.1f}-{i}",
            heterogeneous_gpus=not args.homogeneous_gpus,
        )

        scenarios.append(scenario)

    return scenarios


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fixed vGPU mixed-load dataset"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="DQN2/data_mixed_load_fixed")

    parser.add_argument(
        "--train-target-loads",
        type=str,
        default="0.6,0.8,1.0,1.2,1.5",
    )

    parser.add_argument(
        "--eval-loads",
        type=str,
        default="0.6,0.8,1.0,1.2,1.5",
    )

    parser.add_argument("--train-batches", type=int, default=3000)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--test-batches", type=int, default=50)

    parser.add_argument("--train-min-gpus", type=int, default=6)
    parser.add_argument("--train-max-gpus", type=int, default=12)
    parser.add_argument("--train-min-pods", type=int, default=100)
    parser.add_argument("--train-max-pods", type=int, default=300)

    parser.add_argument("--test-min-gpus", type=int, default=8)
    parser.add_argument("--test-max-gpus", type=int, default=16)
    parser.add_argument("--test-min-pods", type=int, default=100)
    parser.add_argument("--test-max-pods", type=int, default=400)

    parser.add_argument(
        "--gpu-memory-choices",
        type=float,
        nargs="+",
        default=[24.0, 32.0, 40.0, 48.0],
    )

    parser.add_argument("--gpu-cores", type=float, default=100.0)
    parser.add_argument("--homogeneous-gpus", action="store_true")

    args = parser.parse_args()

    if args.train_min_gpus > args.train_max_gpus:
        raise ValueError("--train-min-gpus cannot be larger than --train-max-gpus")

    if args.test_min_gpus > args.test_max_gpus:
        raise ValueError("--test-min-gpus cannot be larger than --test-max-gpus")

    if args.train_min_pods > args.train_max_pods:
        raise ValueError("--train-min-pods cannot be larger than --train-max-pods")

    if args.test_min_pods > args.test_max_pods:
        raise ValueError("--test-min-pods cannot be larger than --test-max-pods")

    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    save_json(vars(args), os.path.join(args.output_dir, "dataset_meta.json"))

    train_scenarios = generate_train_scenarios(args)
    val_scenarios = generate_val_scenarios(args)

    train_path = os.path.join(args.output_dir, "train_scenarios.jsonl")
    val_path = os.path.join(args.output_dir, "val_scenarios.jsonl")

    write_jsonl(train_scenarios, train_path)
    write_jsonl(val_scenarios, val_path)

    print(f"train scenarios saved to: {train_path}")
    print(f"val scenarios saved to  : {val_path}")

    for load in parse_float_list(args.eval_loads):
        test_scenarios = generate_test_scenarios(args, load)

        test_path = os.path.join(
            args.output_dir,
            f"test_load_{load:.1f}.jsonl",
        )

        write_jsonl(test_scenarios, test_path)
        print(f"test load {load:.1f} saved to: {test_path}")

    print(f"dataset meta saved to: {os.path.join(args.output_dir, 'dataset_meta.json')}")


if __name__ == "__main__":
    main()