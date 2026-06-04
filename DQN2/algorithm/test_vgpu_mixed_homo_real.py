#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test DQN model on homogeneous real-style mixed-load scenarios.

输入：
    test_scenarios.jsonl

输出：
    mixed_load_test_detail.csv
    mixed_load_test_summary.csv

示例：
python DQN2/algorithm/test_vgpu_mixed_homo_real.py \
  --model-path DQN2/outputs_mixed_load_fixed_v8_homo_real/vgpu_dqn_mixed_best.pth \
  --test-path DQN2/data_mixed_load_fixed_homo_real/test_scenarios.jsonl \
  --output-dir DQN2/outputs_mixed_load_fixed_v8_homo_real_eval
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

try:
    from DQN2.algorithm.vgpu_dqn_sim import (
        BASELINE_METHODS,
        create_agent,
        device,
        ensure_dir,
        load_jsonl,
        print_summary_table,
        run_baseline,
        run_one_episode,
        save_json,
        seed_everything,
        summarize_detail_rows,
        write_csv,
    )
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    from DQN2.algorithm.vgpu_dqn_sim import (
        BASELINE_METHODS,
        create_agent,
        device,
        ensure_dir,
        load_jsonl,
        print_summary_table,
        run_baseline,
        run_one_episode,
        save_json,
        seed_everything,
        summarize_detail_rows,
        write_csv,
    )


def infer_actual_load_bucket(scenario):
    """
    Group evaluation by actual demand pressure, not requested target_load.

    Some generated scenarios miss their target load, so target_load buckets can
    make low-load curves include high-pressure samples.
    """
    load = scenario.get("actual_load", scenario.get("target_load", 0.0))

    try:
        load = float(load)
    except Exception:
        load = 0.0

    return round(load, 1)


def evaluate_scenarios(agent, scenarios, args):
    detail_rows = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for batch_id, scenario in enumerate(scenarios):
        dqn_row = run_one_episode(
            agent=agent,
            scenario=scenario,
            args=args,
            train=False,
        )

        dqn_row["batch_id"] = batch_id
        dqn_row["load_bucket"] = infer_actual_load_bucket(scenario)
        detail_rows.append(dqn_row)

        for method in BASELINE_METHODS:
            baseline_row = run_baseline(
                method=method,
                scenario=scenario,
                args=args,
            )

            baseline_row["batch_id"] = batch_id
            baseline_row["load_bucket"] = infer_actual_load_bucket(scenario)
            detail_rows.append(baseline_row)

    agent.epsilon = old_epsilon

    return detail_rows


def summarize_by_load_bucket(detail_rows):
    buckets = sorted(
        set(row["load_bucket"] for row in detail_rows)
    )

    all_summary_rows = []

    for bucket in buckets:
        rows = [
            row for row in detail_rows
            if row["load_bucket"] == bucket
        ]

        summary_rows = summarize_detail_rows(
            rows,
            target_load=bucket,
        )

        all_summary_rows.extend(summary_rows)

    return all_summary_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--model-path",
        type=str,
        default="DQN2/outputs_mixed_load_fixed_v8_homo_real/vgpu_dqn_mixed_best.pth",
    )

    parser.add_argument(
        "--test-path",
        type=str,
        default="DQN2/data_mixed_load_fixed_homo_real/test_scenarios.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="DQN2/outputs_mixed_load_fixed_v8_homo_real_eval",
    )

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--disable-job-features", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--priority-beta", type=float, default=0.4)

    parser.add_argument("--epsilon-start", type=float, default=0.0)
    parser.add_argument("--epsilon-min", type=float, default=0.0)
    parser.add_argument("--epsilon-decay", type=float, default=1.0)

    parser.add_argument("--success-weight", type=float, default=2.0)
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--inter-balance-weight", type=float, default=1.0)
    parser.add_argument("--intra-balance-weight", type=float, default=1.0)
    parser.add_argument("--delta-balance-weight", type=float, default=1.0)
    parser.add_argument("--delta-inter-balance-weight", type=float, default=None)
    parser.add_argument("--delta-intra-balance-weight", type=float, default=None)
    parser.add_argument("--failure-weight", type=float, default=2.0)
    parser.add_argument("--action-rerank-topk", type=int, default=0)
    parser.add_argument("--action-rerank-q-weight", type=float, default=1.0)
    parser.add_argument("--action-rerank-balance-weight", type=float, default=1.0)
    parser.add_argument("--action-rerank-inter-weight", type=float, default=0.0)
    parser.add_argument("--action-rerank-intra-weight", type=float, default=0.0)
    parser.add_argument("--train-action-rerank", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    print(f"Using device: {device}")
    print(f"model path: {args.model_path}")
    print(f"test path : {args.test_path}")
    print(f"output dir: {args.output_dir}")
    print(f"baselines : {BASELINE_METHODS}")

    save_json(vars(args), os.path.join(args.output_dir, "test_args.json"))

    scenarios = load_jsonl(args.test_path)
    print(f"loaded test scenarios: {len(scenarios)}")

    if not scenarios:
        raise RuntimeError("empty test scenarios")

    agent = create_agent(args)
    agent.load(args.model_path)
    agent.epsilon = 0.0

    detail_rows = evaluate_scenarios(
        agent=agent,
        scenarios=scenarios,
        args=args,
    )

    summary_rows = summarize_by_load_bucket(detail_rows)

    detail_path = os.path.join(
        args.output_dir,
        "mixed_load_test_detail.csv",
    )

    summary_path = os.path.join(
        args.output_dir,
        "mixed_load_test_summary.csv",
    )

    write_csv(detail_rows, detail_path)
    write_csv(summary_rows, summary_path)

    print("\n=== Test comparison summary ===")
    print_summary_table(summary_rows)

    print(f"\ntest detail saved to : {detail_path}")
    print(f"test summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
