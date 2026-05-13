"""
Test one unified DQN model on fixed test scenarios.

Input:
    vgpu_dqn_mixed_best.pth
    test_load_0.6.jsonl
    test_load_0.8.jsonl
    test_load_1.0.jsonl
    test_load_1.2.jsonl
    test_load_1.5.jsonl

Output:
    eval_load_*/test_comparison_detail.csv
    eval_load_*/test_comparison_summary.csv
    mixed_load_test_detail.csv
    mixed_load_test_summary.csv
"""

import argparse
import os

try:
    from DQN2.algorithm.vgpu_dqn_sim import (
        BASELINE_METHODS,
        create_agent,
        device,
        ensure_dir,
        load_jsonl,
        parse_float_list,
        print_summary_table,
        run_baseline,
        run_one_episode,
        save_json,
        seed_everything,
        summarize_detail_rows,
        write_csv,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    from DQN2.algorithm.vgpu_dqn_sim import (
        BASELINE_METHODS,
        create_agent,
        device,
        ensure_dir,
        load_jsonl,
        parse_float_list,
        print_summary_table,
        run_baseline,
        run_one_episode,
        save_json,
        seed_everything,
        summarize_detail_rows,
        write_csv,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test mixed-load DQN model from saved scenarios"
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--model-path",
        type=str,
        default="DQN2/outputs_mixed_load_fixed/vgpu_dqn_mixed_best.pth",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="DQN2/data_mixed_load_fixed",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="DQN2/outputs_mixed_load_fixed_eval",
    )

    parser.add_argument(
        "--eval-loads",
        type=str,
        default="0.6,0.8,1.0,1.2,1.5",
    )

    # Must match model structure used in training.
    parser.add_argument("--hidden-dim", type=int, default=256)
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
    parser.add_argument("--failure-weight", type=float, default=2.0)

    return parser.parse_args()


def evaluate_one_load(agent, scenarios, target_load, args):
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
        detail_rows.append(dqn_row)

        for method in BASELINE_METHODS:
            baseline_row = run_baseline(
                method=method,
                scenario=scenario,
                args=args,
            )

            baseline_row["batch_id"] = batch_id
            detail_rows.append(baseline_row)

    agent.epsilon = old_epsilon

    summary_rows = summarize_detail_rows(
        detail_rows,
        target_load=target_load,
    )

    return detail_rows, summary_rows


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    print(f"Using device: {device}")
    print(f"model path: {args.model_path}")
    print(f"data dir  : {args.data_dir}")
    print(f"baselines : {BASELINE_METHODS}")

    save_json(vars(args), os.path.join(args.output_dir, "test_args.json"))

    agent = create_agent(args)
    agent.load(args.model_path)
    agent.epsilon = 0.0

    all_detail_rows = []
    all_summary_rows = []

    for load in parse_float_list(args.eval_loads):
        test_path = os.path.join(
            args.data_dir,
            f"test_load_{load:.1f}.jsonl",
        )

        scenarios = load_jsonl(test_path)

        print(f"\n========== Testing target_load={load:.1f} ==========")
        print(f"loaded test scenarios: {len(scenarios)} from {test_path}")

        detail_rows, summary_rows = evaluate_one_load(
            agent=agent,
            scenarios=scenarios,
            target_load=load,
            args=args,
        )

        load_dir = os.path.join(
            args.output_dir,
            f"eval_load_{load:.1f}",
        )

        ensure_dir(load_dir)

        detail_path = os.path.join(load_dir, "test_comparison_detail.csv")
        summary_path = os.path.join(load_dir, "test_comparison_summary.csv")

        write_csv(detail_rows, detail_path)
        write_csv(summary_rows, summary_path)

        print("\n=== Final test comparison summary ===")
        print_summary_table(summary_rows)

        print(f"test detail saved to : {detail_path}")
        print(f"test summary saved to: {summary_path}")

        all_detail_rows.extend(detail_rows)
        all_summary_rows.extend(summary_rows)

    combined_detail_path = os.path.join(
        args.output_dir,
        "mixed_load_test_detail.csv",
    )

    combined_summary_path = os.path.join(
        args.output_dir,
        "mixed_load_test_summary.csv",
    )

    write_csv(all_detail_rows, combined_detail_path)
    write_csv(all_summary_rows, combined_summary_path)

    print(f"\ncombined detail saved to : {combined_detail_path}")
    print(f"combined summary saved to: {combined_summary_path}")


if __name__ == "__main__":
    main()