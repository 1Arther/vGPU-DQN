"""
Train one unified DQN model from fixed mixed-load dataset.

Input:
    train_scenarios.jsonl
    val_scenarios.jsonl

Output:
    train_args.json
    vgpu_mixed_training_log.csv
    vgpu_dqn_mixed_best.pth
    vgpu_dqn_mixed_final.pth
"""

import argparse
import os
import random

try:
    from DQN2.algorithm.vgpu_dqn_sim import (
        create_agent,
        device,
        ensure_dir,
        evaluate_dqn_on_scenarios,
        load_jsonl,
        run_one_episode,
        save_json,
        seed_everything,
        write_csv,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    from DQN2.algorithm.vgpu_dqn_sim import (
        create_agent,
        device,
        ensure_dir,
        evaluate_dqn_on_scenarios,
        load_jsonl,
        run_one_episode,
        save_json,
        seed_everything,
        write_csv,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train mixed-load DQN model from saved scenarios"
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--train-path",
        type=str,
        default="DQN2/data_mixed_load_fixed/train_scenarios.jsonl",
    )

    parser.add_argument(
        "--val-path",
        type=str,
        default="DQN2/data_mixed_load_fixed/val_scenarios.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="DQN2/outputs_mixed_load_fixed",
    )

    parser.add_argument("--episodes", type=int, default=8000)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--priority-beta", type=float, default=0.4)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)

    parser.add_argument("--success-weight", type=float, default=2.0)
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--failure-weight", type=float, default=2.0)

    parser.add_argument("--target-update-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--print-interval", type=int, default=20)
    parser.add_argument("--log-save-interval", type=int, default=100)

    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--min-episodes", type=int, default=1000)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    print(f"Using device: {device}")
    print(f"train path: {args.train_path}")
    print(f"val path  : {args.val_path}")

    save_json(vars(args), os.path.join(args.output_dir, "train_args.json"))

    train_scenarios = load_jsonl(args.train_path)
    val_scenarios = load_jsonl(args.val_path)

    print(f"loaded train scenarios: {len(train_scenarios)}")
    print(f"loaded val scenarios  : {len(val_scenarios)}")

    if not train_scenarios:
        raise RuntimeError("empty train scenarios")

    if not val_scenarios:
        raise RuntimeError("empty val scenarios")

    agent = create_agent(args)

    log_path = os.path.join(args.output_dir, "vgpu_mixed_training_log.csv")
    best_model_path = os.path.join(args.output_dir, "vgpu_dqn_mixed_best.pth")
    final_model_path = os.path.join(args.output_dir, "vgpu_dqn_mixed_final.pth")

    log_rows = []
    best_eval_objective = -1e18
    no_improve_count = 0

    order = list(range(len(train_scenarios)))
    random.shuffle(order)

    for episode in range(1, args.episodes + 1):
        pos = (episode - 1) % len(order)

        if pos == 0 and episode > 1:
            random.shuffle(order)

        scenario = train_scenarios[order[pos]]

        result = run_one_episode(
            agent=agent,
            scenario=scenario,
            args=args,
            train=True,
        )

        loss = agent.replay(
            args.batch_size,
            beta=args.priority_beta,
        )

        if episode % args.target_update_interval == 0:
            agent.update_target_model()

        agent.update_epsilon()

        eval_objective = ""
        eval_balance = ""
        eval_success = ""
        eval_failure = ""

        if episode % args.eval_interval == 0:
            eval_result = evaluate_dqn_on_scenarios(
                agent=agent,
                scenarios=val_scenarios,
                args=args,
            )

            eval_objective = eval_result["avg_objective"]
            eval_balance = eval_result["avg_balance_score"]
            eval_success = eval_result["avg_success_rate"]
            eval_failure = eval_result["avg_failure_rate"]

            if eval_objective > best_eval_objective:
                best_eval_objective = eval_objective
                no_improve_count = 0
                agent.save(best_model_path, args=args)
                improved = "*"
            else:
                no_improve_count += 1
                improved = ""

            print(
                f"episode={episode:05d} "
                f"scenario={scenario['scenario_id']} "
                f"load={scenario['target_load']:.1f} "
                f"reward={result['reward']:.4f} "
                f"objective={result['objective']:.4f} "
                f"balance_score={result['balance_score']:.4f} "
                f"success_rate={result['success_rate']:.4f} "
                f"failure_rate={result['failure_rate']:.4f} "
                f"loss={loss:.6f} "
                f"epsilon={agent.epsilon:.4f} "
                f"eval_objective={eval_objective:.4f} "
                f"eval_success={eval_success:.4f} "
                f"{improved}"
            )

            if (
                args.early_stop_patience > 0
                and episode >= args.min_episodes
                and no_improve_count >= args.early_stop_patience
            ):
                print(
                    f"early stopped at episode={episode}, "
                    f"best_eval_objective={best_eval_objective:.6f}"
                )
                break

        elif episode % args.print_interval == 0:
            print(
                f"episode={episode:05d} "
                f"scenario={scenario['scenario_id']} "
                f"load={scenario['target_load']:.1f} "
                f"reward={result['reward']:.4f} "
                f"objective={result['objective']:.4f} "
                f"balance_score={result['balance_score']:.4f} "
                f"success_rate={result['success_rate']:.4f} "
                f"failure_rate={result['failure_rate']:.4f} "
                f"loss={loss:.6f} "
                f"epsilon={agent.epsilon:.4f}"
            )

        log_rows.append(
            {
                "episode": episode,
                "scenario_id": scenario["scenario_id"],
                "target_load": scenario["target_load"],
                "actual_load": result["actual_load"],
                "memory_load": result["memory_load"],
                "core_load": result["core_load"],
                "reward": result["reward"],
                "objective": result["objective"],
                "balance_score": result["balance_score"],
                "success_rate": result["success_rate"],
                "failure_rate": result["failure_rate"],
                "allocated_count": result["allocated_count"],
                "failure_count": result["failure_count"],
                "num_gpus": result["num_gpus"],
                "num_pods": result["num_pods"],
                "steps": result["steps"],
                "loss": loss,
                "epsilon": agent.epsilon,
                "eval_objective": eval_objective,
                "eval_balance_score": eval_balance,
                "eval_success_rate": eval_success,
                "eval_failure_rate": eval_failure,
                "best_eval_objective": best_eval_objective,
            }
        )

        if episode % args.log_save_interval == 0:
            write_csv(log_rows, log_path)

    write_csv(log_rows, log_path)
    agent.save(final_model_path, args=args)

    print(f"\ntraining log saved to : {log_path}")
    print(f"final model saved to  : {final_model_path}")
    print(f"best model saved to   : {best_model_path}")


if __name__ == "__main__":
    main()