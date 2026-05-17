#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Job-batch training for vGPU-DQN with hard cases.

Volcano 语义：
一个 Job 包含一批 Pods；
每个 Pod 可以申请任意个 vGPU；
build_vgpu_graph(gpus, pods, allocations) 表示当前 GPU 状态与当前 Job Pod 批次之间的图；
DQN 动作仍然是 (gpu_idx, pod_idx)。

这个脚本保留原来的 run_one_episode 训练逻辑，只额外加入 hard cases。
重点验证：
4 GPU, 8 Pod, each pod requests 1 vGPU, memory=8192, core=25
期望最终分布接近 2,2,2,2。
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
    构造一个 Volcano Job 场景：
    一个 Job 内有 num_pods 个 Pod；
    每个 Pod 申请 vgpu_number 个 vGPU；
    每个 vGPU slice 消耗 memory_demand / core_demand。
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


def build_hard_cases(args) -> List[Dict]:
    """
    加入几个固定 Job case。
    第一个就是你当前真实问题：
    4 GPU，8 Pod，每个 Pod 申请 1 vGPU。
    """

    cases = []

    cases.append(
        build_exact_job_scenario(
            scenario_id="hard-job-4gpu-8pod-1vgpu",
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
            scenario_id="hard-job-4gpu-4pod-1vgpu",
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
            scenario_id="hard-job-4gpu-12pod-small",
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
            scenario_id="hard-job-8gpu-16pod-1vgpu",
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
            scenario_id="hard-job-4gpu-4pod-2vgpu",
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train mixed-load DQN model with job-batch hard cases"
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
        default="DQN2/outputs_mixed_load_fixed_v8_job_hardcase",
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

    parser.add_argument("--disable-hard-cases", action="store_true")
    parser.add_argument("--hard-case-repeat", type=int, default=80)

    parser.add_argument("--hard-memory-total", type=float, default=49152.0)
    parser.add_argument("--hard-memory-demand", type=float, default=8192.0)
    parser.add_argument("--hard-core-total", type=float, default=100.0)
    parser.add_argument("--hard-core-demand", type=float, default=25.0)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    print(f"Using device: {device}")
    print(f"train path: {args.train_path}")
    print(f"val path  : {args.val_path}")
    print(f"output dir: {args.output_dir}")

    save_json(vars(args), os.path.join(args.output_dir, "train_args.json"))

    train_scenarios = load_jsonl(args.train_path)
    val_scenarios = load_jsonl(args.val_path)

    print(f"loaded train scenarios: {len(train_scenarios)}")
    print(f"loaded val scenarios  : {len(val_scenarios)}")

    if not train_scenarios:
        raise RuntimeError("empty train scenarios")

    if not val_scenarios:
        raise RuntimeError("empty val scenarios")

    if not args.disable_hard_cases:
        hard_cases = build_hard_cases(args)

        for _ in range(args.hard_case_repeat):
            train_scenarios.extend(copy.deepcopy(hard_cases))

        val_scenarios.extend(copy.deepcopy(hard_cases))

        print(f"hard cases added      : {len(hard_cases)}")
        print(f"hard case repeat      : {args.hard_case_repeat}")
        print(f"train scenarios final : {len(train_scenarios)}")
        print(f"val scenarios final   : {len(val_scenarios)}")

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
        eval_vgpu_success = ""
        eval_vgpu_failure = ""

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
            eval_vgpu_success = eval_result["avg_vgpu_success_rate"]
            eval_vgpu_failure = eval_result["avg_vgpu_failure_rate"]

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
                f"load={float(scenario.get('target_load', 0.0)):.3f} "
                f"reward={result['reward']:.4f} "
                f"objective={result['objective']:.4f} "
                f"balance_score={result['balance_score']:.4f} "
                f"success_rate={result['success_rate']:.4f} "
                f"vgpu_success_rate={result['vgpu_success_rate']:.4f} "
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
                f"load={float(scenario.get('target_load', 0.0)):.3f} "
                f"reward={result['reward']:.4f} "
                f"objective={result['objective']:.4f} "
                f"balance_score={result['balance_score']:.4f} "
                f"success_rate={result['success_rate']:.4f} "
                f"vgpu_success_rate={result['vgpu_success_rate']:.4f} "
                f"failure_rate={result['failure_rate']:.4f} "
                f"loss={loss:.6f} "
                f"epsilon={agent.epsilon:.4f}"
            )

        log_rows.append(
            {
                "episode": episode,
                "scenario_id": scenario["scenario_id"],
                "target_load": scenario.get("target_load", ""),
                "actual_load": result["actual_load"],
                "memory_load": result["memory_load"],
                "core_load": result["core_load"],
                "reward": result["reward"],
                "objective": result["objective"],
                "balance_score": result["balance_score"],
                "success_rate": result["success_rate"],
                "failure_rate": result["failure_rate"],
                "vgpu_success_rate": result["vgpu_success_rate"],
                "vgpu_failure_rate": result["vgpu_failure_rate"],
                "allocated_count": result["allocated_count"],
                "failure_count": result["failure_count"],
                "allocated_vgpu_count": result["allocated_vgpu_count"],
                "failure_vgpu_count": result["failure_vgpu_count"],
                "total_vgpu_count": result["total_vgpu_count"],
                "num_gpus": result["num_gpus"],
                "num_pods": result["num_pods"],
                "steps": result["steps"],
                "loss": loss,
                "epsilon": agent.epsilon,
                "eval_objective": eval_objective,
                "eval_balance_score": eval_balance,
                "eval_success_rate": eval_success,
                "eval_failure_rate": eval_failure,
                "eval_vgpu_success_rate": eval_vgpu_success,
                "eval_vgpu_failure_rate": eval_vgpu_failure,
                "best_eval_objective": best_eval_objective,
            }
        )

        if episode % args.log_save_interval == 0:
            write_csv(log_rows, log_path)

    write_csv(log_rows, log_path)
    agent.save(final_model_path, args=args)

    if not os.path.exists(best_model_path):
        agent.save(best_model_path, args=args)

    print(f"\ntraining log saved to : {log_path}")
    print(f"final model saved to  : {final_model_path}")
    print(f"best model saved to   : {best_model_path}")


if __name__ == "__main__":
    main()