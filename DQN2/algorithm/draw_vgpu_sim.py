"""
vGPU-DQN 训练日志画图脚本。

读取：
    DQN2/outputs/vgpu_sim_training_log.csv

输出：
    DQN2/outputs/figures/reward_curve.png
    DQN2/outputs/figures/balance_score_curve.png
    DQN2/outputs/figures/loss_curve.png
    DQN2/outputs/figures/epsilon_curve.png
    DQN2/outputs/figures/success_rate_curve.png

运行方式：
    python DQN2/algorithm/draw_vgpu_sim.py

也可以指定路径：
    python DQN2/algorithm/draw_vgpu_sim.py \
        --log-path DQN2/outputs/vgpu_sim_training_log.csv \
        --output-dir DQN2/outputs/figures
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def smooth_series(series: pd.Series, window: int = 20) -> pd.Series:
    """
    平滑曲线，避免训练曲线抖动太厉害。
    window 越大，曲线越平滑。
    """
    if window <= 1:
        return series

    return series.rolling(window=window, min_periods=1).mean()


def plot_metric(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_dir: str,
    smooth_window: int = 20,
):
    """
    画单个指标曲线。

    参数：
        df: 训练日志 DataFrame
        x_col: 横轴，一般是 episode
        y_col: 纵轴，例如 reward / balance_score / loss
        output_dir: 图片保存目录
        smooth_window: 平滑窗口
    """
    if y_col not in df.columns:
        print(f"[skip] column not found: {y_col}")
        return

    os.makedirs(output_dir, exist_ok=True)

    x = df[x_col]
    y = df[y_col]
    y_smooth = smooth_series(y, smooth_window)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, alpha=0.35, label=f"raw {y_col}")
    plt.plot(x, y_smooth, linewidth=2, label=f"smoothed {y_col}")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    save_path = os.path.join(output_dir, f"{y_col}_curve.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[saved] {save_path}")


def plot_training_log(
    log_path: str = "DQN2/outputs/vgpu_sim_training_log.csv",
    output_dir: str = "DQN2/outputs/figures",
    smooth_window: int = 20,
):
    """
    统一画 vGPU-DQN 训练曲线。

    主要看：
        reward: 训练奖励，越高越好
        balance_score: 最终负载均衡指标，越低越好
        loss: Q 网络拟合误差，只看训练是否稳定
        epsilon: 探索率，应该逐渐下降
        success_rate: Pod 调度成功率，越高越好
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"training log not found: {log_path}")

    df = pd.read_csv(log_path)

    if "episode" not in df.columns:
        raise ValueError("CSV must contain column: episode")

    metrics = [
        "reward",
        "balance_score",
        "loss",
        "epsilon",
        "success_rate",
        "allocated_count",
        "steps",
    ]

    for metric in metrics:
        plot_metric(
            df=df,
            x_col="episode",
            y_col=metric,
            output_dir=output_dir,
            smooth_window=smooth_window,
        )


def main():
    parser = argparse.ArgumentParser(description="Plot vGPU-DQN training curves")

    parser.add_argument(
        "--log-path",
        type=str,
        default="DQN2/outputs/vgpu_sim_training_log.csv",
        help="训练日志 CSV 路径",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="DQN2/outputs/figures",
        help="图片输出目录",
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=20,
        help="曲线平滑窗口",
    )

    args = parser.parse_args()

    plot_training_log(
        log_path=args.log_path,
        output_dir=args.output_dir,
        smooth_window=args.smooth_window,
    )


if __name__ == "__main__":
    main()