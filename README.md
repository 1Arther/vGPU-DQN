下面这版可以直接作为 `README.md`。你当前仓库里的 README 还是很简单的占位内容。

你可以直接在服务器里执行：

````bash
cd ~/vGPU-DQN
cat > README.md <<'EOF'
# vGPU-DQN：基于 GNN-DQN 的单节点 vGPU 负载均衡仿真实验

本项目基于原有 DQN 调度代码，扩展实现了一个面向单节点多 GPU 场景的 vGPU 负载均衡仿真实验框架。实验目标是模拟多个 Pod 申请 vGPU 资源时，调度器如何在多张 GPU 之间进行分配，使 GPU 间显存和算力负载尽量均衡，同时尽可能提高 Pod 调度成功率。

当前版本主要用于仿真实验，不直接调用 Kubernetes / Volcano API，也不会真实创建 Pod。后续可以进一步扩展为接入 Volcano deviceshare 插件的真实调度实验。

---

## 1. 实验背景

在 Kubernetes + Volcano 的 vGPU 场景中，一个 Pod 可以通过如下资源字段申请 vGPU：

```yaml
resources:
  limits:
    volcano.sh/vgpu-number: 1
    volcano.sh/vgpu-memory: 4096
    volcano.sh/vgpu-cores: 10
````

当单个节点上存在多张物理 GPU 时，不同 Pod 的 vGPU 请求需要被分配到具体 GPU 上。如果调度策略不合理，可能出现部分 GPU 资源被快速打满，而其他 GPU 仍有较多剩余资源的情况，从而造成资源碎片和负载不均衡。

本项目尝试使用 GNN-DQN 学习 GPU-Pod 分配策略，并与启发式算法和随机算法进行对比。

---

## 2. 当前仿真假设

当前实验阶段做了如下简化：

1. 只研究单节点内部的多 GPU 负载均衡。
2. 一个 Pod 当前只分配到一张 GPU 上。
3. 一个 Pod 不支持跨节点分配。
4. GPU 资源包括：

   * `memory_total`
   * `memory_free`
   * `core_total`
   * `core_free`
   * `pod_count`
   * `util`
5. Pod 请求包括：

   * `vgpu_number`
   * `memory_demand`
   * `core_demand`
6. 当前训练中 Pod 批次不同，但 GPU 模板固定。
7. 测试阶段 DQN 关闭探索，即 `epsilon = 0`。
8. DQN、启发式算法、随机算法使用同一批测试 Pod 进行比较。

---

## 3. 项目结构

```text
vGPU-DQN/
├── DQN2/
│   ├── algorithm/
│   │   ├── vgpu_dqn_sim.py          # vGPU GNN-DQN 仿真实验主程序
│   │   └── draw_vgpu_sim.py         # 训练日志画图脚本
│   ├── vgpu_gpu_generator.py        # GPU 信息生成器
│   ├── vgpu_pod_generator.py        # Pod 任务生成器
│   ├── data/                        # 自动生成的实验数据
│   ├── outputs/                     # 默认输出目录
│   ├── outputs_30pods/              # 30 Pods 实验输出
│   └── outputs_8gpu_80pods/         # 8 GPU / 80 Pods 实验输出
└── README.md
```

---

## 4. 核心方法

### 4.1 图建模

将单节点内 GPU 和 Pod 建模为二分图：

```text
GPU 节点  <---- 可分配边 ---->  Pod 节点
```

如果某个 Pod 的资源请求可以被某张 GPU 满足，则二者之间存在一条有效边。

GPU 节点特征包括：

```text
memory_used_ratio
core_used_ratio
pod_count_ratio
util_ratio
memory_free_ratio
core_free_ratio
```

Pod 节点特征包括：

```text
memory_demand_ratio
core_demand_ratio
vgpu_number
allocated_flag
```

### 4.2 动作空间

每一步动作定义为：

```text
action = (gpu_idx, pod_idx)
```

含义是：将某个 Pod 分配到某张 GPU 上。

### 4.3 奖励函数

训练过程中同时考虑调度成功率和负载均衡。

即时奖励：

```python
reward = 1.0 - balance_score + 0.2 * success_rate
```

终止奖励：

```python
terminal_reward = 2.0 * success_rate - balance_score
```

其中：

```python
balance_score = std(memory_usage) + std(core_usage)
```

`balance_score` 越低表示 GPU 之间负载越均衡；`success_rate` 越高表示成功调度的 Pod 越多。

### 4.4 Best Checkpoint 选择

当前版本的 best model 选择逻辑不是简单按照训练 reward，也不是单纯按照 balance_score。

当前逻辑是：

```text
优先选择 success_rate 更高的模型；
当 success_rate 接近时，选择 balance_score 更低的模型。
```

因此当前 best model 更偏向保证较高调度成功率，同时兼顾负载均衡。

---

## 5. 对比算法

当前实验中使用了三类策略：

### 5.1 DQN

使用 GNN 编码 GPU-Pod 二分图，再通过 Q 网络输出每个 `(GPU, Pod)` 动作的 Q 值，选择 Q 值最高且合法的动作。

### 5.2 Least-loaded 启发式算法

每次选择当前负载最低的 GPU：

```text
score = memory_used_ratio + core_used_ratio
选择 score 最小的 GPU
```

该方法是一个简单启发式 baseline，接近 spread 思路，但不是严格等同于 Volcano 官方的 deviceshare spread 策略。

### 5.3 Random

从所有可分配 GPU 中随机选择一个 GPU。

---

## 6. 运行方式

### 6.1 3 GPU / 20 Pods 实验

```bash
python DQN2/algorithm/vgpu_dqn_sim.py \
  --episodes 2000 \
  --gpus 3 \
  --pods 20 \
  --lr 0.0003 \
  --train-batches 200 \
  --test-batches 20 \
  --output-dir DQN2/outputs \
  --data-dir DQN2/data/vgpu_sim \
  --regenerate-data
```

画图：

```bash
python DQN2/algorithm/draw_vgpu_sim.py \
  --log-path DQN2/outputs/vgpu_sim_training_log.csv \
  --output-dir DQN2/outputs/figures \
  --smooth-window 5
```

---

### 6.2 3 GPU / 30 Pods 实验

```bash
python DQN2/algorithm/vgpu_dqn_sim.py \
  --episodes 3000 \
  --gpus 3 \
  --pods 30 \
  --lr 0.0003 \
  --train-batches 500 \
  --test-batches 50 \
  --output-dir DQN2/outputs_30pods \
  --data-dir DQN2/data/vgpu_sim_30pods \
  --regenerate-data
```

画图：

```bash
python DQN2/algorithm/draw_vgpu_sim.py \
  --log-path DQN2/outputs_30pods/vgpu_sim_training_log.csv \
  --output-dir DQN2/outputs_30pods/figures \
  --smooth-window 5
```

---

### 6.3 8 GPU / 80 Pods 实验

```bash
python DQN2/algorithm/vgpu_dqn_sim.py \
  --episodes 6000 \
  --gpus 8 \
  --pods 80 \
  --lr 0.0003 \
  --train-batches 1200 \
  --test-batches 100 \
  --batch-size 64 \
  --hidden-dim 256 \
  --output-dir DQN2/outputs_8gpu_80pods \
  --data-dir DQN2/data/vgpu_sim_8gpu_80pods \
  --regenerate-data
```

画图：

```bash
python DQN2/algorithm/draw_vgpu_sim.py \
  --log-path DQN2/outputs_8gpu_80pods/vgpu_sim_training_log.csv \
  --output-dir DQN2/outputs_8gpu_80pods/figures \
  --smooth-window 5
```

---

## 7. 实验结果

### 7.1 最终测试结果汇总

| 场景              | 方法           | avg balance_score ↓ | avg success_rate ↑ |
| --------------- | ------------ | ------------------: | -----------------: |
| 3 GPU / 20 Pods | DQN          |              0.0597 |             0.9975 |
| 3 GPU / 20 Pods | Least-loaded |              0.1311 |             0.9925 |
| 3 GPU / 20 Pods | Random       |              0.2157 |             0.9950 |
| 3 GPU / 30 Pods | DQN          |              0.0941 |             0.8480 |
| 3 GPU / 30 Pods | Least-loaded |              0.1284 |             0.8180 |
| 3 GPU / 30 Pods | Random       |              0.1384 |             0.8167 |
| 8 GPU / 80 Pods | DQN          |              0.1440 |             0.8474 |
| 8 GPU / 80 Pods | Least-loaded |              0.1433 |             0.8294 |
| 8 GPU / 80 Pods | Random       |              0.1516 |             0.8238 |

---

## 8. 实验分析

### 8.1 3 GPU / 20 Pods

在 3 GPU / 20 Pods 场景下，DQN 的平均 `balance_score` 为 `0.0597`，明显低于 Least-loaded 的 `0.1311` 和 Random 的 `0.2157`。

同时，三种方法的 `success_rate` 都接近 1.0，说明该场景资源压力相对较低，主要考察的是调度后的负载均衡效果。

该实验说明：在可调度性较高的场景下，DQN 能够学习到更优的 GPU-Pod 分配策略，使 GPU 间显存和算力负载更加均衡。

---

### 8.2 3 GPU / 30 Pods

在 3 GPU / 30 Pods 场景下，任务压力明显增大。DQN 的平均 `balance_score` 为 `0.0941`，低于 Least-loaded 的 `0.1284` 和 Random 的 `0.1384`。

同时，DQN 的平均 `success_rate` 为 `0.8480`，也高于 Least-loaded 的 `0.8180` 和 Random 的 `0.8167`。

该实验说明：在更高负载场景下，DQN 不仅能够降低 GPU 间负载差异，还能提高 Pod 调度成功率。

---

### 8.3 8 GPU / 80 Pods

在 8 GPU / 80 Pods 的大规模场景下，DQN 的平均 `success_rate` 为 `0.8474`，高于 Least-loaded 的 `0.8294` 和 Random 的 `0.8238`。

但在 `balance_score` 上，DQN 为 `0.1440`，Least-loaded 为 `0.1433`，二者基本持平，DQN 略差于 Least-loaded，但优于 Random 的 `0.1516`。

该实验说明：在更大动作空间和更高资源压力下，DQN 主要优势体现在提高调度成功率；负载均衡能力与启发式方法接近，但没有明显超过 Least-loaded。

因此，8 GPU / 80 Pods 更适合作为扩展性实验，说明 DQN 在大规模场景下仍具有一定调度能力，但 success_rate 和 balance_score 之间存在明显权衡。

---

## 9. 主要结论

当前实验可以得到以下结论：

1. 在 3 GPU / 20 Pods 场景下，DQN 明显提升了负载均衡效果。
2. 在 3 GPU / 30 Pods 场景下，DQN 同时提升了调度成功率和负载均衡效果。
3. 在 8 GPU / 80 Pods 场景下，DQN 提高了调度成功率，但负载均衡效果与 Least-loaded 基本持平。
4. 随着规模增大，动作空间扩大，DQN 训练难度明显增加。
5. 高负载场景下，调度成功率和负载均衡之间存在一定冲突。

---

## 10. 当前局限

当前版本仍有以下不足：

1. Least-loaded 只是简单启发式 baseline，不是严格的 Volcano 官方 binpack / spread 实现。
2. 当前 GPU 模板固定，训练中不同 episode 使用同一组 GPU 初始资源。
3. 当前 Pod 可以变化，但 GPU 资源配置尚未按 batch 动态生成。
4. 当前只做单节点内 GPU 间负载均衡，没有做多节点调度。
5. 当前一个 Pod 默认分配到一张 GPU，没有实现单 Pod 多 GPU 分配。
6. 当前没有接入真实 Kubernetes / Volcano 调度链路。
7. 只使用单随机种子实验，后续需要多随机种子重复实验增强稳定性。

---

## 11. 后续工作

后续可以从以下方向继续改进：

### 11.1 增加 Volcano 风格 baseline

将当前 Least-loaded 替换或扩展为：

```text
Volcano-binpack
Volcano-spread
Random
DQN
```

其中：

```text
binpack：倾向于将任务压实到已有负载 GPU 上，减少碎片。
spread：倾向于将任务分散到低负载 GPU 上，提高均衡性。
```

### 11.2 动态 GPU + 动态 Pod

当前训练是：

```text
不同 Pod 批次 + 固定 GPU 模板
```

后续可以改为：

```text
不同 Pod 批次 + 不同 GPU 批次
```

即每个 episode 使用不同 GPU 初始配置，以增强模型对异构 GPU 环境的泛化能力。

### 11.3 多目标 best checkpoint

当前 best model 选择更偏向 success_rate。后续可以同时保存：

```text
best_success_model
best_balance_model
best_weighted_model
```

其中 weighted objective 可以定义为：

```python
eval_objective = 2.0 * eval_success_rate - eval_balance_score
```

这样可以分别分析不同优化目标下的调度行为。

### 11.4 接入真实 Volcano

后续可以将 DQN 策略接入真实调度流程，与 Volcano deviceshare 的 binpack / spread 策略进行真实集群对比。

---

## 12. 结果文件说明

训练后会生成：

```text
vgpu_sim_training_log.csv
```

其中主要字段包括：

```text
episode
reward
balance_score
loss
epsilon
steps
allocated_count
success_rate
eval_balance_score
eval_success_rate
best_eval_score
best_eval_success
```

画图脚本会输出：

```text
reward_curve.png
balance_score_curve.png
loss_curve.png
epsilon_curve.png
success_rate_curve.png
allocated_count_curve.png
eval_balance_score_curve.png
eval_success_rate_curve.png
best_eval_score_curve.png
best_eval_success_curve.png
```

其中：

* `reward_curve` 反映训练奖励变化。
* `balance_score_curve` 反映训练批次上的负载均衡情况。
* `eval_balance_score_curve` 反映固定测试集上的负载均衡效果。
* `success_rate_curve` 反映训练批次上的调度成功率。
* `eval_success_rate_curve` 反映固定测试集上的调度成功率。
* `loss_curve` 只用于观察训练是否稳定，不作为最终性能指标。
* `epsilon_curve` 反映探索率衰减过程。

---

## 13. 注意事项

1. `balance_score` 越低越好。
2. `success_rate` 越高越好。
3. `loss` 不是最终性能指标。
4. 训练曲线波动是正常现象，因为不同 episode 的 Pod 批次不同。
5. 最终结论应以固定测试集上的 `eval_balance_score` 和 `eval_success_rate` 为主。
6. 当前 best model 更偏向调度成功率，因此大规模场景下可能出现 success_rate 提升但 balance_score 不明显下降的情况。

EOF

````

然后提交：

```bash
git add README.md
git commit -m "Update README with vGPU DQN experiment results"
git push origin feature/vgpu-sim-generator
````

这版 README 已经把三个实验、命令、结果表、结论和局限都写进去了。
