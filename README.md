可以，先做两件事：

1. **把当前分支代码提交并 push 到 GitHub**
2. **把 README.md 改成当前 hami-core / Volcano baseline / 动态 batch 版本说明**

下面直接照着终端执行。

---

## 1. 先确认当前分支

```bash
cd ~/vGPU-DQN

git branch --show-current
git status
```

如果当前分支就是你现在改代码的分支，继续。
如果你还在 `feature/vgpu-sim-generator`，也可以直接提交到这个分支。

---

## 2. 覆盖 README.md

执行：

````bash
cat > README.md <<'EOF'
# vGPU-DQN：面向 hami-core 的单节点 vGPU 负载均衡仿真实验

本项目基于原有 DQN 调度代码，扩展实现了一个面向 **hami-core / vGPU 场景** 的单节点多 GPU 负载均衡仿真实验框架。

实验目标是：在多个 Pod 申请 vGPU 资源时，学习一个 GPU-Pod 分配策略，使调度器在提高 Pod 调度成功率的同时，尽量降低 GPU 之间的显存和算力负载差异。

当前版本是仿真实验版本，不直接调用 Kubernetes / Volcano API，也不会真实创建 Pod。后续可以进一步接入 Volcano deviceshare 或 HAMi 调度链路。

---

## 1. 当前研究场景

当前主要研究的是单节点内多 GPU 的 vGPU 分配问题。

Pod 请求形式参考 hami-core / Volcano vGPU 风格：

```yaml
resources:
  limits:
    volcano.sh/vgpu-number: 1
    volcano.sh/vgpu-memory: 4096
    volcano.sh/vgpu-cores: 10
````

在单节点多 GPU 场景下，调度器需要决定每个 Pod 分配到哪张 GPU 上。若分配策略不合理，可能导致部分 GPU 被快速打满，而其他 GPU 仍有空闲资源，从而造成资源碎片、调度失败和负载不均衡。

---

## 2. 当前版本主要改动

相比原始版本，当前分支主要做了以下改动：

### 2.1 支持 hami-core 风格 vGPU 资源

Pod 任务包含：

```text
vgpu_number
memory_demand
core_demand
```

GPU 节点包含：

```text
memory_total
memory_free
core_total
core_free
pod_count
util
```

当前默认一个 Pod 分配到一张 GPU 上，暂不做跨节点分配，也暂不做单 Pod 跨多 GPU 分配。

---

### 2.2 支持动态 GPU batch 和动态 Pod batch

当前版本不再只使用固定 GPU 模板，而是支持每个 batch 随机生成 GPU 和 Pod。

也就是说，训练时：

```text
episode 1: 一组随机 GPU + 一组随机 Pod
episode 2: 另一组随机 GPU + 另一组随机 Pod
...
```

测试时也是：

```text
test batch 1: 一组随机 GPU + 一组随机 Pod
test batch 2: 另一组随机 GPU + 另一组随机 Pod
...
```

这样可以避免模型只适应固定 GPU 和固定 Pod 分布，提高实验泛化性。

---

### 2.3 增加 scenario 数据结构

当前每个实验 batch 被保存为一个 scenario：

```json
{
  "batch_id": 0,
  "mode": "hami-core",
  "num_gpus": 3,
  "num_pods": 30,
  "gpus": [],
  "pods": []
}
```

训练集和测试集分别保存为：

```text
train_scenarios.json
test_scenarios.json
```

---

### 2.4 增加多种 baseline

当前对比方法包括：

```text
DQN
Volcano-binpack
Volcano-spread
Simple-spread
Random
```

各方法含义如下：

| 方法              | 含义                         |
| --------------- | -------------------------- |
| DQN             | 使用 GNN-DQN 学习 GPU-Pod 分配策略 |
| Volcano-binpack | 倾向于将任务压实到已有负载较高的 GPU 上     |
| Volcano-spread  | 倾向于将任务分散到放置后负载较低的 GPU 上    |
| Simple-spread   | 简单选择当前负载最低的 GPU            |
| Random          | 从所有可行 GPU 中随机选择            |

其中 `Volcano-binpack` 和 `Volcano-spread` 是仿照 Volcano deviceshare 思路实现的启发式 baseline，不是直接调用 Volcano 源码。

---

### 2.5 增加综合目标函数

当前版本不再只看成功率或只看负载均衡，而是使用统一的综合目标：

```text
objective = α * success_rate - β * balance_score - γ * failure_rate
```

默认权重为：

```text
α = 2.0
β = 1.0
γ = 2.0
```

其中：

```text
success_rate：调度成功率，越高越好
balance_score：GPU 间负载不均衡程度，越低越好
failure_rate：调度失败率，越低越好
```

因此：

```text
objective 越高越好
```

当前 best checkpoint 也是根据 DQN 在固定测试集上的 `objective` 选择，而不是根据 baseline 结果反向挑模型。

---

### 2.6 增加失败惩罚

当 Pod 无法继续分配时，会根据失败 Pod 数量进行惩罚：

```text
failed_pod_penalty * failure_count
```

这样可以避免模型只追求表面上的均衡，而忽略大量 Pod 分配失败的问题。

---

### 2.7 增加 early stopping

当前版本支持早停机制：

```bash
--early-stop-patience 10
--early-stop-min-delta 1e-4
```

含义是：如果连续多次 evaluation 中 DQN 的综合 objective 没有明显提升，则提前停止训练。

---

### 2.8 保存完整测试比较结果

测试阶段会在同一批测试场景上比较所有方法，并保存：

```text
test_comparison_detail.csv
test_comparison_summary.csv
```

其中：

```text
test_comparison_detail.csv
```

保存每个 batch、每种方法的详细结果。

```text
test_comparison_summary.csv
```

保存每种方法在所有测试 batch 上的平均结果。

---

## 3. 项目结构

```text
vGPU-DQN/
├── DQN2/
│   ├── algorithm/
│   │   ├── vgpu_dqn_sim.py          # hami-core vGPU GNN-DQN 主训练脚本
│   │   ├── draw_vgpu_sim.py         # 训练曲线绘图脚本
│   │   └── draw_vgpu_compare.py     # 多方法对比绘图脚本
│   ├── vgpu_gpu_generator.py        # GPU batch 生成器
│   ├── vgpu_pod_generator.py        # Pod batch 生成器
│   ├── vgpu_scenario_generator.py   # GPU + Pod scenario 生成器
│   ├── data/                        # 实验数据
│   └── outputs*/                    # 实验输出目录
└── README.md
```

---

## 4. 核心建模方法

### 4.1 图建模

将单节点内 GPU 和 Pod 建模为二分图：

```text
GPU 节点  <---- 可分配边 ---->  Pod 节点
```

如果某个 Pod 的资源请求可以被某张 GPU 满足，则二者之间存在一条有效边。

---

### 4.2 GPU 节点特征

```text
memory_used_ratio
core_used_ratio
pod_count_ratio
util_ratio
memory_free_ratio
core_free_ratio
```

---

### 4.3 Pod 节点特征

```text
memory_demand_ratio
core_demand_ratio
vgpu_number
allocated_flag
```

---

### 4.4 动作空间

每一步动作定义为：

```text
action = (gpu_idx, pod_idx)
```

含义是：将某个 Pod 分配到某张 GPU 上。

---

### 4.5 负载均衡指标

当前负载均衡指标为：

```python
balance_score = std(memory_usage) + std(core_usage)
```

其中：

```text
memory_usage = 1 - memory_free / memory_total
core_usage   = 1 - core_free / core_total
```

`balance_score` 越低，说明 GPU 之间负载越均衡。

---

## 5. 运行方式

### 5.1 固定规模实验：3GPU / 30Pods

```bash
python DQN2/algorithm/vgpu_dqn_sim.py \
  --episodes 3000 \
  --gpus 3 \
  --pods 30 \
  --train-batches 500 \
  --test-batches 50 \
  --batch-size 32 \
  --hidden-dim 128 \
  --lr 0.0003 \
  --success-weight 2.0 \
  --balance-weight 1.0 \
  --failure-weight 2.0 \
  --output-dir DQN2/outputs_hami_3gpu_30pods \
  --data-dir DQN2/data/hami_3gpu_30pods \
  --regenerate-data
```

---

### 5.2 动态规模实验：3~8 GPU / 20~80 Pods

```bash
python DQN2/algorithm/vgpu_dqn_sim.py \
  --episodes 6000 \
  --min-gpus 3 \
  --max-gpus 8 \
  --min-pods 20 \
  --max-pods 80 \
  --train-batches 1200 \
  --test-batches 50 \
  --batch-size 64 \
  --hidden-dim 256 \
  --lr 0.0003 \
  --success-weight 2.0 \
  --balance-weight 1.0 \
  --failure-weight 2.0 \
  --output-dir DQN2/outputs_hami_dynamic \
  --data-dir DQN2/data/hami_dynamic \
  --regenerate-data
```

---

## 6. 画图方式

### 6.1 训练曲线

```bash
python DQN2/algorithm/draw_vgpu_sim.py \
  --log-path DQN2/outputs_hami_3gpu_30pods/vgpu_sim_training_log.csv \
  --output-dir DQN2/outputs_hami_3gpu_30pods/figures \
  --smooth-window 5
```

---

### 6.2 多方法对比图

```bash
python DQN2/algorithm/draw_vgpu_compare.py \
  --detail-path DQN2/outputs_hami_3gpu_30pods/test_comparison_detail.csv \
  --summary-path DQN2/outputs_hami_3gpu_30pods/test_comparison_summary.csv \
  --output-dir DQN2/outputs_hami_3gpu_30pods/compare_figures
```

会输出：

```text
avg_balance_score_bar.png
avg_success_rate_bar.png
avg_failure_rate_bar.png
avg_objective_bar.png
balance_score_test_batches.png
success_rate_test_batches.png
failure_rate_test_batches.png
objective_test_batches.png
```

---

## 7. 当前样例结果：3GPU / 30Pods

当前一次 3GPU / 30Pods 实验结果如下：

| 方法              | avg_balance_score ↓ | avg_success_rate ↑ | avg_failure_rate ↓ | avg_objective ↑ |
| --------------- | ------------------: | -----------------: | -----------------: | --------------: |
| DQN             |              0.1858 |             0.6633 |             0.3367 |          0.4675 |
| Random          |              0.1876 |             0.5573 |             0.4427 |          0.0418 |
| Simple-spread   |              0.1744 |             0.5693 |             0.4307 |          0.1029 |
| Volcano-binpack |              0.2137 |             0.5407 |             0.4593 |         -0.0510 |
| Volcano-spread  |              0.1583 |             0.5833 |             0.4167 |          0.1750 |

可以看出：

1. DQN 的调度成功率最高。
2. DQN 的综合 objective 最高。
3. Volcano-spread 的纯负载均衡指标最好。
4. DQN 在高负载场景下更偏向提高调度成功率，负载均衡效果没有完全超过 spread。

---

## 8. 当前实验结论

当前阶段可以得到以下结论：

1. DQN 能够学习到比随机策略和简单启发式策略更高的综合调度收益。
2. 在高负载场景下，DQN 更倾向于提高 Pod 调度成功率。
3. Volcano-spread 在单纯负载均衡指标上较强，但调度成功率低于 DQN。
4. Volcano-binpack 在单节点 GPU 负载均衡目标下表现较弱，因为它倾向于压实资源。
5. 固定 Pod 数量不能完全代表调度压力，后续需要引入负载强度控制实验。

---

## 9. 下一步计划：负载强度控制实验

后续实验将引入 `target_load` 控制数据生成：

```text
target_load = Pod 总资源需求 / GPU 总资源容量
```

计划对比：

```text
target_load = 0.6 / 0.8 / 1.0 / 1.2 / 1.5
```

每组负载强度下生成 50 个测试 batch，并比较：

```text
DQN
Volcano-binpack
Volcano-spread
Simple-spread
Random
```

重点观察：

```text
success_rate
balance_score
failure_rate
objective
```

这样可以更清楚地分析不同资源压力下各调度方法的表现。

---

## 10. 当前局限

当前版本仍有以下不足：

1. Volcano-binpack 和 Volcano-spread 是仿真 baseline，不是直接调用 Volcano 源码。
2. 当前只研究单节点内 GPU 分配，没有研究多节点调度。
3. 当前一个 Pod 默认分配到一张 GPU，没有实现单 Pod 多 GPU 分配。
4. 当前尚未接入真实 Kubernetes / Volcano / HAMi 调度链路。
5. 当前训练结果仍受随机种子和数据分布影响，后续需要多随机种子实验。
6. 当前固定 Pod 数量实验无法严格控制负载压力，后续需要做负载强度控制实验。

---

## 11. 输出文件说明

训练后会生成：

```text
vgpu_sim_training_log.csv
vgpu_dqn_sim_best.pth
vgpu_dqn_sim_final.pth
test_comparison_detail.csv
test_comparison_summary.csv
```

其中：

```text
vgpu_sim_training_log.csv
```

记录训练过程中的 reward、objective、balance_score、success_rate、loss、epsilon 等信息。

```text
test_comparison_detail.csv
```

记录每个测试 batch 上每种方法的详细结果。

```text
test_comparison_summary.csv
```

记录所有测试 batch 上每种方法的平均结果。

---

## 12. 指标说明

| 指标            | 含义           | 趋势         |
| ------------- | ------------ | ---------- |
| success_rate  | Pod 调度成功率    | 越高越好       |
| failure_rate  | Pod 调度失败率    | 越低越好       |
| balance_score | GPU 间负载不均衡程度 | 越低越好       |
| objective     | 综合目标函数       | 越高越好       |
| loss          | Q 网络训练损失     | 只用于观察训练稳定性 |

最终方法比较应主要看：

```text
success_rate
balance_score
failure_rate
objective
```

其中 `objective` 是当前版本的综合评价指标。
EOF

````

---

## 3. 提交当前代码和 README

先看当前改动：

```bash
git status
````

如果你不想上传模型 `.pth` 和太大的实验输出，建议先只提交代码和 README：

```bash
git add README.md
git add DQN2/algorithm/vgpu_dqn_sim.py
git add DQN2/algorithm/draw_vgpu_sim.py
git add DQN2/algorithm/draw_vgpu_compare.py
git add DQN2/vgpu_gpu_generator.py
git add DQN2/vgpu_pod_generator.py
git add DQN2/vgpu_scenario_generator.py

git commit -m "Add hami-core dynamic vGPU simulation and baseline comparison"
```

如果你还想把当前实验结果也一起提交：

```bash
git add -f DQN2/outputs_hami_3gpu_30pods/test_comparison_detail.csv
git add -f DQN2/outputs_hami_3gpu_30pods/test_comparison_summary.csv
git add -f DQN2/outputs_hami_3gpu_30pods/vgpu_sim_training_log.csv
git add -f DQN2/outputs_hami_3gpu_30pods/figures/
git add -f DQN2/outputs_hami_3gpu_30pods/compare_figures/

git commit -m "Add hami-core 3GPU 30Pods experiment results"
```

不建议上传 `.pth` 模型文件，除非老师明确要求。

---

## 4. push 到 GitHub

你之前 HTTPS push 不稳定，SSH 已经通了，所以先改 remote：

```bash
git remote set-url origin git@github.com:1Arther/vGPU-DQN.git
git remote -v
```

确认显示：

```text
origin  git@github.com:1Arther/vGPU-DQN.git (fetch)
origin  git@github.com:1Arther/vGPU-DQN.git (push)
```

然后推送当前分支：

```bash
CURRENT_BRANCH=$(git branch --show-current)
git push -u origin "$CURRENT_BRANCH"
```

---

## 5. 如果 commit 提示 nothing to commit

说明你之前已经提交过代码，只需要 push：

```bash
CURRENT_BRANCH=$(git branch --show-current)
git push -u origin "$CURRENT_BRANCH"
```

---

## 6. 最后检查

```bash
git status
git log --oneline -5
```

GitHub 上刷新当前分支，确认 README.md 已经变成新版说明。
