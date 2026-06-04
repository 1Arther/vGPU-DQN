# vGPU-DQN

本项目用于研究基于 DQN 的 vGPU 调度策略。当前分支主要围绕 **Volcano vGPU 场景下的 Job 批次调度** 进行建模、训练和评估。

## 1. 项目目标

在 Volcano vGPU 场景中，一个 Job 通常包含一批 Pod，每个 Pod 可以申请一个或多个 vGPU。调度器需要根据当前节点内 GPU 的资源状态，为 Job 内的 Pod 选择合适的 GPU，从而提升资源利用率、降低分配失败率，并尽量保持负载均衡。

当前实验主要关注：

- 同一节点内 GPU 同规格；
- 不同场景可以对应不同型号 GPU 节点；
- 每个场景从空节点开始；
- 一个 Job 内包含一批 Pod；
- Pod 可以具有不同的 vGPU 数量和显存需求；
- 使用 DQN 学习 Job 内 Pod 与 GPU 之间的分配策略；
- 与固定索引扫描、显存启发式策略和随机策略进行对比。

## 2. 目录结构

```text
DQN2/
├── algorithm/
│   ├── vgpu_dqn_sim.py
│   ├── generate_mixed_load_fixed_homogeneous.py
│   ├── train_vgpu_mixed_job_hardcase.py
│   ├── test_vgpu_mixed_homo_real.py
│   ├── test_exact_4gpu_8pod_direct.py
│   └── draw_eval_formal.py
├── data_mixed_load_fixed_homo_real_hard_clean/
├── outputs_mixed_load_fixed_v9_homo_real_hard/
├── outputs_mixed_load_fixed_v9_homo_real_hard_eval/
└── grpc/

核心文件说明：

文件	说明
vgpu_dqn_sim.py	DQN 环境、图构造、模型、reward、baseline 与评估逻辑
generate_mixed_load_fixed_homogeneous.py	生成同构 GPU 节点下的 Job 批次数据
train_vgpu_mixed_job_hardcase.py	使用生成数据训练 DQN 模型
test_vgpu_mixed_homo_real.py	在测试集上评估 DQN 和 baseline
test_exact_4gpu_8pod_direct.py	测试 4GPU/8Pod 固定场景
draw_eval_formal.py	绘制正式实验图，并过滤 hard case
3. 建模方式

一个 scenario 表示一次 Job 调度实验：

当前节点 GPU 状态 + 当前 Job 的 Pod 批次

其中 GPU 初始为空闲状态：

memory_free = memory_total
pod_count = 0

同一 scenario 内 GPU 同规格，但不同 scenario 可以对应不同型号节点，例如：

scenario A: 4 张 24576 MB GPU
scenario B: 8 张 49152 MB GPU
scenario C: 4 张 81920 MB GPU

Pod 具有以下资源需求：

vgpu_number
memory_demand

其中 vgpu_number 表示该 Pod 需要的 vGPU 数量，memory_demand 表示每个 vGPU slice 的显存需求。

4. Baseline 策略

当前实验使用以下 baseline：

方法	说明
index-desc	按 GPU 索引从高到低扫描，近似固定设备顺序分配
used-mem-asc	优先选择已使用显存较少的 GPU，类似显存 spread
used-mem-desc	优先选择已使用显存较多的 GPU，类似显存 binpack
random	随机选择可行 GPU
dqn	使用 DQN 根据当前图状态选择 GPU 与 Pod

当前阶段暂时以显存资源为主要瓶颈进行实验，core 维度后续可以继续加入更细粒度的建模和 baseline。

5. Balance 指标

当前 `balance_score` 同时包含两部分：

- GPU 间负载均衡：所有 GPU 的 memory 使用率标准差 + core 使用率标准差；
- GPU 内负载均衡：每张 GPU 内部 `abs(memory_used_ratio - core_used_ratio)` 的平均值。

默认权重：

```text
balance_score = inter_gpu_balance_score + intra_gpu_balance_score
```

训练和测试可通过以下参数调整：

```text
--inter-balance-weight 1.0
--intra-balance-weight 1.0
--delta-balance-weight 1.0
```

Reward 中额外加入每一步的 balance 改善量：

```text
balance_delta = previous_balance_score - current_balance_score
step_reward += delta_balance_weight * balance_delta
```

模型输入也显式加入 memory/core 失衡特征：

- GPU 特征包含 `abs(memory_used_ratio - core_used_ratio)` 和 `memory_used_ratio - core_used_ratio`；
- Pod 特征包含 `abs(memory_demand_ratio - core_demand_ratio)` 和 `memory_demand_ratio - core_demand_ratio`。

6. 数据生成

生成更真实的随机测试数据：

python DQN2/algorithm/generate_mixed_load_fixed_homogeneous.py \
  --output-dir DQN2/data_mixed_load_fixed_homo_real_hard_clean \
  --loads 0.6,0.8,1.0,1.2,1.5,1.8 \
  --train-per-load 800 \
  --val-per-load 120 \
  --test-per-load 120 \
  --min-gpus 4 \
  --max-gpus 8 \
  --min-pods 8 \
  --max-pods 28 \
  --gpu-memory-choices 24576 40960 49152 81920 \
  --gpu-core-total 100 \
  --hard-memory-total 49152 \
  --hard-memory-demand 8192 \
  --hard-core-total 100 \
  --hard-core-demand 25 \
  --hard-case-splits train \
  --strict-actual-load \
  --actual-load-tolerance 0.12

说明：

- 随机场景会按 actual_load 过滤，避免 target_load=0.6 的桶混入实际高负载样本；
- 固定 hard case 默认只进入 train，不进入 val/test；
- 如果需要复现实验中的旧行为，可以添加 `--hard-case-splits train val test --no-strict-actual-load`。

生成结果：

DQN2/data_mixed_load_fixed_homo_real_hard_clean/
├── train_scenarios.jsonl
├── val_scenarios.jsonl
└── test_scenarios.jsonl

7. Conflict 展示测试集

用于单独验证 memory/core 互补调度能力。每个 Job 都混合：

- 高 mem / 低 core Pod；
- 低 mem / 高 core Pod；
- 少量均衡 Pod。

该测试集不替代 clean test，只用于展示模型在冲突型资源组合上的优势。

```bash
python DQN2/algorithm/generate_conflict_homogeneous.py \
  --output-dir DQN2/data_conflict_homo_real \
  --loads 0.6,0.8,1.0,1.2,1.3 \
  --per-load 120
```

输出：

```text
DQN2/data_conflict_homo_real/conflict_test_scenarios.jsonl
```

评估：

```bash
python DQN2/algorithm/test_vgpu_mixed_homo_real.py \
  --model-path DQN2/outputs_mixed_load_fixed_v13_homo_real_hard_clean_delta_feature/vgpu_dqn_mixed_best.pth \
  --test-path DQN2/data_conflict_homo_real/conflict_test_scenarios.jsonl \
  --output-dir DQN2/outputs_mixed_load_fixed_v13_conflict_eval \
  --inter-balance-weight 1.0 \
  --intra-balance-weight 1.0 \
  --delta-balance-weight 2.0
```

8. 已有负载场景

用于模拟非空节点和 GPU 碎片。生成器会给随机场景中的每张 GPU 注入已有
memory/core 使用率和 pod_count，包含 balanced、memory-heavy、core-heavy
等已有负载形态。

```bash
python DQN2/algorithm/generate_mixed_load_fixed_homogeneous.py \
  --output-dir DQN2/data_mixed_load_preload_homo_real \
  --loads 0.6,0.8,1.0,1.2,1.3 \
  --train-per-load 900 \
  --val-per-load 150 \
  --test-per-load 150 \
  --hard-case-splits train \
  --enable-existing-load \
  --existing-load-min 0.05 \
  --existing-load-max 0.40
```

大模型训练示例：

```bash
python DQN2/algorithm/train_vgpu_mixed_job_hardcase.py \
  --train-path DQN2/data_mixed_load_preload_homo_real/train_scenarios.jsonl \
  --val-path DQN2/data_mixed_load_preload_homo_real/val_scenarios.jsonl \
  --output-dir DQN2/outputs_mixed_load_preload_v14_big_delta_feature \
  --episodes 12000 \
  --hidden-dim 512 \
  --batch-size 128 \
  --buffer-size 50000 \
  --lr 2e-4 \
  --inter-balance-weight 1.0 \
  --intra-balance-weight 1.0 \
  --delta-balance-weight 2.0
```
9. 模型训练
python DQN2/algorithm/train_vgpu_mixed_job_hardcase.py \
  --train-path DQN2/data_mixed_load_fixed_homo_real_hard_clean/train_scenarios.jsonl \
  --val-path DQN2/data_mixed_load_fixed_homo_real_hard_clean/val_scenarios.jsonl \
  --output-dir DQN2/outputs_mixed_load_fixed_v9_homo_real_hard \
  --episodes 10000 \
  --hard-case-repeat 20 \
  --hard-memory-total 49152 \
  --hard-memory-demand 8192 \
  --hard-core-total 100 \
  --hard-core-demand 25

训练输出：

DQN2/outputs_mixed_load_fixed_v9_homo_real_hard/
├── vgpu_dqn_mixed_best.pth
├── vgpu_dqn_mixed_final.pth
├── train_args.json
└── vgpu_mixed_training_log.csv
10. 固定场景测试

用于验证模型在简单规整场景下的行为。

场景：

4 GPU
8 Pod
每个 Pod 申请 1 vGPU
每个 Pod memory_demand = 8192

运行：

python DQN2/algorithm/test_exact_4gpu_8pod_direct.py \
  --model DQN2/outputs_mixed_load_fixed_v9_homo_real_hard/vgpu_dqn_mixed_best.pth \
  --memory-total 49152 \
  --memory-demand 8192 \
  --core-total 100 \
  --core-demand 25

理想分配结果：

final distribution: [2, 2, 2, 2]
11. 测试集评估
python DQN2/algorithm/test_vgpu_mixed_homo_real.py \
  --model-path DQN2/outputs_mixed_load_fixed_v9_homo_real_hard/vgpu_dqn_mixed_best.pth \
  --test-path DQN2/data_mixed_load_fixed_homo_real_hard_clean/test_scenarios.jsonl \
  --output-dir DQN2/outputs_mixed_load_fixed_v9_homo_real_hard_eval

输出：

DQN2/outputs_mixed_load_fixed_v9_homo_real_hard_eval/
├── mixed_load_test_detail.csv
├── mixed_load_test_summary.csv
└── test_args.json
12. 绘制实验图

过滤 hard case，只保留随机测试负载：

python DQN2/algorithm/draw_eval_formal.py \
  --summary DQN2/outputs_mixed_load_fixed_v9_homo_real_hard_eval/mixed_load_test_summary.csv \
  --detail DQN2/outputs_mixed_load_fixed_v9_homo_real_hard_eval/mixed_load_test_detail.csv \
  --output-dir DQN2/outputs_mixed_load_fixed_v9_homo_real_hard_eval/formal_figures \
  --random-loads 0.6,0.8,1.0,1.2,1.5,1.8

生成图像和表格：

DQN2/outputs_mixed_load_fixed_v9_homo_real_hard_eval/formal_figures/
├── avg_allocated_count.png
├── avg_allocated_vgpu_count.png
├── avg_balance_score.png
├── avg_failure_count.png
├── avg_failure_rate.png
├── avg_success_rate.png
├── avg_vgpu_failure_rate.png
├── avg_vgpu_success_rate.png
├── random_load_summary.csv
└── hard_case_detail_table.csv
13. 当前实验结论

当前结果表明：

在低负载下，各种方法差距较小；
随着负载升高，DQN 在 allocated_count 和 allocated_vgpu_count 上逐渐优于启发式方法；
DQN 在高负载下具有更低的 failure_count 和 failure_rate；
DQN 在保持较好 balance_score 的同时，提高了资源接纳能力；
used-mem-asc 是当前最强的显存启发式 baseline，DQN 的优势主要体现在高负载和复杂 Job 场景下。

简要结论：

DQN 并非在所有负载下碾压启发式方法，但在高负载和复杂 Pod 组合下表现出更好的接纳能力和更低的失败率。
14. 后续工作

后续可以继续完善：

引入 core-aware baseline；
加入更真实的节点已有负载场景；
增加 memory/core 冲突型 Job；
优化 reward，使模型更关注高负载下的 vGPU 接纳量；
与真实 Volcano vGPU 调度流程进一步对齐；
将训练后的模型接入 gRPC 推理服务。
