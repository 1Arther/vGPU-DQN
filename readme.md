# vGPU-DQN：支持单 Pod 多 vGPU 的 GNN-DQN 调度仿真实验

本项目研究单节点多 GPU 场景下的 vGPU 资源分配问题。目标是在多个 Pod 申请 vGPU 资源时，学习一个 GPU-Pod 分配策略，使调度结果同时满足：

1. 提高 Pod 调度成功率；
2. 提高 vGPU 分配成功率；
3. 降低 GPU 间显存与算力使用率差异；
4. 在高负载和超载场景下减少资源碎片与失败 Pod 数量。

当前版本是离线仿真实验代码，不直接调用 Kubernetes / Volcano API，也不会真实创建 Pod。实验主要用于对比 GNN-DQN 策略与 Volcano vGPU 源码风格启发式策略。

---

## 1. v5 版本主要改动

v5 在 v4 的基础上增加了单 Pod 多 vGPU 支持。

### 1.1 单 Pod 支持多个 vGPU

v4 中，一个 Pod 默认只分配到一张 GPU。v5 中，一个 Pod 可以申请多个 vGPU：

```text
vgpu_number = 1 / 2 / 3 / 4

例如：

{
  "task_id": "pod-7",
  "vgpu_number": 3,
  "memory_demand": 8.0,
  "core_demand": 20.0
}

含义是：

该 Pod 需要 3 个 vGPU；
每个 vGPU 需要 8GB 显存；
每个 vGPU 需要 20% core。

因此总需求为：

total_memory = vgpu_number * memory_demand
total_core   = vgpu_number * core_demand
1.2 单节点内多 GPU 分配

当前实验仍然是单节点多 GPU 场景，不涉及多节点调度。

允许：

pod-7 -> GPU1 + GPU4 + GPU6

不允许：

pod-7 -> node-0/GPU1 + node-1/GPU2
1.3 all-or-nothing 分配语义

多 vGPU Pod 必须一次性分配成功。

如果一个 Pod 需要 3 个 vGPU，但当前只能找到 2 张满足条件的 GPU，则该 Pod 整体调度失败，不允许部分成功。

1.4 负载强度重新计算

v5 中，负载强度按照总 vGPU 需求计算：

memory_load = sum(vgpu_number * memory_demand) / total_gpu_memory
core_load   = sum(vgpu_number * core_demand)   / total_gpu_core
actual_load = max(memory_load, core_load)

因此 v5 不能复用 v4 的数据集，必须重新生成固定数据。

2. 代码结构
DQN2/algorithm/vgpu_dqn_sim.py
DQN2/algorithm/generate_vgpu_dataset.py
DQN2/algorithm/train_vgpu_mixed_from_file.py
DQN2/algorithm/test_vgpu_mixed_from_file.py
DQN2/algorithm/draw_mixed_load_v5.py
文件	作用
vgpu_dqn_sim.py	公共模型、环境、多 vGPU 分配逻辑、baseline 和评估函数
generate_vgpu_dataset.py	提前生成固定 GPU/Pod 场景数据
train_vgpu_mixed_from_file.py	从固定训练集读取数据并训练统一 DQN 模型
test_vgpu_mixed_from_file.py	加载同一个 best model，在固定测试集上评估
draw_mixed_load_v5.py	绘制 v5 多 vGPU 实验结果图
3. 固定数据集

为了保证实验可复现，v5 使用提前生成并保存的数据集：

DQN2/data_mixed_load_fixed_v5/
├── dataset_meta.json
├── train_scenarios.jsonl
├── val_scenarios.jsonl
├── test_load_0.6.jsonl
├── test_load_0.8.jsonl
├── test_load_1.0.jsonl
├── test_load_1.2.jsonl
└── test_load_1.5.jsonl

训练、验证、测试均从上述固定文件读取。

训练阶段混合负载：

target_load = 0.6 / 0.8 / 1.0 / 1.2 / 1.5

测试阶段使用同一个 DQN best model 分别测试五个固定负载测试集。

4. GNN-DQN 建模方式
4.1 图建模

将单节点内的 GPU 和 Pod 建模为二分图：

GPU 节点  <---- 可分配边 ---->  Pod 节点

如果某个 GPU 可以作为该 Pod 的 anchor GPU，并且当前节点内能够为该 Pod 凑够 vgpu_number 张可用 GPU，则 GPU-Pod 之间存在可分配边。

4.2 GPU 节点特征
memory_used_ratio
core_used_ratio
pod_count_ratio
util_ratio
memory_free_ratio
core_free_ratio
4.3 Pod 节点特征
memory_demand_ratio
core_demand_ratio
vgpu_number_ratio
allocated_flag
4.4 DQN 动作空间

DQN 动作仍然是：

action = (gpu_idx, pod_idx)

其中 gpu_idx 被解释为 anchor GPU。

如果该 Pod 只需要 1 个 vGPU，则直接分配到该 GPU。

如果该 Pod 需要多个 vGPU，则必须包含 anchor GPU，并从其他可行 GPU 中补齐剩余 vGPU。补齐时选择分配后 balance_score 最低的 GPU 组合。

5. 对比方法

当前对比方法包括：

dqn
volcano-vgpu-binpack
volcano-vgpu-spread
random
5.1 DQN

使用 GNN 编码 GPU-Pod 二分图，通过 Q 网络选择合法动作中 Q 值最高的 (GPU, Pod) 动作。测试阶段关闭探索：

epsilon = 0
5.2 volcano-vgpu-binpack

对齐 Volcano vGPU 源码中的设备排序思想：

UsedMem 更大的 GPU 优先；
UsedMem 相同则 GPU index 小的优先；
按优先级从高到低扫描；
能放就加入组合；
直到凑够 pod.vgpu_number 张 GPU；
如果凑不够，则该 Pod 分配失败。
5.3 volcano-vgpu-spread

对齐 Volcano vGPU 源码中的设备排序思想：

UsedNum / pod_count 更小的 GPU 优先；
UsedNum 相同则 GPU index 小的优先；
按优先级从高到低扫描；
能放就加入组合；
直到凑够 pod.vgpu_number 张 GPU；
如果凑不够，则该 Pod 分配失败。
5.4 random

从所有可行 GPU 中随机扫描并选择，直到凑够 vgpu_number 张 GPU。如果凑不够，则该 Pod 分配失败。

6. 评价指标
6.1 Pod 级成功率
success_rate = allocated_pods / total_pods

越高越好。

6.2 Pod 级失败率
failure_rate = 1 - success_rate

越低越好。

6.3 vGPU 级成功率
vgpu_success_rate = allocated_vgpu_count / total_vgpu_count

越高越好。

6.4 vGPU 级失败率
vgpu_failure_rate = 1 - vgpu_success_rate

越低越好。

6.5 负载均衡指标
balance_score = std(memory_usage) + std(core_usage)

越低表示 GPU 间负载越均衡。

6.6 综合目标
objective = success_weight * success_rate
          - balance_weight * balance_score
          - failure_weight * failure_rate

当前默认：

success_weight = 2.0
balance_weight = 1.0
failure_weight = 2.0

因此：

objective 越高越好
7. 运行方式
7.1 生成 v5 固定数据集
python DQN2/algorithm/generate_vgpu_dataset.py \
  --seed 42 \
  --output-dir DQN2/data_mixed_load_fixed_v5 \
  --train-target-loads 0.6,0.8,1.0,1.2,1.5 \
  --eval-loads 0.6,0.8,1.0,1.2,1.5 \
  --train-batches 3000 \
  --val-batches 20 \
  --test-batches 50 \
  --train-min-gpus 6 \
  --train-max-gpus 12 \
  --train-min-pods 100 \
  --train-max-pods 300 \
  --test-min-gpus 8 \
  --test-max-gpus 16 \
  --test-min-pods 100 \
  --test-max-pods 400
7.2 训练统一 DQN 模型
python DQN2/algorithm/train_vgpu_mixed_from_file.py \
  --seed 42 \
  --train-path DQN2/data_mixed_load_fixed_v5/train_scenarios.jsonl \
  --val-path DQN2/data_mixed_load_fixed_v5/val_scenarios.jsonl \
  --episodes 8000 \
  --batch-size 64 \
  --hidden-dim 256 \
  --lr 0.0003 \
  --success-weight 2.0 \
  --balance-weight 1.0 \
  --failure-weight 2.0 \
  --early-stop-patience 20 \
  --output-dir DQN2/outputs_mixed_load_fixed_v5
7.3 测试统一模型
python DQN2/algorithm/test_vgpu_mixed_from_file.py \
  --seed 42 \
  --model-path DQN2/outputs_mixed_load_fixed_v5/vgpu_dqn_mixed_best.pth \
  --data-dir DQN2/data_mixed_load_fixed_v5 \
  --output-dir DQN2/outputs_mixed_load_fixed_v5_eval \
  --eval-loads 0.6,0.8,1.0,1.2,1.5 \
  --hidden-dim 256 \
  --success-weight 2.0 \
  --balance-weight 1.0 \
  --failure-weight 2.0
7.4 画图
python DQN2/algorithm/draw_mixed_load_v5.py \
  --summary-path DQN2/outputs_mixed_load_fixed_v5_eval/mixed_load_test_summary.csv \
  --train-log DQN2/outputs_mixed_load_fixed_v5/vgpu_mixed_training_log.csv \
  --output-dir DQN2/outputs_mixed_load_fixed_v5_eval/summary_figures \
  --smooth-window 20
8. 实验结果

测试结果来自固定测试集，并保存于：

DQN2/outputs_mixed_load_fixed_v5_eval/mixed_load_test_summary.csv
8.1 Pod success_rate 对比
target_load	DQN	volcano-vgpu-binpack	volcano-vgpu-spread	random
0.6	1.0000	1.0000	1.0000	1.0000
0.8	0.9998	0.9952	1.0000	1.0000
1.0	0.9800	0.9260	0.9115	0.9136
1.2	0.9142	0.7865	0.7724	0.7710
1.5	0.8161	0.6308	0.6172	0.6222
8.2 vGPU success_rate 对比
target_load	DQN	volcano-vgpu-binpack	volcano-vgpu-spread	random
0.6	1.0000	1.0000	1.0000	1.0000
0.8	0.9999	0.9882	1.0000	1.0000
1.0	0.9534	0.8892	0.9059	0.9072
1.2	0.8335	0.7553	0.7662	0.7637
1.5	0.6905	0.6031	0.6089	0.6128
8.3 balance_score 对比
target_load	DQN	volcano-vgpu-binpack	volcano-vgpu-spread	random
0.6	0.0821	0.8046	0.2304	0.2982
0.8	0.0976	0.5552	0.2612	0.3015
1.0	0.1153	0.2674	0.2455	0.2442
1.2	0.1588	0.2519	0.2365	0.2440
1.5	0.1833	0.2612	0.2446	0.2436
8.4 objective 对比
target_load	DQN	volcano-vgpu-binpack	volcano-vgpu-spread	random
0.6	1.9179	1.1954	1.7696	1.7018
0.8	1.9016	1.4257	1.7388	1.6985
1.0	1.8047	1.4365	1.4006	1.4103
1.2	1.4980	0.8939	0.8533	0.8399
1.5	1.0813	0.2622	0.2241	0.2452
9. 实验分析
9.1 低负载场景

在 target_load=0.6 下，所有方法均达到 100% 调度成功率。但 DQN 的 balance_score=0.0821，明显低于 volcano-vgpu-spread 的 0.2304 和 random 的 0.2982。这说明在资源充足时，DQN 也能学习到更均衡的 GPU 使用方式。

在 target_load=0.8 下，volcano-vgpu-spread 和 random 的成功率略高于 DQN，但 DQN 的负载均衡明显更好，最终 objective 仍然最高。

9.2 接近满载场景

在 target_load=1.0 下，DQN 的 Pod success_rate 达到 0.9800，高于所有 baseline。同时，DQN 的 balance_score 仍然最低。说明 DQN 没有通过牺牲均衡性来换取成功率，而是在成功率和负载均衡之间取得了更好的综合效果。

9.3 超载场景

在 target_load=1.2 下，DQN 的 success_rate 为 0.9142，而最强 baseline volcano-vgpu-binpack 为 0.7865。DQN 平均每批成功调度 243.22 个 Pod，而最强 baseline 为 208.60 个 Pod，平均每批多调度约 34.62 个 Pod。

vGPU 维度上，DQN 平均每批分配 317.90 个 vGPU，而最强 baseline 为 290.98 个 vGPU，平均每批多分配约 26.92 个 vGPU。

9.4 重度超载场景

在 target_load=1.5 下，DQN 的 success_rate 为 0.8161，而最强 baseline volcano-vgpu-binpack 为 0.6308。DQN 平均每批成功调度 222.72 个 Pod，而最强 baseline 为 171.18 个 Pod，平均每批多调度约 51.54 个 Pod。

vGPU 维度上，DQN 平均每批分配 268.80 个 vGPU，而最强 baseline random 为 237.44 个 vGPU，平均每批多分配约 31.36 个 vGPU。

这说明在多 vGPU Pod 和重度超载场景下，DQN 更善于做资源组合选择和调度取舍。

10. 总体结论

v5 实验表明，在支持单 Pod 多 vGPU 的场景下，GNN-DQN 相比 Volcano vGPU 源码风格 binpack、spread 和 random baseline 具有更好的综合性能。

在低负载下，DQN 主要优势体现为更低的负载不均衡；在高负载和超载场景下，DQN 同时取得更高的 Pod 成功率、更高的 vGPU 成功率、更低的 failure_rate 和更高的 objective。

因此，GNN-DQN 在复杂 vGPU 组合分配和高负载资源取舍场景中具有更强的调度能力。

11. 当前局限
当前实验仍为单节点多 GPU 仿真，没有接入真实 Kubernetes / Volcano 调度链路。
当前不涉及多节点调度，也不支持跨节点分配。
当前一个 Pod 的多个 vGPU 可以分配到同一节点内多张 GPU，但不允许同一个 Pod 在同一张 GPU 上重复分配多个 vGPU。
当前 baseline 复现的是 Volcano vGPU 选卡策略的核心排序逻辑，不包含真实系统中的所有调度细节。
当前只使用单随机种子，后续可以增加多随机种子实验和置信区间。
当前 DQN replay 仍为逐样本训练，GPU 利用率不高，后续可以进一步 batch 化优化。
12. 后续工作
接入真实 Volcano / HAMi 调度环境；
扩展到多节点 GPU 调度；
支持同一 Pod 在同一张 GPU 上申请多个 vGPU slice 的场景；
增加多随机种子实验；
进一步对齐 Volcano vGPU 的完整调度链路；
优化 replay buffer 训练效率。
EOF