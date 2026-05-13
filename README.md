# vGPU-DQN：基于 GNN-DQN 的 Volcano vGPU 负载均衡调度仿真实验

本项目研究单节点多 GPU 场景下的 vGPU 资源分配问题。实验目标是在多个 Pod 申请 vGPU 资源时，学习一个 GPU-Pod 分配策略，使调度结果同时满足：

1. 提高 Pod 调度成功率；
2. 降低 GPU 间显存与算力使用率差异；
3. 在高负载和超载场景下减少资源碎片与失败 Pod 数量。

当前版本是离线仿真实验代码，不直接调用 Kubernetes / Volcano API，也不会真实创建 Pod。实验主要用于对比 DQN 策略与 Volcano vGPU 源码风格启发式策略。

---

## 1. 当前分支主要改动

本分支完成了混合负载统一模型实验，并对实验流程做了规范化拆分。

### 1.1 训练、测试、数据生成分离

当前实验代码拆分为：

```text
DQN2/algorithm/vgpu_dqn_sim.py
DQN2/algorithm/generate_vgpu_dataset.py
DQN2/algorithm/train_vgpu_mixed_from_file.py
DQN2/algorithm/test_vgpu_mixed_from_file.py
DQN2/algorithm/draw_mixed_load_fixed.py

各文件作用如下：

文件	作用
vgpu_dqn_sim.py	公共模型、环境、reward、baseline 和评估函数
generate_vgpu_dataset.py	提前生成固定 GPU/Pod 场景数据
train_vgpu_mixed_from_file.py	从固定训练集读取数据并训练统一 DQN 模型
test_vgpu_mixed_from_file.py	加载同一个 best model，在固定测试集上评估
draw_mixed_load_fixed.py	绘制混合负载实验结果图
2. 实验设计
2.1 固定随机数据集

为了保证实验可复现，本版本不再在训练和测试时临时生成随机场景，而是先生成固定数据集：

DQN2/data_mixed_load_fixed/
├── dataset_meta.json
├── train_scenarios.jsonl
├── val_scenarios.jsonl
├── test_load_0.6.jsonl
├── test_load_0.8.jsonl
├── test_load_1.0.jsonl
├── test_load_1.2.jsonl
└── test_load_1.5.jsonl

训练、验证、测试均从上述固定文件读取。

2.2 混合负载统一模型

训练阶段使用混合负载：

target_load = 0.6 / 0.8 / 1.0 / 1.2 / 1.5

训练时只训练一个统一 DQN 模型。

测试阶段使用同一个 best model 分别测试：

test_load_0.6.jsonl
test_load_0.8.jsonl
test_load_1.0.jsonl
test_load_1.2.jsonl
test_load_1.5.jsonl

也就是说，本实验不是每个负载强度单独训练一个模型，而是：

一个混合负载训练模型 + 五组固定负载测试集
3. 负载强度定义

对每个测试场景，定义：

memory_load = total_pod_memory / total_gpu_memory
core_load   = total_pod_core   / total_gpu_core
actual_load = max(memory_load, core_load)

不同负载强度含义：

target_load	含义
0.6	低负载
0.8	中低负载
1.0	接近满载
1.2	超载
1.5	重度超载
4. GNN-DQN 建模方式
4.1 图建模

将单节点内的 GPU 和 Pod 建模为二分图：

GPU 节点  <---- 可分配边 ---->  Pod 节点

如果某个 Pod 的 vGPU 显存和 core 请求可以被某张 GPU 当前剩余资源满足，则二者之间存在一条可分配边。

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
vgpu_number
allocated_flag
4.4 动作空间

每一步动作定义为：

action = (gpu_idx, pod_idx)

含义是将某个 Pod 分配到某张 GPU 上。

5. 评价指标
5.1 success_rate
success_rate = allocated_pods / total_pods

越高越好。

5.2 failure_rate
failure_rate = 1 - success_rate

越低越好。

5.3 balance_score
balance_score = std(memory_usage) + std(core_usage)

越低表示 GPU 间负载越均衡。

5.4 objective

综合目标函数：

objective = success_weight * success_rate
          - balance_weight * balance_score
          - failure_weight * failure_rate

当前实验默认：

success_weight = 2.0
balance_weight = 1.0
failure_weight = 2.0

因此：

objective 越高越好
6. 对比方法

当前对比方法包括：

dqn
volcano-vgpu-binpack
volcano-vgpu-spread
random
6.1 DQN

使用 GNN 编码 GPU-Pod 二分图，通过 Q 网络从合法动作中选择 Q 值最高的 GPU-Pod 分配动作。测试阶段关闭探索，即：

epsilon = 0
6.2 volcano-vgpu-binpack

该 baseline 对齐 Volcano vGPU 源码中的 sortedDeviceIndicesByPolicy(binpack) 思路：

UsedMem 更大的 GPU 优先；
UsedMem 相同则 GPU index 小的优先。

也就是倾向于把新的 vGPU 请求继续压入已使用显存更多的 GPU。

6.3 volcano-vgpu-spread

该 baseline 对齐 Volcano vGPU 源码中的 sortedDeviceIndicesByPolicy(spread) 思路：

UsedNum 更小的 GPU 优先；
UsedNum 相同则 GPU index 小的优先。

也就是优先选择当前共享 Pod 数更少的 GPU，空卡优先。

6.4 random

从所有可行 GPU 中随机选择。

7. 运行方式
7.1 生成固定数据集
python DQN2/algorithm/generate_vgpu_dataset.py \
  --seed 42 \
  --output-dir DQN2/data_mixed_load_fixed \
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
  --train-path DQN2/data_mixed_load_fixed/train_scenarios.jsonl \
  --val-path DQN2/data_mixed_load_fixed/val_scenarios.jsonl \
  --episodes 8000 \
  --batch-size 64 \
  --hidden-dim 256 \
  --lr 0.0003 \
  --success-weight 2.0 \
  --balance-weight 1.0 \
  --failure-weight 2.0 \
  --early-stop-patience 20 \
  --output-dir DQN2/outputs_mixed_load_fixed
7.3 测试统一模型
python DQN2/algorithm/test_vgpu_mixed_from_file.py \
  --seed 42 \
  --model-path DQN2/outputs_mixed_load_fixed/vgpu_dqn_mixed_best.pth \
  --data-dir DQN2/data_mixed_load_fixed \
  --output-dir DQN2/outputs_mixed_load_fixed_eval \
  --eval-loads 0.6,0.8,1.0,1.2,1.5 \
  --hidden-dim 256 \
  --success-weight 2.0 \
  --balance-weight 1.0 \
  --failure-weight 2.0
7.4 画图
python DQN2/algorithm/draw_mixed_load_fixed.py \
  --root-dir DQN2/outputs_mixed_load_fixed_eval \
  --summary-path DQN2/outputs_mixed_load_fixed_eval/mixed_load_test_summary.csv \
  --train-log DQN2/outputs_mixed_load_fixed/vgpu_mixed_training_log.csv \
  --loads 0.6,0.8,1.0,1.2,1.5 \
  --output-dir DQN2/outputs_mixed_load_fixed_eval/summary_figures \
  --smooth-window 20
8. 实验结果

测试阶段平均 GPU 数约为 11 到 12 张，平均 Pod 数约为 239 到 259 个。测试结果保存于：

DQN2/outputs_mixed_load_fixed_eval/mixed_load_test_summary.csv
8.1 success_rate 对比
target_load	DQN	volcano-vgpu-binpack	volcano-vgpu-spread	random
0.6	1.0000	1.0000	1.0000	1.0000
0.8	1.0000	1.0000	1.0000	0.9999
1.0	0.9491	0.9071	0.9088	0.9116
1.2	0.8487	0.7573	0.7643	0.7706
1.5	0.7367	0.6130	0.6215	0.6231
8.2 balance_score 对比
target_load	DQN	volcano-vgpu-binpack	volcano-vgpu-spread	random
0.6	0.2193	0.8372	0.2717	0.3418
0.8	0.2040	0.5920	0.2783	0.3177
1.0	0.2002	0.2527	0.2482	0.2444
1.2	0.1894	0.2553	0.2479	0.2490
1.5	0.1722	0.2564	0.2408	0.2488
8.3 objective 对比
target_load	DQN	volcano-vgpu-binpack	volcano-vgpu-spread	random
0.6	1.7807	1.1628	1.7283	1.6582
0.8	1.7960	1.4080	1.7217	1.6817
1.0	1.5961	1.3755	1.3871	1.4022
1.2	1.2055	0.7738	0.8094	0.8335
1.5	0.7745	0.1957	0.2452	0.2438
9. 实验结论
9.1 低负载场景

在 target_load=0.6 和 0.8 下，各方法几乎都可以完成全部 Pod 调度，success_rate 接近或达到 1.0。此时主要差异体现在 GPU 间负载均衡。

实验结果显示，DQN 在低负载下取得最低 balance_score，说明统一 DQN 模型不仅能够成功完成调度，也能够实现更均衡的 GPU 资源使用。

9.2 接近满载场景

在 target_load=1.0 下，DQN 的 success_rate 为 0.9491，高于 volcano-vgpu-binpack、volcano-vgpu-spread 和 random。同时，DQN 的 balance_score 也最低，说明 DQN 在接近满载场景下能够同时提高调度成功率和负载均衡效果。

9.3 超载场景

在 target_load=1.2 下，DQN 的 success_rate 为 0.8487，高于最强 baseline random 的 0.7706，平均每个测试 batch 多调度约 20.72 个 Pod。同时，DQN 的 balance_score 仍然最低，说明其没有通过牺牲负载均衡来换取成功率。

9.4 重度超载场景

在 target_load=1.5 下，DQN 的 success_rate 为 0.7367，而 random、volcano-vgpu-spread 和 volcano-vgpu-binpack 分别为 0.6231、0.6215 和 0.6130。DQN 平均每个 batch 比最强 baseline 多调度约 29.80 个 Pod。

这说明在资源严重不足的情况下，DQN 更会做 GPU-Pod 组合选择，能够减少资源碎片并提升可调度 Pod 数量。

10. 总体结论

本实验表明，混合负载训练得到的统一 DQN 模型能够在不同负载强度下泛化。在低负载场景下，DQN 能够保持 100% 调度成功率并获得更低的负载不均衡；在高负载和超载场景下，DQN 相比 Volcano vGPU 源码风格 baseline 具有更高调度成功率、更低失败率和更高综合目标。

因此，DQN 在复杂高压 vGPU 调度场景下具有较好的资源分配能力。

11. 当前局限
当前实验仍为单节点多 GPU 仿真，没有接入真实 Kubernetes / Volcano 调度链路。
当前 baseline 复现的是 Volcano vGPU 选卡策略的核心排序逻辑，不包含真实系统中的所有调度细节。
当前一个 Pod 默认分配到一张 GPU，不考虑单 Pod 跨 GPU 拆分。
当前只使用单随机种子，后续可以增加多随机种子实验。
当前 DQN replay 仍为逐样本训练，GPU 利用率不高，后续可进一步 batch 化优化。
12. 后续工作
接入真实 Volcano / HAMi 调度环境；
扩展到多节点 GPU 调度；
增加多随机种子实验和置信区间；
优化 DQN replay 训练效率；
进一步对齐 Volcano vGPU 的完整调度链路。
EOF