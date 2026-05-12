# vGPU-DQN：基于 GNN-DQN 的 hami-core vGPU 负载均衡仿真实验

本项目研究单节点多 GPU 场景下的 vGPU 资源分配问题。实验目标是在多个 Pod 申请 vGPU 资源时，学习一个 GPU-Pod 分配策略，使调度结果同时满足：

1. 尽可能提高 Pod 调度成功率；
2. 尽可能降低 GPU 间显存与算力使用率差异；
3. 在高负载和超载场景下减少资源碎片与失败 Pod 数量。

当前版本是离线仿真实验代码，不直接调用 Kubernetes / Volcano API，也不会真实创建 Pod。实验主要用于对比 DQN 策略、Volcano 风格启发式策略和随机策略。

---

## 1. 当前分支主要改动

本分支围绕 hami-core / Volcano vGPU 调度仿真做了如下改动：

1. 重构 `DQN2/algorithm/vgpu_dqn_sim.py`，实现 GNN-DQN 训练、验证、测试流程。
2. 支持按负载强度 `target_load` 生成 GPU / Pod 测试场景。
3. DQN 和所有 baseline 使用同一批测试 batch，保证比较公平。
4. 测试阶段关闭探索，即 DQN 使用 `epsilon = 0`。
5. 新增综合评价指标 `objective`，同时考虑成功率、失败率和负载均衡。
6. 新增早停机制，按验证集综合目标保存 best checkpoint。
7. 新增 Volcano 风格 baseline：
   - `volcano-binpack`
   - `volcano-spread`
   - `simple-spread`
   - `random`
8. 新增训练日志、测试明细、测试汇总结果输出。
9. 新增画图脚本，用于生成训练曲线、方法对比图和负载强度趋势图。

---

## 2. 仿真实验设定

### 2.1 GPU 资源

每张 GPU 记录以下资源状态：

```text
memory_total
memory_free
core_total
core_free
pod_count
util
2.2 Pod 请求

每个 Pod 包含以下 vGPU 请求：

vgpu_number
memory_demand
core_demand

当前仿真中，一个 Pod 分配到一张 GPU 上，不做跨节点拆分，也不做多节点调度。

2.3 负载强度

实验使用 target_load 控制 Pod 总需求强度：

memory_load = total_pod_memory / total_gpu_memory
core_load   = total_pod_core   / total_gpu_core
actual_load = max(memory_load, core_load)

不同负载强度含义如下：

target_load = 0.6  低负载
target_load = 0.8  中低负载
target_load = 1.0  接近满载
target_load = 1.2  超载
target_load = 1.5  重度超载

由于 Pod 是离散生成的，最终 actual_load 通常会略高于 target_load。

3. GNN-DQN 方法
3.1 图建模

将单节点内的 GPU 和 Pod 建模为二分图：

GPU 节点  <---- 可分配边 ---->  Pod 节点

如果某个 Pod 的显存和 core 请求可以被某张 GPU 当前剩余资源满足，则二者之间存在一条有效边。

GPU 节点特征包括：

memory_used_ratio
core_used_ratio
pod_count_ratio
util_ratio
memory_free_ratio
core_free_ratio

Pod 节点特征包括：

memory_demand_ratio
core_demand_ratio
vgpu_number
allocated_flag
3.2 动作空间

每一步动作定义为：

action = (gpu_idx, pod_idx)

表示将某个 Pod 分配到某张 GPU 上。

3.3 评价指标
success_rate
success_rate = allocated_pods / total_pods

越高越好。

failure_rate
failure_rate = 1 - success_rate

越低越好。

balance_score
balance_score = std(memory_usage) + std(core_usage)

越低表示 GPU 间负载越均衡。

objective

综合目标为：

objective = success_weight * success_rate
          - balance_weight * balance_score
          - failure_weight * failure_rate

当前实验默认：

success_weight = 2.0
balance_weight = 1.0
failure_weight = 2.0

因此该指标同时奖励高成功率，并惩罚负载不均衡和调度失败。

4. 对比方法
4.1 DQN

使用 GNN 编码 GPU-Pod 二分图，通过 Q 网络选择合法动作中 Q 值最高的 (GPU, Pod) 分配。

4.2 Volcano-binpack

模拟压实策略，倾向将 Pod 放到放置后负载更高的 GPU 上，以减少资源碎片。

4.3 Volcano-spread

模拟分散策略，倾向将 Pod 放到放置后负载更低的 GPU 上，以提高 GPU 间负载均衡。

4.4 Simple-spread

简单 spread baseline，只根据当前 GPU 已用负载选择较空闲 GPU。

4.5 Random

在所有可行 GPU 中随机选择。

5. 运行方式
5.1 单个负载强度实验

以 target_load=1.2 为例：

python DQN2/algorithm/vgpu_dqn_sim.py \
  --episodes 3000 \
  --gpus 3 \
  --min-pods 1 \
  --max-pods 200 \
  --target-load 1.2 \
  --train-batches 500 \
  --test-batches 50 \
  --batch-size 32 \
  --hidden-dim 128 \
  --lr 0.0003 \
  --success-weight 2.0 \
  --balance-weight 1.0 \
  --failure-weight 2.0 \
  --early-stop-patience 10 \
  --output-dir DQN2/outputs_load_exp/load_1.2 \
  --data-dir DQN2/data_load_exp/load_1.2 \
  --regenerate-data
5.2 负载强度控制实验
mkdir -p DQN2/outputs_load_exp
mkdir -p DQN2/data_load_exp

for LOAD in 0.6 0.8 1.0 1.2 1.5
do
  echo "========== Running target_load=${LOAD} =========="

  python DQN2/algorithm/vgpu_dqn_sim.py \
    --episodes 3000 \
    --gpus 3 \
    --min-pods 1 \
    --max-pods 200 \
    --target-load ${LOAD} \
    --train-batches 500 \
    --test-batches 50 \
    --batch-size 32 \
    --hidden-dim 128 \
    --lr 0.0003 \
    --success-weight 2.0 \
    --balance-weight 1.0 \
    --failure-weight 2.0 \
    --early-stop-patience 10 \
    --output-dir DQN2/outputs_load_exp/load_${LOAD} \
    --data-dir DQN2/data_load_exp/load_${LOAD} \
    --regenerate-data
done

说明：当前已完成的是“每个 target_load 单独训练一个 DQN 模型”的专用模型实验。后续更合理的主实验是使用混合负载训练一个统一 DQN 模型，然后分别在不同负载强度测试集上评估泛化能力。

6. 画图方式
6.1 单个负载强度内部对比图
for LOAD in 0.6 0.8 1.0 1.2 1.5
do
  python DQN2/algorithm/draw_vgpu_compare.py \
    --detail-path DQN2/outputs_load_exp/load_${LOAD}/test_comparison_detail.csv \
    --summary-path DQN2/outputs_load_exp/load_${LOAD}/test_comparison_summary.csv \
    --output-dir DQN2/outputs_load_exp/load_${LOAD}/compare_figures
done
6.2 训练曲线
for LOAD in 0.6 0.8 1.0 1.2 1.5
do
  python DQN2/algorithm/draw_vgpu_sim.py \
    --log-path DQN2/outputs_load_exp/load_${LOAD}/vgpu_sim_training_log.csv \
    --output-dir DQN2/outputs_load_exp/load_${LOAD}/figures \
    --smooth-window 5
done
6.3 多负载强度趋势图
python DQN2/algorithm/collect_load_exp.py \
  --root-dir DQN2/outputs_load_exp \
  --loads 0.6,0.8,1.0,1.2,1.5 \
  --output-dir DQN2/outputs_load_exp/summary_figures

主要输出：

DQN2/outputs_load_exp/summary_figures/load_experiment_summary.csv
DQN2/outputs_load_exp/summary_figures/avg_success_rate_vs_load.png
DQN2/outputs_load_exp/summary_figures/avg_balance_score_vs_load.png
DQN2/outputs_load_exp/summary_figures/avg_failure_rate_vs_load.png
DQN2/outputs_load_exp/summary_figures/avg_objective_vs_load.png
DQN2/outputs_load_exp/summary_figures/avg_allocated_count_vs_load.png
7. 实验结果
7.1 不同负载强度下 DQN 结果
target_load	avg_actual_load	avg_memory_load	avg_core_load	DQN avg_balance_score ↓	DQN avg_success_rate ↑	DQN avg_failure_rate ↓	DQN avg_allocated_count	DQN avg_objective ↑
0.6	0.6395	0.6091	0.5034	0.1567	1.0000	0.0000	10.34	1.8433
0.8	0.8436	0.7998	0.7018	0.2229	0.9940	0.0060	14.46	1.7530
1.0	1.0383	1.0117	0.8639	0.1896	0.9135	0.0865	15.94	1.4645
1.2	1.2343	1.1920	1.0166	0.1783	0.8368	0.1632	17.18	1.1690
1.5	1.5356	1.4947	1.2877	0.1818	0.7358	0.2642	19.30	0.7614
7.2 target_load=0.6

低负载下所有方法 success_rate=1.0，说明资源足够。DQN 的优势主要体现在负载均衡。

method	avg_balance_score ↓	avg_success_rate ↑	avg_objective ↑
dqn	0.1567	1.0000	1.8433
volcano-spread	0.2202	1.0000	1.7798
simple-spread	0.2602	1.0000	1.7398
random	0.3930	1.0000	1.6070
volcano-binpack	0.6433	1.0000	1.3567

结论：低负载场景下，DQN 能在全部调度成功的同时获得最低 balance_score。

7.3 target_load=0.8

中低负载下，各方法成功率都接近 1。DQN 的 balance_score 和 objective 最优。

method	avg_balance_score ↓	avg_success_rate ↑	avg_objective ↑
dqn	0.2229	0.9940	1.7530
volcano-spread	0.2281	0.9903	1.7330
simple-spread	0.2462	0.9876	1.7041
random	0.2913	0.9941	1.6851
volcano-binpack	0.3483	0.9972	1.6407

结论：DQN 的成功率接近最优，且负载均衡最好，因此综合目标最高。

7.4 target_load=1.0

接近满载时，DQN 的 balance_score 最低，objective 与 volcano-binpack 几乎持平并略高。

method	avg_balance_score ↓	avg_success_rate ↑	avg_objective ↑
dqn	0.1896	0.9135	1.4645
volcano-binpack	0.2224	0.9216	1.4641
volcano-spread	0.1937	0.9074	1.4358
simple-spread	0.2253	0.9042	1.3915
random	0.2490	0.8998	1.3503

结论：binpack 的成功率略高，但 DQN 的负载均衡更好，综合目标略优。

7.5 target_load=1.2

超载场景下，DQN 同时取得最高成功率、最低失败率、最低负载不均衡和最高综合目标。

method	avg_balance_score ↓	avg_success_rate ↑	avg_failure_rate ↓	avg_objective ↑
dqn	0.1783	0.8368	0.1632	1.1690
volcano-spread	0.2027	0.7983	0.2017	0.9903
volcano-binpack	0.2254	0.8022	0.1978	0.9835
random	0.2204	0.7982	0.2018	0.9725
simple-spread	0.2631	0.7848	0.2152	0.8762

结论：在超载场景下，DQN 的综合调度能力明显强于启发式方法。

7.6 target_load=1.5

重度超载场景下，DQN 优势进一步扩大。

method	avg_balance_score ↓	avg_success_rate ↑	avg_failure_rate ↓	avg_allocated_count ↑	avg_objective ↑
dqn	0.1818	0.7358	0.2642	19.30	0.7614
volcano-spread	0.1886	0.6802	0.3198	17.84	0.5320
random	0.1969	0.6608	0.3392	17.30	0.4465
simple-spread	0.2292	0.6600	0.3400	17.28	0.4107
volcano-binpack	0.2092	0.6503	0.3497	17.02	0.3918

结论：DQN 在重度超载下平均每个 batch 比最强 baseline 多调度约 1.46 个 Pod，同时保持最低 balance_score。

8. 趋势分析

从 0.6 到 1.5 的负载强度实验可以看到：

低负载阶段，所有方法都能成功调度全部或大部分 Pod，DQN 的优势主要体现在负载均衡。
接近满载时，启发式方法开始在成功率和负载均衡之间出现权衡，DQN 综合目标开始占优。
超载和重度超载时，DQN 不仅成功率更高，而且 balance_score 更低，说明它没有通过牺牲均衡性来换取成功率。
volcano-binpack 在部分场景下成功率较高，但负载不均衡较明显。
volcano-spread 的均衡性较好，但高负载下成功率不如 DQN。
random 和 simple-spread 在高负载下失败率明显高于 DQN。

综合来看，DQN 在高负载和超载场景下优势最明显。

9. 当前实验局限
当前结果是“单负载专用模型”实验，即每个 target_load 单独训练一个 DQN 模型。
后续应补充“混合负载统一模型”实验：用 0.6 / 0.8 / 1.0 / 1.2 / 1.5 混合训练一个模型，再分别测试不同负载强度。
当前仍是单节点仿真，没有接入真实 Kubernetes / Volcano 调度链路。
当前 baseline 是 Volcano 风格近似实现，不是直接调用 Volcano 源码。
当前一个 Pod 不做跨节点拆分，也不模拟多节点调度。
当前只展示单随机种子结果，后续需要多随机种子重复实验增强稳定性。
10. 后续工作
实现混合负载训练：
train_target_loads = [0.6, 0.8, 1.0, 1.2, 1.5]

训练一个统一 DQN 模型，再分别在各负载测试集上评估。

支持动态 GPU 数量实验，例如：
min_gpus = 3
max_gpus = 8
将 baseline 进一步贴近 Volcano deviceshare 的真实策略。
接入真实 Volcano / HAMi 环境，做在线调度实验。
增加多随机种子实验和置信区间。
11. 文件输出说明

每次实验输出：

vgpu_sim_training_log.csv       # 训练日志
test_comparison_detail.csv      # 50 个测试 batch 的逐批次结果
test_comparison_summary.csv     # 各方法的平均结果
vgpu_dqn_sim_final.pth          # 最终模型
vgpu_dqn_sim_best.pth           # 按 objective 保存的 best 模型

多负载汇总输出：

load_experiment_summary.csv
avg_success_rate_vs_load.png
avg_balance_score_vs_load.png
avg_failure_rate_vs_load.png
avg_objective_vs_load.png
avg_allocated_count_vs_load.png
12. 注意事项
balance_score 越低越好。
success_rate 越高越好。
failure_rate 越低越好。
objective 越高越好。
loss 只用于观察训练是否稳定，不作为最终性能指标。
最终实验结论应以测试集上的 test_comparison_summary.csv 为准。
EOF