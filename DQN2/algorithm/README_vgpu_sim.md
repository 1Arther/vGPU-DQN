# vGPU 单节点负载均衡仿真实验说明

本分支新增 `vgpu_dqn_sim.py`，用于先做 vGPU 负载均衡的 Python 仿真实验。该脚本暂时不直接接入 Kubernetes / Volcano，而是把调度过程抽象出来，验证 GNN-DQN 算法是否适合做单节点多 GPU 之间的 vGPU 任务分配。

## 为什么先做仿真

真实接入 Volcano 调度器需要处理调度插件、设备插件、Pod annotation、GPU 设备绑定等问题，工程复杂度较高。为了先验证算法是否有效，本阶段先做仿真：

```text
输入：单节点内多张物理 GPU 的资源状态 + 一批申请 vGPU 的 Pod
输出：每个 Pod 应该分配到哪张物理 GPU
目标：让多张 GPU 的显存使用率、core 使用率、Pod 数量尽量均衡
```

## 和原算法的对应关系

原 `DQN_LB2.py` 的核心思想是：

```text
服务器节点 node  ——  任务 task
```

本分支改成：

```text
物理 GPU  ——  Pod 任务
```

因此动作仍然保持为：

```text
action = (gpu_idx, pod_idx)
```

表示把某个 Pod 分配到某张物理 GPU。

## 主要修改点

代码里已经用 `[vGPU修改点-x]` 做了注释，方便答辩或后续继续改。

### 修改点 1：节点含义变化

原来的节点是集群中的服务器节点，现在改成单节点内的一张物理 GPU。

GPU 状态包括：

```text
memory_total
memory_free
core_total
core_free
pod_count
util
```

### 修改点 2：任务含义变化

原来的任务是普通资源任务，现在改成申请 vGPU 资源的 Pod。

Pod 需求包括：

```text
vgpu_number
memory_demand
core_demand
```

### 修改点 3：构建 GPU-Pod 二分图

`build_vgpu_graph()` 会生成：

```text
node_feats: GPU 特征
task_feats: Pod 特征
adj: GPU-Pod 可行分配矩阵
```

当某张 GPU 的剩余显存和剩余 core 都满足 Pod 需求时：

```text
adj[gpu_idx, pod_idx] = 1
```

否则为 0。

### 修改点 4：资源扣减方式变化

原算法扣减服务器节点的 CPU、内存、GPU 数量。

本分支改为扣减物理 GPU 上的：

```text
memory_free
core_free
pod_count
```

对应函数：

```text
update_gpu_resources()
```

### 修改点 5：奖励函数变化

本分支的奖励函数是：

```text
reward = 1 - balance_penalty
```

其中 `balance_penalty` 由以下指标组成：

```text
显存使用率标准差
core 使用率标准差
Pod 数量标准差
```

标准差越小，说明多张 GPU 越均衡，奖励越高。

## 运行方式

在仓库根目录下执行：

```bash
python DQN2/algorithm/vgpu_dqn_sim.py --episodes 300 --gpus 3 --pods 6
```

参数说明：

```text
--episodes       训练轮数
--gpus           单节点物理 GPU 数量，默认 3
--pods           每轮仿真的 Pod 数量，默认 6
--hidden-dim     GNN 和 Q 网络隐藏层维度
--batch-size     经验回放 batch size
--target-update  target 网络更新间隔
--log-interval   日志打印间隔
--output-dir     训练日志和模型输出目录
```

## 输出结果

运行后会生成：

```text
DQN2/outputs/vgpu_sim_training_log.csv
DQN2/outputs/vgpu_dqn_sim.pth
```

其中：

```text
vgpu_sim_training_log.csv 记录每轮 reward、balance_score、loss、epsilon
vgpu_dqn_sim.pth 保存训练后的模型参数
```

`balance_score` 越小，说明最终分配越均衡。

## 后续可以继续做的事情

1. 把仿真的 GPU 状态替换成真实 `nvidia-smi` / DCGM 指标。
2. 把 Pod 需求替换成 Kubernetes YAML 中的 `volcano.sh/vgpu-memory` 和 `volcano.sh/vgpu-cores`。
3. 加入更多基线策略，例如轮询、最少 Pod 数、最少显存占用。
4. 将算法输出结果写入 Pod annotation 或调度扩展点，进一步接入 Volcano。
5. 画出 reward、balance_score、loss 曲线，用于论文实验展示。
