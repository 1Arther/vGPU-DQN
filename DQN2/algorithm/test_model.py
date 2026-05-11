import copy
import numpy as np
import torch
import torch.nn as nn
import random
from DQN.init import parse_node_info, parse_tasks_info

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义DQN模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义DQN代理
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.policy_net.eval()
        self.target_net.eval()

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()
        print(f"Model loaded from {path}")

    # def act(self, state, nodeNumber, taskNumber, tasks, allocated_tasks):
    #     self.policy_net.eval()
    #     with torch.no_grad():
    #         state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(dim=0).to(device)
    #         q_values = self.policy_net(state)
    #         for i in range(len(q_values[0])):
    #             task_index = i % taskNumber
    #             node_index = i // taskNumber
    #             if task_index in allocated_tasks or not is_task_allocatable(nodes_info[node_index], tasks[task_index]):
    #                 q_values[0][i] = -float('inf')  # 过滤无效动作
    #         action = q_values.max(1)[1].item()
    #     return action

    def act(self, state, nodeNumber, taskNumber, tasks, allocated_tasks, max_tasks=15):
        """
        nodeNumber: 节点数量
        taskNumber: 实际任务数量（可能小于 max_tasks）
        """
        self.policy_net.eval()
        with torch.no_grad():
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(dim=0).to(device)
            q_values = self.policy_net(state)  # (1, nodeNumber*max_tasks)

            # 遍历所有可能动作
            for i in range(q_values.shape[1]):
                node_index = i // max_tasks
                task_index = i % max_tasks

                # 若 task_index >= taskNumber，说明该动作对应的是“不存在的任务”，置为 -inf
                if task_index >= taskNumber:
                    q_values[0][i] = -float('inf')
                    continue

                # 若此任务已经分配或资源不足，也置为 -inf
                if task_index in allocated_tasks or not is_task_allocatable(nodes_info[node_index], tasks[task_index]):
                    q_values[0][i] = -float('inf')

            # 最终选择 Q 值最大的动作
            action = q_values.max(1)[1].item()
        return action


# 计算负载均衡度和资源平衡度
def calculate_load_balance_and_resource_balance(node_start, nodes_info):
    # 确保节点数量一致
    assert len(node_start) == len(nodes_info), "node_start 和 nodes_info 长度不一致！"

    # 初始化列表
    cpu_usages = []
    memory_usages = []
    gpu_usages = []
    resource_balances = 0.0 # 记录每个节点的资源平衡度

    # 遍历节点，计算资源利用率和资源平衡度
    for start, current in zip(node_start, nodes_info):
        # 计算各项利用率
        cpu_usage = (start['cpu'] - current['cpu']) / start['cpu'] if start['cpu'] > 0 else 0
        memory_usage = (start['memory'] - current['memory']) / start['memory'] if start['memory'] > 0 else 0
        gpu_usage = (start['gpu'] - current['gpu']) / start['gpu'] if start['gpu'] > 0 else 0

        # 添加到总列表中
        cpu_usages.append(cpu_usage)
        memory_usages.append(memory_usage)
        gpu_usages.append(gpu_usage)

        # 计算该节点的资源平衡度（CPU 和内存利用率的标准差）
        resource_balances += np.std([cpu_usage, memory_usage])

    # 打印调试信息
    print("CPU利用率: ", cpu_usages)
    print("内存利用率: ", memory_usages)
    print("GPU利用率: ", gpu_usages)

    # 计算整体负载均衡度（各节点 CPU、内存、GPU 利用率的标准差）
    std_cpu = np.std(cpu_usages)
    std_memory = np.std(memory_usages)
    std_gpu = np.std(gpu_usages)
    load_balance = (std_cpu + std_memory + std_gpu) / 3
    resource_balances /= len(nodes_info)
    print("每个节点的负载均衡度: ", load_balance)
    print("每个节点的资源平衡度: ", resource_balances)
    # 返回整体负载均衡度和资源平衡度的平均值
    return load_balance * 0.5 + resource_balances * 0.5



# 更新节点资源
def update_node_resources(node, task, revert=False):
    factor = -1 if revert else 1
    node['cpu'] -= factor * task['cpu_demand']
    node['memory'] -= factor * task['memory_demand']
    node['gpu'] -= factor * task['gpu_demand']

# 检查任务是否可以分配到节点
def is_task_allocatable(node, task):
    return node['cpu'] >= task['cpu_demand'] and node['memory'] >= task['memory_demand'] and node['gpu'] >= task['gpu_demand']

def get_state(agent, nodes_info, tasks_info, allocations, max_tasks=15):
    """
    将状态向量限制为最多包含 max_tasks 个任务，多余的忽略，不足的用占位值补齐。
    """
    task_state = []
    node_state = []
    task_number = len(tasks_info)
    node_number = len(nodes_info)

    # 拷贝节点信息，防止修改原始数据
    updated_nodes = [node.copy() for node in nodes_info]

    # 根据 allocations 更新节点剩余资源
    for task_id, node_id in allocations.items():
        task = next((t for t in tasks_info if t['task_id'] == task_id), None)
        if task:
            node = updated_nodes[node_id]
            node['cpu']    -= task['cpu_demand']
            node['memory'] -= task['memory_demand']
            node['gpu']    -= task['gpu_demand']
            updated_nodes[node_id] = node

    # ---- 构建节点状态 ----
    # 每个节点的状态包含：
    #  [针对前 max_tasks 个任务的分配标记(0/1)] + [cpu_usage, mem_usage, gpu_usage, 剩余cpu, 剩余mem, 剩余gpu]
    for node_index, (node_origin, node_after) in enumerate(zip(node_start, updated_nodes)):
        # (1) 任务分配标记：只对前 max_tasks 个任务
        alloc_vector = []
        for i in range(max_tasks):
            if i < task_number:
                # 判断该任务 i 是否分配给了此 node_index
                task_id = tasks_info[i]['task_id']
                alloc_vector.append(1 if allocations.get(task_id) == node_index else 0)
            else:
                # 超过实际任务数，用 0 填充
                alloc_vector.append(0)

        # (2) 计算资源使用率
        cpu_usage = (
            (node_origin['cpu'] - node_after['cpu']) / node_origin['cpu']
            if node_origin['cpu'] > 0 else 0
        )
        mem_usage = (
            (node_origin['memory'] - node_after['memory']) / node_origin['memory']
            if node_origin['memory'] > 0 else 0
        )
        gpu_usage = (
            (node_origin['gpu'] - node_after['gpu']) / node_origin['gpu']
            if node_origin['gpu'] > 0 else 0
        )

        # (3) 剩余资源量
        remaining_cpu    = node_after['cpu']
        remaining_memory = node_after['memory']
        remaining_gpu    = node_after['gpu']

        # 拼接节点状态
        node_state.extend(alloc_vector)
        node_state.extend([
            cpu_usage,
            mem_usage,
            gpu_usage,
            remaining_cpu,
            remaining_memory,
            remaining_gpu
        ])

    # ---- 构建任务状态 ----
    # 对前 max_tasks 个任务记录 [已分配节点ID, cpu_demand, mem_demand, gpu_demand]
    # 超过的任务不进状态向量
    for i in range(max_tasks):
        if i < task_number:
            task = tasks_info[i]
            node_id = allocations.get(task['task_id'], -1)
            task_state.extend([
                node_id,
                task['cpu_demand'],
                task['memory_demand'],
                task['gpu_demand']
            ])
        else:
            # 若不足 max_tasks，就填充占位值
            task_state.extend([-1, 0, 0, 0])

    # 最终状态拼接
    state = node_state + task_state
    return state

def get_task_batches(tasks_info):
    """根据任务的Arrival Time属性将任务分批"""
    batches = {}
    for task in tasks_info:
        arrival_time = task['arrival_time']
        if arrival_time not in batches:
            batches[arrival_time] = []
        batches[arrival_time].append(task)
    # 返回按照 arrival_time 排序后的批次列表
    return [batches[t] for t in sorted(batches.keys())]

# ========== 3) 定义 Spread 和 Binpack 两种分配策略，用于对比 ==========

def spread_allocation(nodes_info, tasks_info):
    """
    Spread: 每次把任务分配到“最空闲”的节点。
    这里通过 (cpu + memory + gpu) 的剩余资源总量来衡量空闲程度，越大越空闲。
    """
    allocations = {}
    for task in tasks_info:
        best_node = -1
        best_leftover = -1  # 记录最大剩余资源总量
        for i, node in enumerate(nodes_info):
            if is_task_allocatable(node, task):
                # 计算剩余资源总量
                leftover = node['cpu'] + node['memory'] + node['gpu']
                if leftover > best_leftover:
                    best_leftover = leftover
                    best_node = i
        # 若找到可分配节点，则分配
        if best_node >= 0:
            allocations[task['task_id']] = best_node
            update_node_resources(nodes_info[best_node], task)
    return allocations


def binpack_allocation(nodes_info, tasks_info):
    """
    Binpack: 每次把任务分配到“最紧凑”的节点。
    这里通过 (cpu + memory + gpu) 的剩余资源总量来衡量紧凑程度，越小越紧。
    """
    allocations = {}
    for task in tasks_info:
        best_node = -1
        best_leftover = 1e9  # 记录最小剩余资源总量
        for i, node in enumerate(nodes_info):
            if is_task_allocatable(node, task):
                # 计算剩余资源总量
                leftover = node['cpu'] + node['memory'] + node['gpu']
                if leftover < best_leftover:
                    best_leftover = leftover
                    best_node = i
        # 若找到可分配节点，则分配
        if best_node >= 0:
            allocations[task['task_id']] = best_node
            update_node_resources(nodes_info[best_node], task)
    return allocations


# ========== 3) 推理功能：按批次推理分配 ==========
if __name__ == "__main__":
    # 1. 加载节点和任务数据
    nodes_info = parse_node_info('../data/nodes_info.txt')
    tasks_info = parse_tasks_info('../data/100Job/tasks_info.txt')
    node_start = copy.deepcopy(nodes_info)  # 用于计算负载均衡度
    task_start = copy.deepcopy(tasks_info)

    # 2. 分批加载任务（与训练保持一致）
    batch_tasks = get_task_batches(tasks_info)

    # 3. 初始化 DQNAgent（与训练时的参数对应）
    max_tasks    = 15
    node_number  = len(nodes_info)
    state_size   = node_number * (max_tasks + 6) + max_tasks * 4
    action_size  = node_number * max_tasks
    agent        = DQNAgent(state_size, action_size)

    # 4. 加载训练好的模型
    model_path = 'best_model_LB0113.pth'
    try:
        agent.load_model(model_path)
    except FileNotFoundError:
        print(f"模型文件未找到: {model_path}")
        exit(1)

    print("\n开始推理任务分配（分批）...")

    # 5. 依次处理每个批次
    #    注意：如果想让批次之间互不干扰，可以在分完一个批次后，"还原" 节点资源
    #    或者你可以模拟“在线”场景，上一批次的分配影响下一批次的可用资源
    #    这里先演示与训练时相似的“独立批次”测试，即每个批次都从初始节点状态开始
    all_allocations = {}  # 用于记录所有批次的分配结果

    # 为了演示批次间独立，先把原始节点信息保存一下
    global_nodes_backup = copy.deepcopy(nodes_info)
    #打开文件，用于记录每批的三种算法负载均衡度
    output_file = open("compare.txt", "w")
    for batch_index, batch in enumerate(batch_tasks):
        print(f"\n===== 推理批次 {batch_index+1}/{len(batch_tasks)}，到达任务数：{len(batch)} =====")
        # (a) 还原节点到最初状态（若想独立评估每个批次）
        nodes_info = copy.deepcopy(global_nodes_backup)

        # (b) 初始化该批次
        allocations = {}
        allocated_tasks = set()
        # 生成初始 state
        state = get_state(agent, nodes_info, batch, allocations, max_tasks=max_tasks)

        # (c) 分配任务
        while len(allocated_tasks) < len(batch):
            taskNumber = len(batch)
            nodeNumber = len(nodes_info)

            action = agent.act(
                state,
                nodeNumber=nodeNumber,
                taskNumber=taskNumber,
                tasks=batch,
                allocated_tasks=allocated_tasks,
                max_tasks=max_tasks
            )

            node_idx  = action // max_tasks
            task_idx  = action % max_tasks
            node      = nodes_info[node_idx]
            task_info = batch[task_idx]

            # 如果该任务已经分配，跳过
            if task_idx in allocated_tasks:
                print(f" 任务 {task_idx} 已分配，跳过")
                continue

            # 如果资源不足，跳过
            if not is_task_allocatable(node, task_info):
                print(f" 节点 {node_idx} 资源不足，无法分配任务 {task_idx}")
                continue

            # 执行分配
            update_node_resources(node, task_info)
            allocations[task_info['task_id']] = node_idx
            allocated_tasks.add(task_idx)
            print(f" -> 任务 {task_idx} 分配到节点 {node_idx}")

            # (d) 更新状态
            state = get_state(agent, nodes_info, batch, allocations, max_tasks=max_tasks)

        # (e) 记录本批次的分配结果
        all_allocations[f"batch_{batch_index+1}"] = allocations
        # 6. 若希望查看整体负载均衡度，可在**所有批次**分配完成后计算
        #    这里仅示例：节点已被多次覆盖，如果想统一衡量，可自行定义合并逻辑
        # print("\n===== 所有批次分配完成，开始计算负载均衡度(示例) =====")
        dqn_balance = calculate_load_balance_and_resource_balance(node_start, nodes_info)
        print(f"  [DQN] 负载均衡度: {dqn_balance:.4f}, 分配数: {len(allocated_tasks)}/{len(batch)}")

        # 2) ---------- Spread 分配 ----------
        nodes_spread = copy.deepcopy(global_nodes_backup)
        allocations_spread = spread_allocation(nodes_spread, batch)
        spread_balance = calculate_load_balance_and_resource_balance(node_start, nodes_spread)
        print(f"  [Spread] 负载均衡度: {spread_balance:.4f}, 分配数: {len(allocations_spread)}/{len(batch)}")

        # 3) ---------- Binpack 分配 ----------
        nodes_binpack = copy.deepcopy(global_nodes_backup)
        allocations_binpack = binpack_allocation(nodes_binpack, batch)
        binpack_balance = calculate_load_balance_and_resource_balance(node_start, nodes_binpack)
        print(f"  [Binpack] 负载均衡度: {binpack_balance:.4f}, 分配数: {len(allocations_binpack)}/{len(batch)}")
        # 将结果写入 compare.txt，格式:  批次编号  DQN  Spread  Binpack
        output_file.write(
            f"{batch_index + 1} {dqn_balance:.4f} {spread_balance:.4f} {binpack_balance:.4f}\n"
        )

    # 关闭文件
    output_file.close()

    print("\n===== 推理完成，结果已写入 compare.txt =====")
    # 7. 打印结果
    # print("\n===== 最终分配结果（按批次）=====")
    # for batch_key, alloc in all_allocations.items():
    #     print(f"[{batch_key}]")
    #     for task_id, node_id in alloc.items():
    #         print(f"  任务 {task_id} 分配到节点 {node_id}")

