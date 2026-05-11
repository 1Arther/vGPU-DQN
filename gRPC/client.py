import copy
import grpc
import scheduler_pb2
import scheduler_pb2_grpc
import numpy as np

def run_batch_inference():
    channel = grpc.insecure_channel('localhost:50051')
    stub = scheduler_pb2_grpc.DQNServiceStub(channel)

    # 假设有 10 个节点, 11 个任务:
    # 注意，这里 server 只需要 nodes_info / tasks_info 来构造 state 并做决策
    # 但是我们还想在客户端自己再次计算负载均衡，所以要保留 "node_start" 里
    # cpu / memory / gpu => 分别对应节点的 "总资源"；node_current 用来表示最新剩余资源

    # ---------- 1) 构建节点初始信息 (node_start) ----------
    # 假设把 NodeInfo(node_id=..., cpuRem=64, memoryRem=256, gpuRem=8) 视作“总资源”
    # 或者你可以另外有字段来写“总资源”与“剩余资源”
    # 这里为了演示，将 node.cpuRem 看作节点总资源:
    node_start_list = [
        # node_id=0
        {'node_id': 0, 'cpu': 64, 'memory': 256, 'gpu': 8},
        {'node_id': 1, 'cpu': 48, 'memory': 512, 'gpu': 8},
        {'node_id': 2, 'cpu': 32, 'memory': 128, 'gpu': 8},
        {'node_id': 3, 'cpu': 32, 'memory': 256, 'gpu': 8},
        {'node_id': 4, 'cpu': 48, 'memory': 128, 'gpu': 4},
        {'node_id': 5, 'cpu': 64, 'memory': 128, 'gpu': 8},
        {'node_id': 6, 'cpu': 48, 'memory': 128, 'gpu': 4},
        {'node_id': 7, 'cpu': 48, 'memory': 512, 'gpu': 8},
        {'node_id': 8, 'cpu': 32, 'memory': 512, 'gpu': 8},
        {'node_id': 9, 'cpu': 32, 'memory': 256, 'gpu': 4},
    ]
    # 用于发给服务器
    nodes_proto = []
    for n in node_start_list:
        info = scheduler_pb2.NodeInfo(
            node_id=n['node_id'],
            cpuUsage=0,   # 先设 0
            memoryUsage=0,
            gpuUsage=0,
            cpuRem=n['cpu'],      # 这里把 cpuRem 初始化为节点总资源(示例)
            memoryRem=n['memory'],
            gpuRem=n['gpu'],
        )
        nodes_proto.append(info)

    tasks_proto = [
        scheduler_pb2.TaskInfo(task_id=0, cpu_demand=3,  memory_demand=26, gpu_demand=4),
        scheduler_pb2.TaskInfo(task_id=1, cpu_demand=4,  memory_demand=6,  gpu_demand=4),
        scheduler_pb2.TaskInfo(task_id=2, cpu_demand=5,  memory_demand=30, gpu_demand=4),
        scheduler_pb2.TaskInfo(task_id=3, cpu_demand=4,  memory_demand=24, gpu_demand=2),
        scheduler_pb2.TaskInfo(task_id=4, cpu_demand=4,  memory_demand=4,  gpu_demand=2),
        scheduler_pb2.TaskInfo(task_id=5, cpu_demand=7,  memory_demand=15, gpu_demand=2),
        scheduler_pb2.TaskInfo(task_id=6, cpu_demand=5,  memory_demand=8,  gpu_demand=4),
        scheduler_pb2.TaskInfo(task_id=7, cpu_demand=3,  memory_demand=13, gpu_demand=2),
        scheduler_pb2.TaskInfo(task_id=8, cpu_demand=1,  memory_demand=11, gpu_demand=2),
        scheduler_pb2.TaskInfo(task_id=9, cpu_demand=6,  memory_demand=11, gpu_demand=4),
        scheduler_pb2.TaskInfo(task_id=10,cpu_demand=6,  memory_demand=5,  gpu_demand=4),
    ]

    # ---------- 2) 向服务端发送请求 ----------
    request = scheduler_pb2.BatchInferenceRequest(
        nodes=nodes_proto,
        tasks=tasks_proto
    )
    response = stub.PredictBatch(request)

    print("Server BatchInferenceReply:")
    for i, t_id in enumerate(response.task_ids):
        print(f"  Task {t_id} -> Node {response.node_ids[i]}")

    # ---------- 3) 在客户端本地更新节点资源 ----------
    # 以便计算最终负载均衡度
    # 先复制一份 node_current 用于表示“调度后”节点剩余资源
    node_current_list = copy.deepcopy(node_start_list)

    # 将 service 返回的 (task_id -> node_id) 映射应用到本地
    # 这里为了更新资源，需要知道每个 task_id 的资源需求
    # 因此可先把 tasks_proto 转成一个 {task_id: (cpu_demand, mem_demand, gpu_demand)} 字典
    task_demands = {}
    for t in tasks_proto:
        task_demands[t.task_id] = (t.cpu_demand, t.memory_demand, t.gpu_demand)

    # 遍历所有任务分配
    for i, t_id in enumerate(response.task_ids):
        assigned_node = response.node_ids[i]
        if assigned_node == -1:
            # 表示资源不足 或 服务端无法分配
            continue
        cpu_d, mem_d, gpu_d = task_demands[t_id]
        # 找到对应 node_current
        node_c = node_current_list[assigned_node]
        # 扣减资源
        node_c['cpu']    -= cpu_d
        node_c['memory'] -= mem_d
        node_c['gpu']    -= gpu_d

    # ---------- 4) 调用 calculate_load_balance_and_resource_balance ----------
    final_balance = calculate_load_balance_and_resource_balance(
        node_start_list,  # 初始节点资源
        node_current_list # 当前节点剩余资源
    )
    print(f"[客户端] 最终负载均衡度: {final_balance:.4f}")

def calculate_load_balance_and_resource_balance(node_start, nodes_info):
    """
    node_start: [{'cpu':..., 'memory':..., 'gpu':...}, ...]
    nodes_info: [{'cpu':..., 'memory':..., 'gpu':...}, ...]
      - 这里 'cpu' 代表剩余资源
    """
    assert len(node_start) == len(nodes_info), "node_start 和 nodes_info 长度不一致！"

    cpu_usages = []
    memory_usages = []
    gpu_usages = []
    resource_balances = 0.0

    for start, current in zip(node_start, nodes_info):
        # 计算使用率 = (start - current) / start
        # 如果 start['cpu'] > 0, usage = (start['cpu'] - current['cpu']) / start['cpu']
        # 否则 usage=0
        cpu_usage = (start['cpu'] - current['cpu'])/start['cpu'] if start['cpu'] > 0 else 0
        mem_usage = (start['memory'] - current['memory'])/start['memory'] if start['memory'] > 0 else 0
        gpu_usage = (start['gpu'] - current['gpu'])/start['gpu'] if start['gpu'] > 0 else 0

        cpu_usages.append(cpu_usage)
        memory_usages.append(mem_usage)
        gpu_usages.append(gpu_usage)

        resource_balances += np.std([cpu_usage, mem_usage])

    # 计算整体负载均衡度
    std_cpu = np.std(cpu_usages)
    std_mem = np.std(memory_usages)
    std_gpu = np.std(gpu_usages)

    load_balance = (std_cpu + std_mem + std_gpu)/3
    resource_balances /= len(nodes_info)

    print("CPU利用率: ", cpu_usages)
    print("内存利用率: ", memory_usages)
    print("GPU利用率: ", gpu_usages)
    print("每个节点的负载均衡度: ", load_balance)
    print("每个节点的资源平衡度: ", resource_balances)

    return load_balance*0.5 + resource_balances*0.5


if __name__ == "__main__":
    run_batch_inference()
