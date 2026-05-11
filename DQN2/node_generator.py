import numpy as np

# 定义节点数量和资源参数
num_nodes = 50

# 生成节点资源和通信效率信息
nodes_info = []

for node_id in range(num_nodes):
    list_of_cpu = [32,48,64]
    list_of_mem = [128,256,512]
    list_of_gpu = [4,8]
    cpu = np.random.choice(list_of_cpu)  # 随机生成CPU数量（假设范围为4到16）
    memory = np.random.choice(list_of_mem)  # 随机生成内存大小（128~1024）
    gpu = np.random.choice(list_of_gpu)  # 随机生成GPU数量（假设范围为1到4）

    # 生成GPU通信效率矩阵
    gpu_comm_matrix = np.random.rand(gpu, gpu)*10
    gpu_comm_matrix = (gpu_comm_matrix + gpu_comm_matrix.T) / 2
    np.fill_diagonal(gpu_comm_matrix, 0)  # 自己与自己通信效率设为1

    node_info = {
        "node_id": node_id,
        "cpu": cpu,
        "memory": memory,
        "gpu": gpu,
        "gpu_comm_matrix": gpu_comm_matrix
    }
    nodes_info.append(node_info)

# 将节点信息写入txt文件
with open('data/nodes_info.txt', 'w') as file:
    for node_info in nodes_info:
        file.write(f"Node ID: {node_info['node_id']}\n")
        file.write(f"CPU: {node_info['cpu']} cores\n")
        file.write(f"Memory: {node_info['memory']} GB\n")
        file.write(f"GPUs: {node_info['gpu']} cards\n")
        file.write("GPU Communication Efficiency Matrix:\n")
        for row in node_info['gpu_comm_matrix']:
            file.write(" ".join(f"{val:.2f}" for val in row) + "\n")
        file.write("\n")
