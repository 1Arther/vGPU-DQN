import numpy as np

def parse_node_info(file_path):
    nodes_info = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    node_info = {}
    gpu_comm_matrix = []
    reading_matrix = False

    for line in lines:
        if line.startswith("Node ID"):
            if node_info:
                node_info["gpu_comm_matrix"] = np.array(gpu_comm_matrix)
                nodes_info.append(node_info)
                node_info = {}
                gpu_comm_matrix = []
                reading_matrix = False

            node_info["node_id"] = int(line.split(": ")[1].strip())
        elif line.startswith("CPU"):
            node_info["cpu"] = int(line.split(": ")[1].split()[0])
        elif line.startswith("Memory"):
            node_info["memory"] = int(line.split(": ")[1].split()[0])
        elif line.startswith("GPUs"):
            node_info["gpu"] = int(line.split(": ")[1].split()[0])
            node_info["gpu_idles"] = []
            for i in range(node_info["gpu"]):
                node_info["gpu_idles"].append(i)
        elif line.startswith("GPU Communication Efficiency Matrix"):
            reading_matrix = True
        elif reading_matrix:
            if line.strip() == "":
                reading_matrix = False
            else:
                gpu_comm_matrix.append(list(map(float, line.strip().split())))

    if node_info:
        node_info["gpu_comm_matrix"] = np.array(gpu_comm_matrix)
        nodes_info.append(node_info)

    return nodes_info

def parse_tasks_info(file_path):
    tasks_info = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    task_info = {}

    for line in lines:
        if line.startswith("Task ID"):
            if task_info:
                tasks_info.append(task_info)
                task_info = {}
            task_info["task_id"] = int(line.split(": ")[1].strip())
        elif line.startswith("Job ID"):
            task_info["job_id"] = int(line.split(": ")[1].strip())
        elif line.startswith("CPU Demand"):
            task_info["cpu_demand"] = int(line.split(": ")[1].split()[0])
        elif line.startswith("Memory Demand"):
            task_info["memory_demand"] = int(line.split(": ")[1].split()[0])
        elif line.startswith("GPU Demand"):
            task_info["gpu_demand"] = int(line.split(": ")[1].split()[0])
        elif line.startswith("Arrival Time"):
            task_info["arrival_time"] = int(line.split(": ")[1].split()[0])
        elif line.startswith("Execution Time"):
            task_info["execution_time"] = int(line.split(": ")[1].split()[0])

    if task_info:
        tasks_info.append(task_info)

    return tasks_info

def parse_node_comm(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    matrix = np.array([list(map(float, line.split())) for line in lines])
    return matrix
