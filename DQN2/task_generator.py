import random
import os
from datetime import datetime

def generate_vgpu_tasks(num_tasks=150, output_file="data/vgpu_tasks.txt"):
    """生成VGPU任务配置"""
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    cpu_range = [1, 2, 4, 6, 8, 12, 16]
    memory_range = [2, 4, 6, 8, 12, 16, 24, 32]
    vgpu_demand_range = [1, 2, 4, 6, 8]
    vgpu_memory_range = [2, 4, 6, 8, 12, 16]
    execution_time_range = [10, 15, 20, 23, 30, 45, 60, 90, 120]
    
    with open(output_file, 'w') as f:
        
        for i in range(num_tasks):
            cpu_demand = random.choice(cpu_range)
            memory_demand = random.choice(memory_range)
            vgpu_demand = random.choice(vgpu_demand_range)
            vgpu_memory = random.choice(vgpu_memory_range)
            execution_time = random.choice(execution_time_range)
            
            f.write(f"Task ID: {i}\n")
            f.write(f"Job ID: {i}\n")
            f.write(f"CPU Demand: {cpu_demand} cores\n")
            f.write(f"Memory Demand: {memory_demand} GB\n")
            f.write(f"vGPU Demand: {vgpu_demand} cards\n")
            f.write(f"vGPU Memory: {vgpu_memory} GB\n")
            f.write(f"Arrival Time: 0 units\n")
            f.write(f"Execution Time: {execution_time} units\n")
            f.write("\n")
    
    print(f"✅ 生成 {num_tasks} 个任务 -> {output_file}")

if __name__ == "__main__":
    generate_vgpu_tasks(150, "data/vgpu_tasks.txt")