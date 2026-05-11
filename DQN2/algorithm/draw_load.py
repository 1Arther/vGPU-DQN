import matplotlib.pyplot as plt

def plot_compare_results(file_path):
    # 存储批次号、DQN、Spread、Binpack 数据
    batches = []
    dqn_vals = []
    spread_vals = []
    binpack_vals = []

    # 1. 读取 compare.txt 文件
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 每行格式: batch dqn spread binpack
            parts = line.split()
            batch = int(parts[0])
            dqn = float(parts[1])
            spread = float(parts[2])
            binpack = float(parts[3])

            batches.append(batch)
            dqn_vals.append(dqn)
            spread_vals.append(spread)
            binpack_vals.append(binpack)

    # 2. 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(batches, dqn_vals, color='blue', label='DQN')
    plt.plot(batches, spread_vals, color='green', label='Spread')
    plt.plot(batches, binpack_vals, color='red', label='Binpack')

    # 3. 设置坐标轴、图例等
    plt.xlabel('Task Batches')
    plt.ylabel('Load Balance')
    plt.title('Comparison of Load Balance for DQN / Spread / Binpack')
    plt.legend()
    plt.grid(True)

    # 4. 设置横坐标刻度间隔为 5
    max_batch = max(batches)
    min_batch = min(batches)
    # 每隔 5 个批次显示一个刻度
    plt.xticks(range(min_batch-1, max_batch+1, 5))

    # 5. 显示或保存图形
    plt.show()
    # 若想直接保存，可用:
    # plt.savefig('compare_plot.png', dpi=300)

if __name__ == "__main__":
    # 假设 compare.txt 与本脚本在同一目录，也可以改为绝对路径
    file_path = 'compare.txt'
    plot_compare_results(file_path)
