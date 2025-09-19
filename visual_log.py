import re
import pandas as pd
import matplotlib.pyplot as plt


def parse_and_plot_log(log_file_path, output_image_path):
    """
    解析一个训练日志文件，提取指定的指标并绘图。
    """
    print(f"正在读取日志文件: {log_file_path}")

    # 定义需要提取的所有键
    keys = [
        'loss', 'loss_I', 'loss_II', 'lossI_a', 'lossI_r',
        'lossI_b', 'lossII_a', 'lossII_r', 'lossII_b', 'lr'
    ]

    # 正则表达式，用于捕获键值对
    pattern = re.compile(r"(\w+)=([0-9.e+-]+)")

    all_epoch_data = []

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 检查是否是包含所需信息的日志行
                if 'loss=' in line and 'lr=' in line:
                    matches = dict(pattern.findall(line))

                    # 确保所有需要的键都存在，然后将值转换为浮点数
                    try:
                        epoch_data = {key: float(matches[key]) for key in keys}
                        all_epoch_data.append(epoch_data)
                    except KeyError:
                        # 如果某一行不包含所有需要的键，则跳过
                        continue
    except FileNotFoundError:
        print(f"\n错误: 文件 '{log_file_path}' 未找到。")
        print("请确保脚本和日志文件在同一个文件夹中。")
        return
    except Exception as e:
        print(f"读取或解析文件时发生错误: {e}")
        return

    if not all_epoch_data:
        print("未在日志中找到有效的条目进行绘图。")
        return

    print(f"成功解析了 {len(all_epoch_data)} 个轮次的数据。")

    # 将收集的数据转换为Pandas DataFrame
    df = pd.DataFrame(all_epoch_data)

    print("正在生成图表...")

    # 创建2x2的子图，以便更清晰地可视化
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Training Metrics Over Epochs', fontsize=20)

    # 图1: 主要的损失部分
    df[['loss', 'loss_I', 'loss_II']].plot(ax=axes[0, 0], grid=True)
    axes[0, 0].set_title('Main Loss Components')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss Value')
    axes[0, 0].legend()

    # 图2: Loss I 的细分
    df[['lossI_a', 'lossI_r', 'lossI_b']].plot(ax=axes[0, 1], grid=True)
    axes[0, 1].set_title('Loss I Breakdown')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss Value')
    axes[0, 1].legend()

    # 图3: Loss II 的细分
    df[['lossII_a', 'lossII_r', 'lossII_b']].plot(ax=axes[1, 0], grid=True)
    axes[1, 0].set_title('Loss II Breakdown')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Value')
    axes[1, 0].legend()

    # 图4: 学习率
    df[['lr']].plot(ax=axes[1, 1], grid=True, color='purple')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].legend()
    # 如果学习率变化范围很大，使用对数坐标轴
    if not df['lr'].empty and df['lr'].max() / df['lr'].min() > 100:
        axes[1, 1].set_yscale('log')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_image_path)

    print(f"图表已成功保存至 {output_image_path}")


# --- 主程序执行区 ---
if __name__ == '__main__':
    log_file = 'debug (1).log'  # 确保这是您上传的文件名
    output_image = 'training_loss_curves.png'
    parse_and_plot_log(log_file, output_image)