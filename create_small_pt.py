import torch
import os


def create_small_dataset(input_path: str, output_path: str, num_items: int):
    """
    加载一个 .pt 文件，提取列表中的前 num_items 个元素，并保存为新的 .pt 文件。

    参数:
    - input_path (str): 输入的原始 .pt 文件路径。
    - output_path (str): 输出的子集 .pt 文件路径。
    - num_items (int): 需要提取的元素数量。
    """
    print(f"\n--- 正在处理文件: {input_path} ---")

    # 1. 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件 '{input_path}' 不存在。请检查文件名和路径。")
        return

    # 2. 加载数据
    try:
        print("正在加载数据...")
        original_data = torch.load(input_path)
        print("数据加载成功。")
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    # 3. 验证数据类型是否为列表
    if not isinstance(original_data, list):
        print(f"错误: 文件 '{input_path}' 中的数据不是一个列表，而是一个 {type(original_data)}。")
        return

    print(f"原始数据集包含 {len(original_data)} 个图。")

    # 4. 提取前 num_items 个图
    # Python的列表切片会自动处理列表长度不足的情况，不会报错
    subset_data = original_data[:num_items]

    print(f"已提取前 {len(subset_data)} 个图。")

    # 5. 保存新的小数据集
    try:
        print(f"正在将子集保存到: {output_path} ...")
        torch.save(subset_data, output_path)
        print(f"成功保存！")
    except Exception as e:
        print(f"保存文件时出错: {e}")


# --- 主程序 ---
if __name__ == "__main__":
    # --- 1. 定义文件路径和参数 ---

    original_file_1 = 'prepared_data/gdb9_pyg_dataset_fully_connected.pt'
    original_file_2 = 'prepared_data/gdb9_pyg_dataset_with_absorbing_state.pt'

    # 定义新生成的小数据集文件名
    subset_file_1 = 'prepared_data/small_fully_connected.pt'
    subset_file_2 = 'prepared_data/small.pt'

    # 定义要提取的图的数量
    num_graphs_to_extract = 1000

    # --- 2. 执行任务 ---

    # 处理第一个文件
    create_small_dataset(original_file_1, subset_file_1, num_graphs_to_extract)

    # 处理第二个文件
    create_small_dataset(original_file_2, subset_file_2, num_graphs_to_extract)

    print("\n所有操作完成！")