import os
import glob
import random
import shutil
from tqdm import tqdm

# --- 1. 请在这里配置您的文件夹路径 ---
source_folder = '/root/autodl-tmp/molecular_generation_project/prepared_data/autodl-tmp/gdb9_unique_subgraphs_json'  # 替换为您的原始数据文件夹路径
destination_folder = '/root/autodl-tmp/molecular_generation_project/prepared_data/gdb9_unique_subgraphs_json'  # 替换为您想存放新数据的文件夹路径
num_to_select = 10000  # 您想要选取的 JSON 文件数量
# -----------------------------------------

def select_and_copy_files(src, dst, num):
    """
    从源文件夹随机选取指定数量的.json文件并复制到目标文件夹。
    """
    # 检查源文件夹是否存在
    if not os.path.isdir(src):
        print(f"错误：源文件夹 '{src}' 不存在。请检查路径。")
        return

    # 如果目标文件夹不存在，则创建它
    os.makedirs(dst, exist_ok=True)
    print(f"目标文件夹 '{dst}' 已准备好。")

    # 使用 glob 高效查找所有 .json 文件
    # os.path.join 确保路径在不同操作系统上都能正确拼接
    search_pattern = os.path.join(src, '*.json')
    all_json_files = glob.glob(search_pattern)
    print(f"在源文件夹中找到了 {len(all_json_files)} 个 JSON 文件。")

    # 检查文件数量是否足够
    if len(all_json_files) < num:
        print(f"错误：文件数量不足。需要选取 {num} 个，但只找到了 {len(all_json_files)} 个。")
        return

    # 从文件列表中随机选取指定数量的文件
    print(f"正在随机选取 {num} 个文件...")
    selected_files = random.sample(all_json_files, num)
    print("选取完成！")

    # 复制文件并显示进度条
    print("开始复制文件...")
    # 使用 tqdm 创建一个进度条，让您可以看到复制进度
    for file_path in tqdm(selected_files, desc="复制进度"):
        shutil.copy2(file_path, dst) # 使用 copy2 可以同时复制文件的元数据

    print(f"\n成功！ {num} 个随机 JSON 文件已复制到 '{dst}'。")

# --- 运行主函数 ---
if __name__ == "__main__":
    select_and_copy_files(source_folder, destination_folder, num_to_select)