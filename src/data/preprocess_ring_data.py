import os
import json
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from pathlib import Path

def preprocess_json_to_pt(source_dir, target_dir, batch_size=10000):
    """
    将包含分子图信息的 JSON 文件预处理并保存为 PyTorch Geometric 的 .pt 文件。

    Args:
        source_dir (str): 包含源 .json 文件的目录路径。
        target_dir (str): 保存输出 .pt 文件的目录路径。
        batch_size (int): 每个 .pt 文件中包含的图样本数量。
    """
    # 1. 确保目标目录存在
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    print(f"源数据目录: {source_dir}")
    print(f"目标数据目录: {target_dir}")
    print(f"每个 .pt 文件包含 {batch_size} 个样本")

    # 2. 查找所有源 JSON 文件
    try:
        json_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
        if not json_files:
            print(f"错误: 在 '{source_dir}' 中没有找到任何 .json 文件。请检查路径。")
            return
        print(f"找到了 {len(json_files)} 个 .json 文件，开始处理...")
    except FileNotFoundError:
        print(f"错误: 源目录 '{source_dir}' 不存在。请检查路径。")
        return

    # 3. 循环处理并分批保存
    processed_data_batch = []
    batch_counter = 1
    
    for filename in tqdm(json_files, desc="正在转换 JSON 文件"):
        file_path = os.path.join(source_dir, filename)
        
        try:
            with open(file_path, 'r') as f:
                graph_data = json.load(f)

            # 将JSON中的列表转换为PyTorch张量
            # 注意：PyG 要求 edge_index 的数据类型是 torch.long
            x = torch.tensor(graph_data['x'], dtype=torch.float)
            edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(graph_data['edge_attr'], dtype=torch.float).view(-1, 4)
            pos = torch.tensor(graph_data['pos'], dtype=torch.float)
            pring_out = torch.tensor(graph_data['pring_out'], dtype=torch.float).view(-1, 1)
            
            # 创建 PyG Data 对象
            data_obj = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                pring_out=pring_out
            )
            
            processed_data_batch.append(data_obj)
            
            # 当达到批次大小时，保存并重置列表
            if len(processed_data_batch) >= batch_size:
                save_path = os.path.join(target_dir, f'data_batch_{batch_counter}.pt')
                torch.save(processed_data_batch, save_path)
                tqdm.write(f"已保存批次 {batch_counter} 到 {save_path}")
                
                processed_data_batch = []
                batch_counter += 1

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            continue

    # 4. 保存最后一批不足 batch_size 的数据
    if processed_data_batch:
        save_path = os.path.join(target_dir, f'data_batch_{batch_counter}.pt')
        torch.save(processed_data_batch, save_path)
        tqdm.write(f"已保存最后一个批次 {batch_counter} 到 {save_path}")

    print("\n预处理完成！")


if __name__ == '__main__':
    # --- 配置路径 ---
    # 你的 JSON 文件所在的目录
    SOURCE_JSON_DIR = 'gdb9_preprocessed_data' 
    
    # 你想把生成的 .pt 文件保存到的新目录
    TARGET_PT_DIR = 'gdb9_pt_data_for_ring_predictor' 

    source_path = 'prepared_data/autodl-tmp/gdb9_unique_subgraphs_json'
    target_path = 'src/models/ring_network/gdb9_pt_data_for_ring_predictor'

    preprocess_json_to_pt(source_dir=source_path, target_dir=target_path)
