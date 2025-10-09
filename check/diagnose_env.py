import sys
import os
from pathlib import Path

print("--- 环境诊断开始 ---")

# 1. 打印当前正在运行的 Python 解释器的确切路径
print(f"\n[1] 当前 Python 解释器 (sys.executable):")
print(f"    {sys.executable}")

# 2. 尝试导入 RDKit 并打印其位置
try:
    import rdkit
    print(f"\n[2] 成功导入 RDKit 库。")
    
    # 获取 RDKit 包的 __init__.py 文件所在的目录
    rdkit_path = Path(rdkit.__file__).parent
    print(f"    RDKit 安装位置 (rdkit.__file__):")
    print(f"    {rdkit_path}")

    # 3. 检查在该位置下是否存在 'Contrib' 文件夹
    contrib_path = rdkit_path / 'Contrib'
    print(f"\n[3] 检查 'Contrib' 文件夹是否存在:")
    if contrib_path.is_dir():
        print(f"    [成功] 在 {rdkit_path} 中找到了 'Contrib' 文件夹。")
        
        # 尝试列出 'Contrib' 文件夹中的内容，看看 'SA_Score' 是否在里面
        sa_score_path = contrib_path / 'SA_Score'
        if sa_score_path.is_dir():
            print(f"    [成功] 在 'Contrib' 文件夹中找到了 'SA_Score' 子文件夹。")
        else:
            print(f"    [警告] 'Contrib' 存在，但未找到 'SA_Score' 子文件夹。")
            
    else:
        print(f"    [失败] 在 {rdkit_path} 中未找到 'Contrib' 文件夹。")

except ImportError:
    print("\n[!] 严重错误: 在当前 Python 环境中无法导入 RDKit 库。")
    print("    请确保你已在激活的 Conda 环境中正确安装了 RDKit。")

print("\n--- 环境诊断结束 ---")