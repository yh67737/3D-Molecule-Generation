import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import argparse
import os
import matplotlib.pyplot as plt
import math

# --- 常量定义 ---
ATOM_MAP = ['H', 'C', 'N', 'O', 'F', 'Absorbing']
BOND_TYPE_MAP = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    None
]
ATOM_COLOR_MAP_3D = {'H': 'yellow', 'C': 'grey', 'N': 'blue', 'O': 'red', 'F': 'green'}


def pyg_to_rdkit_mol(data):
    """将单个 PyG Data 对象转换为 RDKit Mol 对象。"""
    emol = Chem.EditableMol(Chem.Mol())
    pyg_idx_to_rdkit_idx = {}

    # 添加原子
    for i in range(data.x.size(0)):
        atom_type_idx = torch.argmax(data.x[i]).item()
        atom_symbol = ATOM_MAP[atom_type_idx]
        if atom_symbol == 'Absorbing': continue
        atom = Chem.Atom(atom_symbol)
        rdkit_idx = emol.AddAtom(atom)
        pyg_idx_to_rdkit_idx[i] = rdkit_idx

    # 添加键
    for i in range(data.edge_index.size(1)):
        start_node_pyg, end_node_pyg = data.edge_index[:, i].tolist()
        if start_node_pyg not in pyg_idx_to_rdkit_idx or end_node_pyg not in pyg_idx_to_rdkit_idx: continue

        start_node_rdkit = pyg_idx_to_rdkit_idx[start_node_pyg]
        end_node_rdkit = pyg_idx_to_rdkit_idx[end_node_pyg]

        bond_type_idx = torch.argmax(data.edge_attr[i]).item()
        bond_type = BOND_TYPE_MAP[bond_type_idx]
        if bond_type is None: continue

        if start_node_rdkit < end_node_rdkit and emol.GetBondBetweenAtoms(start_node_rdkit, end_node_rdkit) is None:
            emol.AddBond(start_node_rdkit, end_node_rdkit, bond_type)

    try:
        mol = emol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def draw_3d_molecule_on_ax(data, ax, title):
    """在给定的 Matplotlib 3D Axes 上绘制单个分子。"""
    pos = data.pos.cpu().numpy()
    atom_types_indices = torch.argmax(data.x, dim=1).cpu().numpy()

    for i in range(pos.shape[0]):
        atom_symbol = ATOM_MAP[atom_types_indices[i]]
        if atom_symbol == 'Absorbing': continue
        color = ATOM_COLOR_MAP_3D.get(atom_symbol, 'black')
        ax.scatter(pos[i, 0], pos[i, 1], pos[i, 2], c=color, s=150, edgecolors='black')

    for i in range(data.edge_index.size(1)):
        start_idx, end_idx = data.edge_index[:, i]
        if ATOM_MAP[atom_types_indices[start_idx]] != 'Absorbing' and ATOM_MAP[
            atom_types_indices[end_idx]] != 'Absorbing':
            p1, p2 = pos[start_idx], pos[end_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', linewidth=1)

    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(False)
    ax.view_init(elev=20., azim=-60)
    plt.axis('off')


def visualize_from_pyg_file(file_path, args):
    """从 PyG 文件加载分子列表并以网格形式可视化。"""
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 -> {file_path}");
        return

    try:
        molecule_list = torch.load(file_path)
        if not isinstance(molecule_list, list):  # 保证兼容只含单个分子的文件
            molecule_list = [molecule_list]
        total_mols_in_file = len(molecule_list)
        print(f"成功加载文件: {os.path.basename(file_path)}，共包含 {total_mols_in_file} 个分子。")
    except Exception as e:
        print(f"错误: 无法加载或解析文件 -> {file_path}\n{e}");
        return

    # 根据参数筛选要处理的分子范围
    start = args.start_index
    end = min(start + args.max_mols, total_mols_in_file)
    mols_to_process = molecule_list[start:end]

    if not mols_to_process:
        print("指定范围内没有要处理的分子。");
        return

    print(f"将处理索引从 {start} 到 {end - 1} 的分子，共 {len(mols_to_process)} 个。")

    # 分页处理
    base_name = os.path.splitext(args.output or file_path)[0]
    for i in range(0, len(mols_to_process), args.mols_per_image):
        chunk = mols_to_process[i: i + args.mols_per_image]
        page_num = (i // args.mols_per_image)
        output_path = f"{base_name}_{page_num}.png"

        print(f"正在生成第 {page_num + 1} 张图，包含 {len(chunk)} 个分子...")

        if args.view3d:
            # --- 生成 3D 网格图 ---
            cols = 4  # 每行固定显示4个分子
            rows = math.ceil(len(chunk) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), subplot_kw={'projection': '3d'})
            axes = axes.flatten()
            for j, data in enumerate(chunk):
                draw_3d_molecule_on_ax(data, axes[j], f"Mol_{start + i + j}")
            for j in range(len(chunk), len(axes)):  # 隐藏多余的子图
                axes[j].set_visible(False)
            fig.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        else:
            # --- 生成 2D 网格图 ---
            mol_list = [pyg_to_rdkit_mol(data) for data in chunk]
            mol_list = [m for m in mol_list if m is not None]  # 过滤掉转换失败的
            for mol in mol_list: AllChem.Compute2DCoords(mol)  # 为每个分子生成2D坐标

            legends = [f"Mol_{start + i + j}" for j in range(len(chunk))]
            img = Draw.MolsToGridImage(mol_list, molsPerRow=4, subImgSize=(300, 300), legends=legends)
            img.save(output_path)

        print(f"成功！已保存至: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从包含分子列表的 PyTorch Geometric (.pt) 文件中可视化分子网格。")
    parser.add_argument("file_path", type=str, help="包含一个或多个分子图数据的 .pt 文件的路径。")
    parser.add_argument("-o", "--output", type=str, help="输出图像的基础文件名 (不含后缀和页码)。")
    parser.add_argument("-n", "--mols_per_image", type=int, default=12, help="每张输出图像中包含的分子数量。")
    parser.add_argument("--start_index", type=int, default=0, help="从文件中的哪个索引位置开始处理分子。")
    parser.add_argument("--max_mols", type=int, default=120, help="从起始位置开始，最多处理多少个分子。")
    parser.add_argument("--view3d", action="store_true", help="生成 3D 视图而不是默认的 2D 视图。")

    args = parser.parse_args()
    visualize_from_pyg_file(args.file_path, args)