# utils.py
"""
工具函数模块

generate_demo_pointcloud(path, radius, n)
    生成一个均匀球形点云用于演示，保存为 .npy 文件

stl_to_pointcloud(stl_path, ...)
    把 STL 文件转换为点云，支持均匀面采样 / 泊松盘采样

parse_args()
    解析命令行参数，返回 (config_path, pc_file_or_None)
"""

import sys
import math
import numpy as np
import open3d as o3d
import os


# ════════════════════════════════════════════════════════════
#  generate_demo_pointcloud
# ════════════════════════════════════════════════════════════

def generate_demo_pointcloud(
    path:   str   = "demo_ball.npy",
    radius: float = 0.015,
    n:      int   = 2000,
) -> str:
    """
    生成均匀分布的球形点云并保存为 .npy 文件（黄金螺旋算法）。

    参数
    ----
    path   : 保存路径
    radius : 球半径 (m)
    n      : 点数

    返回
    ----
    path   : 实际保存路径
    """
    pts    = []
    golden = 2.399963229728653   # 黄金角（弧度）

    for i in range(n):
        theta = math.acos(1.0 - 2.0 * (i + 0.5) / n)
        phi   = golden * i
        pts.append([
            radius * math.sin(theta) * math.cos(phi),
            radius * math.cos(theta),
            radius * math.sin(theta) * math.sin(phi),
        ])

    np.save(path, np.array(pts, dtype=np.float32))
    print(f"[utils] 已生成演示点云: {path}  ({n} 点, 半径={radius} m)")
    return path


# ════════════════════════════════════════════════════════════
#  stl_to_pointcloud
# ════════════════════════════════════════════════════════════

def stl_to_pointcloud(
    stl_path:      str,
    output_path:   str   = None,
    n_points:      int   = 5000,
    method:        str   = "uniform",       # "uniform" | "poisson"
    poisson_radius:float = None,            # poisson 专用，None = 自动
    include_normals: bool = False,
    scale:         float = 1.0,
    center:        bool  = True,
) -> np.ndarray:
    """
    将 STL 文件转换为点云。

    参数
    ----
    stl_path        : 输入 STL 文件路径（ASCII 或 Binary 均可）
    output_path     : 输出文件路径（.npy / .ply / .pcd / .txt），
                      None = 不保存，仅返回数组
    n_points        : 目标采样点数（uniform 模式精确，poisson 模式近似）
    method          : 采样方法
                        "uniform"  — 按面积权重均匀随机采样（推荐）
                        "poisson"  — 泊松盘采样（点间距更均匀，但较慢）
    poisson_radius  : 泊松盘采样最小点间距 (m)，None = 自动从 n_points 估算
    include_normals : 是否在返回数组中附加法向量（True → shape=(N,6)）
    scale           : 整体缩放系数（STL 常用 mm，传 0.001 转换为 m）
    center          : 是否将点云平移到局部原点（True = 减去包围盒中心）

    返回
    ----
    pts : np.ndarray，shape=(N,3) 或 (N,6)，dtype=float32

    依赖
    ----
    open3d >= 0.13

    示例
    ----
    # STL 单位 mm，转换为 m，均匀采样 3000 点，保存为 PLY
    pts = stl_to_pointcloud(
        stl_path    = "tool.stl",
        output_path = "tool.ply",
        n_points    = 3000,
        scale       = 0.001,
        method      = "uniform",
    )

    # 泊松盘采样，点间距约 1 mm
    pts = stl_to_pointcloud(
        stl_path       = "tool.stl",
        method         = "poisson",
        poisson_radius = 0.001,
        scale          = 0.001,
    )
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "stl_to_pointcloud 需要 open3d，请先安装：pip install open3d"
        )

    # ── 1. 读取 STL（作为三角网格）────────────────────────────
    print(f"[stl_to_pointcloud] 读取 STL: {stl_path}")
    mesh = o3d.io.read_triangle_mesh(stl_path)

    if not mesh.has_vertices():
        raise ValueError(f"STL 文件读取失败或无顶点: {stl_path}")

    n_tri = len(mesh.triangles)
    print(f"[stl_to_pointcloud] 网格: {len(mesh.vertices)} 顶点, {n_tri} 三角面")

    # ── 2. 缩放 ──────────────────────────────────────────────
    if scale != 1.0:
        mesh.scale(scale, center=mesh.get_center())
        print(f"[stl_to_pointcloud] 已缩放 ×{scale}")

    # ── 3. 法向量（泊松采样必须有法向量）────────────────────────
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # ── 4. 采样 ──────────────────────────────────────────────
    if method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
        print(f"[stl_to_pointcloud] 均匀采样: {len(pcd.points)} 点")

    elif method == "poisson":
        # 自动估算 poisson_radius
        if poisson_radius is None:
            bbox   = mesh.get_axis_aligned_bounding_box()
            vol    = float(np.prod(np.asarray(bbox.get_extent())))
            area   = vol ** (2.0 / 3.0)           # 粗略估计表面积
            poisson_radius = math.sqrt(area / (n_points * math.pi)) * 2.0
            print(f"[stl_to_pointcloud] 自动估算泊松半径: {poisson_radius:.5f} m")

        pcd = mesh.sample_points_poisson_disk(
            number_of_points = n_points,
            init_factor      = 5,                 # 先均匀采 5× 再筛选
            pcl              = None,
        )
        print(f"[stl_to_pointcloud] 泊松采样: {len(pcd.points)} 点")

    else:
        raise ValueError(f"未知采样方法: {method!r}，请选 'uniform' 或 'poisson'")

    # ── 5. 中心化 ─────────────────────────────────────────────
    pts = np.asarray(pcd.points, dtype=np.float32)

    if center:
        pts -= pts.mean(axis=0)
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        print(f"[stl_to_pointcloud] 已中心化到局部原点")

    # ── 6. 法向量拼接 ─────────────────────────────────────────
    if include_normals:
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=poisson_radius or 0.01, max_nn=30
                )
            )
        normals = np.asarray(pcd.normals, dtype=np.float32)
        pts = np.concatenate([pts, normals], axis=1)   # (N, 6)
        print(f"[stl_to_pointcloud] 已附加法向量，输出 shape: {pts.shape}")

    # ── 7. 保存 ───────────────────────────────────────────────
    if output_path is not None:
        _save_pointcloud(pcd, pts, output_path)

    return pts


def _save_pointcloud(pcd, pts: np.ndarray, output_path: str):
    """将点云保存为指定格式。"""
    
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".npy":
        np.save(output_path, pts)

    elif ext == ".txt":
        np.savetxt(output_path, pts, fmt="%.6f")

    elif ext in (".ply", ".pcd"):
        # 确保 pcd.points 与 pts 的 XYZ 同步（pts 可能已经中心化）
        pcd.points = o3d.utility.Vector3dVector(
            pts[:, :3].astype(np.float64)
        )
        o3d.io.write_point_cloud(output_path, pcd)

    else:
        raise ValueError(
            f"不支持的输出格式: {ext}，请使用 .npy / .ply / .pcd / .txt"
        )

    print(f"[stl_to_pointcloud] 已保存: {output_path}  ({len(pts)} 点)")


# ════════════════════════════════════════════════════════════
#  parse_args
# ════════════════════════════════════════════════════════════

def parse_args():
    """
    解析命令行参数。

    用法
    ----
    python main.py                        → config="config.yaml", pc=None
    python main.py my.yaml                → config="my.yaml",     pc=None
    python main.py config.yaml ball.npy  → config + pc
    python main.py tool.stl              → 自动转换 STL → 点云

    返回
    ----
    (config_path: str, pc_file: str | None)
    """
    config_path = "config.yaml"
    pc_file     = None

    for arg in sys.argv[1:]:
        if arg.endswith((".yaml", ".yml")):
            config_path = arg
        else:
            pc_file = arg

    return config_path, pc_file
