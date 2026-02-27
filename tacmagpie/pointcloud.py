# pointcloud.py
"""
点云碰撞器模块

类
--
PointCloudCollider
    管理点云碰撞体：
      - 从文件加载初始形状（.npy / .txt / .xyz / .ply）
      - 接受外部控制（位置直接设定 / 速度积分）
      - 将当前世界坐标同步到 Taichi fields，供 MPM kernel 使用
      - 累积碰撞力供外部读取
"""

import os
import taichi as ti
import open3d as o3d
import numpy as np
from config_loader import SimConfig


class PointCloudCollider:
    """
    点云碰撞体。

    参数
    ----
    filepath    : 点云文件路径（.npy / .txt / .xyz / .ply）
    init_center : 点云包围盒中心的初始世界坐标，shape=(3,)
    cfg         : SimConfig 实例（提供 DIM、MAX_COLLIDER_PTS、COLLIDER_RADIUS）
    """

    def __init__(self, filepath: str, init_center: np.ndarray, cfg: SimConfig):
        self.cfg          = cfg
        self.point_radius = cfg.COLLIDER_RADIUS

        self._raw_pts = self._load(filepath)          # (N,3) 局部坐标
        n = len(self._raw_pts)
        assert n <= cfg.MAX_COLLIDER_PTS, (
            f"点云点数 {n} 超过上限 {cfg.MAX_COLLIDER_PTS}，"
            f"请调大 config.yaml 中的 collider.max_pts"
        )
        self.n_pts = n

        # 将点云中心移至局部原点
        local_center   = self._raw_pts.mean(axis=0)
        self._raw_pts -= local_center

        # 当前状态
        self.position = np.array(init_center, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)

        # ── Taichi fields ──────────────────────────────────
        DIM = cfg.DIM
        MAX = cfg.MAX_COLLIDER_PTS
        self.ti_pts    = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX)
        self.ti_n      = ti.field(dtype=ti.i32, shape=())
        self.ti_radius = ti.field(dtype=ti.f32, shape=())
        self.ti_vel    = ti.Vector.field(DIM, dtype=ti.f32, shape=())
        self.ti_force  = ti.Vector.field(DIM, dtype=ti.f32, shape=())

        self.ti_n[None]      = self.n_pts
        self.ti_radius[None] = self.point_radius

        self.sync_to_taichi()

    # ── 文件加载 ─────────────────────────────────────────────

    @staticmethod
    def _load(filepath: str) -> np.ndarray:
        """加载点云文件，返回 (N,3) float32 数组。"""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".npy":
            pts = np.load(filepath).astype(np.float32)
        elif ext in (".txt", ".xyz"):
            pts = np.loadtxt(filepath, dtype=np.float32)
        elif ext == ".ply":
            pts = PointCloudCollider._load_ply(filepath)
        else:
            raise ValueError(
                f"不支持的文件格式: {ext}，请使用 .npy / .txt / .xyz / .ply"
            )
        assert pts.ndim == 2 and pts.shape[1] >= 3, \
            "点云必须是 (N, 3+) 数组"
        return pts[:, :3].astype(np.float32)

    @staticmethod
    def _load_ply(filepath: str) -> np.ndarray:
        """使用 Open3D 读取 PLY 文件（支持 ASCII 和 Binary 格式）。"""
        
        
        pcd = o3d.io.read_point_cloud(filepath)
        return np.asarray(pcd.points, dtype=np.float32)

    # ── 外部控制接口 ─────────────────────────────────────────

    def set_velocity(self, vel: np.ndarray):
        """设置碰撞体速度 (m/s)，shape=(3,)"""
        self.velocity = np.array(vel, dtype=np.float32)

    def set_position(self, pos: np.ndarray):
        """直接设定碰撞体质心世界坐标，shape=(3,)"""
        self.position = np.array(pos, dtype=np.float32)

    def step(self, dt_val: float):
        """按当前速度积分更新位置并同步到 Taichi。"""
        self.position += self.velocity * dt_val
        self.sync_to_taichi()

    def sync_to_taichi(self):
        """将当前点云世界坐标批量写入 Taichi field。"""
        MAX = self.cfg.MAX_COLLIDER_PTS
        DIM = self.cfg.DIM
        world_pts = self._raw_pts + self.position          # (N,3) 广播
        tmp = np.zeros((MAX, DIM), dtype=np.float32)
        tmp[: self.n_pts] = world_pts
        self.ti_pts.from_numpy(tmp)
        self.ti_vel[None] = ti.Vector(self.velocity.tolist())

    # ── 力读写 ───────────────────────────────────────────────

    def get_force(self) -> np.ndarray:
        """读取上一步累积的碰撞力 (N)，shape=(3,)"""
        return self.ti_force.to_numpy()

    def clear_force(self):
        """清零碰撞力累加器。"""
        self.ti_force[None] = [0.0, 0.0, 0.0]

    # ── 几何查询 ─────────────────────────────────────────────

    def get_current_points(self) -> np.ndarray:
        """返回当前世界坐标点云，shape=(N,3)。"""
        return self._raw_pts + self.position

    def get_bbox(self):
        """返回包围盒 (min_xyz, max_xyz)，各 shape=(3,)。"""
        pts = self.get_current_points()
        return pts.min(axis=0), pts.max(axis=0)
