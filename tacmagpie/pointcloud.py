"""Point cloud collider module for MPM simulation.

This module provides collision handling using point cloud representations.
Supports loading from various file formats and synchronization with Taichi fields.

Classes
-------
PointCloudCollider
    Manages point cloud collision bodies with external control and force feedback.
"""

import os
import taichi as ti
import open3d as o3d
import numpy as np
from config_loader import SimConfig


class PointCloudCollider:
    """Point cloud collision body for MPM simulation.

    Loads point cloud geometry from file, manages position/velocity state,
    and synchronizes with Taichi fields for MPM kernel collision detection.

    Args:
        filepath: Path to point cloud file (.npy / .txt / .xyz / .ply)
        init_center: Initial world coordinates of point cloud bounding box center, shape=(3,)
        cfg: SimConfig instance providing DIM, MAX_COLLIDER_PTS, COLLIDER_RADIUS

    Attributes:
        position: Current world position of collider center, shape=(3,)
        velocity: Current velocity in m/s, shape=(3,)
        ti_pts: Taichi field containing world-space point positions
        ti_n: Taichi field containing number of points
        ti_radius: Taichi field containing collision radius
        ti_vel: Taichi field containing collider velocity
        ti_force: Taichi field accumulating collision forces
    """

    def __init__(self, filepath: str, init_center: np.ndarray, cfg: SimConfig):
        self.cfg = cfg
        self.point_radius = cfg.COLLIDER_RADIUS

        self._raw_pts = self._load(filepath)
        n = len(self._raw_pts)
        assert n <= cfg.MAX_COLLIDER_PTS, (
            f"Point count {n} exceeds limit {cfg.MAX_COLLIDER_PTS}. "
            f"Increase collider.max_pts in config.yaml"
        )
        self.n_pts = n

        local_center = self._raw_pts.mean(axis=0)
        self._raw_pts -= local_center

        self.position = np.array(init_center, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)

        DIM = cfg.DIM
        MAX = cfg.MAX_COLLIDER_PTS
        self.ti_pts = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX)
        self.ti_n = ti.field(dtype=ti.i32, shape=())
        self.ti_radius = ti.field(dtype=ti.f32, shape=())
        self.ti_vel = ti.Vector.field(DIM, dtype=ti.f32, shape=())
        self.ti_force = ti.Vector.field(DIM, dtype=ti.f32, shape=())

        self.ti_n[None] = self.n_pts
        self.ti_radius[None] = self.point_radius

        self.sync_to_taichi()

    @staticmethod
    def _load(filepath: str) -> np.ndarray:
        """Load point cloud from file.

        Args:
            filepath: Path to point cloud file

        Returns:
            Point cloud array of shape (N, 3) with dtype float32

        Raises:
            ValueError: If file format is unsupported
            AssertionError: If point cloud shape is invalid
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".npy":
            pts = np.load(filepath).astype(np.float32)
        elif ext in (".txt", ".xyz"):
            pts = np.loadtxt(filepath, dtype=np.float32)
        elif ext == ".ply":
            pts = PointCloudCollider._load_ply(filepath)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. Use .npy / .txt / .xyz / .ply"
            )
        assert pts.ndim == 2 and pts.shape[1] >= 3, \
            "Point cloud must be (N, 3+) array"
        return pts[:, :3].astype(np.float32)

    @staticmethod
    def _load_ply(filepath: str) -> np.ndarray:
        """Load PLY file using Open3D (supports ASCII and binary formats).

        Args:
            filepath: Path to PLY file

        Returns:
            Point cloud array of shape (N, 3) with dtype float32
        """
        pcd = o3d.io.read_point_cloud(filepath)
        return np.asarray(pcd.points, dtype=np.float32)

    def set_velocity(self, vel: np.ndarray):
        """Set collider velocity.

        Args:
            vel: Velocity vector in m/s, shape=(3,)
        """
        self.velocity = np.array(vel, dtype=np.float32)

    def set_position(self, pos: np.ndarray):
        """Set collider center position directly.

        Args:
            pos: World coordinates of collider center, shape=(3,)
        """
        self.position = np.array(pos, dtype=np.float32)

    def step(self, dt_val: float):
        """Update position by integrating velocity and sync to Taichi.

        Args:
            dt_val: Time step in seconds
        """
        self.position += self.velocity * dt_val
        self.sync_to_taichi()

    def sync_to_taichi(self):
        """Synchronize current world-space point positions to Taichi fields."""
        MAX = self.cfg.MAX_COLLIDER_PTS
        DIM = self.cfg.DIM
        world_pts = self._raw_pts + self.position
        tmp = np.zeros((MAX, DIM), dtype=np.float32)
        tmp[: self.n_pts] = world_pts
        self.ti_pts.from_numpy(tmp)
        self.ti_vel[None] = ti.Vector(self.velocity.tolist())

    def get_force(self) -> np.ndarray:
        """Get accumulated collision force from last timestep.

        Returns:
            Force vector in Newtons, shape=(3,)
        """
        return self.ti_force.to_numpy()

    def clear_force(self):
        """Reset collision force accumulator to zero."""
        self.ti_force[None] = [0.0, 0.0, 0.0]

    def get_current_points(self) -> np.ndarray:
        """Get current world-space point cloud.

        Returns:
            Point cloud array, shape=(N, 3)
        """
        return self._raw_pts + self.position

    def get_bbox(self):
        """Get axis-aligned bounding box of current point cloud.

        Returns:
            tuple: (min_xyz, max_xyz), each with shape=(3,)
        """
        pts = self.get_current_points()
        return pts.min(axis=0), pts.max(axis=0)
