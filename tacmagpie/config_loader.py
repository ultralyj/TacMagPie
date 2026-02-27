# config_loader.py
"""
配置加载模块
负责读取 YAML 文件并派生所有仿真参数。
"""

import yaml
import numpy as np


def load_config(path: str = "./config/default.yaml") -> dict:
    """加载并返回 YAML 配置字典"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SimConfig:
    """
    从配置字典派生所有仿真参数（含二次计算量）。
    使用时通过 cfg.xxx 访问。
    """

    def __init__(self, cfg: dict):
        # ── 物理 ──────────────────────────────────────────
        phy = cfg["physics"]
        self.DIM      = int(phy["dim"])
        self.E        = float(phy["E"])
        self.nu       = float(phy["nu"])
        self.rho      = float(phy["rho"])
        self.mu_0     = self.E / (2 * (1 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        # ── 硅胶板 ────────────────────────────────────────
        slab = cfg["slab"]
        self.slab_x = float(slab["x"])
        self.slab_y = float(slab["y"])
        self.slab_z = float(slab["z"])

        # ── 网格 ──────────────────────────────────────────
        grid = cfg["grid"]
        self.domain = float(grid["domain"])
        self.n_grid = int(grid["n_grid"])
        self.dt     = float(grid["dt"])
        self.psf    = float(grid.get("particle_spacing_factor", 0.5))

        self.dx     = self.domain / self.n_grid
        self.inv_dx = 1.0 / self.dx

        # ── 粒子 ──────────────────────────────────────────
        spacing            = self.dx * self.psf
        self.n_particles_x = max(1, int(self.slab_x / spacing))
        self.n_particles_y = max(1, int(self.slab_y / spacing))
        self.n_particles_z = max(1, int(self.slab_z / spacing))
        self.n_particles   = (self.n_particles_x
                              * self.n_particles_y
                              * self.n_particles_z)
        self.p_vol  = (self.slab_x * self.slab_y * self.slab_z) / self.n_particles
        self.p_mass = self.rho * self.p_vol

        # ── 本构 ──────────────────────────────────────────
        con = cfg["constitutive"]
        self.C1    = float(con["C1"])
        self.C2    = float(con["C2"])
        self.C3    = float(con["C3"])
        self.kappa = float(con["kappa"])

        # ── 磁场 ──────────────────────────────────────────
        mag = cfg["magnetic"]
        self.MU_0_4PI = float(mag["mu_0_4pi"])
        self.B0       = float(mag["B0"])
        sp = mag["sensor_pos"]
        sx = sp.get("x") or self.domain * 0.5
        sy = float(sp["y"])
        sz = sp.get("z") or self.domain * 0.5
        self.SENSOR_POS = (float(sx), float(sy), float(sz))

        # ── 碰撞器 ────────────────────────────────────────
        col = cfg["collider"]
        self.MAX_COLLIDER_PTS = int(col["max_pts"])
        self.COLLIDER_RADIUS  = self.dx * float(col["radius_factor"])
        self.COLLIDER_VIS_RADIUS = self.dx * float(col.get("vis_radius_factor", 0.15))
        self._collider_offset_y = col.get("init_offset_y")
        
        # ── 控制器 ────────────────────────────────────────
        self.controller_cfg = cfg["controller"]

        # ── 仿真运行 ──────────────────────────────────────
        sim = cfg["simulation"]
        self.total_frames   = int(sim["total_frames"])
        self.substeps       = int(sim["substeps"])
        self.print_interval = int(sim["print_interval"])
        self.use_gui        = bool(sim["use_gui"])

        # ── 点云 ──────────────────────────────────────────
        self.pointcloud_cfg = cfg["pointcloud"]

        # ── 衍生几何 ──────────────────────────────────────
        self.slab_top_y = 2 * self.dx + self.slab_y

        offset = (float(self._collider_offset_y)
                  if self._collider_offset_y is not None
                  else self.COLLIDER_RADIUS + 0.001)

        self.collider_init_center = np.array([
            self.domain * 0.5,
            self.slab_top_y + offset,
            self.domain * 0.5,
        ], dtype=np.float32)

    def __repr__(self):
        return (f"SimConfig(n_particles={self.n_particles}, "
                f"n_grid={self.n_grid}, dt={self.dt}, "
                f"collider_radius={self.COLLIDER_RADIUS:.4f})")