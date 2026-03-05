"""Configuration loader for MPM simulation.

This module handles loading YAML configuration files and deriving all simulation
parameters including physics properties, grid settings, particle counts, and
magnetic field configurations.
"""

import yaml
import numpy as np


def load_config(path: str = "./config/default.yaml") -> dict:
    """Load YAML configuration file.
    
    Args:
        path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SimConfig:
    """Simulation configuration with derived parameters.
    
    Parses configuration dictionary and computes derived quantities such as
    Lamé parameters, particle counts, and geometric properties. All parameters
    are accessible via cfg.xxx notation.
    
    Args:
        cfg: Configuration dictionary loaded from YAML.
    """

    def __init__(self, cfg: dict):
        phy = cfg["physics"]
        self.DIM = int(phy["dim"])
        self.E = float(phy["E"])
        self.nu = float(phy["nu"])
        self.rho = float(phy["rho"])
        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        slab = cfg["slab"]
        self.slab_x = float(slab["x"])
        self.slab_y = float(slab["y"])
        self.slab_z = float(slab["z"])

        grid = cfg["grid"]
        self.domain = float(grid["domain"])
        self.n_grid = int(grid["n_grid"])
        self.dt = float(grid["dt"])
        self.psf = float(grid.get("particle_spacing_factor", 0.5))
        self.dx = self.domain / self.n_grid
        self.inv_dx = 1.0 / self.dx

        spacing = self.dx * self.psf
        self.n_particles_x = max(1, int(self.slab_x / spacing))
        self.n_particles_y = max(1, int(self.slab_y / spacing))
        self.n_particles_z = max(1, int(self.slab_z / spacing))
        self.n_particles = (self.n_particles_x * 
                           self.n_particles_y * 
                           self.n_particles_z)
        self.p_vol = (self.slab_x * self.slab_y * self.slab_z) / self.n_particles
        self.p_mass = self.rho * self.p_vol

        con = cfg["constitutive"]
        self.C1 = float(con["C1"])
        self.C2 = float(con["C2"])
        self.C3 = float(con["C3"])
        self.kappa = float(con["kappa"])

        mag = cfg["magnetic"]
        self.MU_0_4PI = float(mag["mu_0_4pi"])
        self.B0 = float(mag["B0"])
        self.SENSOR_POS = []
        for sp in mag["sensors"]:
            sx = (sp.get("x") + self.domain * 0.5 
                  if sp.get("x") is not None 
                  else self.domain * 0.5)
            sy = float(sp["y"])
            sz = (sp.get("z") + self.domain * 0.5 
                  if sp.get("z") is not None 
                  else self.domain * 0.5)
            self.SENSOR_POS.append((float(sx), float(sy), float(sz)))

        col = cfg["collider"]
        self.MAX_COLLIDER_PTS = int(col["max_pts"])
        self.COLLIDER_RADIUS = self.dx * float(col["radius_factor"])
        self.COLLIDER_VIS_RADIUS = self.dx * float(col.get("vis_radius_factor", 0.15))
        self._collider_offset_y = col.get("init_offset_y")
        
        self.controller_cfg = cfg["controller"]

        sim = cfg["simulation"]
        self.total_frames = int(sim["total_frames"])
        self.substeps = int(sim["substeps"])
        self.print_interval = int(sim["print_interval"])
        self.use_gui = bool(sim["use_gui"])

        self.pointcloud_cfg = cfg["pointcloud"]

        self.slab_top_y = 2 * self.dx + self.slab_y
        offset = (float(self._collider_offset_y) 
                  if self._collider_offset_y is not None 
                  else self.COLLIDER_RADIUS + 0.001)
        self.collider_init_center = np.array([
            self.domain * 0.5,
            self.slab_top_y + offset,
            self.domain * 0.5,
        ], dtype=np.float32)

        ws = cfg["websocket"]
        self.websocket_enable = bool(ws.get("enable", False))
        self.websocket_host = ws.get("host", "localhost")
        self.websocket_port = int(ws.get("port", 8765))

    def __repr__(self):
        return (f"SimConfig(n_particles={self.n_particles}, "
                f"n_grid={self.n_grid}, dt={self.dt}, "
                f"collider_radius={self.COLLIDER_RADIUS:.4f})")
