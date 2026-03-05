"""Main MPM simulation class integrating all subsystems.

This module provides the unified simulation driver interface that:
  - Creates Taichi fields for particles and grid
  - Assembles kernels from constitutive, MPM core, and magnetic modules
  - Advances simulation frame-by-frame
  - Computes magnetic field changes
  - Renders GUI visualization
"""

import taichi as ti
import numpy as np
import asyncio

from config_loader import SimConfig
from constitutive import build_yeoh_piola
from mpm_core import build_mpm_kernels
from dmm import build_magnetic_kernels
from pointcloud import PointCloudCollider
from controllers import MotionController, build_controller
from websocket import MagneticDataServer


class MPMSimulator:
    """Main MPM simulation class integrating all subsystems.

    Args:
        pc_file: Path to point cloud file for collider
        cfg: Simulation configuration object
        controller: Motion controller (None = auto-build from cfg)
        ws_server: WebSocket server for magnetic data streaming (optional)

    Attributes:
        t: Current simulation time in seconds
        frame: Current frame number
        x, v, C, F: Particle fields (position, velocity, affine, deformation gradient)
        grid_v, grid_m: Grid fields (velocity, mass)
        B_sensor: Magnetic field sensor reading
        collider: Point cloud collision body
        controller: Motion controller for collider
        B_baseline: Baseline magnetic field readings at sensor positions
    """

    def __init__(self, pc_file: str, cfg: SimConfig,
                 controller: MotionController = None,
                 ws_server: MagneticDataServer = None):
        self.cfg = cfg
        self.t = 0.0
        self.frame = 0

        DIM = cfg.DIM
        n_grid = cfg.n_grid
        NP = cfg.n_particles

        self.x = ti.Vector.field(DIM, dtype=ti.f32, shape=NP)
        self.v = ti.Vector.field(DIM, dtype=ti.f32, shape=NP)
        self.C = ti.Matrix.field(DIM, DIM, dtype=ti.f32, shape=NP)
        self.F = ti.Matrix.field(DIM, DIM, dtype=ti.f32, shape=NP)
        self.grid_v = ti.Vector.field(DIM, dtype=ti.f32,
                                      shape=(n_grid, n_grid, n_grid))
        self.grid_m = ti.field(dtype=ti.f32,
                               shape=(n_grid, n_grid, n_grid))
        self.B_sensor = ti.Vector.field(DIM, dtype=ti.f64, shape=())

        yeoh_piola = build_yeoh_piola(cfg.C1, cfg.C2, cfg.C3, cfg.kappa)

        self._init_particles, self._substep = build_mpm_kernels(
            cfg, self.x, self.v, self.C, self.F,
            self.grid_v, self.grid_m, yeoh_piola
        )

        self._clear_B, self._compute_B = build_magnetic_kernels(
            cfg, self.x, self.B_sensor
        )

        self.collider = PointCloudCollider(
            filepath=pc_file,
            init_center=cfg.collider_init_center,
            cfg=cfg,
        )

        self.controller = controller or build_controller(cfg)

        self._init_particles()

        self.B_baseline = []
        for sensor_pos in cfg.SENSOR_POS:
            self._clear_B()
            self._compute_B(*sensor_pos)
            self.B_baseline.append(self.B_sensor.to_numpy().copy())

        if cfg.use_gui:
            self._init_gui()
        if cfg.websocket_enable:
            self.ws_server = ws_server

    def _init_gui(self):
        """Initialize Taichi GUI window and camera."""
        self.window = ti.ui.Window("TacMagPie Inner GUI", (900, 700))
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0.075, 0.12, 0.28)
        self.camera.lookat(0.075, 0.03, 0.075)
        self.camera.up(0, 1, 0)
        self.vis_pts = ti.Vector.field(
            self.cfg.DIM, dtype=ti.f32, shape=self.collider.n_pts
        )

    def _update_vis_pts(self):
        """Update visualization point cloud from collider."""
        pts = self.collider.get_current_points().astype(np.float32)
        self.vis_pts.from_numpy(pts)

    def _render(self):
        """Render current simulation state to GUI."""
        self._update_vis_pts()
        self.camera.track_user_inputs(
            self.window, movement_speed=0.003, hold_key=ti.ui.LMB
        )
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.4, 0.4, 0.4))
        self.scene.point_light(pos=(0.1, 0.3, 0.1), color=(1, 1, 1))
        dx = self.cfg.dx
        self.scene.particles(self.x, radius=dx * 0.30, color=(0.2, 0.8, 0.6))
        self.scene.particles(self.vis_pts, radius=self.cfg.COLLIDER_VIS_RADIUS,color=(0.9, 0.3, 0.2))
        self.canvas.scene(self.scene)
        self.window.show()

    def step_frame(self):
        """Advance simulation by one frame (cfg.substeps physics timesteps)."""
        col = self.collider
        cfg = self.cfg

        for _ in range(cfg.substeps):
            new_pos = self.controller.get_position(self.t, col)
            if new_pos is not None:
                col.set_position(new_pos)
                col.sync_to_taichi()
            else:
                vel = self.controller.get_velocity(self.t, col)
                col.set_velocity(vel)
                col.step(cfg.dt)

            col.clear_force()
            self._substep(
                col.ti_pts, col.ti_n, col.ti_radius,
                col.ti_vel, col.ti_force
            )
            self.t += cfg.dt

        self.frame += 1

    def get_B_delta(self) -> list:
        """Compute magnetic field changes at all sensor positions.

        Returns:
            List of magnetic field change vectors (current - baseline) for each sensor
        """
        B_deltas = []
        for i, sensor_pos in enumerate(self.cfg.SENSOR_POS):
            self._clear_B()
            self._compute_B(*sensor_pos)
            B_deltas.append(self.B_sensor.to_numpy() - self.B_baseline[i])
        return B_deltas

    def run(self):
        """Run complete simulation according to cfg parameters."""
        if self.ws_server:
            asyncio.run(self._run_with_ws())
        else:
            self._run_sync()

    async def _run_with_ws(self):
        """Run simulation asynchronously with WebSocket broadcasting."""
        cfg = self.cfg

        for _ in range(cfg.total_frames):
            self.step_frame()
            B_list = self.get_B_delta()
            self.ws_server.update_data(B_list)
            await self.ws_server.broadcast()

            if self.frame % cfg.print_interval == 0:
                f = self.collider.get_force()
                print(f"Frame {self.frame} | t={self.t:.4f}s | F={f} N")
                if cfg.use_gui:
                    self._render()
            await asyncio.sleep(0)

    def _run_sync(self):
        """Run simulation synchronously (original logic)."""
        for _ in range(self.cfg.total_frames):
            self.step_frame()
            if self.frame % self.cfg.print_interval == 0:
                f = self.collider.get_force()
                B_list = self.get_B_delta()
                print(f"Frame {self.frame} | t={self.t:.4f}s | F={f} N | B={B_list}")
            if self.cfg.use_gui:
                self._render()
