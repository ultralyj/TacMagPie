# simulator.py
"""
主仿真类 MPMSimulator

整合所有子模块，提供统一的仿真驱动接口：
  - 创建 Taichi fields
  - 装配 kernels
  - 逐帧推进（step_frame / run）
  - 磁场读取（get_B_delta）
  - GUI 渲染
"""

import taichi as ti
import numpy as np

from config_loader  import SimConfig
from constitutive   import build_yeoh_piola
from mpm_core       import build_mpm_kernels
from magnetic       import build_magnetic_kernels
from pointcloud     import PointCloudCollider
from controllers    import MotionController, build_controller


class MPMSimulator:
    """
    MPM 仿真主类。

    参数
    ----
    pc_file    : 点云文件路径
    cfg        : SimConfig 实例
    controller : 运动控制器（None → 从 cfg 自动构建）
    """

    def __init__(self, pc_file: str, cfg: SimConfig,
                 controller: MotionController = None):
        self.cfg   = cfg
        self.t     = 0.0
        self.frame = 0

        DIM    = cfg.DIM
        n_grid = cfg.n_grid
        NP     = cfg.n_particles

        # ── Taichi fields ──────────────────────────────────
        self.x        = ti.Vector.field(DIM, dtype=ti.f32, shape=NP)
        self.v        = ti.Vector.field(DIM, dtype=ti.f32, shape=NP)
        self.C        = ti.Matrix.field(DIM, DIM, dtype=ti.f32, shape=NP)
        self.F        = ti.Matrix.field(DIM, DIM, dtype=ti.f32, shape=NP)
        self.grid_v   = ti.Vector.field(DIM, dtype=ti.f32,
                                        shape=(n_grid, n_grid, n_grid))
        self.grid_m   = ti.field(dtype=ti.f32,
                                  shape=(n_grid, n_grid, n_grid))
        self.B_sensor = ti.Vector.field(DIM, dtype=ti.f64, shape=())

        # ── 装配本构函数 ────────────────────────────────────
        yeoh_piola = build_yeoh_piola(cfg.C1, cfg.C2, cfg.C3, cfg.kappa)

        # ── 装配 MPM kernels ────────────────────────────────
        self._init_particles, self._substep = build_mpm_kernels(
            cfg, self.x, self.v, self.C, self.F,
            self.grid_v, self.grid_m, yeoh_piola
        )

        # ── 装配磁场 kernels ────────────────────────────────
        self._clear_B, self._compute_B = build_magnetic_kernels(
            cfg, self.x, self.B_sensor
        )

        # ── 点云碰撞器 ──────────────────────────────────────
        self.collider = PointCloudCollider(
            filepath    = pc_file,
            init_center = cfg.collider_init_center,
            cfg         = cfg,
        )

        # ── 运动控制器 ──────────────────────────────────────
        self.controller = controller or build_controller(cfg)

        # ── 初始化粒子 ──────────────────────────────────────
        self._init_particles()

        # ── 磁场基准（未变形状态）──────────────────────────
        self._clear_B()
        self._compute_B(*cfg.SENSOR_POS)
        self.B_baseline = self.B_sensor.to_numpy().copy()

        # ── GUI ─────────────────────────────────────────────
        if cfg.use_gui:
            self._init_gui()

    # ════════════════════════════════════════════════════════
    #  GUI
    # ════════════════════════════════════════════════════════

    def _init_gui(self):
        self.window = ti.ui.Window("MPM 点云按压仿真", (900, 700))
        self.canvas = self.window.get_canvas()
        self.scene  = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0.075, 0.12, 0.28)
        self.camera.lookat(0.075, 0.03, 0.075)
        self.camera.up(0, 1, 0)
        # 独立的点云可视化 field（避免修改碰撞器内部 field）
        self.vis_pts = ti.Vector.field(
            self.cfg.DIM, dtype=ti.f32, shape=self.collider.n_pts
        )

    def _update_vis_pts(self):
        pts = self.collider.get_current_points().astype(np.float32)
        self.vis_pts.from_numpy(pts)

    def _render(self):
        self._update_vis_pts()
        self.camera.track_user_inputs(
            self.window, movement_speed=0.003, hold_key=ti.ui.LMB
        )
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.4, 0.4, 0.4))
        self.scene.point_light(pos=(0.1, 0.3, 0.1), color=(1, 1, 1))
        dx = self.cfg.dx
        self.scene.particles(self.x,       radius=dx * 0.30, color=(0.2, 0.8, 0.6))
        self.scene.particles(self.vis_pts, radius=self.cfg.COLLIDER_VIS_RADIUS, color=(0.9, 0.3, 0.2))
        self.canvas.scene(self.scene)
        self.window.show()

    # ════════════════════════════════════════════════════════
    #  推进接口
    # ════════════════════════════════════════════════════════

    def step_frame(self):
        """推进一帧（cfg.substeps 个物理时间步）。"""
        col = self.collider
        cfg = self.cfg

        for _ in range(cfg.substeps):
            # 询问控制器
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

    # ════════════════════════════════════════════════════════
    #  磁场接口
    # ════════════════════════════════════════════════════════

    def get_B_delta(self) -> np.ndarray:
        """返回相对初始状态的传感器磁场变化量 (T)，shape=(3,)。"""
        self._clear_B()
        self._compute_B(*self.cfg.SENSOR_POS)
        return self.B_sensor.to_numpy() - self.B_baseline

    # ════════════════════════════════════════════════════════
    #  完整运行
    # ════════════════════════════════════════════════════════

    def run(self):
        """按照 cfg 中的参数运行完整仿真。"""
        cfg   = self.cfg
        col   = self.collider
        pint  = cfg.print_interval
        total = cfg.total_frames

        def _log():
            if self.frame % pint == 0:
                B = self.get_B_delta()
                f = col.get_force()
                print(
                    f"Frame {self.frame:4d} | t={self.t:.4f}s | "
                    f"pos={col.position} | "
                    f"F={f} N | "
                    f"ΔB=[{B[0]:.3e},{B[1]:.3e},{B[2]:.3e}] T"
                )

        if cfg.use_gui:
            while self.window.running and self.frame < total:
                self.step_frame()
                _log()
                self._render()
        else:
            for _ in range(total):
                self.step_frame()
                _log()
