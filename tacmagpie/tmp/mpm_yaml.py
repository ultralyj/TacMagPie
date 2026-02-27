# mpm_sim_pointcloud.py
"""
MPM 点云按压仿真
所有参数通过 config.yaml 配置，使用方法：
    python mpm_sim_pointcloud.py                      # 使用默认 config.yaml
    python mpm_sim_pointcloud.py my_config.yaml       # 指定配置文件
    python mpm_sim_pointcloud.py config.yaml ball.npy # 指定配置 + 点云文件
"""

import sys
import os
import math
import taichi as ti
import numpy as np
import yaml
import time

# ════════════════════════════════════════════════════════════
#  配置加载
# ════════════════════════════════════════════════════════════

def load_config(path: str = "config.yaml") -> dict:
    """加载并返回 YAML 配置字典"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


class SimConfig:
    """
    从配置字典派生所有仿真参数（含二次计算量）。
    使用时通过 cfg.xxx 访问，与原全局变量同名。
    """
    def __init__(self, cfg: dict):
        # ── 物理 ──────────────────────────────────────────
        phy = cfg["physics"]
        self.DIM   = int(phy["dim"])
        self.E     = float(phy["E"])
        self.nu    = float(phy["nu"])
        self.rho   = float(phy["rho"])
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
        self.psf    = float(grid.get("particle_spacing_factor", 0.5))  # 粒子间距因子

        self.dx     = self.domain / self.n_grid
        self.inv_dx = 1.0 / self.dx

        # ── 粒子 ──────────────────────────────────────────
        spacing = self.dx * self.psf
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
        self._collider_offset_y = col.get("init_offset_y")  # None or float

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

        if self._collider_offset_y is not None:
            offset = float(self._collider_offset_y)
        else:
            offset = self.COLLIDER_RADIUS + 0.001

        self.collider_init_center = np.array([
            self.domain * 0.5,
            self.slab_top_y + offset,
            self.domain * 0.5,
        ], dtype=np.float32)

    def __repr__(self):
        return (f"SimConfig(n_particles={self.n_particles}, "
                f"n_grid={self.n_grid}, dt={self.dt}, "
                f"collider_radius={self.COLLIDER_RADIUS:.4f})")


# ════════════════════════════════════════════════════════════
#  点云碰撞器类  PointCloudCollider
# ════════════════════════════════════════════════════════════

class PointCloudCollider:
    """管理点云碰撞体：加载文件、外部控制、同步到 Taichi field。"""

    def __init__(self, filepath: str, init_center: np.ndarray,
                 cfg: SimConfig):
        self.cfg          = cfg
        self.point_radius = cfg.COLLIDER_RADIUS
        self._raw_pts     = self._load(filepath)          # (N,3) 局部坐标
        n = len(self._raw_pts)
        assert n <= cfg.MAX_COLLIDER_PTS, \
            f"点云点数 {n} 超过上限 {cfg.MAX_COLLIDER_PTS}，请调大 collider.max_pts"

        self.n_pts = n

        # 将点云中心移到原点（局部坐标）
        local_center  = self._raw_pts.mean(axis=0)
        self._raw_pts -= local_center

        self.position = np.array(init_center, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)

        # ── Taichi fields ──
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
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".npy":
            pts = np.load(filepath).astype(np.float32)
        elif ext in (".txt", ".xyz"):
            pts = np.loadtxt(filepath, dtype=np.float32)
        elif ext == ".ply":
            pts = PointCloudCollider._load_ply(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {ext}，请使用 .npy / .txt / .ply")
        assert pts.ndim == 2 and pts.shape[1] >= 3, "点云必须是 (N,3+) 数组"
        return pts[:, :3].astype(np.float32)

    @staticmethod
    def _load_ply(filepath: str) -> np.ndarray:
        pts, in_data = [], False
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line == "end_header":
                    in_data = True
                    continue
                if in_data:
                    vals = line.split()
                    if len(vals) >= 3:
                        pts.append([float(vals[0]), float(vals[1]), float(vals[2])])
        return np.array(pts, dtype=np.float32)

    # ── 外部控制接口 ─────────────────────────────────────────
    def set_velocity(self, vel: np.ndarray):
        self.velocity = np.array(vel, dtype=np.float32)

    def set_position(self, pos: np.ndarray):
        self.position = np.array(pos, dtype=np.float32)

    def step(self, dt_val: float):
        self.position += self.velocity * dt_val
        self.sync_to_taichi()

    def sync_to_taichi(self):
        MAX = self.cfg.MAX_COLLIDER_PTS
        DIM = self.cfg.DIM
        world_pts = self._raw_pts + self.position
        tmp = np.zeros((MAX, DIM), dtype=np.float32)
        tmp[: self.n_pts] = world_pts
        self.ti_pts.from_numpy(tmp)
        self.ti_vel[None] = ti.Vector(self.velocity.tolist())

    def get_force(self) -> np.ndarray:
        return self.ti_force.to_numpy()

    def clear_force(self):
        self.ti_force[None] = [0.0, 0.0, 0.0]

    def get_current_points(self) -> np.ndarray:
        return self._raw_pts + self.position

    def get_bbox(self):
        pts = self.get_current_points()
        return pts.min(axis=0), pts.max(axis=0)


# ════════════════════════════════════════════════════════════
#  Taichi 核函数工厂  —  根据 SimConfig 动态构建
# ════════════════════════════════════════════════════════════

def build_kernels(cfg: SimConfig,
                  x, v, C, F,
                  grid_v, grid_m,
                  B_sensor):
    """
    返回一组已绑定参数的 Taichi 核函数：
      init_particles, substep, clear_B_sensor, compute_sensor_field
    """

    # ── 从 cfg 提取标量（避免在 kernel 中引用 Python 对象）──
    DIM         = cfg.DIM
    n_particles = cfg.n_particles
    npx         = cfg.n_particles_x
    npy         = cfg.n_particles_y
    npz         = cfg.n_particles_z
    slab_x      = cfg.slab_x
    slab_y      = cfg.slab_y
    slab_z      = cfg.slab_z
    domain      = cfg.domain
    dx          = cfg.dx
    inv_dx      = cfg.inv_dx
    dt          = cfg.dt
    p_mass      = cfg.p_mass
    p_vol       = cfg.p_vol
    n_grid      = cfg.n_grid
    C1          = cfg.C1
    C2          = cfg.C2
    C3          = cfg.C3
    kappa       = cfg.kappa
    MU_0_4PI    = cfg.MU_0_4PI
    B0          = cfg.B0

    # ── 本构函数（内联在 substep 中，这里单独定义供复用）──
    @ti.func
    def yeoh_piola(F_mat: ti.template()):
        J = F_mat.determinant()
        if J < 0.1:
            F_mat = ti.pow(0.1 / J, 1.0 / 3.0) * F_mat
            J = 0.1
        F_inv_T = F_mat.inverse().transpose()
        I1      = (F_mat.transpose() @ F_mat).trace()
        J23     = ti.pow(J, -2.0 / 3.0)
        I1b     = J23 * I1
        i1bm3   = I1b - 3.0
        dWdI1b  = C1 + 2.0 * C2 * i1bm3 + 3.0 * C3 * i1bm3 * i1bm3
        P_iso   = 2.0 * dWdI1b * J23 * (F_mat - (I1 / 3.0) * F_inv_T)
        P_vol   = kappa * (J - 1.0) * J * F_inv_T
        return P_iso + P_vol

    # ── init_particles ────────────────────────────────────
    @ti.kernel
    def init_particles():
        ox = (domain - slab_x) * 0.5
        oy = 2 * dx
        oz = (domain - slab_z) * 0.5
        for i in range(n_particles):
            ix = i % npx
            iy = (i // npx) % npy
            iz = i // (npx * npy)
            x[i] = ti.Vector([
                ox + (ix + 0.5) * slab_x / npx,
                oy + (iy + 0.5) * slab_y / npy,
                oz + (iz + 0.5) * slab_z / npz,
            ])
            v[i] = ti.Vector([0.0, 0.0, 0.0])
            F[i] = ti.Matrix.identity(ti.f32, DIM)
            C[i] = ti.Matrix.zero(ti.f32, DIM, DIM)

    # ── substep ───────────────────────────────────────────
    @ti.kernel
    def substep(
        col_pts:    ti.template(),
        col_n:      ti.template(),
        col_radius: ti.template(),
        col_vel:    ti.template(),
        col_force:  ti.template(),
    ):
        # 1. 清空网格
        for i, j, k in grid_m:
            grid_v[i, j, k] = ti.Vector.zero(ti.f32, DIM)
            grid_m[i, j, k] = 0.0

        # 2. P2G
        for p in range(n_particles):
            Xp   = x[p] * inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx   = Xp - ti.cast(base, ti.f32)
            w = [0.5 * (1.5 - fx) ** 2,
                 0.75 - (fx - 1.0) ** 2,
                 0.5 * (fx - 0.5) ** 2]
            stress = ((-dt * p_vol * 4 * inv_dx * inv_dx)
                      * yeoh_piola(F[p]) @ F[p].transpose())
            affine = stress + p_mass * C[p]
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos   = (ti.cast(offset, ti.f32) - fx) * dx
                weight = w[i][0] * w[j][1] * w[k][2]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

        # 3. 网格更新 + 点云碰撞
        radius = col_radius[None]
        n_col  = col_n[None]
        c_vel  = col_vel[None]

        for i, j, k in grid_m:
            if grid_m[i, j, k] > 0:
                grid_v[i, j, k] /= grid_m[i, j, k]
                gpos = ti.Vector([i, j, k], dt=ti.f32) * dx

                min_dist  = radius * 10.0
                closest_n = ti.Vector.zero(ti.f32, DIM)

                for q in range(n_col):
                    diff = gpos - col_pts[q]
                    dist = diff.norm()
                    if dist < min_dist:
                        min_dist = dist
                        if dist > 1e-6:
                            closest_n = diff / dist
                        else:
                            closest_n = ti.Vector([0.0, 1.0, 0.0])

                if min_dist < radius:
                    rel_v = grid_v[i, j, k] - c_vel
                    vn    = rel_v.dot(closest_n)
                    if vn < 0:
                        delta_v = -vn * closest_n
                        grid_v[i, j, k] += delta_v
                        f_contrib = -grid_m[i, j, k] * delta_v / dt
                        ti.atomic_add(col_force[None][0], f_contrib[0])
                        ti.atomic_add(col_force[None][1], f_contrib[1])
                        ti.atomic_add(col_force[None][2], f_contrib[2])

                # 边界条件
                if j < 3:
                    grid_v[i, j, k] = ti.Vector.zero(ti.f32, DIM)
                if j > n_grid - 3 and grid_v[i, j, k][1] > 0:
                    grid_v[i, j, k][1] = 0.0
                if i < 3 and grid_v[i, j, k][0] < 0:
                    grid_v[i, j, k][0] = 0.0
                if i > n_grid - 3 and grid_v[i, j, k][0] > 0:
                    grid_v[i, j, k][0] = 0.0
                if k < 3 and grid_v[i, j, k][2] < 0:
                    grid_v[i, j, k][2] = 0.0
                if k > n_grid - 3 and grid_v[i, j, k][2] > 0:
                    grid_v[i, j, k][2] = 0.0

        # 4. G2P
        for p in range(n_particles):
            Xp   = x[p] * inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx   = Xp - ti.cast(base, ti.f32)
            w = [0.5 * (1.5 - fx) ** 2,
                 0.75 - (fx - 1.0) ** 2,
                 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(ti.f32, DIM)
            new_C = ti.Matrix.zero(ti.f32, DIM, DIM)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos   = (ti.cast(offset, ti.f32) - fx) * dx
                weight = w[i][0] * w[j][1] * w[k][2]
                g_v    = grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            v[p]  = new_v
            C[p]  = new_C
            x[p] += dt * new_v
            F[p]  = (ti.Matrix.identity(ti.f32, DIM) + dt * new_C) @ F[p]

    # ── 磁场 kernel ───────────────────────────────────────
    @ti.kernel
    def clear_B_sensor():
        B_sensor[None] = [0.0, 0.0, 0.0]

    @ti.kernel
    def compute_sensor_field(sx: ti.f32, sy: ti.f32, sz: ti.f32):
        m0 = ti.cast(B0, ti.f64)
        my = m0
        _sx = ti.cast(sx, ti.f64)
        _sy = ti.cast(sy, ti.f64)
        _sz = ti.cast(sz, ti.f64)
        for p in range(n_particles):
            rx = _sx - ti.cast(x[p][0], ti.f64)
            ry = _sy - ti.cast(x[p][1], ti.f64)
            rz = _sz - ti.cast(x[p][2], ti.f64)
            r2 = rx * rx + ry * ry + rz * rz
            if r2 < 1e-30:
                continue
            r_norm = ti.sqrt(r2)
            r3 = r_norm * r2
            r5 = r3 * r2
            m_dot_r = my * ry
            ti.atomic_add(B_sensor[None][0], MU_0_4PI * (3.0 * m_dot_r * rx / r5))
            ti.atomic_add(B_sensor[None][1], MU_0_4PI * (3.0 * m_dot_r * ry / r5 - my / r3))
            ti.atomic_add(B_sensor[None][2], MU_0_4PI * (3.0 * m_dot_r * rz / r5))

    return init_particles, substep, clear_B_sensor, compute_sensor_field


# ════════════════════════════════════════════════════════════
#  运动控制器
# ════════════════════════════════════════════════════════════

class MotionController:
    def get_velocity(self, t: float, collider) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def get_position(self, t: float, collider):
        return None


class PressAndHoldController(MotionController):
    def __init__(self, press_speed: float, press_depth: float, init_y: float):
        self.press_speed = press_speed
        self.press_depth = press_depth
        self.init_y      = init_y

    def get_velocity(self, t, collider):
        current_depth = self.init_y - collider.position[1]
        if current_depth < self.press_depth:
            return np.array([0.0, -self.press_speed, 0.0], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)


class TrajectoryController(MotionController):
    def __init__(self, times: np.ndarray, positions: np.ndarray):
        self.times     = times
        self.positions = positions.astype(np.float32)

    def get_position(self, t, collider):
        return np.array([
            np.interp(t, self.times, self.positions[:, i])
            for i in range(3)
        ], dtype=np.float32)

    def get_velocity(self, t, collider):
        return np.zeros(3, dtype=np.float32)


def build_controller(cfg: SimConfig) -> MotionController:
    """根据配置构建运动控制器"""
    ctrl_cfg  = cfg.controller_cfg
    ctrl_type = ctrl_cfg.get("type", "none")
    init_y    = float(cfg.collider_init_center[1])

    if ctrl_type == "press_and_hold":
        c = ctrl_cfg["press_and_hold"]
        return PressAndHoldController(
            press_speed = float(c["press_speed"]),
            press_depth = float(c["press_depth"]),
            init_y      = init_y,
        )
    elif ctrl_type == "trajectory":
        c      = ctrl_cfg["trajectory"]
        t_arr  = np.linspace(float(c["t_start"]), float(c["t_end"]), int(c["n_points"]))
        pos    = np.zeros((len(t_arr), 3), dtype=np.float32)
        pos[:, 0] = cfg.domain * 0.5
        pos[:, 2] = cfg.domain * 0.5
        depth     = float(c["press_depth"])
        t_press   = float(c["press_time"])
        pos[:, 1] = init_y - np.clip(t_arr / t_press, 0, 1) * depth
        return TrajectoryController(t_arr, pos)
    else:
        return MotionController()


# ════════════════════════════════════════════════════════════
#  主仿真类  MPMSimulator
# ════════════════════════════════════════════════════════════

class MPMSimulator:
    def __init__(self, pc_file: str, cfg: SimConfig,
                 controller: MotionController = None):
        self.cfg  = cfg
        self.t    = 0.0
        self.frame = 0

        DIM    = cfg.DIM
        n_grid = cfg.n_grid
        NP     = cfg.n_particles

        # ── Taichi fields ──────────────────────────────────
        self.x      = ti.Vector.field(DIM, dtype=ti.f32, shape=NP)
        self.v      = ti.Vector.field(DIM, dtype=ti.f32, shape=NP)
        self.C      = ti.Matrix.field(DIM, DIM, dtype=ti.f32, shape=NP)
        self.F      = ti.Matrix.field(DIM, DIM, dtype=ti.f32, shape=NP)
        self.grid_v = ti.Vector.field(DIM, dtype=ti.f32,
                                      shape=(n_grid, n_grid, n_grid))
        self.grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
        self.B_sensor = ti.Vector.field(DIM, dtype=ti.f64, shape=())

        # ── 编译 kernels ────────────────────────────────────
        (self._init_particles,
         self._substep,
         self._clear_B,
         self._compute_B) = build_kernels(
            cfg, self.x, self.v, self.C, self.F,
            self.grid_v, self.grid_m, self.B_sensor
        )

        # ── 碰撞器 ─────────────────────────────────────────
        self.collider = PointCloudCollider(
            filepath    = pc_file,
            init_center = cfg.collider_init_center,
            cfg         = cfg,
        )

        # ── 控制器 ─────────────────────────────────────────
        self.controller = controller or build_controller(cfg)

        # ── 初始化粒子 ──────────────────────────────────────
        self._init_particles()

        # ── 磁场基准 ────────────────────────────────────────
        self._clear_B()
        self._compute_B(*cfg.SENSOR_POS)
        self.B_baseline = self.B_sensor.to_numpy().copy()

        # ── GUI ─────────────────────────────────────────────
        if cfg.use_gui:
            self._init_gui()

    # ── GUI ─────────────────────────────────────────────────
    def _init_gui(self):
        self.window = ti.ui.Window("MPM 点云按压仿真", (900, 700))
        self.canvas = self.window.get_canvas()
        self.scene  = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0.075, 0.12, 0.28)
        self.camera.lookat(0.075, 0.03, 0.075)
        self.camera.up(0, 1, 0)
        n = self.collider.n_pts
        self.vis_pts = ti.Vector.field(self.cfg.DIM, dtype=ti.f32, shape=n)

    def _update_vis_pts(self):
        pts = self.collider.get_current_points().astype(np.float32)
        self.vis_pts.from_numpy(pts)

    def _render(self):
        self._update_vis_pts()
        self.camera.track_user_inputs(
            self.window, movement_speed=0.003, hold_key=ti.ui.LMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.4, 0.4, 0.4))
        self.scene.point_light(pos=(0.1, 0.3, 0.1), color=(1, 1, 1))
        dx = self.cfg.dx
        self.scene.particles(self.x,            radius=dx * 0.3,  color=(0.2, 0.8, 0.6))
        self.scene.particles(self.vis_pts,       radius=dx * 0.15, color=(0.9, 0.3, 0.2))
        self.canvas.scene(self.scene)
        self.window.show()

    # ── 单帧推进 ────────────────────────────────────────────
    def step_frame(self):
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
            self._substep(col.ti_pts, col.ti_n, col.ti_radius,
                          col.ti_vel, col.ti_force)
            
            self.t += cfg.dt

        self.frame += 1

    # ── 磁场读取 ────────────────────────────────────────────
    def get_B_delta(self) -> np.ndarray:
        self._clear_B()
        self._compute_B(*self.cfg.SENSOR_POS)
        return self.B_sensor.to_numpy() - self.B_baseline

    # ── 完整运行 ────────────────────────────────────────────
    def run(self):
        cfg  = self.cfg
        col  = self.collider
        pint = cfg.print_interval
        total = cfg.total_frames

        def _print():
            if self.frame % pint == 0:
                B = self.get_B_delta()
                f = col.get_force()
                print(f"Frame {self.frame:4d} | t={self.t:.4f}s | "
                      f"pos={col.position} | "
                      f"F={f} N | "
                      f"ΔB=[{B[0]:.3e},{B[1]:.3e},{B[2]:.3e}] T")

        if cfg.use_gui:
            while self.window.running and self.frame < total:
                self.step_frame()
                _print()
                self._render()
        else:
            for _ in range(total):
                self.step_frame()
                _print()


# ════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════

def generate_demo_pointcloud(path: str = "demo_ball.npy",
                             radius: float = 0.015,
                             n: int = 2000) -> str:
    pts    = []
    golden = 2.399963229728653
    for i in range(n):
        theta = math.acos(1.0 - 2.0 * (i + 0.5) / n)
        phi   = golden * i
        pts.append([radius * math.sin(theta) * math.cos(phi),
                    radius * math.cos(theta),
                    radius * math.sin(theta) * math.sin(phi)])
    np.save(path, np.array(pts, dtype=np.float32))
    print(f"已生成演示点云: {path}  ({n} 点, 半径={radius}m)")
    return path


# ════════════════════════════════════════════════════════════
#  入口
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── 解析命令行 ──────────────────────────────────────────
    config_path = "config.yaml"
    pc_file_arg = None

    args = sys.argv[1:]
    for a in args:
        if a.endswith(".yaml") or a.endswith(".yml"):
            config_path = a
        else:
            pc_file_arg = a

    # ── 加载配置 ────────────────────────────────────────────
    print(f"加载配置: {config_path}")
    cfg = SimConfig(load_config(config_path))
    print(cfg)

    ti.init(arch=ti.gpu)

    # ── 准备点云文件 ────────────────────────────────────────
    if pc_file_arg:
        pc_file = pc_file_arg
        print(f"使用外部点云文件: {pc_file}")
    else:
        pc_cfg  = cfg.pointcloud_cfg
        pc_file = pc_cfg.get("file") or None
        if pc_file is None:
            demo  = pc_cfg["demo_ball"]
            pc_file = generate_demo_pointcloud(
                path   = demo.get("save_path", "demo_ball.npy"),
                radius = float(demo.get("radius", 0.015)),
                n      = int(demo.get("n_points", 2000)),
            )

    # ── 创建仿真并运行 ───────────────────────────────────────
    sim = MPMSimulator(pc_file=pc_file, cfg=cfg)
    sim.run()
