# mpm_sim_pointcloud.py
import taichi as ti
import numpy as np
import os
import time
ti.init(arch=ti.gpu)

# ─── 物理参数 ───────────────────────────────────────────────
DIM = 3

E        = 5e5
nu       = 0.49
mu_0     = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))
rho      = 1000.0

slab_x = 0.16
slab_y = 0.02
slab_z = 0.16

domain = 0.2
n_grid = 32
dx     = domain / n_grid
inv_dx = 1.0 / dx
dt     = 1e-4

n_particles_x = int(slab_x / (dx * 0.5))
n_particles_y = int(slab_y / (dx * 0.5))
n_particles_z = int(slab_z / (dx * 0.5))
n_particles   = n_particles_x * n_particles_y * n_particles_z
p_vol  = (slab_x * slab_y * slab_z) / n_particles
p_mass = rho * p_vol

# ─── 磁场参数 ────────────────────────────────────────────────
MU_0_4PI   = 1e-7
B0         = 1e-3
SENSOR_POS = (domain * 0.5, -0.005, domain * 0.5)

# ─── 点云碰撞器参数 ──────────────────────────────────────────
MAX_COLLIDER_PTS = 50000          # 最多支持的点云点数
COLLIDER_RADIUS  = dx * 1.5      # 每个点云点的影响半径（碰撞检测用）


# ════════════════════════════════════════════════════════════
#  点云碰撞器类  PointCloudCollider
# ════════════════════════════════════════════════════════════
class PointCloudCollider:
    """
    管理点云碰撞体：
      - 从文件加载初始形状
      - 接受外部控制（位移 / 速度）
      - 将当前点云位置同步到 Taichi field
    """
    def __init__(self, filepath: str, init_center: np.ndarray,
                 point_radius: float = COLLIDER_RADIUS):
        """
        参数
        ----
        filepath    : 点云文件路径（.npy / .ply / .txt）
        init_center : 点云包围盒中心的初始世界坐标 shape=(3,)
        point_radius: 碰撞检测时每个点的影响半径
        """
        self.point_radius = point_radius
        self._raw_pts     = self._load(filepath)          # (N,3) 局部坐标
        n = len(self._raw_pts)
        assert n <= MAX_COLLIDER_PTS, \
            f"点云点数 {n} 超过上限 {MAX_COLLIDER_PTS}，请调大 MAX_COLLIDER_PTS"

        self.n_pts = n

        # 将点云中心移到原点（局部坐标）
        local_center = self._raw_pts.mean(axis=0)
        self._raw_pts -= local_center

        # 世界坐标 = 局部坐标 + position
        self.position = np.array(init_center, dtype=np.float32)  # 当前质心世界坐标
        self.velocity = np.zeros(3, dtype=np.float32)             # 当前速度 m/s

        # ── Taichi fields ──
        self.ti_pts = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_COLLIDER_PTS)
        self.ti_n   = ti.field(dtype=ti.i32, shape=())
        self.ti_n[None] = self.n_pts
        self.ti_radius  = ti.field(dtype=ti.f32, shape=())
        self.ti_radius[None] = self.point_radius
        self.ti_vel     = ti.Vector.field(DIM, dtype=ti.f32, shape=())

        # 力累加
        self.ti_force = ti.Vector.field(DIM, dtype=ti.f32, shape=())

        self.sync_to_taichi()

    # ── 文件加载 ─────────────────────────────────────────────
    @staticmethod
    def _load(filepath: str) -> np.ndarray:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.npy':
            pts = np.load(filepath).astype(np.float32)
        elif ext == '.txt' or ext == '.xyz':
            pts = np.loadtxt(filepath, dtype=np.float32)
        elif ext == '.ply':
            pts = PointCloudCollider._load_ply(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {ext}，请使用 .npy / .txt / .ply")
        assert pts.ndim == 2 and pts.shape[1] >= 3, "点云必须是 (N,3+) 数组"
        return pts[:, :3].astype(np.float32)

    @staticmethod
    def _load_ply(filepath: str) -> np.ndarray:
        """极简 PLY 读取（仅 ASCII 格式，只取 xyz）"""
        pts = []
        in_data = False
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line == 'end_header':
                    in_data = True
                    continue
                if in_data:
                    vals = line.split()
                    if len(vals) >= 3:
                        pts.append([float(vals[0]),
                                    float(vals[1]),
                                    float(vals[2])])
        return np.array(pts, dtype=np.float32)

    # ── 外部控制接口 ─────────────────────────────────────────
    def set_velocity(self, vel: np.ndarray):
        """设置碰撞体速度 (m/s)，shape=(3,)"""
        self.velocity = np.array(vel, dtype=np.float32)

    def set_position(self, pos: np.ndarray):
        """直接设定碰撞体质心世界坐标，shape=(3,)"""
        self.position = np.array(pos, dtype=np.float32)

    def step(self, dt_val: float):
        """按当前速度推进位置"""
        self.position += self.velocity * dt_val
        self.sync_to_taichi()

    def sync_to_taichi(self):
        """将当前点云世界坐标写入 Taichi field"""
        world_pts = self._raw_pts + self.position          # (N,3)
        # 用 numpy 批量写入（比逐点快）
        tmp = np.zeros((MAX_COLLIDER_PTS, DIM), dtype=np.float32)
        tmp[:self.n_pts] = world_pts
        self.ti_pts.from_numpy(tmp)
        self.ti_vel[None] = ti.Vector(self.velocity.tolist())

    def get_force(self) -> np.ndarray:
        """读取上一步碰撞力 (N)"""
        return self.ti_force.to_numpy()

    def clear_force(self):
        self.ti_force[None] = [0.0, 0.0, 0.0]

    def get_current_points(self) -> np.ndarray:
        """返回当前世界坐标点云 (N,3)"""
        return self._raw_pts + self.position

    def get_bbox(self):
        """返回包围盒 (min_xyz, max_xyz)"""
        pts = self.get_current_points()
        return pts.min(axis=0), pts.max(axis=0)


# ════════════════════════════════════════════════════════════
#  Taichi Fields
# ════════════════════════════════════════════════════════════
x  = ti.Vector.field(DIM, dtype=ti.f32, shape=n_particles)
v  = ti.Vector.field(DIM, dtype=ti.f32, shape=n_particles)
C  = ti.Matrix.field(DIM, DIM, dtype=ti.f32, shape=n_particles)
F  = ti.Matrix.field(DIM, DIM, dtype=ti.f32, shape=n_particles)

grid_v = ti.Vector.field(DIM, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))

B_sensor  = ti.Vector.field(DIM, dtype=ti.f64, shape=())
total_force = ti.Vector.field(DIM, dtype=ti.f32, shape=())   # 碰撞合力

# ─── 初始化 ─────────────────────────────────────────────────
@ti.kernel
def init_particles():
    ox = (domain - slab_x) * 0.5
    oy = 2 * dx
    oz = (domain - slab_z) * 0.5
    for i in range(n_particles):
        ix = i % n_particles_x
        iy = (i // n_particles_x) % n_particles_y
        iz = i // (n_particles_x * n_particles_y)
        x[i] = ti.Vector([
            ox + (ix + 0.5) * slab_x / n_particles_x,
            oy + (iy + 0.5) * slab_y / n_particles_y,
            oz + (iz + 0.5) * slab_z / n_particles_z,
        ])
        v[i] = ti.Vector([0.0, 0.0, 0.0])
        F[i] = ti.Matrix.identity(ti.f32, DIM)
        C[i] = ti.Matrix.zero(ti.f32, DIM, DIM)

# ─── 本构模型 ────────────────────────────────────────────────
C1 = 1e-1  * 1e6
C2 = 6.42e-2 * 1e6
C3 = 8.08e-5 * 1e6
kappa = 0.0

@ti.func
def yeoh_piola(F_mat: ti.template()):
    J = F_mat.determinant()
    if J < 0.1:
        F_mat = ti.pow(0.1 / J, 1.0/3.0) * F_mat
        J = 0.1
    F_inv_T = F_mat.inverse().transpose()
    I1      = (F_mat.transpose() @ F_mat).trace()
    J23     = ti.pow(J, -2.0 / 3.0)
    I1b     = J23 * I1
    i1bm3   = I1b - 3.0
    dWdI1b  = C1 + 2.0*C2*i1bm3 + 3.0*C3*i1bm3*i1bm3
    P_iso   = 2.0 * dWdI1b * J23 * (F_mat - (I1 / 3.0) * F_inv_T)
    P_vol   = kappa * (J - 1.0) * J * F_inv_T
    return P_iso + P_vol


# ════════════════════════════════════════════════════════════
#  MPM substep（带点云碰撞）
# ════════════════════════════════════════════════════════════

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

    # 2. P2G（不变）
    for p in range(n_particles):
        Xp   = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx   = Xp - ti.cast(base, ti.f32)
        w = [0.5*(1.5-fx)**2, 0.75-(fx-1.0)**2, 0.5*(fx-0.5)**2]
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * yeoh_piola(F[p]) @ F[p].transpose()
        affine = stress + p_mass * C[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos   = (ti.cast(offset, ti.f32) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_v[base+offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base+offset] += weight * p_mass

    # 3. 网格更新 + 改进的点云碰撞
    radius = col_radius[None]
    n_col  = col_n[None]
    c_vel  = col_vel[None]

    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] /= grid_m[i, j, k]
            gpos = ti.Vector([i, j, k], dt=ti.f32) * dx

            # ── 改进：找最近点云点，用距离判断是否碰撞 ──────
            min_dist  = radius * 10.0   # 初始化为大值
            closest_n = ti.Vector.zero(ti.f32, DIM)

            for q in range(n_col):
                diff = gpos - col_pts[q]
                dist = diff.norm()
                if dist < min_dist:
                    min_dist  = dist
                    # 避免零向量
                    if dist > 1e-6:
                        closest_n = diff / dist
                    else:
                        closest_n = ti.Vector([0.0, 1.0, 0.0])

            # 只要网格节点到最近点云点的距离小于半径就触发碰撞
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

            # ── 边界条件（不变）──────────────────────────────
            if j < 3:
                grid_v[i, j, k] = ti.Vector.zero(ti.f32, DIM)
            if j > n_grid-3 and grid_v[i,j,k][1] > 0: grid_v[i,j,k][1] = 0.0
            if i < 3 and grid_v[i,j,k][0] < 0:        grid_v[i,j,k][0] = 0.0
            if i > n_grid-3 and grid_v[i,j,k][0] > 0: grid_v[i,j,k][0] = 0.0
            if k < 3 and grid_v[i,j,k][2] < 0:        grid_v[i,j,k][2] = 0.0
            if k > n_grid-3 and grid_v[i,j,k][2] > 0: grid_v[i,j,k][2] = 0.0

    # 4. G2P（不变）
    for p in range(n_particles):
        Xp   = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx   = Xp - ti.cast(base, ti.f32)
        w = [0.5*(1.5-fx)**2, 0.75-(fx-1.0)**2, 0.5*(fx-0.5)**2]
        new_v = ti.Vector.zero(ti.f32, DIM)
        new_C = ti.Matrix.zero(ti.f32, DIM, DIM)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos   = (ti.cast(offset, ti.f32) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            g_v    = grid_v[base+offset]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p]  = new_v
        C[p]  = new_C
        x[p] += dt * new_v
        F[p]  = (ti.Matrix.identity(ti.f32, DIM) + dt * new_C) @ F[p]

# ─── 磁场 ────────────────────────────────────────────────────
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
        r2 = rx*rx + ry*ry + rz*rz
        if r2 < 1e-30: continue
        r_norm = ti.sqrt(r2); r3 = r_norm*r2; r5 = r3*r2
        m_dot_r = my * ry
        ti.atomic_add(B_sensor[None][0], MU_0_4PI*(3.0*m_dot_r*rx/r5))
        ti.atomic_add(B_sensor[None][1], MU_0_4PI*(3.0*m_dot_r*ry/r5 - my/r3))
        ti.atomic_add(B_sensor[None][2], MU_0_4PI*(3.0*m_dot_r*rz/r5))


# ════════════════════════════════════════════════════════════
#  运动控制器基类  —  用户继承此类实现自定义轨迹
# ════════════════════════════════════════════════════════════
class MotionController:
    """
    外部运动控制器接口。
    子类实现 get_velocity(t, collider) 即可。
    """
    def get_velocity(self, t: float, collider: PointCloudCollider) -> np.ndarray:
        """
        返回此时刻碰撞体速度 (m/s), shape=(3,)
        t        : 当前仿真时间 (s)
        collider : 碰撞体对象（可读取当前位置）
        """
        return np.zeros(3, dtype=np.float32)

    def get_position(self, t: float, collider: PointCloudCollider):
        """
        可选：直接设置位置（优先级高于速度）。
        返回 None 则使用速度积分。
        """
        return None


class PressAndHoldController(MotionController):
    """
    示例：先向下按压，再保持静止
    press_speed : 按压速度 (m/s)，正值向下
    press_depth : 按压深度 (m)
    init_y      : 初始 Y 坐标
    """
    def __init__(self, press_speed=0.02, press_depth=0.008, init_y=0.0):
        self.press_speed = press_speed
        self.press_depth = press_depth
        self.init_y      = init_y

    def get_velocity(self, t, collider):
        current_depth = self.init_y - collider.position[1]
        if current_depth < self.press_depth:
            return np.array([0.0, -self.press_speed, 0.0], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)


class TrajectoryController(MotionController):
    """
    轨迹控制器：传入时间序列和位置序列，自动插值
    times     : shape=(T,)  时间点 (s)
    positions : shape=(T,3) 对应位置
    """
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


# ════════════════════════════════════════════════════════════
#  主仿真类  MPMSimulator
# ════════════════════════════════════════════════════════════
class MPMSimulator:
    def __init__(self,
                 pointcloud_file: str,
                 controller: MotionController = None,
                 collider_init_center: np.ndarray = None,
                 collider_radius: float = COLLIDER_RADIUS,
                 use_gui: bool = True):
        """
        参数
        ----
        pointcloud_file     : 点云文件路径
        controller          : 运动控制器（None = 静止）
        collider_init_center: 点云质心初始世界坐标，默认悬停在硅胶板上方
        collider_radius     : 碰撞检测半径
        use_gui             : 是否显示窗口
        """
        # 硅胶板信息
        slab_top_y = 2 * dx + slab_y   # 硅胶板顶面 Y

        if collider_init_center is None:
            collider_init_center = np.array([
                domain * 0.5,
                slab_top_y + collider_radius + 0.001,
                domain * 0.5
            ], dtype=np.float32)

        self.collider   = PointCloudCollider(pointcloud_file,
                                             collider_init_center,
                                             collider_radius)
        self.controller = controller or MotionController()
        self.use_gui    = use_gui
        self.t          = 0.0
        self.frame      = 0
        self.substeps   = 20

        init_particles()

        # 磁场基准
        clear_B_sensor()
        compute_sensor_field(*SENSOR_POS)
        self.B_baseline = B_sensor.to_numpy().copy()

        if use_gui:
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

        # 点云可视化 field
        n = self.collider.n_pts
        self.vis_pts = ti.Vector.field(DIM, dtype=ti.f32, shape=n)

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
        self.scene.particles(x,             radius=dx*0.3,  color=(0.2, 0.8, 0.6))
        self.scene.particles(self.vis_pts,  radius=dx*0.15, color=(0.9, 0.3, 0.2))
        self.canvas.scene(self.scene)
        self.window.show()

    # ── 单帧推进 ────────────────────────────────────────────
    def step_frame(self):
        """推进一帧（substeps 个物理步）"""
        col = self.collider

        for _ in range(self.substeps):
            # 询问控制器
            new_pos = self.controller.get_position(self.t, col)
            if new_pos is not None:
                col.set_position(new_pos)
                col.sync_to_taichi()
            else:
                vel = self.controller.get_velocity(self.t, col)
                col.set_velocity(vel)
                col.step(dt)

            # 清力
            col.clear_force()

            # MPM substep
            timestart = time.time()
            substep(col.ti_pts, col.ti_n, col.ti_radius,
                    col.ti_vel, col.ti_force)
            print(f"Substep computed in {(time.time() - timestart)*1000:.2f} ms", flush=True)
            self.t += dt

        self.frame += 1

    # ── 磁场读取 ────────────────────────────────────────────
    def get_B_delta(self):
        """返回相对初始状态的磁场变化量 (T)"""
        clear_B_sensor()
        compute_sensor_field(*SENSOR_POS)
        B = B_sensor.to_numpy()
        return B - self.B_baseline

    # ── 完整运行 ────────────────────────────────────────────
    def run(self, total_frames: int = 500, print_interval: int = 10):
        if self.use_gui:
            while self.window.running and self.frame < total_frames:
                self.step_frame()
                if self.frame % print_interval == 0:
                    B = self.get_B_delta()
                    f = self.collider.get_force()
                    print(f"Frame {self.frame:4d} | t={self.t:.4f}s | "
                          f"pos={self.collider.position} | "
                          f"F={f} N | "
                          f"ΔB=[{B[0]:.3e},{B[1]:.3e},{B[2]:.3e}] T")
                self._render()
        else:
            for _ in range(total_frames):
                # timestart = time.time()
                self.step_frame()
                # print(f"Frame {self.frame:4d} computed in {(time.time() - timestart)*1000:.2f} ms")
                if self.frame % print_interval == 0:
                    B = self.get_B_delta()
                    f = self.collider.get_force()
                    # print(f"Frame {self.frame:4d} | t={self.t:.4f}s | "
                    #       f"pos={self.collider.position} | "
                    #       f"F={f} N | "
                    #       f"ΔB=[{B[0]:.3e},{B[1]:.3e},{B[2]:.3e}] T")


# ════════════════════════════════════════════════════════════
#  入口
# ════════════════════════════════════════════════════════════
def generate_demo_pointcloud(path="demo_ball.npy", radius=0.015, n=2000):
    """生成一个球形点云用于演示（无真实点云文件时使用）"""
    pts = []
    golden = 2.399963229728653
    for i in range(n):
        theta = np.arccos(1.0 - 2.0*(i+0.5)/n)
        phi   = golden * i
        pts.append([radius*np.sin(theta)*np.cos(phi),
                    radius*np.cos(theta),
                                    radius*np.sin(theta)*np.sin(phi)])
    np.save(path, np.array(pts, dtype=np.float32))
    print(f"已生成演示点云: {path}  ({n} 点, 半径={radius}m)")
    return path


if __name__ == "__main__":
    import sys

    # ── 1. 准备点云文件 ──────────────────────────────────────
    if len(sys.argv) > 1:
        pc_file = sys.argv[1]
        print(f"使用外部点云文件: {pc_file}")
    else:
        # 没有提供文件则生成一个球形演示点云
        pc_file = generate_demo_pointcloud("demo_ball.npy", radius=0.015, n=2000)

    # ── 2. 选择运动控制器 ────────────────────────────────────
    # 方案A：简单按压控制器
    slab_top_y   = 2 * dx + slab_y
    init_center  = np.array([domain*0.5,
                              slab_top_y + COLLIDER_RADIUS + 0.008,
                              domain*0.5], dtype=np.float32)

    controller = PressAndHoldController(
        press_speed = 0.02,
        press_depth = 0.008,
        init_y      = float(init_center[1])
    )

    # 方案B：轨迹控制器（取消注释使用）
    # times     = np.linspace(0, 2.0, 200)
    # positions = np.zeros((200, 3), dtype=np.float32)
    # positions[:, 0] = domain * 0.5
    # positions[:, 2] = domain * 0.5
    # positions[:, 1] = init_center[1] - np.clip(times / 0.4, 0, 1) * 0.008
    # controller = TrajectoryController(times, positions)

    # ── 3. 创建并运行仿真 ────────────────────────────────────
    sim = MPMSimulator(
        pointcloud_file      = pc_file,
        controller           = controller,
        collider_init_center = init_center,
        collider_radius      = COLLIDER_RADIUS,
        use_gui              = True,
    )
    sim.run(total_frames=600, print_interval=10)
