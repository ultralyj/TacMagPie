# mpm_core.py
"""
MPM 核心 Kernel 模块

提供 MPM（Material Point Method）的三大步骤：
  1. P2G（Particle to Grid）— 粒子动量/质量投影到网格
  2. 网格更新            — 速度归一化、碰撞响应、边界条件
  3. G2P（Grid to Particle）— 网格速度/梯度插值回粒子，更新变形梯度

对外接口
--------
build_mpm_kernels(cfg, x, v, C, F, grid_v, grid_m, yeoh_piola_fn)
    返回 (init_particles, substep)

    init_particles()
        在仿真域内均匀分布粒子，设置初始速度/变形梯度

    substep(col_pts, col_n, col_radius, col_vel, col_force)
        执行一个物理时间步（P2G → 网格更新 → G2P）
        col_*  : 碰撞器的 Taichi fields（来自 PointCloudCollider）
"""

import taichi as ti
from config_loader import SimConfig


def build_mpm_kernels(cfg: SimConfig, x, v, C, F, grid_v, grid_m,
                      yeoh_piola_fn):
    """
    构建 MPM 核心 Taichi kernels。

    参数
    ----
    cfg           : SimConfig
    x, v, C, F   : 粒子 Taichi fields
    grid_v, grid_m: 网格 Taichi fields
    yeoh_piola_fn : 由 constitutive.build_yeoh_piola() 返回的 @ti.func

    返回
    ----
    (init_particles, substep)
    """

    # ── 从 cfg 提取标量，避免在 kernel 内引用 Python 对象 ──
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

    # ── init_particles ────────────────────────────────────

    @ti.kernel
    def init_particles():
        """
        在硅胶板区域内均匀初始化粒子。
        板偏移量：X/Z 方向居中，Y 方向从 2·dx 起。
        """
        ox = (domain - slab_x) * 0.5
        oy = 2.0 * dx
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
        col_pts:    ti.template(),   # PointCloudCollider.ti_pts
        col_n:      ti.template(),   # PointCloudCollider.ti_n
        col_radius: ti.template(),   # PointCloudCollider.ti_radius
        col_vel:    ti.template(),   # PointCloudCollider.ti_vel
        col_force:  ti.template(),   # PointCloudCollider.ti_force
    ):
        # ── 1. 清空网格 ──────────────────────────────────
        for i, j, k in grid_m:
            grid_v[i, j, k] = ti.Vector.zero(ti.f32, DIM)
            grid_m[i, j, k] = 0.0

        # ── 2. P2G（MLS-MPM APIC 方案）────────────────────
        for p in range(n_particles):
            Xp   = x[p] * inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx   = Xp - ti.cast(base, ti.f32)

            # 二次 B-spline 权重
            w = [0.5 * (1.5 - fx) ** 2,
                 0.75 - (fx - 1.0) ** 2,
                 0.5 * (fx - 0.5) ** 2]

            # 柯西应力 → 动量贡献
            stress = ((-dt * p_vol * 4.0 * inv_dx * inv_dx)
                      * yeoh_piola_fn(F[p]) @ F[p].transpose())
            affine = stress + p_mass * C[p]

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos   = (ti.cast(offset, ti.f32) - fx) * dx
                weight = w[i][0] * w[j][1] * w[k][2]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

        # ── 3. 网格更新 + 点云碰撞 + 边界条件 ───────────────
        radius = col_radius[None]
        n_col  = col_n[None]
        c_vel  = col_vel[None]

        for i, j, k in grid_m:
            if grid_m[i, j, k] > 0:
                grid_v[i, j, k] /= grid_m[i, j, k]
                gpos = ti.Vector([i, j, k], dt=ti.f32) * dx

                # 找最近点云点
                min_dist  = radius * 10.0
                closest_n = ti.Vector.zero(ti.f32, DIM)

                for q in range(n_col):
                    diff = gpos - col_pts[q]
                    dist = diff.norm()
                    if dist < min_dist:
                        min_dist = dist
                        closest_n = (diff / dist
                                     if dist > 1e-6
                                     else ti.Vector([0.0, 1.0, 0.0]))

                # 碰撞响应（只处理穿入）
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

                # 边界条件（黏合底面，其余自由滑动）
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

        # ── 4. G2P（APIC 速度 + 变形梯度更新）─────────────
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
                new_C += 4.0 * inv_dx * weight * g_v.outer_product(dpos)

            v[p]  = new_v
            C[p]  = new_C
            x[p] += dt * new_v
            F[p]  = (ti.Matrix.identity(ti.f32, DIM) + dt * new_C) @ F[p]

    return init_particles, substep
