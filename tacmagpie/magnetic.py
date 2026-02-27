# magnetic.py
"""
磁场计算模块

基于偶极子模型，计算弹性体变形后传感器处的磁场。
每个粒子视为一个沿 +Y 方向的磁偶极子，强度为 B0。

对外接口
--------
build_magnetic_kernels(cfg, x, B_sensor)
    返回 (clear_B_sensor, compute_sensor_field) 两个 Taichi kernel

    clear_B_sensor()
        清零传感器场累加器

    compute_sensor_field(sx, sy, sz)
        在世界坐标 (sx, sy, sz) 处累加所有粒子的磁偶极子贡献
        结果存入 B_sensor[None]（ti.f64，防精度损失）
"""

import taichi as ti
from config_loader import SimConfig


def build_magnetic_kernels(cfg: SimConfig, x, B_sensor):
    """
    构建磁场 Taichi kernels。

    参数
    ----
    cfg      : SimConfig
    x        : ti.Vector.field  粒子位置 (n_particles, DIM)
    B_sensor : ti.Vector.field  传感器磁场累加器 (shape=(), dtype=ti.f64)

    返回
    ----
    (clear_B_sensor, compute_sensor_field)
    """
    n_particles = cfg.n_particles
    MU_0_4PI    = cfg.MU_0_4PI
    B0          = cfg.B0

    @ti.kernel
    def clear_B_sensor():
        """清零传感器磁场。"""
        B_sensor[None] = [0.0, 0.0, 0.0]

    @ti.kernel
    def compute_sensor_field(sx: ti.f32, sy: ti.f32, sz: ti.f32):
        """
        在 (sx, sy, sz) 处计算所有粒子的磁偶极子叠加贡献。

        物理假设
        --------
        每个粒子为沿 +Y 轴的磁偶极子，磁矩大小 = B0。
        使用双精度累加以减少舍入误差。
        """
        m0 = ti.cast(B0, ti.f64)
        my = m0                          # 偶极矩仅 Y 分量
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

            r_norm  = ti.sqrt(r2)
            r3      = r_norm * r2
            r5      = r3 * r2
            m_dot_r = my * ry

            ti.atomic_add(B_sensor[None][0],
                          MU_0_4PI * (3.0 * m_dot_r * rx / r5))
            ti.atomic_add(B_sensor[None][1],
                          MU_0_4PI * (3.0 * m_dot_r * ry / r5 - my / r3))
            ti.atomic_add(B_sensor[None][2],
                          MU_0_4PI * (3.0 * m_dot_r * rz / r5))

    return clear_B_sensor, compute_sensor_field
