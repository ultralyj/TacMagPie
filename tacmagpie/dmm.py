"""Magnetic field calculation module using DMM method.

Computes magnetic field at sensor positions based on deformed elastomer
configuration.

Public Interface
----------------
build_magnetic_kernels(cfg, x, B_sensor)
    Returns (clear_B_sensor, compute_sensor_field) Taichi kernel pair.
    
    clear_B_sensor()
        Clears sensor field accumulator.
    
    compute_sensor_field(sx, sy, sz)
        Accumulates magnetic unit contributions from all particles at
        world coordinates (sx, sy, sz). Result stored in B_sensor[None].
"""

import taichi as ti
from config_loader import SimConfig


def build_magnetic_kernels(cfg: SimConfig, x, B_sensor):
    """Build magnetic field computation Taichi kernels.

    Args:
        cfg: Simulation configuration object.
        x: Particle position field (n_particles, DIM).
        B_sensor: Sensor magnetic field accumulator (shape=(), dtype=ti.f64).

    Returns:
        Tuple of (clear_B_sensor, compute_sensor_field) kernel functions.
    """
    n_particles = cfg.n_particles
    MU_0_4PI = cfg.MU_0_4PI
    B0 = cfg.B0

    @ti.kernel
    def clear_B_sensor():
        """Clear sensor magnetic field accumulator."""
        B_sensor[None] = [0.0, 0.0, 0.0]

    @ti.kernel
    def compute_sensor_field(sx: ti.f32, sy: ti.f32, sz: ti.f32):
        """Compute magnetic unit superposition at sensor position.

        Each particle is modeled as a magnetic unit along +Y axis with
        moment magnitude B0. Uses double precision accumulation to reduce
        rounding errors.

        Args:
            sx: Sensor X coordinate.
            sy: Sensor Y coordinate.
            sz: Sensor Z coordinate.
        """
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

            ti.atomic_add(B_sensor[None][0],
                         MU_0_4PI * (3.0 * m_dot_r * rx / r5))
            ti.atomic_add(B_sensor[None][1],
                         MU_0_4PI * (3.0 * m_dot_r * ry / r5 - my / r3))
            ti.atomic_add(B_sensor[None][2],
                         MU_0_4PI * (3.0 * m_dot_r * rz / r5))

    return clear_B_sensor, compute_sensor_field
