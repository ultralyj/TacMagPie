"""Material Point Method (MPM) core kernel module.

This module implements the three main steps of the MPM algorithm:
  1. P2G (Particle to Grid) - Project particle momentum/mass to grid
  2. Grid Update - Velocity normalization, collision response, boundary conditions
  3. G2P (Grid to Particle) - Interpolate grid velocity/gradient back to particles

Public Interface
----------------
build_mpm_kernels(cfg, x, v, C, F, grid_v, grid_m, yeoh_piola_fn)
    Returns (init_particles, substep) kernel functions.
"""

import taichi as ti
from config_loader import SimConfig


def build_mpm_kernels(cfg: SimConfig, x, v, C, F, grid_v, grid_m,
                      yeoh_piola_fn):
    """Build MPM core Taichi kernels.

    Args:
        cfg: Simulation configuration object
        x: Particle position field
        v: Particle velocity field
        C: Particle affine velocity field (APIC)
        F: Particle deformation gradient field
        grid_v: Grid velocity field
        grid_m: Grid mass field
        yeoh_piola_fn: Yeoh constitutive model function from constitutive.build_yeoh_piola()

    Returns:
        tuple: (init_particles, substep) kernel functions
            - init_particles(): Initialize particles in simulation domain
            - substep(): Execute one physics timestep (P2G → Grid Update → G2P)
    """
    DIM = cfg.DIM
    n_particles = cfg.n_particles
    npx, npy, npz = cfg.n_particles_x, cfg.n_particles_y, cfg.n_particles_z
    slab_x, slab_y, slab_z = cfg.slab_x, cfg.slab_y, cfg.slab_z
    domain = cfg.domain
    dx, inv_dx = cfg.dx, cfg.inv_dx
    dt = cfg.dt
    p_mass, p_vol = cfg.p_mass, cfg.p_vol
    n_grid = cfg.n_grid

    @ti.kernel
    def init_particles():
        """Initialize particles uniformly within the silicone slab region.
        
        Slab is centered in X/Z directions and starts at 2*dx in Y direction.
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
            v[i] = ti.Vector.zero(ti.f32, DIM)
            F[i] = ti.Matrix.identity(ti.f32, DIM)
            C[i] = ti.Matrix.zero(ti.f32, DIM, DIM)

    @ti.kernel
    def substep(
        col_pts: ti.template(),
        col_n: ti.template(),
        col_radius: ti.template(),
        col_vel: ti.template(),
        col_force: ti.template(),
    ):
        """Execute one MPM timestep with collision handling.

        Args:
            col_pts: Collider point cloud positions
            col_n: Collider point count
            col_radius: Collider interaction radius
            col_vel: Collider velocity
            col_force: Collider force accumulator (output)
        """
        # Clear grid
        for i, j, k in grid_m:
            grid_v[i, j, k] = ti.Vector.zero(ti.f32, DIM)
            grid_m[i, j, k] = 0.0

        # P2G: Particle to Grid (MLS-MPM APIC)
        for p in range(n_particles):
            Xp = x[p] * inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - ti.cast(base, ti.f32)

            w = [0.5 * (1.5 - fx) ** 2,
                 0.75 - (fx - 1.0) ** 2,
                 0.5 * (fx - 0.5) ** 2]

            stress = ((-dt * p_vol * 4.0 * inv_dx * inv_dx)
                      * yeoh_piola_fn(F[p]) @ F[p].transpose())
            affine = stress + p_mass * C[p]

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (ti.cast(offset, ti.f32) - fx) * dx
                weight = w[i][0] * w[j][1] * w[k][2]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

        # Grid Update: Normalize velocity, collision response, boundary conditions
        radius = col_radius[None]
        n_col = col_n[None]
        c_vel = col_vel[None]

        for i, j, k in grid_m:
            if grid_m[i, j, k] > 0:
                grid_v[i, j, k] /= grid_m[i, j, k]
                gpos = ti.Vector([i, j, k], dt=ti.f32) * dx

                min_dist = radius * 10.0
                closest_n = ti.Vector.zero(ti.f32, DIM)

                for q in range(n_col):
                    diff = gpos - col_pts[q]
                    dist = diff.norm()
                    if dist < min_dist:
                        min_dist = dist
                        closest_n = diff / dist if dist > 1e-6 else ti.Vector([0.0, 1.0, 0.0])

                if min_dist < radius:
                    rel_v = grid_v[i, j, k] - c_vel
                    vn = rel_v.dot(closest_n)
                    if vn < 0:
                        delta_v = -vn * closest_n
                        grid_v[i, j, k] += delta_v
                        f_contrib = -grid_m[i, j, k] * delta_v / dt
                        ti.atomic_add(col_force[None][0], f_contrib[0])
                        ti.atomic_add(col_force[None][1], f_contrib[1])
                        ti.atomic_add(col_force[None][2], f_contrib[2])

                # Boundary conditions: stick to bottom, free slip elsewhere
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

        # G2P: Grid to Particle (APIC velocity + deformation gradient update)
        for p in range(n_particles):
            Xp = x[p] * inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - ti.cast(base, ti.f32)

            w = [0.5 * (1.5 - fx) ** 2,
                 0.75 - (fx - 1.0) ** 2,
                 0.5 * (fx - 0.5) ** 2]

            new_v = ti.Vector.zero(ti.f32, DIM)
            new_C = ti.Matrix.zero(ti.f32, DIM, DIM)

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (ti.cast(offset, ti.f32) - fx) * dx
                weight = w[i][0] * w[j][1] * w[k][2]
                g_v = grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4.0 * inv_dx * weight * g_v.outer_product(dpos)

            v[p] = new_v
            C[p] = new_C
            x[p] += dt * new_v
            F[p] = (ti.Matrix.identity(ti.f32, DIM) + dt * new_C) @ F[p]

    return init_particles, substep
