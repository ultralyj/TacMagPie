"""Utility functions for point cloud generation and file conversion.

Functions
---------
generate_demo_pointcloud(path, radius, n)
    Generate uniform spherical point cloud for demonstration and save as .npy

stl_to_pointcloud(stl_path, ...)
    Convert STL file to point cloud with uniform or Poisson disk sampling

parse_args()
    Parse command line arguments, returns (config_path, pc_file_or_None)
"""

import sys
import math
import numpy as np
import open3d as o3d
import os


def generate_demo_pointcloud(
    path: str = "demo_ball.npy",
    radius: float = 0.015,
    n: int = 2000,
) -> str:
    """Generate uniformly distributed spherical point cloud using golden spiral algorithm.

    Args:
        path: Output file path
        radius: Sphere radius in meters
        n: Number of points

    Returns:
        Actual save path
    """
    pts = []
    golden = 2.399963229728653

    for i in range(n):
        theta = math.acos(1.0 - 2.0 * (i + 0.5) / n)
        phi = golden * i
        pts.append([
            radius * math.sin(theta) * math.cos(phi),
            radius * math.cos(theta),
            radius * math.sin(theta) * math.sin(phi),
        ])

    np.save(path, np.array(pts, dtype=np.float32))
    print(f"[utils] Generated demo point cloud: {path} ({n} points, radius={radius} m)")
    return path


def stl_to_pointcloud(
    stl_path: str,
    output_path: str = None,
    n_points: int = 5000,
    method: str = "uniform",
    poisson_radius: float = None,
    include_normals: bool = False,
    scale: float = 10.0,
    center: bool = True,
) -> np.ndarray:
    """Convert STL file to point cloud.

    Args:
        stl_path: Input STL file path (ASCII or binary)
        output_path: Output file path (.npy / .ply / .pcd / .txt), None = no save
        n_points: Target number of sample points (exact for uniform, approximate for poisson)
        method: Sampling method
            "uniform" - Area-weighted uniform random sampling (recommended)
            "poisson" - Poisson disk sampling (more uniform spacing, slower)
        poisson_radius: Minimum point spacing for Poisson sampling in meters, None = auto-estimate
        include_normals: Include normals in output array (True = shape (N,6))
        scale: Overall scale factor (STL often in mm, use 0.001 to convert to m)
        center: Center point cloud at local origin (True = subtract bounding box center)

    Returns:
        Point cloud array, shape=(N,3) or (N,6), dtype=float32

    Raises:
        ImportError: If open3d is not installed
        ValueError: If STL file is invalid or method is unknown

    Examples:
        # STL in mm, convert to m, uniform sample 3000 points, save as PLY
        pts = stl_to_pointcloud(
            stl_path="tool.stl",
            output_path="tool.ply",
            n_points=3000,
            scale=0.001,
            method="uniform",
        )

        # Poisson disk sampling with ~1mm spacing
        pts = stl_to_pointcloud(
            stl_path="tool.stl",
            method="poisson",
            poisson_radius=0.001,
            scale=0.001,
        )
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "stl_to_pointcloud requires open3d. Install with: pip install open3d"
        )

    print(f"[stl_to_pointcloud] Reading STL: {stl_path}")
    mesh = o3d.io.read_triangle_mesh(stl_path)

    if not mesh.has_vertices():
        raise ValueError(f"Failed to read STL or no vertices: {stl_path}")

    n_tri = len(mesh.triangles)
    print(f"[stl_to_pointcloud] Mesh: {len(mesh.vertices)} vertices, {n_tri} triangles")

    if scale != 1.0:
        mesh.scale(scale, center=mesh.get_center())
        print(f"[stl_to_pointcloud] Scaled by ×{scale}")

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    if method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
        print(f"[stl_to_pointcloud] Uniform sampling: {len(pcd.points)} points")

    elif method == "poisson":
        if poisson_radius is None:
            bbox = mesh.get_axis_aligned_bounding_box()
            vol = float(np.prod(np.asarray(bbox.get_extent())))
            area = vol ** (2.0 / 3.0)
            poisson_radius = math.sqrt(area / (n_points * math.pi)) * 2.0
            print(f"[stl_to_pointcloud] Auto-estimated Poisson radius: {poisson_radius:.5f} m")

        pcd = mesh.sample_points_poisson_disk(
            number_of_points=n_points,
            init_factor=5,
            pcl=None,
        )
        print(f"[stl_to_pointcloud] Poisson sampling: {len(pcd.points)} points")

    else:
        raise ValueError(f"Unknown sampling method: {method!r}, use 'uniform' or 'poisson'")

    pts = np.asarray(pcd.points, dtype=np.float32)

    if center:
        pts -= pts.mean(axis=0)
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        print(f"[stl_to_pointcloud] Centered at local origin")

    if include_normals:
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=poisson_radius or 0.01, max_nn=30
                )
            )
        normals = np.asarray(pcd.normals, dtype=np.float32)
        pts = np.concatenate([pts, normals], axis=1)
        print(f"[stl_to_pointcloud] Added normals, output shape: {pts.shape}")

    pts[:, 1] *= -1
    if include_normals:
        pts[:, 4] *= -1

    y_percentile = np.percentile(pts[:, 1], 3)
    mask = pts[:, 1] <= y_percentile
    pts = pts[mask]

    pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
    if include_normals and pcd.has_normals():
        pcd.normals = o3d.utility.Vector3dVector(pts[:, 3:6].astype(np.float64))

    if output_path is not None:
        _save_pointcloud(pcd, pts, output_path)

    return pts


def _save_pointcloud(pcd, pts: np.ndarray, output_path: str):
    """Save point cloud to specified format.

    Args:
        pcd: Open3D point cloud object
        pts: Point cloud array
        output_path: Output file path

    Raises:
        ValueError: If output format is unsupported
    """
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".npy":
        np.save(output_path, pts)

    elif ext == ".txt":
        np.savetxt(output_path, pts, fmt="%.6f")

    elif ext in (".ply", ".pcd"):
        pcd.points = o3d.utility.Vector3dVector(
            pts[:, :3].astype(np.float64)
        )
        o3d.io.write_point_cloud(output_path, pcd)

    else:
        raise ValueError(
            f"Unsupported output format: {ext}, use .npy / .ply / .pcd / .txt"
        )

    print(f"[stl_to_pointcloud] Saved: {output_path} ({len(pts)} points)")


def parse_args():
    """Parse command line arguments.

    Usage:
        python MagPie.py                        → config="config.yaml", pc=None
        python MagPie.py my.yaml                → config="my.yaml", pc=None
        python MagPie.py config.yaml ball.npy   → config + pc
        python MagPie.py tool.stl               → auto-convert STL to point cloud

    Returns:
        tuple: (config_path: str, pc_file: str | None)
    """
    config_path = "config/default.yaml"
    pc_file = None

    for arg in sys.argv[1:]:
        if arg.endswith((".yaml", ".yml")):
            config_path = arg
        else:
            pc_file = arg

    return config_path, pc_file


if __name__ == "__main__":
    path = "../model/asset/indenter_type1.STL"
    stl_to_pointcloud(path, output_path="../model/indenter_type1.npy",
                      n_points=50000, method="uniform", scale=0.01)
