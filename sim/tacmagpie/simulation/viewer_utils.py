import os
import numpy as np
import mujoco


def setup_viewer(viewer):
    """
    Configure MuJoCo viewer for wireframe-based visualization.

    This setup is intended for:
    - Clear observation of deformation
    - Reduced rendering overhead
    - Consistent camera pose across runs
    """
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 0.2
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -20
    viewer.cam.lookat = [0.0, 0.0, 0.03]


def initialize_simulation_state(data, runner):
    """
    Initialize joint positions and control targets before simulation starts.

    This ensures deterministic initial conditions regardless of
    model defaults or previous runs.
    """
    data.qpos[runner.joint_id] = runner.initial_position
    data.ctrl[runner.position_control_id] = runner.initial_position


def capture_screenshot(
    capturer,
    data,
    runner,
    screenshot_dir: str,
    timestamp: str,
):
    """
    Capture and save an offscreen-rendered frame.

    Screenshots are timestamped using simulation time (milliseconds)
    to preserve temporal ordering.
    """
    try:
        image = capturer.capture_frame(data)
        if image is None:
            return

        filename = f"frame_{timestamp}_{int(data.time * 1000):06d}.png"
        filepath = os.path.join(screenshot_dir, filename)
        capturer.save_frame(image, filepath)

    except Exception as exc:
        print(f"[Screenshot] Capture failed: {exc}")


def print_detailed_status(data, runner, sensor_data):
    """
    Print detailed runtime status for debugging and monitoring.

    Includes:
    - Wall-clock elapsed time
    - Simulation time
    - Indenter depth
    - Aggregate elastic deformation
    - Actuator force and velocity
    """
    performance = runner.get_performance_info()
    elastic_deformation = np.sum(runner.grid_vec[:, :, :, 2])

    print(
        f"[{performance['elapsed_time']:.2f}s] "
        f"SimTime: {performance['simulation_time']:.4f}s | "
        f"Depth: {data.qpos[runner.joint_id]:.3f} m | "
        f"Elastic: {elastic_deformation:.4f} | "
        f"ActForce: {sensor_data['actuator_force']:.2f} N | "
        f"NetForce: {sensor_data['force']:.2f} N | "
        f"Velocity: {sensor_data['velocity']:.4f} m/s"
    )
