from multiprocessing import Process, Queue
import numpy as np
import mujoco
import mujoco.viewer
import os
import config

from simulation.simulator import Simulator
from plotting.live_plot import live_plot_process
from sensors.dmm import compute_sensor_baseline, get_magnetic_data
from sensors.logger import update_logger, save_grid_vec
from utils.screenshot_capturer import ScreenshotCapturer
from simulation.viewer_utils import (
    setup_viewer,
    initialize_simulation_state,
    capture_screenshot,
    print_detailed_status,
)


def run_simulation_with_viewer(
    model,
    data,
    screenshot_dir: str,
    timestamp: str,
):
    """
    Run MuJoCo simulation with an interactive viewer, offscreen rendering,
    real-time plotting (in a separate process), and data logging.

    This function is intended for:
    - Debugging
    - Data collection
    - Qualitative inspection

    Not optimized for high-throughput training.
    """

    # Core components initialization
    runner = Simulator(model, data)
    capturer = ScreenshotCapturer(model)

    # Magnetic sensor data buffer
    mag_data = np.zeros((config.SENSOR_NUMBER, 3))

    # Asynchronous plotting process
    plot_queue: Queue = Queue(maxsize=100)
    plot_process = Process(
        target=live_plot_process,
        args=(plot_queue, config.SENSOR_NUMBER),
        daemon=True,
    )
    plot_process.start()

    # File system & logging setup
    os.makedirs("data", exist_ok=True)

    log_file_path = os.path.join("data", f"data_{timestamp}.csv")
    log_fp = open(log_file_path, "w+", encoding="utf-8")

    # Sensor baseline (initialized at simulation start)
    sensor_baseline = np.zeros((config.SENSOR_NUMBER, 3))

    print("========================================")
    print(" MuJoCo simulation started")
    print(" Press ESC in viewer to exit")
    print("========================================")

    try:
        # Viewer lifecycle
        with mujoco.viewer.launch_passive(model, data) as viewer:
            setup_viewer(viewer)
            initialize_simulation_state(data, runner)

            # ---------------- Main simulation loop ----------------
            while viewer.is_running() and not runner.is_finished():

                # Step physics
                runner.step()
                
                # Update control policy                
                runner.update_control()

                # Read proprioceptive sensors
                sensor_data = runner.get_sensor_data()
                # Magnetic sensing & logging (downsampled)                
                if runner.should_update_data():
                    sim_time = runner.get_performance_info()["simulation_time"]

                    # Initialize magnetic baseline at early stage
                    if sim_time < 0.01:
                        sensor_baseline = compute_sensor_baseline(runner)
                    else:
                        mag_data = get_magnetic_data(runner, sensor_baseline)

                        # Send data to plotting process (non-blocking intent)
                        try:
                            plot_queue.put_nowait(
                                (sim_time, mag_data.copy())
                            )
                        except Exception:
                            # Plotting is best-effort; dropping data is acceptable
                            pass
                    print_detailed_status(data, runner, sensor_data)
                    update_logger(
                        log_fp,
                        data,
                        runner,
                        sensor_data,
                        mag_data,
                    )
                    save_grid_vec(runner, screenshot_dir, timestamp)

                # Offscreen rendering & screenshot capture                
                if runner.should_capture_screenshot():
                    capture_screenshot(
                        capturer,
                        data,
                        runner,
                        screenshot_dir,
                        timestamp,
                    )

                # Synchronize viewer (real-time pacing)
                viewer.sync()

            # ---------------- Simulation finished ----------------
            perf = runner.get_performance_info()
            print("\nSimulation finished")
            print(f"Total steps      : {perf['step_count']}")
            print(f"Simulation time  : {perf['simulation_time']:.2f}s")
            print(f"Average FPS      : {perf['fps']:.1f}")

    finally:
        # Cleanup (guaranteed)
        print("\nCleaning up resources...")

        # Close renderer
        capturer.cleanup()

        # Close logger
        log_fp.close()

        # Gracefully terminate plotting process
        try:
            plot_queue.put(None)
            plot_process.join(timeout=2.0)
            if plot_process.is_alive():
                plot_process.terminate()
        except Exception:
            pass

        print("Cleanup completed.")
