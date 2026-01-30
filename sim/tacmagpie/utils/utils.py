import os
import datetime

import config
from utils.model_generator import generate_pin_ids


def create_screenshot_directory(base_dir=None):
    """
    Create a directory for storing simulation outputs.

    Parameters
    ----------
    base_dir : str or None
        Base directory where simulation folders are created.
        If None, uses config.DATA_DIR.

    Returns
    -------
    screenshot_dir : str
        Path to the created simulation directory.
    timestamp : str
        Timestamp identifier for the simulation run.
    """
    if base_dir is None:
        base_dir = config.DATA_DIR

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    screenshot_dir = os.path.join(base_dir, f"simulation_{timestamp}")
    os.makedirs(screenshot_dir, exist_ok=True)

    return screenshot_dir, timestamp


def print_simulation_info(model, grid_size):
    """
    Print a summary of the simulation configuration.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model instance.
    grid_size : tuple
        Grid resolution in (x, y, z).
    """
    pin_count = len(generate_pin_ids(grid_size).split())

    print("=" * 60)
    print("MuJoCo Tactile Sensor Simulation")
    print("=" * 60)
    print(f"Grid resolution      : {grid_size}")
    print(f"Number of pinned nodes: {pin_count}")
    print(f"Total joints         : {model.njnt}")
    print(f"Simulation duration  : {config.SIMULATION_DURATION} s")
    print(f"Time step            : {config.TIMESTEP}")
    print("=" * 60)


def cleanup_files(filename, should_delete=True):
    """
    Remove temporary files generated during simulation.

    Parameters
    ----------
    filename : str
        Path to the file to be removed.
    should_delete : bool
        Whether the file should be deleted.
    """
    if should_delete and filename and os.path.exists(filename):
        os.remove(filename)
        print(f"Temporary file removed: {filename}")
