"""
Main entry point for the MuJoCo simulation pipeline.

This script performs the following steps:
1. Configure simulation parameters.
2. Generate and save a MuJoCo model XML.
3. Load the model and run the simulation with a viewer.
4. Merge collected simulation data.
5. Generate a video from rendered frames.

The script is designed to be the top-level executable for the project.
"""

import os
import mujoco

import config
from simulation.runner import run_simulation_with_viewer
from utils.model_generator import create_model_xml, save_model_to_file
from utils.video_generator import generate_video_from_frames
from utils.data_merger import merge_npy_files
from utils.utils import (
    create_screenshot_directory,
    print_simulation_info,
    cleanup_files,
)


def setup_simulation_config():
    """
    Initialize and override key simulation parameters.

    This function modifies global values in the config module
    to control the indenter position and target location.
    """
    config.INDENTER_X = 0
    config.INDENTER_Y = 0
    config.TARGET_POSITION = 0.006


def build_and_load_model():
    """
    Generate the MuJoCo model XML, save it to disk, and load it.

    Returns
    -------
    model : mujoco.MjModel
        Loaded MuJoCo model.
    data : mujoco.MjData
        Runtime data associated with the model.
    model_file : str
        Path to the saved XML model file.
    """
    xml_content = create_model_xml(config.GRID_SIZE)
    model_file = save_model_to_file(xml_content)

    print(f"Model XML saved to: {os.path.abspath(model_file)}")

    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)

    return model, data, model_file


def run_simulation_pipeline(model, data, screenshot_dir, timestamp):
    """
    Execute the main simulation loop with visualization enabled.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model to simulate.
    data : mujoco.MjData
        Runtime data for the simulation.
    screenshot_dir : str
        Directory where rendered frames and data are stored.
    timestamp : str
        Timestamp identifier for this simulation run.
    """
    print_simulation_info(model, config.GRID_SIZE)
    run_simulation_with_viewer(model, data, screenshot_dir, timestamp)


def postprocess_results(screenshot_dir, timestamp):
    """
    Merge collected data files and generate a visualization video.

    Parameters
    ----------
    screenshot_dir : str
        Directory containing saved frames and data.
    timestamp : str
        Timestamp identifier for this simulation run.
    """
    print("\nMerging grid data...")
    success = merge_npy_files(screenshot_dir, timestamp)
    print("✓ Data merge completed" if success else "✗ Data merge failed")

    print("\nGenerating video...")
    success = generate_video_from_frames(screenshot_dir, timestamp)
    print("✓ Video generation completed" if success else "✗ Video generation failed")


def main():
    """
    Main execution function for the simulation workflow.
    """
    try:
        screenshot_dir, timestamp = create_screenshot_directory()
        print(f"Simulation outputs will be saved to: {screenshot_dir}")

        setup_simulation_config()
        model, data, model_file = build_and_load_model()

        run_simulation_pipeline(model, data, screenshot_dir, timestamp)
        postprocess_results(screenshot_dir, timestamp)

    except Exception as e:
        print(f"✗ An error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cleanup_files(config.MODEL_OUTPUT_FILE, should_delete=True)


if __name__ == "__main__":
    main()
