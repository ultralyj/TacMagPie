import time
import numpy as np
import mujoco
import config
from sensors import dmm


class Simulator:
    """
    Core MuJoCo simulation runner.

    This class is responsible for:
    - Stepping the MuJoCo physics
    - Applying control commands
    - Reading sensor data
    - Tracking simulation performance
    - Managing termination conditions
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Initialize the simulator with a MuJoCo model and data.

        Parameters
        ----------
        model : mujoco.MjModel
            Loaded MuJoCo model.
        data : mujoco.MjData
            Runtime simulation data.
        """
        self.model = model
        self.data = data

        self.step_count = 0
        self.start_time = time.time()
        self.timeout = 0

        self._resolve_ids()

        self.initial_position = 0.0
        self.target_position = config.TARGET_POSITION
        self.movement_started = False

        self.movement_speed = config.MOVEMENT_SPEED * config.TIMESTEP

        self.grid_vec = np.zeros(
            (
                config.GRID_SIZE[0],
                config.GRID_SIZE[1],
                config.GRID_SIZE[2] - 1,
                3,
            ),
            dtype=np.float64,
        )

        self.grid_pos = dmm.get_grid_positions()

    def _resolve_ids(self):
        """
        Resolve MuJoCo object IDs for joints, actuators, and sensors.

        These IDs are cached to avoid repeated name lookups during simulation.
        """
        self.joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "indenter_slider"
        )
        self.position_control_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "position_control"
        )

        self.force_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor"
        )
        self.actuator_force_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "actuator_force"
        )
        self.position_sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "indenter_position"
        )
        self.velocity_sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "indenter_velocity"
        )

    def step(self):
        """
        Advance the simulation by one MuJoCo timestep.

        Also updates the internal grid displacement representation.
        """
        mujoco.mj_step(self.model, self.data)

        self.grid_vec = self.data.qpos[:-1].reshape(
            config.GRID_SIZE[0],
            config.GRID_SIZE[1],
            config.GRID_SIZE[2] - 1,
            3,
        )

        self.step_count += 1

    def update_control(self):
        """
        Apply position control to the indenter joint.

        A short delay is used at the beginning to allow the simulation
        to stabilize before motion starts.
        """
        if self.data.time > 0.01:
            self.data.ctrl[self.position_control_id] = self.target_position
        else:
            self.data.ctrl[self.position_control_id] = self.initial_position

    def get_sensor_data(self) -> dict:
        """
        Read all relevant sensor values.

        Returns
        -------
        dict
            Dictionary containing force, actuator force, position, and velocity.
        """
        return {
            "force": self.data.sensordata[self.force_id + 2],
            "actuator_force": self.data.sensordata[self.actuator_force_id + 2],
            "position": self.data.sensordata[self.position_sensor_id + 2],
            "velocity": self.data.sensordata[self.velocity_sensor_id + 2],
        }

    def get_performance_info(self) -> dict:
        """
        Return runtime performance statistics.

        Returns
        -------
        dict
            Elapsed wall time, simulation time, FPS, and step count.
        """
        elapsed = time.time() - self.start_time
        return {
            "elapsed_time": elapsed,
            "simulation_time": self.data.time,
            "fps": self.step_count / elapsed if elapsed > 0 else 0.0,
            "step_count": self.step_count,
        }

    def should_capture_screenshot(self) -> bool:
        """
        Check whether a screenshot should be captured at this step.
        """
        return self.step_count % config.SCREENSHOT_INTERVAL == 0

    def should_update_data(self) -> bool:
        """
        Check whether sensor and grid data should be recorded at this step.
        """
        return self.step_count % config.DATA_UPDATE_INTERVAL == 0

    def is_finished(self) -> bool:
        """
        Determine whether the simulation should terminate.
        """
        return self.data.time >= config.SIMULATION_DURATION
