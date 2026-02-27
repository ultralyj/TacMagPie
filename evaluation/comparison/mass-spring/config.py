# config.py
import platform
from dataclasses import dataclass, field
from typing import Tuple, List
import math


# -----------------------------
# Simulation & Model Parameters
# -----------------------------

@dataclass
class SimulationConfig:
    MODEL_TEMPLATE_FILE: str = "./model/template.xml"
    MODEL_OUTPUT_FILE: str = "./model/dynamic.xml"

    GRID_SIZE: Tuple[int, int, int] = (17, 17, 4)
    GRID_SPACING: float = 0.001

    TIMESTEP: float = 1e-5
    SIMULATION_DURATION: float = 3.0

    MOVEMENT_SPEED: float = 0.01
    TARGET_POSITION: float = 0.006
    EARLY_STOP_FLAG: bool = True


# -----------------------------
# Rendering & Screenshot Config
# -----------------------------

@dataclass
class RenderConfig:
    SCREENSHOT_FREQUENCY: float = 0.001
    SCREENSHOT_WIDTH: int = 1920
    SCREENSHOT_HEIGHT: int = 1080
    SCREENSHOT_QUALITY: int = 100

    VIDEO_FPS: int = 100
    DELETE_FRAMES_AFTER_VIDEO: bool = False

    def screenshot_interval(self, timestep: float) -> int:
        return max(1, int(self.SCREENSHOT_FREQUENCY / timestep))


# -----------------------------
# Data Logging Config
# -----------------------------

@dataclass
class DataConfig:
    DATA_UPDATE_FREQUENCY: float = 0.0001
    DATA_DIR: str = "data"

    def data_update_interval(self, timestep: float) -> int:
        return max(1, int(self.DATA_UPDATE_FREQUENCY / timestep))


# -----------------------------
# Physics Parameters
# -----------------------------

@dataclass
class PhysicsConfig:
    INDENTER_RADIUS: float = 0.05
    INDENTER_DENSITY: float = 1000

    FLEXCOMP_MASS_MULTIPLIER: float = field(init=False)

    JOINT_DAMPING: float = 1.0
    JOINT_STIFFNESS: float = 0.0

    INDENTER_LIMIT_L: float = 0.015
    INDENTER_LIMIT_H: float = 0.035

    INDENTER_X: float = 0.0
    INDENTER_Y: float = 0.0

    def __post_init__(self):
        self.FLEXCOMP_MASS_MULTIPLIER = 5.2


# -----------------------------
# Magnetic Sensor Parameters
# -----------------------------

@dataclass
class SensorConfig:
    SENSOR_NUMBER: int = 5
    SENSOR_ARRAY: List[List[float]] = field(
        default_factory=lambda: [
            [0.004,  0.004, 0],
            [-0.004, 0.004, 0],
            [0.004, -0.004, 0],
            [-0.004, -0.004, 0],
            [0.0,    0.0,   0],
        ]
    )

    MU0: float = 1e-8
    B0: float = 1.0


# -----------------------------
# External Tools
# -----------------------------

def _detect_ffmpeg_path() -> str:
    if platform.system() == "Windows":
        return r"C:\Users\admin\AppData\Local\ffmpeg\bin\ffmpeg.exe"
    return "ffmpeg"


# -----------------------------
# Global Config Object
# -----------------------------

_sim = SimulationConfig()
_render = RenderConfig()
_data = DataConfig()
_phys = PhysicsConfig()
_sensor = SensorConfig()

FFMPEG_PATH = _detect_ffmpeg_path()

# -----------------------------
# Backward-Compatible Exports
# -----------------------------

MODEL_TEMPLATE_FILE = _sim.MODEL_TEMPLATE_FILE
MODEL_OUTPUT_FILE = _sim.MODEL_OUTPUT_FILE

GRID_SIZE = _sim.GRID_SIZE
GRID_SPACING = _sim.GRID_SPACING

TIMESTEP = _sim.TIMESTEP
SIMULATION_DURATION = _sim.SIMULATION_DURATION

MOVEMENT_SPEED = _sim.MOVEMENT_SPEED
TARGET_POSITION = _sim.TARGET_POSITION
EARLY_STOP_FLAG = _sim.EARLY_STOP_FLAG

SCREENSHOT_FREQUENCY = _render.SCREENSHOT_FREQUENCY
SCREENSHOT_INTERVAL = _render.screenshot_interval(TIMESTEP)
SCREENSHOT_WIDTH = _render.SCREENSHOT_WIDTH
SCREENSHOT_HEIGHT = _render.SCREENSHOT_HEIGHT
SCREENSHOT_QUALITY = _render.SCREENSHOT_QUALITY

VIDEO_FPS = _render.VIDEO_FPS
DELETE_FRAMES_AFTER_VIDEO = _render.DELETE_FRAMES_AFTER_VIDEO

DATA_UPDATE_FREQUENCY = _data.DATA_UPDATE_FREQUENCY
DATA_UPDATE_INTERVAL = _data.data_update_interval(TIMESTEP)
DATA_DIR = _data.DATA_DIR

INDENTER_RADIUS = _phys.INDENTER_RADIUS
INDENTER_DENSITY = _phys.INDENTER_DENSITY
FLEXCOMP_MASS_MULTIPLIER = _phys.FLEXCOMP_MASS_MULTIPLIER * GRID_SPACING * GRID_SPACING
JOINT_DAMPING = _phys.JOINT_DAMPING
JOINT_STIFFNESS = _phys.JOINT_STIFFNESS
INDENTER_LIMIT_L = _phys.INDENTER_LIMIT_L
INDENTER_LIMIT_H = _phys.INDENTER_LIMIT_H
INDENTER_X = _phys.INDENTER_X
INDENTER_Y = _phys.INDENTER_Y

SENSOR_NUMBER = _sensor.SENSOR_NUMBER
SENSOR_ARRAY = _sensor.SENSOR_ARRAY
MU0 = _sensor.MU0
B0 = _sensor.B0
