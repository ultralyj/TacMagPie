"""Motion controller module for collider trajectory control.

Provides various controller implementations for defining collider motion in
MPM simulations, including static, press-and-hold, trajectory-based, and
MuJoCo joint interface controllers.

Classes
-------
MotionController
    Base class (stationary by default).
PressAndHoldController
    Simple press-and-hold motion.
TrajectoryController
    Time-position interpolation trajectory.
MuJoCoJointController
    MuJoCo slide joint input interface.

Factory Function
----------------
build_controller(cfg: SimConfig) -> MotionController
    Creates controller instance based on configuration.
"""

import numpy as np
from config_loader import SimConfig
from typing import Union, List


class MotionController:
    """Base motion controller class with stationary behavior."""

    def get_velocity(self, t: float, collider) -> np.ndarray:
        """Get velocity at time t.
        
        Args:
            t: Current simulation time.
            collider: Collider object.
            
        Returns:
            Velocity vector (3D).
        """
        return np.zeros(3, dtype=np.float32)

    def get_position(self, t: float, collider):
        """Get position at time t.
        
        Args:
            t: Current simulation time.
            collider: Collider object.
            
        Returns:
            Position vector (3D) or None for velocity-based control.
        """
        return None


class PressAndHoldController(MotionController):
    """Press downward at constant speed until target depth, then hold."""

    def __init__(self, press_speed: float, press_depth: float, init_y: float):
        """Initialize press-and-hold controller.
        
        Args:
            press_speed: Downward pressing speed (m/s).
            press_depth: Target pressing depth (m).
            init_y: Initial Y position of collider.
        """
        self.press_speed = press_speed
        self.press_depth = press_depth
        self.init_y = init_y

    def get_velocity(self, t: float, collider) -> np.ndarray:
        current_depth = self.init_y - collider.position[1]
        if current_depth < self.press_depth:
            return np.array([0.0, -self.press_speed, 0.0], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)


class TrajectoryController(MotionController):
    """Time-position interpolation trajectory controller."""

    def __init__(self, times: np.ndarray, positions: np.ndarray):
        """Initialize trajectory controller.
        
        Args:
            times: Time points array.
            positions: Position array (N×3) corresponding to time points.
        """
        self.times = np.asarray(times, dtype=np.float64)
        self.positions = np.asarray(positions, dtype=np.float32)

    def get_position(self, t: float, collider) -> np.ndarray:
        return np.array([
            np.interp(t, self.times, self.positions[:, i])
            for i in range(3)
        ], dtype=np.float32)

    def get_velocity(self, t: float, collider) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


class MuJoCoJointController(MotionController):
    """MuJoCo slide joint to MPM collider position mapping interface.
    
    Maps three independent MuJoCo slide joints (X/Y/Z axes) to collider world
    coordinates with support for scaling, flipping, offsets, limits, and filtering.
    Coordinate mapping formula (per axis):
        raw = clamp(mujoco_val, limit_min, limit_max)
        disp = raw * scale * (-1 if flip else 1)
        world = origin + disp
    """

    _DEFAULT_AXIS_MAP = {
        "slide_x": 0,
        "slide_y": 1,
        "slide_z": 2,
    }

    def __init__(
        self,
        origin: np.ndarray,
        axis_map: dict = None,
        scale: Union[np.ndarray, float, List[float], None] = None,
        flip: Union[np.ndarray, bool, List[bool], None] = None,
        joint_limits: dict = None,
        filter_alpha: float = 1.0,
    ):
        """Initialize MuJoCo joint controller.
        
        Args:
            origin: Joint zero position in MPM world coordinates (m).
            axis_map: Maps joint names to world axis indices.
                     e.g., {"slide_z": 1} maps MuJoCo slide_z to MPM Y axis.
            scale: Per-axis scaling factors (default 1.0).
            flip: Per-axis direction flip flags (default False).
            joint_limits: Joint travel limits {"slide_x": (min, max), ...}.
            filter_alpha: Low-pass filter coefficient in (0, 1].         1.0 = no filtering, lower = stronger smoothing.
        """
        self.origin = np.array(origin, dtype=np.float64)
        self.axis_map = axis_map or dict(self._DEFAULT_AXIS_MAP)
        self.joint_limits = joint_limits or {}
        self.filter_alpha = float(np.clip(filter_alpha, 1e-6, 1.0))

        if scale is None:
            self._scale = np.ones(3, dtype=np.float64)
        else:
            self._scale = np.broadcast_to(
                np.asarray(scale, dtype=np.float64), (3,)
            ).copy()

        if flip is None:
            self._flip = np.zeros(3, dtype=bool)
        else:
            self._flip = np.broadcast_to(
                np.asarray(flip, dtype=bool), (3,)
            ).copy()

        self._target = self.origin.copy()
        self._filtered = self.origin.copy()
        self._last_raw: dict[str, float] = {}

    def update(self, joint_values: dict[str, float]):
        """Update target position from MuJoCo joint values.
        
        Args:
            joint_values: Dictionary mapping joint names to qpos values.
                         Only joints defined in axis_map are used.
        """
        self._last_raw = {k: float(v) for k, v in joint_values.items()}
        new_target = self.origin.copy()

        for joint_name, axis_idx in self.axis_map.items():
            if joint_name not in joint_values:
                continue

            raw = float(joint_values[joint_name])

            if joint_name in self.joint_limits:
                lo, hi = self.joint_limits[joint_name]
                if lo is not None:
                    raw = max(raw, float(lo))
                if hi is not None:
                    raw = min(raw, float(hi))

            sign = -1.0 if self._flip[axis_idx] else 1.0
            disp = raw * self._scale[axis_idx] * sign
            new_target[axis_idx] = self.origin[axis_idx] + disp

        self._target = new_target
        alpha = self.filter_alpha
        self._filtered = alpha * self._target + (1.0 - alpha) * self._filtered

    def get_position(self, t: float, collider) -> np.ndarray:
        return self._filtered.astype(np.float32)

    def get_velocity(self, t: float, collider) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def get_state(self) -> dict:
        """Get current controller internal state for debugging.
        
        Returns:
            Dictionary containing raw_joints, target, filtered, and origin.
        """
        return {
            "raw_joints": dict(self._last_raw),
            "target": self._target.copy(),
            "filtered": self._filtered.copy(),
            "origin": self.origin.copy(),
        }

    def reset(self, origin: np.ndarray = None):
        """Reset controller state.
        
        Args:
            origin: New zero position, or None to keep current origin.
        """
        if origin is not None:
            self.origin = np.array(origin, dtype=np.float64)
        self._target = self.origin.copy()
        self._filtered = self.origin.copy()
        self._last_raw = {}


def build_controller(cfg: SimConfig) -> MotionController:
    """Build motion controller from configuration.
    
    Args:
        cfg: Simulation configuration object.
        
    Returns:
        Controller instance based on cfg.controller_cfg["type"].
        Supported types: "press_and_hold", "trajectory", "mujoco_joint", "none".
    """
    ctrl_cfg = cfg.controller_cfg
    ctrl_type = ctrl_cfg.get("type", "none")
    init_y = float(cfg.collider_init_center[1])

    if ctrl_type == "press_and_hold":
        c = ctrl_cfg["press_and_hold"]
        return PressAndHoldController(
            press_speed=float(c["press_speed"]),
            press_depth=float(c["press_depth"]),
            init_y=init_y,
        )

    elif ctrl_type == "trajectory":
        c = ctrl_cfg["trajectory"]
        t_arr = np.linspace(float(c["t_start"]), float(c["t_end"]),
                int(c["n_points"]))
        pos = np.zeros((len(t_arr), 3), dtype=np.float32)
        pos[:, 0] = cfg.domain * 0.5
        pos[:, 2] = cfg.domain * 0.5
        depth = float(c["press_depth"])
        t_press = float(c["press_time"])
        pos[:, 1] = init_y - np.clip(t_arr / t_press, 0.0, 1.0) * depth
        return TrajectoryController(t_arr, pos)

    elif ctrl_type == "mujoco_joint":
        c = ctrl_cfg["mujoco_joint"]

        axis_map = {str(k): int(v)
                   for k, v in c.get("axis_map", {}).items()} or None

        raw_limits = c.get("joint_limits", {})
        joint_limits = {}
        for jname, lims in raw_limits.items():
            if lims is None:
                joint_limits[str(jname)] = (None, None)
            else:
                joint_limits[str(jname)] = (
                    float(lims[0]) if lims[0] is not None else None,
                    float(lims[1]) if lims[1] is not None else None,
                )

        return MuJoCoJointController(
            origin=cfg.collider_init_center,
            axis_map=axis_map,
            scale=c.get("scale", 1.0),
            flip=c.get("flip", False),
            joint_limits=joint_limits,
            filter_alpha=float(c.get("filter_alpha", 1.0)),
        )

    else:
        return MotionController()
