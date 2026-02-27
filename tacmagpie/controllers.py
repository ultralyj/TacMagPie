# controllers.py
"""
运动控制器模块

类层次
------
MotionController                基类（默认静止）
  ├── PressAndHoldController    简单按压 → 保持
  ├── TrajectoryController      时间-位置插值轨迹
  └── MuJoCoJointController     MuJoCo 直线关节输入接口

工厂函数
--------
build_controller(cfg: SimConfig) -> MotionController
"""

import numpy as np
from config_loader import SimConfig
from typing import Union, List

# ════════════════════════════════════════════════════════════
#  基类
# ════════════════════════════════════════════════════════════

class MotionController:
    def get_velocity(self, t: float, collider) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def get_position(self, t: float, collider):
        return None


# ════════════════════════════════════════════════════════════
#  PressAndHoldController
# ════════════════════════════════════════════════════════════

class PressAndHoldController(MotionController):
    """先以固定速度向下按压，到达目标深度后保持静止。"""

    def __init__(self, press_speed: float, press_depth: float, init_y: float):
        self.press_speed = press_speed
        self.press_depth = press_depth
        self.init_y      = init_y

    def get_velocity(self, t: float, collider) -> np.ndarray:
        current_depth = self.init_y - collider.position[1]
        if current_depth < self.press_depth:
            return np.array([0.0, -self.press_speed, 0.0], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)


# ════════════════════════════════════════════════════════════
#  TrajectoryController
# ════════════════════════════════════════════════════════════

class TrajectoryController(MotionController):
    """时间-位置插值轨迹控制器。"""

    def __init__(self, times: np.ndarray, positions: np.ndarray):
        self.times     = np.asarray(times,     dtype=np.float64)
        self.positions = np.asarray(positions, dtype=np.float32)

    def get_position(self, t: float, collider) -> np.ndarray:
        return np.array([
            np.interp(t, self.times, self.positions[:, i])
            for i in range(3)
        ], dtype=np.float32)

    def get_velocity(self, t: float, collider) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


# ════════════════════════════════════════════════════════════
#  MuJoCoJointController
# ════════════════════════════════════════════════════════════

class MuJoCoJointController(MotionController):
    """
    MuJoCo 直线关节（slide joint）→ MPM 碰撞体位置 映射接口。

    设计思路
    --------
    MuJoCo 中的直线关节（type="slide"）输出一个标量位移值，
    本控制器将三个独立的直线关节（X / Y / Z 轴）映射为碰撞体
    的世界坐标，并支持：
      - 比例缩放（scale）         : mujoco_val * scale = 实际位移 (m)
      - 方向翻转（flip）          : 处理坐标系朝向差异
      - 轴偏移（origin）         : MuJoCo 关节零点对应的 MPM 世界坐标
      - 行程限制（joint_limits） : 硬件/模型关节行程保护
      - 低通滤波（filter_alpha） : 平滑突变输入，防止仿真爆炸

    坐标映射公式（每轴独立）
    --------
        raw   = clamp(mujoco_val, limit_min, limit_max)
        disp  = raw * scale * (-1 if flip else 1)
        world = origin + disp

    参数
    ----
    origin       : (3,) MPM世界坐标中关节零点的位置 (m)
                   通常设为碰撞体的初始中心坐标
    axis_map     : dict，将关节名映射到世界轴索引
                   e.g. {"slide_z": 1} → MuJoCo slide_z 控制 MPM Y 轴
                   支持键: "slide_x", "slide_y", "slide_z"
                   支持值: 0(X), 1(Y), 2(Z)
    scale        : (3,) 或 float，各轴缩放系数（默认 1.0）
                   用于单位换算（如 mm→m 则填 0.001）
    flip         : (3,) bool，各轴是否取反（默认 False）
    joint_limits : {"slide_x": (min,max), ...} 各关节行程上下限 (MuJoCo单位)
                   超限值将被 clamp，None 表示不限制
    filter_alpha : float ∈ (0,1]，低通滤波系数
                   1.0 = 无滤波，0.1 = 强平滑
                   filtered = alpha*new + (1-alpha)*old

    使用示例
    --------
    # 在 MuJoCo 仿真循环中：
    ctrl = MuJoCoJointController(
        origin       = sim.collider_init_center,
        axis_map     = {"slide_x": 0, "slide_z": 1, "slide_y": 2},
        scale        = [1.0, 1.0, 1.0],
        flip         = [False, True, False],   # MuJoCo Y 与 MPM Y 方向相反
        joint_limits = {"slide_x": (-0.05, 0.05),
                        "slide_z": (-0.08, 0.0),   # 只能向下
                        "slide_y": (-0.05, 0.05)},
        filter_alpha = 0.3,
    )

    # MuJoCo step 后：
    ctrl.update({
        "slide_x": mj_data.joint("slide_x").qpos[0],
        "slide_z": mj_data.joint("slide_z").qpos[0],
        "slide_y": mj_data.joint("slide_y").qpos[0],
    })

    # MPM substep 中自动调用 get_position()
    """

    # 支持的关节名 → 默认世界轴映射
    _DEFAULT_AXIS_MAP = {
        "slide_x": 0,
        "slide_y": 1,
        "slide_z": 2,
    }

    def __init__(
        self,
        origin:       np.ndarray,
        axis_map:     dict        = None,
        scale: Union[np.ndarray, float, List[float], None] = None,  # 添加类型注解
        flip: Union[np.ndarray, bool, List[bool], None] = None,     # 添加类型注解
        joint_limits: dict        = None,
        filter_alpha: float       = 1.0,
    ):
        self.origin       = np.array(origin, dtype=np.float64)
        self.axis_map     = axis_map or dict(self._DEFAULT_AXIS_MAP)
        self.joint_limits = joint_limits or {}
        self.filter_alpha = float(np.clip(filter_alpha, 1e-6, 1.0))

        # 缩放与翻转（per-axis，索引 0/1/2 = X/Y/Z）
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

        # 当前目标世界坐标（初始 = origin）
        self._target   = self.origin.copy()
        # 低通滤波后的平滑坐标
        self._filtered = self.origin.copy()

        # 最新一帧的原始关节值（供调试）
        self._last_raw: dict[str, float] = {}

    # ── 主更新接口 ───────────────────────────────────────────

    def update(self, joint_values: dict[str, float]):
        """
        接收 MuJoCo 关节值字典并更新内部目标位置。

        参数
        ----
        joint_values : { 关节名: qpos 标量 }
                       只需包含 axis_map 中定义的关节，多余的键忽略。

        典型调用时机：MuJoCo mj_step() 执行后，MPM substep 之前。
        """
        self._last_raw = {k: float(v) for k, v in joint_values.items()}
        new_target = self.origin.copy()

        for joint_name, axis_idx in self.axis_map.items():
            if joint_name not in joint_values:
                continue

            raw = float(joint_values[joint_name])

            # 1. 行程限制
            if joint_name in self.joint_limits:
                lo, hi = self.joint_limits[joint_name]
                if lo is not None:
                    raw = max(raw, float(lo))
                if hi is not None:
                    raw = min(raw, float(hi))

            # 2. 缩放 + 翻转
            sign = -1.0 if self._flip[axis_idx] else 1.0
            disp = raw * self._scale[axis_idx] * sign

            # 3. 叠加到对应轴
            new_target[axis_idx] = self.origin[axis_idx] + disp

        self._target = new_target

        # 4. 低通滤波
        alpha = self.filter_alpha
        self._filtered = alpha * self._target + (1.0 - alpha) * self._filtered

    # ── 控制器接口实现 ───────────────────────────────────────

    def get_position(self, t: float, collider) -> np.ndarray:
        """返回平滑后的目标世界坐标（float32）。"""
        return self._filtered.astype(np.float32)

    def get_velocity(self, t: float, collider) -> np.ndarray:
        """位置模式下速度由 MPM 内部差分，此处返回零。"""
        return np.zeros(3, dtype=np.float32)

    # ── 调试接口 ─────────────────────────────────────────────

    def get_state(self) -> dict:
        """
        返回当前控制器内部状态，便于调试和日志记录。

        返回字段
        --------
        raw_joints  : 最新原始关节值
        target      : 未滤波目标位置
        filtered    : 平滑后目标位置（实际输出）
        origin      : 关节零点世界坐标
        """
        return {
            "raw_joints" : dict(self._last_raw),
            "target"     : self._target.copy(),
            "filtered"   : self._filtered.copy(),
            "origin"     : self.origin.copy(),
        }

    def reset(self, origin: np.ndarray = None):
        """
        重置控制器状态（换轨迹或重新仿真时调用）。

        参数
        ----
        origin : 新的零点坐标，None 则保持原零点
        """
        if origin is not None:
            self.origin = np.array(origin, dtype=np.float64)
        self._target   = self.origin.copy()
        self._filtered = self.origin.copy()
        self._last_raw = {}


# ════════════════════════════════════════════════════════════
#  工厂函数
# ════════════════════════════════════════════════════════════

def build_controller(cfg: SimConfig) -> MotionController:
    """
    根据 config.yaml 中的 controller 段构建运动控制器。

    支持的 type 值
    --------------
    press_and_hold  → PressAndHoldController
    trajectory      → TrajectoryController
    mujoco_joint    → MuJoCoJointController
    none / 其他     → 静止基类
    """
    ctrl_cfg  = cfg.controller_cfg
    ctrl_type = ctrl_cfg.get("type", "none")
    init_y    = float(cfg.collider_init_center[1])

    if ctrl_type == "press_and_hold":
        c = ctrl_cfg["press_and_hold"]
        return PressAndHoldController(
            press_speed = float(c["press_speed"]),
            press_depth = float(c["press_depth"]),
            init_y      = init_y,
        )

    elif ctrl_type == "trajectory":
        c     = ctrl_cfg["trajectory"]
        t_arr = np.linspace(float(c["t_start"]), float(c["t_end"]),
                            int(c["n_points"]))
        pos       = np.zeros((len(t_arr), 3), dtype=np.float32)
        pos[:, 0] = cfg.domain * 0.5
        pos[:, 2] = cfg.domain * 0.5
        depth     = float(c["press_depth"])
        t_press   = float(c["press_time"])
        pos[:, 1] = init_y - np.clip(t_arr / t_press, 0.0, 1.0) * depth
        return TrajectoryController(t_arr, pos)

    elif ctrl_type == "mujoco_joint":
        c = ctrl_cfg["mujoco_joint"]

        # axis_map: YAML 中写 {slide_x: 0, slide_z: 1} 等
        axis_map = {str(k): int(v)
                    for k, v in c.get("axis_map", {}).items()} or None

        # joint_limits: {slide_x: [min, max], ...}
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
            origin       = cfg.collider_init_center,
            axis_map     = axis_map,
            scale        = c.get("scale",        1.0),
            flip         = c.get("flip",         False),
            joint_limits = joint_limits,
            filter_alpha = float(c.get("filter_alpha", 1.0)),
        )

    else:
        return MotionController()
