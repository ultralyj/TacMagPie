import mujoco
import mujoco.viewer
import numpy as np
import time
import datetime
import os
import config
from screenshot_capture import ScreenshotCapturer
import magnetic_flux

class Simulatior:
    """模拟运行器"""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.step_count = 0
        self.start_time = time.time()
        
        # 获取传感器和执行器ID
        self._get_sensor_actuator_ids()
        
        # 运动控制参数
        self.movement_started = False
        self.initial_position = 0.035
        self.target_position = 0.016
        self.movement_speed = config.MOVEMENT_SPEED
        
        self.grid_vec = np.zeros((config.GRID_SIZE[0], config.GRID_SIZE[1], config.GRID_SIZE[2]-1, 3))
        print("模拟运行器初始化完成")

    def _get_sensor_actuator_ids(self):
        """获取传感器和执行器ID"""
        self.joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "soft")
        self.position_control_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "position_control")
        
        # 传感器ID
        self.actuator_force_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "actuator_force")
        self.position_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "indenter_position")
        self.velocity_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "indenter_velocity")
        
        # 验证所有ID
        self._validate_ids()

    def _validate_ids(self):
        """验证所有ID是否有效"""
        ids_to_check = [
            (self.joint_id, "关节 'soft'"),
            (self.position_control_id, "执行器 'position_control'"),
            (self.actuator_force_id, "传感器 'actuator_force'"),
            (self.position_sensor_id, "传感器 'indenter_position'"),
            (self.velocity_sensor_id, "传感器 'indenter_velocity'")
        ]
        
        for id_val, name in ids_to_check:
            if id_val == -1:
                print(f"警告: 未找到 {name}")

    def get_sensor_data(self):
        """获取传感器数据"""
        
        actuator_force = self.data.sensordata[self.actuator_force_id]
        position = self.data.sensordata[self.position_sensor_id]
        velocity = self.data.sensordata[self.velocity_sensor_id]
        
        return {
            'actuator_force': actuator_force,
            'position': position,
            'velocity': velocity,
        }

    def update_control(self):
        """更新控制逻辑"""
        current_time = self.data.time
        
        # 延迟开始运动，让系统稳定
        if current_time > 0.1 and not self.movement_started:
            self.movement_started = True
            print("开始慢速下压...")
        
        if self.movement_started:
            # 缓慢改变目标位置
            if self.data.qpos[self.joint_id] > self.target_position:
                self.data.ctrl[self.position_control_id] -= self.movement_speed
            else:
                self.data.ctrl[self.position_control_id] = self.target_position

    def should_capture_screenshot(self):
        """判断是否应该捕获截图"""
        return self.step_count % config.SCREENSHOT_INTERVAL == 0

    def should_print_detailed_status(self):
        """判断是否应该打印详细状态"""
        return self.step_count % 1000 == 0

    def get_performance_info(self):
        """获取性能信息"""
        elapsed = time.time() - self.start_time
        sim_time = self.data.time
        fps = self.step_count / elapsed if elapsed > 0 else 0
        
        return {
            'elapsed_time': elapsed,
            'simulation_time': sim_time,
            'fps': fps,
            'step_count': self.step_count
    }

    def step(self):
        """执行一步模拟"""
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

    def is_finished(self):
        """检查模拟是否完成"""
        return self.data.time >= config.SIMULATION_DURATION

def run_simulation_with_viewer(model, data, screenshot_dir, timestamp):
    """使用查看器运行模拟"""
    runner = Simulatior(model, data)
    capturer = ScreenshotCapturer(model)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 配置查看器
            _setup_viewer(viewer)
            
            # 初始化状态
            _initialize_simulation_state(data, runner)
            
            print("模拟开始...")
            print("按ESC退出")
            
            while viewer.is_running() and not runner.is_finished():
                # 执行模拟步骤
                runner.step()
                
                # 更新控制逻辑
                runner.update_control()
                
                # 获取传感器数据
                sensor_data = runner.get_sensor_data()
                runner.grid_vec = data.qpos[:-1].reshape(config.GRID_SIZE[0], config.GRID_SIZE[1], config.GRID_SIZE[2]-1, 3)
                # 捕获截图
                if runner.should_capture_screenshot():
                    _capture_screenshot(capturer, data, runner, screenshot_dir, timestamp, viewer)
                
                if runner.should_print_detailed_status():
                    _print_detailed_status(data, runner, sensor_data)
                
                # 同步查看器
                viewer.sync()
            
            # 模拟完成
            performance = runner.get_performance_info()
            print(f"\n模拟完成!")
            print(f"总步数: {performance['step_count']}")
            print(f"模拟时间: {performance['simulation_time']:.2f}s")
            print(f"真实时间: {performance['elapsed_time']:.2f}s")
            print(f"平均FPS: {performance['fps']:.1f}")
            
    finally:
        # 清理资源
        capturer.cleanup()

def _setup_viewer(viewer):
    """配置查看器"""
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 0.2
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -20
    viewer.cam.lookat = [0, 0, 0.03]

def _initialize_simulation_state(data, runner):
    """初始化模拟状态"""
    data.qpos[runner.joint_id] = runner.initial_position
    data.ctrl[runner.position_control_id] = runner.initial_position

def _capture_screenshot(capturer, data, runner, screenshot_dir, timestamp, viewer):
    """捕获截图"""
    try:
    # 同步相机参数
        # camera_params = {
        # 'azimuth': viewer.cam.azimuth,
        # 'elevation': viewer.cam.elevation,
        # 'distance': viewer.cam.distance,
        # 'lookat': viewer.cam.lookat.copy()
        # }

        # 捕获图像
        image = capturer.capture_frame(data)
        if image is not None:
            filename = f'frame_{timestamp}_{int(data.time*1000):06d}.png'
            filepath = os.path.join(screenshot_dir, filename) 
            capturer.save_frame(image, filepath)

    except Exception as e:
        print(f"✗ 截图捕获失败: {e}")

def _print_detailed_status(data, runner, sensor_data):
    """打印详细状态信息"""
    performance = runner.get_performance_info()
    
    elas_deformation = np.sum(runner.grid_vec[:,:,:,2])

    print(f"\n[{performance['elapsed_time']:.2f}] 模拟时间:{performance['simulation_time']:.2f} "
        f"深度:{data.qpos[runner.joint_id]:.3f}, 弹性变形:{elas_deformation:.4f}", 
        f"执行器力: {sensor_data['actuator_force']:.2f}N, "
    f"速度: {sensor_data['velocity']:.4f}m/s")