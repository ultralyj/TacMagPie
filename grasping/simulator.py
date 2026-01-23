import mujoco
import mujoco.viewer
import numpy as np
import time
import datetime
import os
import config
from screenshot_capture import ScreenshotCapturer
import magnetic_flux
import glob
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt

class Simulatior:
    """模拟运行器"""
    # ... (这部分代码保持您原样即可，为了节省篇幅此处省略内容) ...
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.step_count = 0
        self.timeout = 0
        self.start_time = time.time()
        self._get_sensor_actuator_ids()
        self.movement_started = False
        self.initial_position = 0.0
        self.target_position = config.TARGET_POSITION
        self.movement_speed = config.MOVEMENT_SPEED * config.TIMESTEP
        print(self.movement_speed, self.target_position)
        self.grid_vec = np.zeros((config.GRID_SIZE[0], config.GRID_SIZE[1], config.GRID_SIZE[2]-1, 3))
        self.grid_pos = magnetic_flux.get_grid_positions()
        print("模拟运行器初始化完成")

    def _get_sensor_actuator_ids(self):
        self.joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "indenter_slider")
        self.position_control_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "position_control")
        self.force_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor")
        self.actuator_force_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "actuator_force")
        self.position_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "indenter_position")
        self.velocity_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "indenter_velocity")
        self._validate_ids()

    def _validate_ids(self):
        ids_to_check = [
            (self.joint_id, "关节 'indenter_slider'"),
            (self.position_control_id, "执行器 'position_control'"),
            (self.force_id, "传感器 'force_sensor'"),
            (self.actuator_force_id, "传感器 'actuator_force'"),
            (self.position_sensor_id, "传感器 'indenter_position'"),
            (self.velocity_sensor_id, "传感器 'indenter_velocity'")
        ]
        for id_val, name in ids_to_check:
            if id_val == -1:
                print(f"警告: 未找到 {name}")

    def get_sensor_data(self):
        force = self.data.sensordata[self.force_id+2]
        actuator_force = self.data.sensordata[self.actuator_force_id+2]
        position = self.data.sensordata[self.position_sensor_id+2]
        velocity = self.data.sensordata[self.velocity_sensor_id+2]
        return {
            'force' : force - actuator_force,
            'actuator_force': actuator_force,
            'position': position,
            'velocity': velocity,
        }

    def update_control(self):
        current_time = self.data.time
        if current_time > 0.01 and not self.movement_started:
            self.movement_started = True
            print("开始慢速下压...")
        elif self.movement_started:
            
            self.data.ctrl[self.position_control_id] = self.target_position
        else:
            self.data.ctrl[self.position_control_id] = 0.0
        # print(f"控制目标位置: {self.data.ctrl[self.position_control_id]:.4f}")

    def should_capture_screenshot(self):
        return self.step_count % config.SCREENSHOT_INTERVAL == 0

    def should_update_data(self):
        return self.step_count % config.DATA_UPDATE_INTERVAL == 0
    
    def get_performance_info(self):
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
        mujoco.mj_step(self.model, self.data)
        self.grid_vec = self.data.qpos[:-1].reshape(config.GRID_SIZE[0], config.GRID_SIZE[1], config.GRID_SIZE[2]-1, 3)
        self.step_count += 1
    def is_finished(self):
        if(config.EARLY_STOP_FLAG):
            depth = self.data.qpos[self.joint_id]
            if depth <= self.target_position + 0.00005 and self.timeout>20000:
                return True
            else:
                self.timeout+=1
        else:
            return self.data.time >= config.SIMULATION_DURATION

# ==========================================
# 3. 修改：运行逻辑，加入多进程支持
# ==========================================
def run_simulation_with_viewer(model, data, screenshot_dir, timestamp):
    """使用查看器运行模拟"""
    runner = Simulatior(model, data)
    capturer = ScreenshotCapturer(model)
    mag_data = np.zeros((config.SENSOR_NUMBER,3))
    
    # ---  Multiprocessing Setup ---
    # 创建队列
    plot_queue = Queue() 
    # 创建并启动绘图进程
    plot_process = Process(target=live_plot_process, args=(plot_queue, config.SENSOR_NUMBER))
    plot_process.start()
    # -----------------------------

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 配置查看器
            _setup_viewer(viewer)
            
            # 初始化状态
            _initialize_simulation_state(data, runner)
            
            log_file_name = f'data_{timestamp}.csv'
            log_file_path = os.path.join('data', log_file_name) 
            
            log_fp = open(log_file_path,'w+',encoding='utf-8')
            grid_file_name = f'grid_{timestamp}.csv'
            grid_file_path = os.path.join('data', grid_file_name) 
            grid_fp = open(grid_file_path,'w+',encoding='utf-8')
            print("模拟开始...")
            print("按ESC退出")
            sensor_baseline = np.zeros((config.SENSOR_NUMBER,3))
            
            # --- 可视化限制 ---
            last_plot_time = 0
            plot_interval = 0.05 # 限制向绘图进程发送数据的频率 (秒)
            # -----------------

            while viewer.is_running() and not runner.is_finished():
                # 执行模拟步骤
                runner.step()
                
                # 更新控制逻辑
                runner.update_control()
                
                # 获取传感器数据
                sensor_data = runner.get_sensor_data()
                if runner.should_update_data():
                    if(runner.get_performance_info()['simulation_time']<0.01):
                        sensor_baseline = update_sensor_baseline(runner)
                    else:
                        mag_data = get_magnetic_data(runner, sensor_baseline)
                        
                        # --- 发送数据到绘图进程 ---
                        current_real_time = runner.get_performance_info()['simulation_time']
                        # 这里我们利用 should_capture_screenshot 的节奏发送数据，或者单独判断时间
                        # 由于截图通常频率不高，直接在这里发送是合适的
                        
                        # 深拷贝副本发送，防止numpy引用问题
                        plot_queue.put((current_real_time, mag_data.copy())) 
                        print(f"发送数据到绘图进程，时间: {current_real_time:.2f}s, plot_queue大小: {plot_queue.qsize()}")
                        # -----------------------

                    _print_detailed_status(data, runner, sensor_data)
                    update_logger(log_fp,data, runner, sensor_data, mag_data)
                    save_grid_vec(data, runner, screenshot_dir, timestamp)
                    
                # 捕获截图 & 计算磁通量
                if runner.should_capture_screenshot():
                    _capture_screenshot(capturer, data, runner, screenshot_dir, timestamp, viewer)
                    
                    
                
                # 同步查看器
                viewer.sync()
            
            # 模拟完成
            performance = runner.get_performance_info()
            print(f"\n模拟完成!")
            print(f"总步数: {performance['step_count']}")
            print(f"模拟时间: {performance['simulation_time']:.2f}s")
            # ...
            
    finally:
        # 清理资源
        capturer.cleanup()
        
        # --- 清理多进程 ---
        plot_queue.put(None) # 发送结束信号
        plot_process.join(timeout=2)
        if plot_process.is_alive():
            plot_process.terminate()
        print("绘图进程已关闭")
        # ----------------

# 辅助函数保持不变
def _setup_viewer(viewer):
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 0.2
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -20
    viewer.cam.lookat = [0, 0, 0.03]

def _initialize_simulation_state(data, runner):
    data.qpos[runner.joint_id] = runner.initial_position
    data.ctrl[runner.position_control_id] = runner.initial_position

def _capture_screenshot(capturer, data, runner, screenshot_dir, timestamp, viewer):
    try:
        image = capturer.capture_frame(data)
        if image is not None:
            filename = f'frame_{timestamp}_{int(data.time*1000):06d}.png'
            filepath = os.path.join(screenshot_dir, filename) 
            capturer.save_frame(image, filepath)
    except Exception as e:
        print(f"✗ 截图捕获失败: {e}")

def _print_detailed_status(data, runner, sensor_data):
    performance = runner.get_performance_info()
    elas_deformation = np.sum(runner.grid_vec[:,:,:,2])
    print(f"[{performance['elapsed_time']:.2f}] 模拟时间:{performance['simulation_time']:.2f} "
        f"深度:{data.qpos[runner.joint_id]:.3f}, 弹性变形:{elas_deformation:.4f}", 
        f"执行器力: {sensor_data['actuator_force']:.2f}, {sensor_data['force']:.2f}N, "
    f"速度: {sensor_data['velocity']:.4f}m/s")

def update_sensor_baseline(runner):
    baseline = np.zeros((config.SENSOR_NUMBER,3))
    for i in range(config.SENSOR_NUMBER):
        baseline[i] = magnetic_flux.magnetic_flux_3axis(config.SENSOR_ARRAY[i],runner.grid_vec,runner.grid_pos)
    return baseline

def get_magnetic_data(runner,baseline):
    magdata = np.zeros((config.SENSOR_NUMBER,3))
    for i in range(config.SENSOR_NUMBER):
        b0 = baseline[i]
        magdata[i] = magnetic_flux.magnetic_flux_3axis(config.SENSOR_ARRAY[i],runner.grid_vec,runner.grid_pos)-b0
    return magdata

def update_logger(fp, data, runner, sensor_data, magdata):
    performance = runner.get_performance_info()
    line = f"{performance['elapsed_time']:.2f},{performance['simulation_time']:.2f}, {data.qpos[runner.joint_id]:.3f}," + \
            f"{sensor_data['force']:.2f},{sensor_data['velocity']:.4f},"
    for i in range(config.SENSOR_NUMBER):
        line+=f'{magdata[i][0]:.6f},{magdata[i][1]:.6f},{magdata[i][2]:.6f},'
    line+='\n'
    fp.write(line)
    fp.flush()

def save_grid_vec(data, runner, screenshot_dir, timestamp):
    filename = f'grid_{timestamp}_{int(data.time*1000):06d}.npy'
    filepath = os.path.join(screenshot_dir, filename) 
    np.save(filepath, runner.grid_vec)
