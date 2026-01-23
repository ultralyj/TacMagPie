import mujoco
import numpy as np
import time
import config
import magnetic_flux

class Simulator:
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
        self.grid_vec = np.zeros((config.GRID_SIZE[0], config.GRID_SIZE[1], config.GRID_SIZE[2]-1, 3))
        self.grid_pos = magnetic_flux.get_grid_positions()

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
        elif self.movement_started:
            
            self.data.ctrl[self.position_control_id] = self.target_position
        else:
            self.data.ctrl[self.position_control_id] = 0.0

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
            if self.data.time > 0.2:
                return True
            else:
                self.timeout+=1
        else:
            return self.data.time >= config.SIMULATION_DURATION


def run_simulation(model, data, shutdown_event=None):
    
    if shutdown_event and shutdown_event.is_set():
        return 

    runner = Simulator(model, data)
    mag_data = np.zeros((config.SENSOR_NUMBER,3))

        
    # 初始化状态
    _initialize_simulation_state(data, runner)

    sensor_baseline = np.zeros((config.SENSOR_NUMBER,3))

    while not runner.is_finished():

        if shutdown_event and shutdown_event.is_set():
            break
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
                
            # _print_detailed_status(data, runner, sensor_data)
            
    # 模拟完成
    result = runner.get_performance_info()
    result['magnetic_data'] = mag_data
    result['force'] = sensor_data['force']
    
    return result        

def _initialize_simulation_state(data, runner):
    data.qpos[runner.joint_id] = runner.initial_position
    data.ctrl[runner.position_control_id] = runner.initial_position

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




