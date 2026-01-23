import mujoco
import mujoco.viewer
import numpy as np
import time
import datetime
import os
import config
from screenshot_capture import ScreenshotCapturer
import magnetic_flux
from logger import log_info, log_error, log_warning

class Simulator:
    """模拟运行器"""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.step_count = 0
        self.timeout = 0
        self.start_time = time.time()
        
        # 获取传感器和执行器ID
        self._get_sensor_actuator_ids()
        
        # 运动控制参数
        self.movement_started = False
        self.initial_position = 0.035
        self.target_position = config.TARGET_POSITION
        self.movement_speed = config.MOVEMENT_SPEED * config.TIMESTEP
        
        self.grid_vec = np.zeros((config.GRID_SIZE[0], config.GRID_SIZE[1], config.GRID_SIZE[2]-1, 3))
        self.grid_pos = magnetic_flux.get_grid_positions()
        # log_info("模拟运行器初始化完成")

    def _get_sensor_actuator_ids(self):
        """获取传感器和执行器ID"""
        self.joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "soft")
        self.position_control_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "position_control")
        
        # 传感器ID
        self.force_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor")
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
            (self.force_id, "传感器 'force_sensor'"),
            (self.actuator_force_id, "传感器 'actuator_force'"),
            (self.position_sensor_id, "传感器 'indenter_position'"),
            (self.velocity_sensor_id, "传感器 'indenter_velocity'")
        ]
        
        for id_val, name in ids_to_check:
            if id_val == -1:
                log_warning(f"警告: 未找到 {name}")

    def get_sensor_data(self):
        """获取传感器数据"""
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
        """更新控制逻辑"""
        current_time = self.data.time
        
        # 延迟开始运动，让系统稳定
        if current_time > 0.1 and not self.movement_started:
            self.movement_started = True
            # log_info("开始慢速下压...")
        if self.movement_started:
            # 缓慢改变目标位置
            if self.data.qpos[self.joint_id] > self.target_position:
                self.data.ctrl[self.position_control_id] -= self.movement_speed
            else:
                self.data.ctrl[self.position_control_id] = self.target_position

    def should_capture_screenshot(self):
        """判断是否应该捕获截图"""
        return self.step_count % config.SCREENSHOT_INTERVAL == 0

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
        self.grid_vec = self.data.qpos[:-1].reshape(config.GRID_SIZE[0], config.GRID_SIZE[1], config.GRID_SIZE[2]-1, 3)
        self.step_count += 1
    
    def is_finished(self):
        """检查模拟是否完成"""
        if(config.EARLY_STOP_FLAG):
            depth = self.data.qpos[self.joint_id]
            if depth <= self.target_position + 0.00005 and self.timeout>2000:
                return True
            else:
                self.timeout+=1
        else:
            return self.data.time >= config.SIMULATION_DURATION

def run_simulation_with_viewer(model, data, screenshot_dir, timestamp, case_logger=None, shutdown_event=None):
    """运行模拟而不打开可视化窗口"""
    # 检查是否应该退出
    if shutdown_event and shutdown_event.is_set():
        if case_logger:
            case_logger.info("模拟被取消")
        return
    
    runner = Simulator(model, data)
    mag_data = np.zeros((config.SENSOR_NUMBER, 3))
    
    try:
        # 初始化状态
        _initialize_simulation_state(data, runner)
        
        log_file_name = f'data_{timestamp}.csv'
        log_file_path = os.path.join(screenshot_dir, log_file_name) 
        
        grid_file_name = f'grid_{timestamp}.csv'
        grid_file_path = os.path.join(screenshot_dir, grid_file_name)
        
        # 确保目录存在
        os.makedirs(screenshot_dir, exist_ok=True)
        
        if case_logger:
            case_logger.info("模拟开始...")
        else:
            log_info("模拟开始...")
            
        sensor_baseline = np.zeros((config.SENSOR_NUMBER, 3))
        
        # 打开日志文件
        with open(log_file_path, 'w+', encoding='utf-8') as log_fp, \
             open(grid_file_path, 'w+', encoding='utf-8') as grid_fp:
            
            # 写入CSV头部（如果需要）
            # 例如: log_fp.write("time,step,sensor_data,...\n")
            
            # 主模拟循环
            while not runner.is_finished():
                # 检查退出信号
                if shutdown_event and shutdown_event.is_set():
                    if case_logger:
                        case_logger.info("模拟被中断")
                    break
                
                # 执行模拟步骤
                runner.step()
                
                # 更新控制逻辑
                runner.update_control()
                
                # 获取传感器数据
                sensor_data = runner.get_sensor_data()
                
                # 捕获数据（而不是截图）
                if runner.should_capture_screenshot():
                    if runner.get_performance_info()['simulation_time'] < 0.4:
                        sensor_baseline = update_sensor_baseline(runner)
                    else:
                        mag_data = get_magnetic_data(runner, sensor_baseline)
                    
                    _print_detailed_status(data, runner, sensor_data, case_logger)
                    update_logger(log_fp, data, runner, sensor_data, mag_data)
                    save_grid_vec(data, runner, screenshot_dir, timestamp)
            
            # 模拟完成
            performance = runner.get_performance_info()
            if case_logger:
                case_logger.info(f"模拟完成!")
                case_logger.info(f"总步数: {performance['step_count']}")
                case_logger.info(f"模拟时间: {performance['simulation_time']:.2f}s")
                case_logger.info(f"真实时间: {performance['elapsed_time']:.2f}s")
                case_logger.info(f"平均FPS: {performance['fps']:.1f}")
            else:
                log_info(f"模拟完成!")
                log_info(f"总步数: {performance['step_count']}")
                log_info(f"模拟时间: {performance['simulation_time']:.2f}s")
                log_info(f"真实时间: {performance['elapsed_time']:.2f}s")
                log_info(f"平均FPS: {performance['fps']:.1f}")
                
    except Exception as e:
        if case_logger:
            case_logger.error(f"模拟运行失败: {e}", exc_info=True)
        else:
            log_error(f"模拟运行失败: {e}")
        raise

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
        log_error(f"✗ 截图捕获失败: {e}")

def _print_detailed_status(data, runner, sensor_data, case_logger=None):
    """打印详细状态信息"""
    performance = runner.get_performance_info()
    
    elas_deformation = np.sum(runner.grid_vec[:,:,:,2])
    if case_logger:
        case_logger.info(f"[{performance['elapsed_time']:.2f}] 模拟时间:{performance['simulation_time']:.2f} "
            f"深度:{data.qpos[runner.joint_id]:.3f}, 弹性变形:{elas_deformation:.4f}"
            f"执行器力: {sensor_data['actuator_force']:.2f}, {sensor_data['force']:.2f}N, "
        f"速度: {sensor_data['velocity']:.4f}m/s")
    else:
        log_info(f"[{performance['elapsed_time']:.2f}] 模拟时间:{performance['simulation_time']:.2f} "
            f"深度:{data.qpos[runner.joint_id]:.3f} 弹性变形:{elas_deformation:.4f} "
            f" 执行器力: {sensor_data['actuator_force']:.2f} {sensor_data['force']:.2f}N "
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
    # 运行时间，模拟时间，按压深度，压力，速度，磁力数据
    line = f'{performance['elapsed_time']:.2f},{performance['simulation_time']:.2f}, {data.qpos[runner.joint_id]:.3f},' + \
            f'{sensor_data['force']:.2f},{sensor_data['velocity']:.4f},'
    for i in range(config.SENSOR_NUMBER):
        line+=f'{magdata[i][0]:.6f},{magdata[i][1]:.6f},{magdata[i][2]:.6f},'
    line+='\n'
    
    fp.write(line)
    fp.flush()

def save_grid_vec(data, runner, screenshot_dir, timestamp):
    filename = f'grid_{timestamp}_{int(data.time*1000):06d}.npy'
    filepath = os.path.join(screenshot_dir, filename) 
    np.save(filepath, runner.grid_vec)
  
import numpy as np
import os
import glob
import config

import numpy as np
import os
import glob
import config

def merge_npy_files(data_dir, timestamp, delete_original=config.DELETE_FRAMES_AFTER_VIDEO):
    """
    将多个npy文件合并为一个大的npy文件，每个文件形状为[17,17,3,3]，合并后为[n,17,17,3,3]
    
    参数:
    - data_dir: 包含npy文件的目录
    - timestamp: 时间戳，用于命名输出文件
    - delete_original: 是否删除原始文件
    """
    try:
        # 获取所有npy文件并按名称排序
        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        
        if not npy_files:
            log_error("未找到npy文件，无法合并")
            return False

        # log_info(f"找到 {len(npy_files)} 个npy文件")
        
        # 检查所有文件的兼容性（形状和数据类型）
        first_file = np.load(npy_files[0])
        first_shape = first_file.shape
        first_dtype = first_file.dtype
        
        # log_info(f"第一个文件形状: {first_shape}, 数据类型: {first_dtype}")
        
        # 验证所有文件是否兼容
        compatible = True
        file_shapes = []
        
        for i, npy_file in enumerate(npy_files):
            data = np.load(npy_file)
            file_shapes.append(data.shape)
            
            # 检查数据类型是否一致
            if data.dtype != first_dtype:
                log_warning(f"警告: 文件 {os.path.basename(npy_file)} 的数据类型不匹配")
                compatible = False
            
            # 检查形状是否一致（所有文件应该是[17,17,3,3]）
            if data.shape != (17, 17, 3, 3):
                log_error(f"错误: 文件 {os.path.basename(npy_file)} 的形状不是 [17,17,3,3]")
                compatible = False
                break
        
        if not compatible:
            log_error("文件不兼容，无法合并")
            return False
        
        # 确定合并后的形状 [n, 17, 17, 3, 3]
        n_files = len(npy_files)
        merged_shape = (n_files,) + first_shape  # (n, 17, 17, 3, 3)
        
        # log_info(f"合并后的形状: {merged_shape}")
        # log_info(f"总文件数: {n_files}")
        
        # 创建合并后的数组
        merged_data = np.empty(merged_shape, dtype=first_dtype)
        
        # 逐个加载并合并数据
        for i, npy_file in enumerate(npy_files):
            data = np.load(npy_file)
            
            # 将数据复制到合并数组的相应位置
            merged_data[i] = data
            
            # log_info(f"已合并文件 {i+1}/{len(npy_files)}: {os.path.basename(npy_file)}")
        
        # 保存合并后的文件
        grid_dir = os.path.dirname(data_dir)
        grid_path = os.path.join(grid_dir, f"grid_{timestamp}.npy")
        np.save(grid_path, merged_data)
        # log_info(f"✓ 数据已成功合并: {grid_path}")
        # log_info(f"文件大小: {os.path.getsize(grid_path) / (1024 * 1024):.2f} MB")
        
        # 可选：删除原始文件
        if delete_original:
            for npy_file in npy_files:
                os.remove(npy_file)
            # log_info(f"已删除 {len(npy_files)} 个原始npy文件")
        
        return True
        
    except Exception as e:
        log_error(f"✗ 合并npy文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def merge_npy_files_with_pattern(base_dir, pattern, output_filename, 
                                 delete_original=config.DELETE_FRAMES_AFTER_VIDEO):
    """
    使用文件模式匹配来合并npy文件
    
    参数:
    - base_dir: 基础目录
    - pattern: 文件匹配模式（如 "displacement_*.npy"）
    - output_filename: 输出文件名
    - delete_original: 是否删除原始文件
    """
    # 构建完整的搜索模式
    search_pattern = os.path.join(base_dir, pattern)
    npy_files = sorted(glob.glob(search_pattern))
    
    if not npy_files:
        log_error(f"未找到匹配模式 '{pattern}' 的npy文件")
        return False
    
    log_info(f"找到 {len(npy_files)} 个匹配的文件")
    return merge_npy_files(base_dir, output_filename, delete_original)

def merge_npy_files_by_timestamp(data_dir, timestamp, 
                                delete_original=config.DELETE_FRAMES_AFTER_VIDEO):
    """
    按时间戳合并npy文件（与视频生成函数类似）
    
    参数:
    - data_dir: 数据目录
    - timestamp: 时间戳，用于命名输出文件
    - delete_original: 是否删除原始文件
    """
    output_filename = os.path.join(data_dir, f"merged_displacement_{timestamp}.npy")
    return merge_npy_files(data_dir, output_filename, delete_original)

# 高级版本：支持不同的合并策略
def merge_npy_files_advanced(data_dir, output_filename, 
                           merge_axis=0,  # 沿哪个轴合并
                           delete_original=config.DELETE_FRAMES_AFTER_VIDEO,
                           compression=False):
    """
    高级npy文件合并函数，支持不同的合并策略
    
    参数:
    - data_dir: 包含npy文件的目录
    - output_filename: 输出文件名
    - merge_axis: 合并的轴（0=行方向，1=列方向等）
    - delete_original: 是否删除原始文件
    - compression: 是否使用压缩格式（npz）
    """
    try:
        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        
        if not npy_files:
            log_error("未找到npy文件")
            return False
        
        log_info(f"找到 {len(npy_files)} 个npy文件")
        
        # 加载第一个文件以获取参考信息
        first_data = np.load(npy_files[0])
        first_shape = first_data.shape
        first_dtype = first_data.dtype
        
        # 检查所有文件的兼容性
        compatible = True
        for npy_file in npy_files[1:]:
            data = np.load(npy_file)
            
            # 检查除合并轴外的其他维度是否匹配
            for i in range(len(first_shape)):
                if i != merge_axis and data.shape[i] != first_shape[i]:
                    log_error(f"形状不匹配: {os.path.basename(npy_file)}")
                    compatible = False
                    break
            
            if not compatible:
                break
        
        if not compatible:
            log_error("文件形状不兼容，无法合并")
            return False
        
        # 计算合并后的形状
        total_size_along_axis = sum(np.load(f).shape[merge_axis] for f in npy_files)
        
        new_shape = list(first_shape)
        new_shape[merge_axis] = total_size_along_axis
        new_shape = tuple(new_shape)
        
        log_info(f"合并后的形状: {new_shape}")
        
        # 创建合并数组
        merged_data = np.empty(new_shape, dtype=first_dtype)
        
        # 执行合并
        current_position = 0
        for i, npy_file in enumerate(npy_files):
            data = np.load(npy_file)
            data_length = data.shape[merge_axis]
            
            # 构建切片索引
            slice_obj = [slice(None)] * len(new_shape)
            slice_obj[merge_axis] = slice(current_position, current_position + data_length)
            
            # 将数据复制到合并数组
            merged_data[tuple(slice_obj)] = data
            
            current_position += data_length
            # log_info(f"已合并文件 {i+1}/{len(npy_files)}")
        
        # 保存结果
        if compression:
            output_filename = output_filename.replace('.npy', '.npz')
            np.savez_compressed(output_filename, data=merged_data)
        else:
            np.save(output_filename, merged_data)
        
        log_info(f"✓ 数据已成功合并: {output_filename}")
        
        # 删除原始文件
        if delete_original:
            for npy_file in npy_files:
                os.remove(npy_file)
            log_info(f"已删除 {len(npy_files)} 个原始文件")
        
        return True
        
    except Exception as e:
        log_error(f"✗ 合并npy文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    
   