import os
import datetime
from model_generator import generate_pin_ids
import config

def create_screenshot_directory():
    """创建截图目录"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_dir = f"{config.DATA_DIR}/frame_{timestamp}"
    os.makedirs(screenshot_dir, exist_ok=True)
    return screenshot_dir, timestamp

def print_simulation_info(model, grid_size):
    """打印模拟信息"""
    pin_count = len(generate_pin_ids(grid_size).split())

    print("=" * 50)
    print("MuJoCo 触觉传感器模拟")
    print("=" * 50)
    print(f"网格尺寸: {grid_size}")
    print(f"固定点数量: {pin_count}")
    print(f"模型总关节数: {model.njnt}")
    print(f"模拟时长: {config.SIMULATION_DURATION}秒")
    print(f"时间步长: {config.TIMESTEP}")
    print("=" * 50)

def cleanup_files(filename, should_delete=False):
    """清理临时文件"""
    if should_delete and os.path.exists(filename):
        os.remove(filename)
    print(f"已删除临时文件: {filename}")

