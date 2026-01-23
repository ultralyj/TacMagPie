import mujoco
import os
import config
import multiprocessing as mp
import numpy as np
import time
import csv
import shutil
import traceback
from typing import List, Tuple
from model_generator import create_model_xml, save_model_to_file
from simulator import run_simulation
from logger import log_info, log_error, log_warning, logger
import signal

# 全局控制变量
shutdown_event = mp.Event()
process_pool = None
workers = 24  # 进程数

def signal_handler(sig, frame):
    """处理中断信号"""
    log_warning("收到中断信号，正在退出...")
    shutdown_event.set()
    raise KeyboardInterrupt

def generate_parameter_combinations() -> List[Tuple[float, float, float, int]]:
    """生成所有参数组合"""
    target_positions = [0.0062]  # 固定目标位置
    indenter_x = np.arange(-0.006, 0.006, 0.0005).round(4)  # X方向坐标
    indenter_y = np.arange(-0.006, 0.006, 0.0005).round(4)  # Y方向坐标
    indenter_type = [1,2,3,4,5]
    combinations = [(tp, ix, iy, it) for tp in target_positions 
             for ix in indenter_x for iy in indenter_y for it in indenter_type]
    log_info(f"生成参数组合: {len(combinations)} 个")
    return combinations

def run_simulation_case_process(params):
    """子进程：执行单个模拟任务"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # 忽略子进程中断信号
    target_pos, indenter_x, indenter_y, indenter_type, base_dir = params
    
    try:
        if shutdown_event.is_set():
            return (False, "任务已取消", None)

        # 设置配置参数
        config.TARGET_POSITION = target_pos
        config.INDENTER_X = indenter_x
        config.INDENTER_Y = indenter_y
        config.INDENTER_TYPE = indenter_type
        
        # 创建并保存模型
        xml_content = create_model_xml()
        model_file = save_model_to_file(
            xml_content, 
            f"model_{target_pos:.4f}_{indenter_x:.4f}_{indenter_y:.4f}_{indenter_type}.xml"
        )
        time.sleep(0.1)  # 确保文件写入完成

         # 保存临时CSV（备份用）
        temp_dir = os.path.join(base_dir, "temp_data")
        os.makedirs(temp_dir, exist_ok=True)
        csv_path = os.path.join(temp_dir, f"data_{target_pos:.4f}_{indenter_x:.4f}_{indenter_y:.4f}_{indenter_type}.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    first_row = next(reader, None)
                if first_row:
                    line = np.array([float(x) for x in first_row], dtype=float)
                    log_info(f"已存在CSV，加载并跳过模拟: {csv_path}")
                    return (True, f"已存在数据，跳过: {csv_path}", line.tolist())
            except Exception as e:
                log_warning(f"读取已存在CSV失败: {csv_path}，错误={e}")
        # 加载模型并运行模拟
        model = mujoco.MjModel.from_xml_path(model_file)
        data = mujoco.MjData(model)
        log_info(f"模拟开始: TARGET={target_pos:.4f}, X={indenter_x:.4f}, Y={indenter_y:.4f}, TYPE={indenter_type}")

        if not shutdown_event.is_set():
            result = run_simulation(model, data, shutdown_event=shutdown_event)
            
            # 生成数据行并保存临时文件
            line = np.array([
                target_pos, indenter_x, indenter_y, indenter_type, result['force']
            ] + result['magnetic_data'].flatten().tolist()).tolist()
           
            with open(csv_path, 'w', encoding='utf-8') as f:
                csv.writer(f).writerow(line)

            return (True, f"完成: target={target_pos}, x={indenter_x}, type={indenter_type}", line)
        return (False, "模拟被取消", None)

    except Exception as e:
        if shutdown_event.is_set():
            return (False, "任务已取消", None)
        error_msg = f"失败: target={target_pos}, x={indenter_x}, 错误={str(e)}"
        print(error_msg)
        return (False, error_msg, None)

def main():
    """主函数：多进程任务调度与结果处理"""
    global process_pool
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 初始化目录
    base_dir = "simulation_results"
    os.makedirs(base_dir, exist_ok=True)

    # 生成任务列表
    combinations = generate_parameter_combinations()
    tasks = [(tp, ix, iy, it, base_dir) for tp, ix, iy, it in combinations]
    total_tasks = len(tasks)
    log_info(f"开始处理任务: {total_tasks} 个")

    # 任务统计
    start_time = time.time()
    completed = 0
    success = 0
    all_data = []

    try:
        # 进程池处理任务
        with mp.Pool(processes=workers, maxtasksperchild=10) as pool:
            process_pool = pool
            results = [pool.apply_async(run_simulation_case_process, (task,)) 
                      for task in tasks if not shutdown_event.is_set()]

            # 处理结果
            for result in results:
                if shutdown_event.is_set():
                    log_warning("取消剩余任务")
                    break

                # 等待任务完成（带超时检查中断）
                while not result.ready():
                    result.wait(1)
                    if shutdown_event.is_set():
                        raise KeyboardInterrupt

                # 获取结果
                success_flag, msg, data_line = result.get(timeout=0.1)
                completed += 1

                if success_flag and data_line:
                    success += 1
                    all_data.append(data_line)
                    log_info(f"进度: {completed}/{total_tasks} - {msg}")

                # 进度报告（每10个任务）
                if completed % 10 == 0 and not shutdown_event.is_set():
                    elapsed = time.time() - start_time
                    remaining = (elapsed / completed * total_tasks - elapsed) / 60
                    log_info(f"进度: {completed/total_tasks*100:.1f}% - 剩余时间: {remaining:.1f}分钟")

    except KeyboardInterrupt:
        log_warning("程序被中断")
    except Exception as e:
        log_error(f"程序异常: {str(e)}")
        traceback.print_exc()
    finally:
        # 结果汇总
        total_time = (time.time() - start_time) / 60
        log_info("\n" + "="*60)
        log_info(f"模拟总结:")
        log_info(f"总任务: {total_tasks} | 完成: {completed} | 成功: {success} | 失败: {completed - success}")
        log_info(f"耗时: {total_time:.2f}分钟 | 平均/任务: {total_time*60/completed:.2f}秒" if completed else "无完成任务")

        # 保存最终数据
        if all_data:
            try:
                final_array = np.array(all_data)
                np.save(os.path.join(base_dir, "simulation_data.npy"), final_array)
                np.savetxt(
                    os.path.join(base_dir, "simulation_data.csv"),
                    final_array,
                    delimiter=",",
                    fmt="%.6f"
                )
                log_info(f"数据已保存至: {os.path.abspath(base_dir)}")
            except Exception as e:
                log_error(f"保存数据失败: {str(e)}")
        else:
            log_warning("无有效数据可保存")

        log_info("程序退出" + (" (用户中断)" if shutdown_event.is_set() else ""))
        logger.cleanup()

if __name__ == "__main__":
    main()
    # 正在仿真，请不要关机！