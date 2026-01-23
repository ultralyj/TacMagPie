import mujoco
import os
import config
from model_generator import create_model_xml, save_model_to_file
from video_generator import generate_video_from_frames
from simulator import run_simulation_with_viewer, merge_npy_files
from utils import create_screenshot_directory, print_simulation_info, cleanup_files
import multiprocessing as mp
import numpy as np
from typing import List, Tuple
import time
from logger import log_info, log_error, log_warning, logger
import signal
import sys
import shutil
import traceback
import logging

# 全局变量，用于控制程序退出
shutdown_event = mp.Event()
total_tasks = 0
process_pool = None
workers = 24  # 使用16个进程

def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    log_warning("收到中断信号，正在优雅退出...")
    shutdown_event.set()
    
    # 如果进程池存在，尝试关闭
    global process_pool
    if process_pool:
        log_info("正在关闭进程池...")
        process_pool.terminate()
        process_pool.join()
    
    log_info("退出中...")
    sys.exit(1)

def generate_parameter_combinations() -> List[Tuple[float, float]]:
    """生成所有参数组合"""
    target_positions = np.arange(0.002, 0.0141, 0.001)
    indenter_x_positions = np.arange(0, 0.051, 0.002)
    
    combinations = []
    for target_pos in target_positions:
        for indenter_x in indenter_x_positions:
            combinations.append((0.03 - round(target_pos, 4), round(indenter_x, 3)))
    
    log_info(f"总共生成 {len(combinations)} 个参数组合")
    return combinations

def create_case_directory(base_dir: str, target_position: float, indenter_x: float) -> str:
    """为每个case创建独立的保存目录"""
    case_name = f"target_{target_position:.3f}_indenter_{indenter_x:.3f}"
    case_dir = os.path.join(base_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)
    return case_dir, case_name

def setup_process_logger(case_name, log_dir):
    """为每个进程设置独立的日志记录器"""
    # 创建进程特定的日志记录器
    logger = logging.getLogger(f'process_{os.getpid()}_{case_name}')
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 进程日志文件
        log_filename = os.path.join(log_dir, f"{case_name}.log")
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        
        # 日志格式
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.propagate = False
    
    return logger

def cleanup_process_logger(logger):
    """清理进程特定的日志记录器"""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

def run_simulation_case_process(params):
    """在独立进程中运行单个模拟case"""
    # 在每个进程中重新设置信号处理（忽略中断信号，由主进程处理）
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    target_position, indenter_x, base_output_dir = params
    case_name = f"target_{target_position:.3f}_indenter_{indenter_x:.3f}"
    
    # 为当前进程创建独立的日志记录器
    case_logger = setup_process_logger(case_name, base_output_dir)
    
    try:
        # 检查是否应该退出
        if shutdown_event.is_set():
            case_logger.info("任务被取消")
            return (False, "任务被取消")
        
        # 为当前case创建独立目录
        case_dir, _ = create_case_directory(base_output_dir, target_position, indenter_x)
        
        # 设置当前case的参数
        config.TARGET_POSITION = target_position
        config.INDENTER_X = indenter_x
        log_info(f"模拟开始 配置参数: TARGET_POSITION={config.TARGET_POSITION}, INDENTER_X={config.INDENTER_X}")
        case_logger.info(f"开始处理: TARGET_POSITION={target_position:.3f}, INDENTER_X={indenter_x:.3f}")
        case_logger.info(f"结果保存到: {case_dir}")

        # 创建截图目录
        screenshot_dir, timestamp = create_screenshot_directory(case_dir)
        
        # 创建模型XML
        xml_content = create_model_xml(config.GRID_SIZE, config.INDENTER_RADIUS)

        # 保存模型到文件
        model_filename = f"model_{int(time.time()*1000)}_{os.getpid()}.xml"
        model_file = save_model_to_file(xml_content, model_filename)

        # 加载模型
        model = mujoco.MjModel.from_xml_path(model_file)
        data = mujoco.MjData(model)

        # 打印模拟信息
        # print_simulation_info(model, config.GRID_SIZE)

        # 运行模拟（检查退出信号）
        if not shutdown_event.is_set():
            # 使用无头模式运行模拟
            run_simulation_with_viewer(model, data, screenshot_dir, timestamp, case_logger)
        else:
            case_logger.info("模拟被取消")
            return (False, "模拟被取消")

        # 检查是否应该退出
        if shutdown_event.is_set():
            case_logger.info("跳过后续处理（文件合并和视频生成）")
            return (False, "任务被取消")

        # 合并网格数据
        case_logger.info("开始合并网格数据")
        success = merge_npy_files(screenshot_dir, timestamp)

        if success:
            case_logger.info("数据集合并完成")
        else:
            case_logger.error("数据集合并失败")
        log_info(f"模拟结束 配置参数: TARGET_POSITION={config.TARGET_POSITION}, INDENTER_X={config.INDENTER_X}")
        # # 生成视频
        # case_logger.info("开始生成视频")
        # success = generate_video_from_frames(screenshot_dir, timestamp)

        # if success:
        #     case_logger.info("视频生成完成")
        # else:
        #     case_logger.error("视频生成失败")

        # 清理临时文件
        cleanup_files(model_file, should_delete=True)
        
        return (True, f"Case completed: target={target_position}, indenter_x={indenter_x}")
        
    except Exception as e:
        # 如果是由于退出信号导致的异常，不记录为错误
        if shutdown_event.is_set():
            case_logger.info("任务被取消")
            return (False, "任务被取消")
        else:
            error_msg = f"Case失败: target={target_position}, indenter_x={indenter_x}, error: {e}"
            case_logger.error(error_msg, exc_info=True)
            return (False, error_msg)
    finally:
        # 清理进程特定的日志记录器
        cleanup_process_logger(case_logger)

def main():
    """主函数 - 多进程版本"""
    global process_pool, total_tasks
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 记录程序开始
    log_info("=" * 60)
    log_info("开始多进程模拟处理")
    log_info(f"可用CPU核心数: {mp.cpu_count()}")
    log_info(f"使用进程数: {workers}")
    log_info("提示: 使用 Ctrl+C 可以优雅退出程序")
    log_info("=" * 60)
    
    # 创建基础输出目录
    base_output_dir = "simulation_results"
    if os.path.exists(base_output_dir):
        try:
            log_info(f"输出目录已存在，正在删除 {base_output_dir} 及其所有子目录...")
            shutil.rmtree(base_output_dir)
            log_info("删除完成")
        except Exception as e:
            log_error(f"删除目录失败: {e}")
            sys.exit(1)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 生成所有参数组合
    combinations = generate_parameter_combinations()
    
    # 准备任务参数列表
    tasks = []
    for target_pos, indenter_x in combinations:
        tasks.append((target_pos, indenter_x, base_output_dir))
    
    total_tasks = len(tasks)
    log_info(f"开始处理 {total_tasks} 个任务")
    
    # 使用多进程池处理
    successful_cases = 0
    failed_cases = 0
    
    start_time = time.time()
    completed_count = 0
    
    try:
        # 使用多进程池，设置maxtasksperchild定期重启进程避免内存泄漏
        with mp.Pool(processes=workers, maxtasksperchild=10) as pool:
            process_pool = pool
            
            # 提交所有任务
            results = []
            for task in tasks:
                if shutdown_event.is_set():
                    log_warning("接收到退出信号，停止提交新任务")
                    break
                
                # 异步提交任务
                result = pool.apply_async(run_simulation_case_process, (task,))
                results.append(result)
            
            # 处理完成的任务
            for i, result in enumerate(results):
                if shutdown_event.is_set():
                    log_warning("接收到退出信号，取消剩余任务")
                    # 尝试取消尚未开始的任务
                    for j in range(i+1, len(results)):
                        try:
                            results[j].get(timeout=0.1)
                        except:
                            pass
                    break
                
                try:
                    # 获取结果，设置超时避免无限等待
                    success, message = result.get(timeout=3600)  # 1小时超时
                    
                    completed_count += 1
                    
                    if success:
                        successful_cases += 1
                        log_info(f"进度: {completed_count}/{total_tasks} - 完成: {message}")
                    else:
                        failed_cases += 1
                        # 如果是取消导致的失败，不记录为错误
                        if "取消" not in message and "cancelled" not in message.lower():
                            log_error(f"进度: {completed_count}/{total_tasks} - 失败: {message}")
                        else:
                            log_warning(f"进度: {completed_count}/{total_tasks} - {message}")
                            
                except mp.TimeoutError:
                    failed_cases += 1
                    log_error(f"进度: {completed_count}/{total_tasks} - 任务超时")
                    completed_count += 1
                except Exception as e:
                    failed_cases += 1
                    log_error(f"进度: {completed_count}/{total_tasks} - 异常: {e}")
                    completed_count += 1
                
                # 定期报告进度
                if completed_count % 10 == 0 and not shutdown_event.is_set():
                    elapsed_time = time.time() - start_time
                    if completed_count > 0:
                        estimated_total_time = (elapsed_time / completed_count) * total_tasks
                        remaining_time = estimated_total_time - elapsed_time
                        
                        log_info(f"进度报告: 已完成 {completed_count}/{total_tasks} ({completed_count/total_tasks*100:.1f}%) - "
                                f"预计剩余时间: {remaining_time/60:.1f}分钟")
                        
                        # 报告CPU和内存使用情况
                        try:
                            import psutil
                            cpu_percent = psutil.cpu_percent(interval=1)
                            memory = psutil.virtual_memory()
                            log_info(f"系统状态: CPU使用率 {cpu_percent}%, 内存使用率 {memory.percent}%")
                        except ImportError:
                            pass  # 如果没有psutil，跳过系统状态报告
    
    except KeyboardInterrupt:
        log_warning("程序被用户中断")
    except Exception as e:
        log_error(f"程序发生异常: {e}")
        traceback.print_exc()
    finally:
        # 计算总时间
        total_time = time.time() - start_time
        
        # 打印总结
        log_info("=" * 60)
        log_info("模拟完成总结:")
        log_info(f"总任务数: {total_tasks}")
        log_info(f"已完成: {completed_count}")
        log_info(f"成功: {successful_cases}")
        log_info(f"失败: {failed_cases}")
        if completed_count > 0:
            log_info(f"完成率: {completed_count/total_tasks*100:.2f}%")
            log_info(f"成功率: {successful_cases/completed_count*100:.2f}%" if completed_count > 0 else "成功率: 0%")
        log_info(f"总耗时: {total_time/60:.2f} 分钟")
        if completed_count > 0:
            log_info(f"平均每个任务: {total_time/completed_count:.2f} 秒")
        log_info(f"结果保存到: {os.path.abspath(base_output_dir)}")
        
        if shutdown_event.is_set():
            log_info("程序被用户中断退出")
        else:
            log_info("程序正常完成")
            
        log_info("=" * 60)
        
        # 清理日志资源
        logger.cleanup()

if __name__ == "__main__":
    # 在Windows上，多进程需要这个保护
    main()