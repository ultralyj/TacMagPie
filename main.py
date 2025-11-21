import mujoco
import os
import config
from model_generator import create_model_xml, save_model_to_file
from video_generator import generate_video_from_frames
from simulator import run_simulation_with_viewer
from utils import create_screenshot_directory, print_simulation_info, cleanup_files

def main():
    """主函数"""
    try:
        # 创建截图目录
        screenshot_dir, timestamp = create_screenshot_directory()
        print(f"截图将保存到: {screenshot_dir}")

        # 创建模型XML
        xml_content = create_model_xml(config.GRID_SIZE, config.INDENTER_RADIUS)

        # 保存模型到文件
        model_file = save_model_to_file(xml_content)
        print(f"模型已保存到: {os.path.abspath(model_file)}")

        # 加载模型
        model = mujoco.MjModel.from_xml_path(model_file)
        data = mujoco.MjData(model)

        # 打印模拟信息
        print_simulation_info(model, config.GRID_SIZE)

        # 运行模拟
        run_simulation_with_viewer(model, data, screenshot_dir, timestamp)

        # 生成视频
        print("\n开始生成视频...")
        success = generate_video_from_frames(screenshot_dir, timestamp)

        if success:
            print("✓ 视频生成完成")
        else:
            print("✗ 视频生成失败")

    except Exception as e:
        print(f"✗ 模拟过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理临时文件
        cleanup_files(config.MODEL_OUTPUT_FILE, should_delete=False)

if __name__== "__main__":
    main()