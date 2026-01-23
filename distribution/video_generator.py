import subprocess
import os
import glob
import config

def generate_video_from_frames(screenshot_dir, timestamp, fps=config.VIDEO_FPS,
    delete_frames=config.DELETE_FRAMES_AFTER_VIDEO):
    """
    使用ffmpeg将截图序列合成为MP4视频
    """
    try:
        # 获取所有PNG文件
        frame_files = sorted(glob.glob(os.path.join(screenshot_dir, "*.png")))

        if not frame_files:
            print("未找到截图文件，无法生成视频")
            return False

        print(f"找到 {len(frame_files)} 个截图文件")

        # 设置视频输出路径
        video_dir = os.path.dirname(screenshot_dir)
        video_path = os.path.join(video_dir, f"simulation_{timestamp}.mp4")

        # 创建临时文件列表
        file_list_path = os.path.join(screenshot_dir, "file_list.txt")
        with open(file_list_path, 'w') as f:
            for frame_file in frame_files:
                f.write(f"file '{os.path.abspath(frame_file)}'\n")        

        # 构建ffmpeg命令
        ffmpeg_cmd = [
            config.FFMPEG_PATH,
            '-y',  # 覆盖输出文件
            '-f', 'concat',
            '-safe', '0',
            '-r', str(fps),
            '-i', file_list_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            '-r', str(fps),
            video_path
        ]

        print("开始生成视频...")

        # 执行ffmpeg命令
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # 清理临时文件
        if os.path.exists(file_list_path):
            os.remove(file_list_path)

        if result.returncode == 0:
            print(f"✓ 视频已成功生成: {video_path}")
            print(f"视频大小: {os.path.getsize(video_path) / (1024 * 1024):.2f} MB")
            
            # 可选：删除截图文件
            if delete_frames:
                for frame_file in frame_files:
                    os.remove(frame_file)
                print("截图文件已删除")
            return True
        else:
            print(f"✗ ffmpeg执行失败，返回码: {result.returncode}")
            print(f"错误输出: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ ffmpeg命令执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ 生成视频时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_ffmpeg():
    """查找ffmpeg可执行文件路径"""
    # 尝试直接使用ffmpeg命令
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return 'ffmpeg'
    except:
        pass
    
    # 尝试从环境变量中查找
    import shutil
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    return None