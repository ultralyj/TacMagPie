import subprocess
import os
import glob
import shutil
import config
from typing import List, Optional


def _collect_frame_files(screenshot_dir: str) -> List[str]:
    """
    Collect and sort all PNG frame files in the screenshot directory.
    """
    return sorted(glob.glob(os.path.join(screenshot_dir, "*.png")))


def _write_ffmpeg_file_list(frame_files: List[str], output_dir: str) -> str:
    """
    Create a temporary ffmpeg concat file list.

    Returns
    -------
    file_list_path : str
        Path to the generated file list.
    """
    file_list_path = os.path.join(output_dir, "file_list.txt")

    with open(file_list_path, "w") as f:
        for frame in frame_files:
            f.write(f"file '{os.path.abspath(frame)}'\n")

    return file_list_path


def _build_ffmpeg_command(
    file_list_path: str,
    output_path: str,
    fps: int
) -> List[str]:
    """
    Build ffmpeg command for frame-to-video conversion.
    """
    return [
        config.FFMPEG_PATH,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-r", str(fps),
        "-i", file_list_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "23",
        "-r", str(fps),
        output_path,
    ]


def generate_video_from_frames(
    screenshot_dir: str,
    timestamp: str,
    fps: int = config.VIDEO_FPS,
    delete_frames: bool = config.DELETE_FRAMES_AFTER_VIDEO,
) -> bool:
    """
    Generate an MP4 video from a sequence of PNG frames using ffmpeg.

    Parameters
    ----------
    screenshot_dir : str
        Directory containing rendered frame images.
    timestamp : str
        Timestamp string used for naming the output video.
    fps : int
        Video frame rate.
    delete_frames : bool
        Whether to delete frame images after video generation.

    Returns
    -------
    success : bool
        True if video generation succeeds, False otherwise.
    """
    try:
        frame_files = _collect_frame_files(screenshot_dir)

        if not frame_files:
            print("✗ No frame images found, video generation skipped")
            return False

        print(f"Found {len(frame_files)} frame images")

        video_dir = os.path.dirname(screenshot_dir)
        video_path = os.path.join(video_dir, f"simulation_{timestamp}.mp4")

        file_list_path = _write_ffmpeg_file_list(frame_files, screenshot_dir)
        ffmpeg_cmd = _build_ffmpeg_command(file_list_path, video_path, fps)

        print("Generating video with ffmpeg...")

        subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        if os.path.exists(file_list_path):
            os.remove(file_list_path)

        print(f"✓ Video successfully generated: {video_path}")
        print(f"Video size: {os.path.getsize(video_path) / (1024 * 1024):.2f} MB")

        if delete_frames:
            for frame in frame_files:
                os.remove(frame)
            print("Frame images deleted")

        return True

    except subprocess.CalledProcessError as e:
        print("✗ ffmpeg execution failed")
        print(e.stderr)
        return False

    except Exception as e:
        print(f"✗ Unexpected error during video generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_ffmpeg() -> Optional[str]:
    """
    Locate ffmpeg executable.

    Returns
    -------
    ffmpeg_path : str or None
        Path to ffmpeg executable if found.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return "ffmpeg"
    except Exception:
        pass

    return shutil.which("ffmpeg")
