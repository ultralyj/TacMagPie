import mujoco
import numpy as np
from PIL import Image
import os
import glfw
import config
from OpenGL.GL import *

class ScreenshotCapturer:
    """截图捕获器类"""

    def __init__(self, model, width=config.SCREENSHOT_WIDTH, height=config.SCREENSHOT_HEIGHT):
        self.model = model
        self.width = width
        self.height = height
        
        # 初始化GLFW
        if not glfw.init():
            raise RuntimeError("无法初始化GLFW")
        
        # 创建离屏窗口
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(width, height, "Offscreen", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("无法创建离屏窗口")
        
        glfw.make_context_current(self.window)
        
        # 设置OpenGL线宽和线框渲染参数
        self._setup_opengl_line_style()
        
        # 创建MuJoCo渲染上下文
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # 设置视口
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        
        # 设置相机和选项
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 0.2
        self.camera.azimuth = 45
        self.camera.elevation = -20
        self.camera.lookat = [0, 0, 0.03]
        
        self.option = mujoco.MjvOption()
        self.option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
        
        print(f"截图捕获器初始化完成: {width}x{height}")

    def _setup_opengl_line_style(self):
        """设置OpenGL线框样式"""
        # 设置线宽（默认1.0，可以调整到3.0-5.0以获得更粗的线）
        glLineWidth(10.0)
        
        # 启用线平滑（使线条更平滑）
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # 设置点大小（如果有点的话）
        glPointSize(3.0)
        
        # 启用多边形偏移，防止深度冲突
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)

    def capture_frame(self, data, camera_params=None):
        """捕获当前帧"""
        if camera_params:
            self.camera.azimuth = camera_params.get('azimuth', 45)
            self.camera.elevation = camera_params.get('elevation', -20)
            self.camera.distance = camera_params.get('distance', 0.2)
            self.camera.lookat = camera_params.get('lookat', [0, 0, 0.03])
        
        try:
            # 确保OpenGL上下文正确
            glfw.make_context_current(self.window)
            
            # 在渲染前重新设置线宽（确保每次渲染都应用）
            glLineWidth(3.0)
            
            # 清除缓冲区
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # 更新场景
            mujoco.mjv_updateScene(self.model, data, self.option, None, 
                                self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            
            # 渲染到缓冲区
            mujoco.mjr_render(self.viewport, self.scene, self.context)
            
            # 强制刷新OpenGL命令
            glFlush()
            
            # 读取像素
            rgb = np.zeros((self.viewport.height, self.viewport.width, 3), dtype=np.uint8)
            depth = np.zeros((self.viewport.height, self.viewport.width), dtype=np.float32)
            
            mujoco.mjr_readPixels(rgb, depth, self.viewport, self.context)
            
            # 翻转图像
            rgb = np.flipud(rgb)
            
            return rgb
            
        except Exception as e:
            print(f"截图捕获失败: {e}")
            return None

    def set_line_width(self, width):
        """动态设置线宽"""
        glfw.make_context_current(self.window)
        glLineWidth(float(width))
        print(f"线宽设置为: {width}")

    def save_frame(self, image_array, filepath, quality=config.SCREENSHOT_QUALITY):
        """保存帧到文件"""
        if image_array is None:
            return False
            
        try:
            pil_image = Image.fromarray(image_array)
            pil_image.save(filepath, 'PNG', optimize=True, quality=quality)
            return True
        except Exception as e:
            print(f"保存截图失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'window'):
            glfw.destroy_window(self.window)
        glfw.terminate()
        print("截图捕获器资源已清理")