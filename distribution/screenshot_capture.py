import mujoco
import numpy as np
from PIL import Image
import os
import glfw
import config
from OpenGL.GL import *
import threading
import contextlib

# 线程本地存储，确保每个线程有独立的GLFW上下文
_thread_local = threading.local()

class ScreenshotCapturer:
    """截图捕获器类"""

    def __init__(self, model, width=config.SCREENSHOT_WIDTH, height=config.SCREENSHOT_HEIGHT):
        self.model = model
        self.width = width
        self.height = height
        
        # 为每个线程初始化独立的GLFW上下文
        self._init_glfw_context()
        
        print(f"截图捕获器初始化完成: {width}x{height} (线程: {threading.get_ident()})")

    def _init_glfw_context(self):
        """为当前线程初始化GLFW上下文"""
        thread_id = threading.get_ident()
        
        # 如果当前线程还没有GLFW上下文，则初始化
        if not hasattr(_thread_local, 'glfw_initialized'):
            try:
                if not glfw.init():
                    raise RuntimeError(f"线程 {thread_id} 无法初始化GLFW")
                
                _thread_local.glfw_initialized = True
                _thread_local.windows = []
                print(f"线程 {thread_id} GLFW初始化成功")
                
            except Exception as e:
                print(f"线程 {thread_id} GLFW初始化失败: {e}")
                raise

        # 创建离屏窗口
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Offscreen", None, None)
        if not self.window:
            raise RuntimeError(f"线程 {thread_id} 无法创建离屏窗口")
        
        glfw.make_context_current(self.window)
        _thread_local.windows.append(self.window)
        
        # 设置OpenGL线宽和线框渲染参数
        self._setup_opengl_line_style()
        
        # 创建MuJoCo渲染上下文
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # 设置视口
        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        
        # 设置相机和选项
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 0.2
        self.camera.azimuth = 45
        self.camera.elevation = -20
        self.camera.lookat = [0, 0, 0.03]
        
        self.option = mujoco.MjvOption()
        self.option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1

    def _setup_opengl_line_style(self):
        """设置OpenGL线框样式"""
        try:
            # 设置线宽
            glLineWidth(10.0)
            
            # 启用线平滑
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            
            # 设置点大小
            glPointSize(3.0)
            
            # 启用多边形偏移，防止深度冲突
            glEnable(GL_POLYGON_OFFSET_LINE)
            glPolygonOffset(-1.0, -1.0)
        except Exception as e:
            print(f"OpenGL设置失败: {e}")

    @contextlib.contextmanager
    def _gl_context(self):
        """GL上下文管理器"""
        try:
            glfw.make_context_current(self.window)
            yield
        finally:
            glfw.make_context_current(None)

    def capture_frame(self, data, camera_params=None):
        """捕获当前帧"""
        if camera_params:
            self.camera.azimuth = camera_params.get('azimuth', 45)
            self.camera.elevation = camera_params.get('elevation', -20)
            self.camera.distance = camera_params.get('distance', 0.2)
            self.camera.lookat = camera_params.get('lookat', [0, 0, 0.03])
        
        try:
            with self._gl_context():
                # 在渲染前重新设置线宽
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
        with self._gl_context():
            glLineWidth(float(width))
            print(f"线宽设置为: {width}")

    def save_frame(self, image_array, filepath, quality=config.SCREENSHOT_QUALITY):
        """保存帧到文件"""
        if image_array is None:
            return False
            
        try:
            pil_image = Image.fromarray(image_array)
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            pil_image.save(filepath, 'PNG', optimize=True, quality=quality)
            return True
        except Exception as e:
            print(f"保存截图失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, 'window') and self.window:
                with self._gl_context():
                    glfw.destroy_window(self.window)
                
                # 从线程本地存储中移除窗口引用
                if hasattr(_thread_local, 'windows') and self.window in _thread_local.windows:
                    _thread_local.windows.remove(self.window)
                
                # 如果这是线程的最后一个窗口，终止GLFW
                if (hasattr(_thread_local, 'windows') and 
                    len(_thread_local.windows) == 0 and
                    hasattr(_thread_local, 'glfw_initialized')):
                    glfw.terminate()
                    _thread_local.glfw_initialized = False
                    
        except Exception as e:
            print(f"清理资源时出错: {e}")
        
        print(f"截图捕获器资源已清理 (线程: {threading.get_ident()})")
        
        