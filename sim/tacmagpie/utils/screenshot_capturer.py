import os
import mujoco
import glfw
import numpy as np
from PIL import Image
from OpenGL.GL import *

import config


class ScreenshotCapturer:
    """
    Offscreen screenshot capture utility for MuJoCo simulations.

    This class uses GLFW and OpenGL to render MuJoCo scenes
    without displaying a visible window, and captures RGB frames
    as NumPy arrays.
    """

    def __init__(
        self,
        model,
        width=config.SCREENSHOT_WIDTH,
        height=config.SCREENSHOT_HEIGHT,
    ):
        """
        Initialize the offscreen renderer.

        Parameters
        ----------
        model : mujoco.MjModel
            Loaded MuJoCo model.
        width : int
            Output image width.
        height : int
            Output image height.
        """
        self.model = model
        self.width = width
        self.height = height

        self._init_glfw()
        self._init_mujoco_render_context()
        self._setup_camera()
        self._setup_visual_options()

        print(f"ScreenshotCapturer initialized ({width} x {height})")

    def _init_glfw(self):
        """
        Initialize GLFW and create an invisible offscreen window.
        """
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(
            self.width, self.height, "Offscreen", None, None
        )

        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create offscreen GLFW window")

        glfw.make_context_current(self.window)
        self._setup_opengl_line_style()

    def _init_mujoco_render_context(self):
        """
        Initialize MuJoCo rendering context and scene.
        """
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150
        )

        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)

    def _setup_camera(self):
        """
        Configure the default free camera.
        """
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 0.2
        self.camera.azimuth = 45
        self.camera.elevation = -20
        self.camera.lookat = [0.0, 0.0, 0.03]

    def _setup_visual_options(self):
        """
        Configure MuJoCo visualization options.
        """
        self.option = mujoco.MjvOption()
        self.option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1

    def _setup_opengl_line_style(self):
        """
        Configure OpenGL line rendering style for wireframe visualization.
        """
        glLineWidth(10.0)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glPointSize(3.0)
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)

    def capture_frame(self, data, camera_params=None):
        """
        Render and capture the current frame.

        Parameters
        ----------
        data : mujoco.MjData
            Simulation data corresponding to the model.
        camera_params : dict or None
            Optional camera overrides (azimuth, elevation, distance, lookat).

        Returns
        -------
        np.ndarray or None
            Captured RGB image of shape (H, W, 3), or None on failure.
        """
        if camera_params:
            self.camera.azimuth = camera_params.get("azimuth", self.camera.azimuth)
            self.camera.elevation = camera_params.get("elevation", self.camera.elevation)
            self.camera.distance = camera_params.get("distance", self.camera.distance)
            self.camera.lookat = camera_params.get("lookat", self.camera.lookat)

        try:
            glfw.make_context_current(self.window)

            glLineWidth(3.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            mujoco.mjv_updateScene(
                self.model,
                data,
                self.option,
                None,
                self.camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )

            mujoco.mjr_render(self.viewport, self.scene, self.context)
            glFlush()

            rgb = np.zeros(
                (self.viewport.height, self.viewport.width, 3),
                dtype=np.uint8,
            )
            depth = np.zeros(
                (self.viewport.height, self.viewport.width),
                dtype=np.float32,
            )

            mujoco.mjr_readPixels(rgb, depth, self.viewport, self.context)
            return np.flipud(rgb)

        except Exception as e:
            print(f"Screenshot capture failed: {e}")
            return None

    def save_frame(
        self,
        image_array,
        filepath,
        quality=config.SCREENSHOT_QUALITY,
    ):
        """
        Save a captured frame to disk.

        Parameters
        ----------
        image_array : np.ndarray
            RGB image array.
        filepath : str
            Output file path.
        quality : int
            Image quality parameter for PNG compression.

        Returns
        -------
        bool
            True if save succeeds, False otherwise.
        """
        if image_array is None:
            return False

        try:
            image = Image.fromarray(image_array)
            image.save(filepath, "PNG", optimize=True, quality=quality)
            return True
        except Exception as e:
            print(f"Failed to save screenshot: {e}")
            return False

    def set_line_width(self, width):
        """
        Dynamically update OpenGL line width.

        Parameters
        ----------
        width : float
            Line width value.
        """
        glfw.make_context_current(self.window)
        glLineWidth(float(width))
        print(f"Line width set to {width}")

    def cleanup(self):
        """
        Release all rendering resources.
        """
        if hasattr(self, "window"):
            glfw.destroy_window(self.window)
        glfw.terminate()
        print("ScreenshotCapturer resources released")
