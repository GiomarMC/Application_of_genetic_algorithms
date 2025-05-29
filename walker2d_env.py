import gymnasium as gym
import numpy as np
import os
import time


class Walker2DEnv:
    def __init__(self, render_mode='human'):
        os.environ['MUJOCO_GL'] = 'glfw'
        self.env = gym.make('Walker2d-v5', render_mode=render_mode)
        self.observation, _ = self.env.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.configure_camera()

    def configure_camera(self):
        self.env.env.camera_id = 0
        self.env.env.camera_distance = 1.5
        self.env.env.camera_lookat = np.array([0, 0, 1.0])
        if hasattr(self.env.env, 'viewer'):
            self.env.env.viewer.cam.distance = 1.5
            self.env.env.viewer.cam.lookat = np.array([0, 0, 1.0])
            self.env.env.viewer.cam.elevation = -20
            self.env.env.viewer.cam.azimuth = 180

    def reset(self):
        self.observation, _ = self.env.reset()
        return self.observation

    def step(self, action):
        result = self.env.step(action)
        self.observation, reward, terminated, truncated, info = result
        return self.observation, reward, terminated, truncated, info

    def close(self):
        self.env.close()

    def keep_static(self, sleep_time=0.01):
        print("\nWalker2D estático. Presiona Ctrl+C para cerrar.")
        action = self.action_space.sample() * 0
        try:
            while True:
                _, _, terminated, truncated, _ = self.step(action)
                if terminated or truncated:
                    self.reset()
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nCerrando Walker2D...")
        finally:
            self.close()
