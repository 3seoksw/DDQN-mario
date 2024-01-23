import cv2
import gym
import numpy as np
import torch
from gym import Wrapper
from gym.spaces import Box
from torchvision import transforms as T


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env, monitor):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.cur_score = 0

        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        trunc = None

        if self.monitor:
            self.monitor.record(state)

        state = process_frame(state)
        reward += (info["score"] - self.cur_score) / 50
        self.cur_score = info["score"]

        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50

        return state, reward / 10, done, info

    def reset(self):
        self.cur_score = 0

        return process_frame(self.env.reset()), None


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            trunc = None
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
