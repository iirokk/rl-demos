from typing import Any, Tuple
from nes_py import NESEnv
from gym import Wrapper


class SkipFrames(Wrapper):
    def __init__(self, env: NESEnv, skipped_frames: int):
        super().__init__(env)
        self.skipped_frames = skipped_frames

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        total_reward = 0.0
        done = False

        for _ in range(self.skipped_frames):
            next_state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return next_state, total_reward, done, truncated, info
