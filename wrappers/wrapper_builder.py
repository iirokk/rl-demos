from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from gym.core import Env

from wrappers.skip_frames import SkipFrames

def build(env: Env) -> Env:
    env = SkipFrames(env, skipped_frames=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env
