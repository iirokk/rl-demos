import argparse
import gym_super_mario_bros

from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

def train_model():
    """Train Reinforcement Learning model for Mario"""

    env = get_env()
    done = False
    env.reset()

    while not done:
        action = env.action_space.sample()
        _, _, done, _, _ = env.step(action)
        env.render()


def get_env():
    """Return Super Mario Env"""
    env =  gym_super_mario_bros.make("SuperMarioBros-v0", render_mode="human", apply_api_compatibility=True)
    return JoypadSpace(env, RIGHT_ONLY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Flag to train model')
    parser.add_argument('--test', action='store_true', help='Flag to test model')
    args = parser.parse_args()

    train_model()

    # if args.train:
    #     train_model()
    # elif args.test:
    #     test_model()
    # else:
    #     parser.print_help()
