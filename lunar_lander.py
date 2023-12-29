import argparse
import gymnasium as gym
from agents import dqn_agent

def train_model():
    """Train Reinforcement Learning model for LunarLander"""
    print("Training model for LunarLander")
    dqn_agent.train(get_env(), n_steps=50000, visualize=False)
    print("Training completed for LunarLander")

def test_model():
    """Run LunarLander using trained model"""

    env = get_env()
    agent = dqn_agent.load(env)
    _ = agent.test(env, nb_episodes=5, visualize=True)
    print("Closed LunarLander")


def get_env() -> gym.Env:
    """Return LunarLander Env"""
    return gym.make("LunarLander-v2", render_mode="human")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Flag to train model')
    parser.add_argument('--test', action='store_true', help='Flag to test model')
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.test:
        test_model()
    else:
        parser.print_help()
