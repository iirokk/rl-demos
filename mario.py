import argparse
import gym_super_mario_bros

from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from wrappers import wrapper_builder
import agents.ddqn_agent


N_EPISODES = 50000

def train_model():
    """Train Reinforcement Learning model for Mario"""
    env = wrapper_builder.build(get_env())
    agent = agents.ddqn_agent.from_env(env)

    for episode in range(N_EPISODES):
        done = False
        state, _ = env.reset()
        
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)
            agent.store_in_memory(state, action, reward, new_state, done)
            agent.learn()
            state = new_state
        print("Completed episode: " + str(episode))

    env.close()


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
