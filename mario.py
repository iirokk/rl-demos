import argparse
from pathlib import Path

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from wrappers import wrapper_builder
import agents.ddqn_agent
from agents.ddqn_agent import DdqnAgent


N_EPISODES = 50000
SAVE_INTERVAL = 1000
model_path = Path("models")

def train_model():
    """Train Reinforcement Learning model for Mario"""
    env = wrapper_builder.build(get_env())
    agent:DdqnAgent = agents.ddqn_agent.from_env(env)

    for episode in range(N_EPISODES):
        done = False
        state, _ = env.reset()
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, _, _ = env.step(action)
            agent.store_in_memory(state, action, reward, new_state, done)
            agent.learn()
            state = new_state
            total_reward += reward
            
        print("Episode:", episode, "Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)

        if (episode + 1) % SAVE_INTERVAL == 0:
            file_path = model_path.joinpath("model_ep_" + str(episode + 1) + ".pt")
            agent.save_model(file_path)
            print("Saved model checkpoint: " + str(file_path))

    env.close()


def test_model(checkpoint:int):
    """Test trained model for Mario"""
    
    env = wrapper_builder.build(get_env())
    agent:DdqnAgent = agents.ddqn_agent.from_env(env)
    
    file_path = model_path.joinpath("model_ep_" + str(checkpoint) + ".pt")
    agent.load_model(file_path)
    agent.epsilon = agent.epsilon_min
    agent.eps_min = 0.0
    agent.eps_decay = 0.0
    
    for episode in range(N_EPISODES):
        done = False
        state, _ = env.reset()
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, _, _ = env.step(action)
            state = new_state
            total_reward += reward
            
        print("Episode:", episode, "Total reward:", total_reward, "Epsilon:", agent.epsilon,
              "Size of replay buffer:", len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)

    env.close()


def get_env():
    """Return Super Mario Env"""
    env =  gym_super_mario_bros.make("SuperMarioBros-v0", render_mode="human", apply_api_compatibility=True)
    return JoypadSpace(env, RIGHT_ONLY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Flag to train model')
    parser.add_argument('--test', action='store_true', help='Flag to test model')
    parser.add_argument('--checkpoint', type=int, help='Checkpoint number to load for testing')
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.test:
        test_model(args.checkpoint)
    else:
        parser.print_help()
