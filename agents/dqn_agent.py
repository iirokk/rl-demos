import numpy as np
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from keras.models import Sequential
from keras.optimizers import Adam
from gymnasium import Env
import models.basic_sequential_model as seq_model

MEM_LIMIT = 50000
WEIGHTS_FILE_FORMAT = '_agent_weights.h5f'

def build(model: Sequential, n_actions: int) -> DQNAgent:
    """ 
    Builds and returns a DQNAgent object with the provided model and number of actions.
    Parameters: model (Sequential): The neural network model used by the agent. n_actions (int): The number of possible actions the agent can take.
    Returns: DQNAgent: The constructed DQNAgent object.
    """
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=MEM_LIMIT, window_length=1)
    return DQNAgent(model=model, memory=memory, policy=policy, nb_actions=n_actions, nb_steps_warmup=10, target_model_update=0.01)


def train(env: Env, n_steps=50000, visualize=False):
    """
    Train
    """
    agent = build_env_agent(env)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
    agent.fit(env, nb_steps=n_steps, visualize=visualize, verbose=1)

    scores = agent.test(env, nb_episodes=100, visualize=False)
    print("Mean episode reward from training test: " + np.mean(scores.history['episode_reward']))

    agent.save_weights(get_weigths_file(env), overwrite=True)


def build_env_agent(env: Env) -> DQNAgent:
    """
    Build agent for the environment
    """
    states = env.observation_space.shape[0]
    actions = env.action_space.n
    model = seq_model.build(states, actions)
    agent = build(model, actions)
    return agent


def load(env: Env) -> DQNAgent:
    """
    Load agent with existing weigths
    """
    agent = build_env_agent(env)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
    agent.load_weights(get_weigths_file(env))
    return agent


def get_weigths_file(env: Env):
    """Get weigths file name"""
    return "" + env.metadata.__class__  + WEIGHTS_FILE_FORMAT
