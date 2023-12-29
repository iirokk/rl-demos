import gymnasium as gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

def run():
    """Run LunarLander"""

    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()

    print("Created env. Starting LunarLander...")
    for i in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print("Closed LunarLander")

def build_model(n_states: int, n_actions: int) -> Sequential:
    """
    This function builds a model using the Keras library to approximate the Q-values of a reinforcement learning agent.
    
    Parameters:
    - n_states: The number of states in the environment. This determines the shape of the input layer of the model.
    - n_actions: The number of possible actions in the environment. This determines the shape of the output layer of the model.
    
    Returns:
    - model: The built neural network model.
    
    The model architecture consists of the following layers:
    1. Flatten layer: This layer reshapes the input to a 1D array to be processed by the subsequent fully connected layers.
    2. Dense layer: This layer consists of 24 neurons with the ReLU activation function. It helps to introduce non-linearity to the model.
    3. Another Dense layer: This is also a fully connected layer with 24 neurons and ReLU activation.
    4. Output layer: This layer has the number of neurons equal to the number of possible actions in the environment, and it uses the linear activation function.
    
    The Sequential model is then returned for further usage.
    """
    model = Sequential()
    model.add(Flatten(input_shape=(1,n_states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    return model

if __name__ == "__main__":
    run()
