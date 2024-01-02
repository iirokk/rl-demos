import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torch import Tensor
from gym.core import Env
from neural_networks.ddqn_nn import DdqnNN

REPLAY_BUFFER_CAPACITY = 100000


class DdqnAgent:
    """
    Double Deep Q-Learning Agent.

    Args:
        input_dims (tuple): The dimension of the input observation.
        n_actions (int): The number of possible actions the agent can take.

    Attributes:
        loss (torch.nn.MSELoss): Mean Squared Error loss.
        n_actions (int): The number of possible actions the agent can take.
        learn_step_counter (int): Counter to keep track of the number of learning steps.

        lr (float): Learning rate / alpha.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        epsilon_decay (float): Decay rate of exploration rate.
        epsilon_min (float): Minimum exploration rate.
        batch_size (int): The number of samples per learning iteration.
        sync_network_rate (int): The interval at which the target network is synced with the online network.

        online_network (DdqnNN): The online network used for Q-Learning.
        target_network (DdqnNN): The target network used for Q-Learning.

        optimizer (torch.optim.Adam): Optimizer used for gradient descent.
        loss (torch.nn.MSELoss): Mean Squared Error loss.

        replay_buffer (TensorDictReplayBuffer): Replay buffer for storing and sampling experience.
    """

    loss: torch.nn.MSELoss

    def __init__(self, input_dims, n_actions):
        self.n_actions = n_actions
        self.learn_step_counter = 0

        # DDQN hyperparameters
        self.lr = 0.00025
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.99999975
        self.epsilon_min = 0.1
        self.batch_size = 32
        self.sync_network_rate = 10000

        # Networks
        self.online_network = DdqnNN(input_dims, n_actions)
        self.target_network = DdqnNN(input_dims, n_actions, freeze=True)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        # Replay buffer
        storage = LazyMemmapStorage(max_size=REPLAY_BUFFER_CAPACITY)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def choose_action(self, observation):
        """
        Choose an action based on the observation using epsilon-greedy approach.

        Args:
            observation (numpy.ndarray): The current observation/state.

        Returns:
            int: The selected action.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        observation = (
            torch.tensor(np.array(observation), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.online_network.device)
        )
        action_values: Tensor = self.online_network(observation)
        return action_values.argmax().item()

    def decay_epsilon(self):
        """Decay the exploration rate (epsilon) based on epsilon_decay and epsilon_min."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        """
        Store a transition (state, action, reward, next_state, done) in the replay buffer.

        Args:
            state (numpy.ndarray): The current state.
            action (int): The selected action.
            reward (float): The received reward.
            next_state (numpy.ndarray): The next state.
            done (bool): Whether the episode is done or not.
        """
        self.replay_buffer.add(
            TensorDict(
                {
                    "state": torch.tensor(np.array(state), dtype=torch.float32),
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "next_state": torch.tensor(
                        np.array(next_state), dtype=torch.float32
                    ),
                    "done": torch.tensor(done, dtype=torch.bool),
                },
                batch_size=[],
            )
        )

    def sync_networks(self):
        """Sync the target network with the online network at a given interval."""
        if (
            self.learn_step_counter % self.sync_network_rate == 0
            and self.learn_step_counter > 0
        ):
            self.target_network.load_state_dict(self.online_network.state_dict())

    def learn(self):
        """Perform one learning step by updating the online network using the target network and sampled experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return
        self.sync_networks()

        self.optimizer.zero_grad()  # clear gradients

        samples = self.replay_buffer.sample(self.batch_size).to(
            self.online_network.device
        )

        keys = (
            "state",
            "action",
            "reward",
            "next_state",
            "done",
        )

        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        # Get predicted values from online network
        predicted_q_values = self.online_network(states)
        # Get values by index of taken actions
        predicted_q_values = predicted_q_values[
            np.arange(self.batch_size), actions.squeeze()
        ]

        # Calculate target values
        target_q_values = self.target_network(next_states).max(dim=1)[
            0
        ]  # Get best value of best action
        state_multiplier = 1 - dones.float()  # 0 if in terminal state
        target_q_values = rewards + self.gamma * target_q_values * state_multiplier

        # Back propagation to calculate gradients
        loss_values: Tensor = self.loss(predicted_q_values, target_q_values)
        loss_values.backward()

        # Gradient descent
        self.optimizer.step()
        self.learn_step_counter += 1
        self.decay_epsilon()

    def save_model(self, path):
        """
        Save the model weights of the online network to a file.

        Args:
            path (str): The path to save the model.
        """
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        """
        Load the model weights from a file to the online and target networks.

        Args:
            path (str): The path to load the model from.
        """
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))


def from_env(env: Env) -> DdqnAgent:
    """
    Create a DdqnAgent instance based on the environment.

    Args:
        env (gym.core.Env): The gym environment.

    Returns:
        DdqnAgent: The created DdqnAgent instance.
    """
    return DdqnAgent(
        input_dims=env.observation_space.shape, n_actions=env.action_space.n
    )
