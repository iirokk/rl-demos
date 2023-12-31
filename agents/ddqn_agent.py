import torch
import numpy as np
from neural_networks.ddqn_nn import DdqnNN
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

REPLAY_BUFFER_CAPACITY = 100000
WEIGHTS_FILE_FORMAT = '_agent_weights.h5f'

class DdqnAgent:
    def __init__(self, input_dims, n_actions):
        self.n_actions = n_actions
        self.learn_step_counter = 0
         
        # DDQN hyperparameters
        self.lr = 0.00025  # learning rate / alpha
        self.gamma = 0.9  # discount factor
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
        # Epsilon greedy approach
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        observation = torch.tensor(np.array(observation), dtype=torch.float32).unsqueeze(0).to(self.online_network.device)
        return self.online_network(observation).argmax().item()


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
            "state": torch.tensor(np.array(state), dtype=torch.float32),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
            "next_state": torch.tensor(np.array(next_state),  dtype=torch.float32),
            "done": torch.tensor(done)
        }, batch_size=[]
        ))


    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())


    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        self.sync_networks()
        
        self.optimizer.zero_grad()  # clear gradients
        
        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)
        
        keys = ("state", "action", "reward", "next_state", "done")  # TODO: refactor tensordict keys
        
        