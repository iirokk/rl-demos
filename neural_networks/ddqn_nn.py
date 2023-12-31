import torch
from torch import nn
import numpy as np


class DdqnNN(nn.Module):
    """
    Double Deep Q-Networks (DDQN) Neural Network
    """
    
    device = 'cuda'
    
    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.network = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        if freeze:
            self._freeze()
        
        self.to(self.device)


    def _get_conv_out(self, shape):
        """
        Dynamically claculate number of neurons for linear layer
        """
        return int(np.prod(self.conv_layers(torch.zeros(1, *shape)).size()))


    def _freeze(self):
        """
        Disable calculating gradients (for target network)
        """
        for p in self.network.parameters():
            p.requires_grad = False


    def forward(self, x):
        """
        Forward pass
        """
        return self.network(x)


if __name__ == "__main__":
    print("Test creating neural network")
    # ddqn_nn()
