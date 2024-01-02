import json
from pathlib import Path
import torch
from agents.ddqn_agent import DdqnAgent


def save_model(agent: DdqnAgent, path: str):
    """
    Save the model weights of the online network to a file.

    Args:
        path (str): The path to save the model.
    """
    torch.save(agent.online_network.state_dict(), path)


def load_model(agent: DdqnAgent, path: str):
    """
    Load the model weights from a file to the online and target networks.

    Args:
        path (str): The path to load the model from.
    """
    agent.online_network.load_state_dict(torch.load(path))
    agent.target_network.load_state_dict(torch.load(path))


def training_checkpoint(agent: DdqnAgent, save_dir: str):
    Path.mkdir(save_dir, exist_ok=True)

    # Save the state dictionaries of the networks
    torch.save(agent.online_network.state_dict(), Path(save_dir, "online_network.pth"))
    torch.save(agent.target_network.state_dict(), Path(save_dir, "target_network.pth"))

    # Save the optimizer state
    torch.save(agent.optimizer.state_dict(), Path(save_dir, "optimizer.pth"))

    # Save the learn_step_counter & epsilon
    data = {"learn_step_counter": agent.learn_step_counter, "epsilon": agent.epsilon}
    with open(Path(save_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Save the replay buffer (optional)
    agent.replay_buffer.save(Path(save_dir, "replay_buffer"))


def load_training_checkpoint(agent: DdqnAgent, save_dir: str):
    # Load the state dictionaries of the networks
    agent.online_network.load_state_dict(
        torch.load(Path(save_dir, "online_network.pth"))
    )
    agent.target_network.load_state_dict(
        torch.load(Path(save_dir, "target_network.pth"))
    )

    # Load the optimizer state
    agent.optimizer.load_state_dict(torch.load(Path(save_dir, "optimizer.pth")))

    # Load the learn_step_counter
    with open(Path(save_dir, "training_state.json"), "r", encoding="utf-8") as f:
        traning_state = json.load(f)
        agent.learn_step_counter = traning_state["learn_step_counter"]
        agent.epsilon = traning_state["epsilon"]

    # Load the replay buffer (optional)
    agent.replay_buffer.load(Path(save_dir, "replay_buffer"))
