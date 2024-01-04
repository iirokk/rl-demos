from agents.ddqn_agent import DdqnAgent


def log_episode(agent: DdqnAgent, episode: int, total_reward: int):
    """Print episode stats"""
    print(
        "Episode:",
        episode,
        "Total reward:",
        total_reward,
        "Epsilon:",
        agent.epsilon,
        "Size of replay buffer:",
        len(agent.replay_buffer),
        "Learn step counter:",
        agent.learn_step_counter,
    )
