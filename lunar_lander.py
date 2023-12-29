import gymnasium as gym

def run():
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