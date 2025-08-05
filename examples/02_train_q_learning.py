import numpy as np
from examples.line_world_env import LineWorldEnv
from miniRL.spaces import Space

class QLearningAgent:
    """A simple agent that learns via Q-learning."""
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        # Initialize the Q-table with zeros
        self.q_table = np.zeros((observation_space.n, action_space.n))

    def get_action(self, observation: int) -> int:
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return self.action_space.sample()  # Explore
        else:
            return int(np.argmax(self.q_table[observation]))  # Exploit

    def learn(self, obs: int, action: int, reward: float, next_obs: int) -> None:
        # The Q-learning update rule
        old_value = self.q_table[obs, action]
        next_max = np.max(self.q_table[next_obs])
        
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[obs, action] = new_value

if __name__ == "__main__":
    env = LineWorldEnv()
    agent = QLearningAgent(env.observation_space, env.action_space)

    num_episodes = 10000
    rewards_per_episode = []
    
    print("--- Training Q-Learning Agent ---")
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated, truncated = False, False
        total_reward = 0

        while not (terminated or truncated):
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward

        rewards_per_episode.append(total_reward)
        
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[-1000:])
            print(f"Episode: {episode + 1}, Average Reward: {avg_reward:.3f}")

    print("\nTraining finished!")
    print("Final Q-table:")
    print(agent.q_table)
    env.close()