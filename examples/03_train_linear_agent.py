import numpy as np
from examples.cartpole_env import SimpleCartPoleEnv
from miniRL.spaces import Space
from miniRL.wrappers import StepLimit

class LinearQLearningAgent:
    """
    An agent that learns a linear function to approximate Q-values.
    This is necessary for continuous state spaces like CartPole.
    """
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        lr: float = 0.01,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Our "Q-table" is now a set of weights.
        # One weight vector for each possible action.
        # Shape: (num_actions, observation_dim)
        self.weights = np.zeros((action_space.n, observation_space.shape[0]))

    def get_q_value(self, observation: np.ndarray, action: int) -> float:
        # The Q-value is the dot product of the observation and the action's weights.
        return np.dot(observation, self.weights[action])

    def get_action(self, observation: np.ndarray) -> int:
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return self.action_space.sample()  # Explore
        else:
            # Exploit: calculate Q-values for all actions and pick the best one
            q_values = [self.get_q_value(observation, a) for a in range(self.action_space.n)]
            return int(np.argmax(q_values))

    def learn(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, terminated: bool) -> None:
        # Get the predicted Q-value of the current state-action pair
        prediction = self.get_q_value(obs, action)

        # Calculate the value of the best action from the *next* state.
        # If the episode is over, there are no future actions, so the value is 0.
        if terminated:
            next_q_max = 0.0
        else:
            next_q_values = [self.get_q_value(next_obs, a) for a in range(self.action_space.n)]
            next_q_max = np.max(next_q_values)
            
        # The TD Target: the reward we got + the discounted value of the future.
        target = reward + self.gamma * next_q_max
        
        # The error is the difference between our target and our prediction.
        td_error = target - prediction
        
        # Update the weights.
        self.weights[action] += self.lr * td_error * obs

if __name__ == "__main__":
    base_env = SimpleCartPoleEnv()
    env = StepLimit(base_env, max_steps=200)
    agent = LinearQLearningAgent(env.observation_space, env.action_space)

    num_episodes = 5000
    episode_lengths = []
    
    print("--- Training Linear Q-Learning Agent on CartPole ---")
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated, truncated = False, False
        current_length = 0

        while not (terminated or truncated):
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # In the training loop...
            agent.learn(obs, action, reward, next_obs, terminated)
            obs = next_obs
            current_length += 1
        
        episode_lengths.append(current_length)

        if (episode + 1) % 500 == 0:
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode: {episode + 1}, Average Length (last 100): {avg_length:.1f}")

    print("\nTraining finished!")
    env.close()