from examples.line_world_env import LineWorldEnv
from miniRL.spaces import Space

class RandomAgent:
    """A simple agent that takes random actions."""
    def __init__(self, action_space: Space):
        self.action_space = action_space

    def get_action(self, observation: int) -> int:
        # Ignore the observation and return a random action
        return self.action_space.sample()

if __name__ == "__main__":
    # 1. Create the environment and agent
    env = LineWorldEnv()
    agent = RandomAgent(env.action_space)
    
    # 2. Reset the environment to get the first observation
    observation, info = env.reset(seed=42)
    
    terminated, truncated = False, False
    print("--- Running Random Agent ---")
    env.render()
    
    # 3. The main interaction loop
    while not (terminated or truncated):
        # Get an action from the agent
        action = agent.get_action(observation)
        print(f"Action taken: {'Left' if action == 0 else 'Right'}")
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Render the new state
        env.render()
        
    print("\nEpisode finished!")
    env.close()