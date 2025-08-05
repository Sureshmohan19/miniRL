import numpy as np
from miniRL.core import Env
from miniRL.spaces import Discrete

class LineWorldEnv(Env[int, int]):
    def __init__(self, size: int = 5):
        self.size = size
        self.agent_pos = 0

        # The agent's position on the line (0 to size-1)
        self.observation_space = Discrete(self.size)
        # The agent's actions: 0 for left, 1 for right
        self.action_space = Discrete(2)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[int, dict]:
        # We must call super() to handle seeding
        super().reset(seed=seed)
        
        # Reset the agent to the middle of the line
        self.agent_pos = self.size // 2
        
        # Return the initial observation and an empty info dict
        return self.agent_pos, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        # Move left or right
        if action == 0:
            self.agent_pos -= 1
        elif action == 1:
            self.agent_pos += 1
        else:
            raise ValueError(f"Invalid action: {action}")

        # Clip the position to be within the bounds of the world [0, size-1]
        self.agent_pos = np.clip(self.agent_pos, 0, self.size - 1)

        # Check if the episode is over
        terminated = self.agent_pos == self.size - 1
        truncated = False # This environment does not have a time limit

        # Assign a reward
        reward = 1.0 if terminated else 0.0

        return self.agent_pos, reward, terminated, truncated, {}

    def render(self) -> None:
        # Create a visual representation of the world
        world = ['-'] * self.size
        world[self.agent_pos] = 'A' # Mark the agent's position
        print("".join(world))