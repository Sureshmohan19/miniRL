import miniRL
import numpy as np

# This file ONLY defines the environment. It doesn't register or run anything.
class MyCustomEnv(miniRL.Env[np.ndarray, int]):
    def __init__(self, initial_value=0.0):
        super().__init__()
        self.observation_space = miniRL.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.action_space = miniRL.spaces.Discrete(2)
        self.state = np.array([initial_value], dtype=np.float32)

    def step(self, action: int):
        reward = 1.0 if self.state[0] > 0.5 else 0.0
        terminated = self.state[0] > 0.9
        self.state += (action - 0.5) * 0.1
        self.state = np.clip(self.state, -1, 1)
        return self.state, reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(1,)).astype(np.float32)
        return self.state, {}

    def render(self):
        print(f"State: {self.state}")