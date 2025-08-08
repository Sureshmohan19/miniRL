"""miniRL.tests._dummy_env"""

import numpy as np
from typing import Any

from miniRL.core import Env
from miniRL.spaces import Discrete, Box

class DummyEnv(Env[np.ndarray, int]):
    """Dummy environment for Env class testing"""
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.render_mode = None
        self._t = 0
        self._state = np.array([0.0, 0.0], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self._t = 0
        self._state = np.array([0.0, 0.0], dtype=np.float32)
        return self._state.copy(), {}
    
    def step(self, action: int):
        assert self.action_space.contains(action)
        self._t += 1
        self._state = np.array([float(self._t), float(action)], dtype=np.float32)
        reward = 1.0 if action == 1 else 0.0
        terminated = self._t >= 3
        truncated = False
        return self._state.copy(), reward, terminated, truncated, {}
    
    def render(self) -> None:
        pass

    def close(self) -> None:
        pass