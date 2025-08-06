"""miniRL.wrappers"""

from __future__ import annotations

from typing import SupportsFloat, Any

import miniRL
from miniRL.core import Env
from miniRL.types import ObsType, ActType

__all__ = ["StepLimit"]

class StepLimit(miniRL.Wrapper[ObsType, ActType, ObsType, ActType]):
    """Restricting the environment to the required max steps (in timesteps if you like that word)"""
    def __init__(
            self, 
            env: Env[ObsType, ActType],
            max_steps: int
    ):
        """Initialises StepLimit class"""
        assert isinstance (env, Env), f"StepLimit Wrapper required miniRL.Env instance but got {type(env)}"
        assert isinstance (max_steps, int) and max_steps > 0, f"max_steps parameter in StepLimit wrapper needs a valid python integer input and should be above 0"

        super().__init__(env=env)
        self._max_steps = max_steps 
        self._elapsed_steps: int = 0

    def step(
            self, 
            action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step using the provided action and increment the elapsed steps counter"""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_steps:
            truncated = True

        return observation, reward, terminated, truncated, info
    
    def reset(
            self, 
            *, 
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Just reset the environment and make elapsed steps to 0 again"""
        self._elapsed_steps = 0
        return self.env.reset(seed=seed, options=options)