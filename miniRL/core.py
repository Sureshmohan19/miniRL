"""miniRL.core"""

from abc import ABC, abstractmethod
from typing import Any, SupportsFloat, Generic

import numpy as np

from miniRL import spaces
from miniRL.types import ObsType, ActType
from miniRL.utils import np_random

class Env(ABC, Generic[ObsType, ActType]):
    metadata: dict[str, Any]
    render_mode: str | None = None
    spec: None = None
    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]
    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    @abstractmethod
    def step(
        self, 
        action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment"""
        raise NotImplementedError
    
    def reset(
        self, 
        *, 
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment"""
        if seed is not None:
            self._np_random, self._np_random_seed = np_random(seed)
        
        raise NotImplementedError

    def render(self) -> None:
        """Render the environment if needed and possible"""
        raise NotImplementedError
    
    def close(self) -> None:
        """Close the environment"""
        pass

    @property
    def np_random_seed(self) -> int:
        """Returns random seed"""
        if self._np_random_seed is None:
            self._np_random, self._np_random_seed = np_random()
        
        return self._np_random_seed
    
    @property
    def np_random(self) -> np.random.Generator:
        """Returns random number"""
        if self._np_random is None:
            self._np_random, self._np_random_seed = np_random()
        
        return self._np_random
    
    @np_random.setter
    def np_random(
        self, 
        value: np.random.Generator
    ) -> None:
        """Set the random generator with user provided Generator"""
        self._np_random = value
        self._np_random_seed = -1

    def __str__(self) -> str:
        """String representation of the environment"""
        if self.spec is None:
            return f"<{type(self).__name__} instance>"
        else:
            return f"<{type(self).__name__}<{self.spec.id}>>"