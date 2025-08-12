"""miniRL.core"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, SupportsFloat, Generic

import numpy as np

from miniRL import spaces
from miniRL.types import ObsType, ActType, WrapperObsType, WrapperActType
from miniRL.utils import np_random
from miniRL.envs.registry import EnvSpec

__all__ = ["Env", "Wrapper", "ObservationWrapper", "ActionWrapper", "RewardWrapper"]

class Env(ABC, Generic[ObsType, ActType]):
    metadata: dict[str, Any]
    render_mode: str | None = None
    spec: EnvSpec | None = None
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
        #mypy will complain about the missing return statement here but subclasses will implement these ones. 
        if seed is not None:
            self._np_random, self._np_random_seed = np_random(seed)

    def render(self) -> None:
        """Render the environment if needed and possible"""
        raise NotImplementedError
    
    def close(self) -> None:
        """Close the environment"""
        pass

    @property
    def unwrapped(self) -> Env[ObsType, ActType]:
        """Returns the unwrapped base environment"""
        return self

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
    def np_random(self, value: np.random.Generator) -> None:
        """Set the random generator with user provided Generator"""
        self._np_random = value
        self._np_random_seed = -1

    def __str__(self) -> str:
        """String representation of the environment"""
        if self.spec is None:
            return f"<{type(self).__name__} instance>"
        else:
            return f"<{type(self).__name__}<{self.spec.id}>>"
        
    def __enter__(self):
        """Support with-statement"""
        return self
    
    def __exit__(self, **args: Any):
        """Support with-statement and close the env"""
        self.close()
        return False
    
    def has_wrapper_attr(self, name:str) -> bool:
        """Check if name exists in the env"""
        return hasattr(self, name)
    
    def get_wrapper_attr(self, name:str) -> Any:
        """Get name from the env"""
        return getattr(self, name)
    
    def set_wrapper_attr(self, name:str, value:Any, force:bool = True) ->bool:
        """Sets the attribute name with value"""
        if hasattr(self, name):
            setattr(self, name, value)
            return True
        return False
        
class Wrapper(Env[WrapperObsType, WrapperActType], Generic[WrapperObsType, WrapperActType, ObsType, ActType]):
    def __init__(
            self,
            env: Env[ObsType, ActType]
    ):
        """Initiates the wrapper with an env"""
        self.env = env
        assert isinstance (env, Env), f"Wrapper class can only be initiated with miniRL Env class and not with {type(env)} env"
        self._action_space: spaces.Space[WrapperActType] | None = None
        self._observation_space: spaces.Space[WrapperObsType] | None = None
        self._metadata: dict[str, Any] | None = None

    def step(
            self,
            action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Returns base environment's step method"""
        return self.env.step(action)
    
    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Returns base environment's reset method"""
        return self.env.reset(seed=seed, options=options)
    
    def close(self) -> None:
        """Returns base environment's close method"""
        return self.env.close()
    
    def render(self) -> None:
        """Returns base environment's render method"""
        return self.env.render()
    
    @property
    def unwrapped(self) -> Env[ObsType, ActType]:
        """Returns the base environment"""
        return self.env.unwrapped
    
    @property
    def np_random_seed(self) -> int:
        """Returns base environment's random seed"""
        return self.env.np_random_seed
    
    @property
    def np_random(self) -> np.random.Generator:
        """Returns base environment's np random"""
        return self.env.np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator) -> None:
        """Set np_random value"""
        self.env.np_random = value

    def __str__(self) -> str:
        """Returns the wrapper name and base environment name"""
        return f"<{type(self).__name__}{self.env}>"
    
    def __repr__(self) -> str:
        """Return string representation of the wrapper"""
        return str(self)
    
    @property
    def action_space(self) -> spaces.Space[ActType] | spaces.Space[WrapperActType]:
        """Return the action space"""
        if self._action_space is None:
            return self.env.action_space
        return self._action_space
    
    @action_space.setter
    def action_space(self, space: spaces.Space[WrapperActType]):
        """Set the action space"""
        self._action_space = space

    @property
    def observation_space(self) -> spaces.Space[ObsType] | spaces.Space[WrapperObsType]:
        """Return the observation space"""
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space
    
    @observation_space.setter
    def observation_space(self, space: spaces.Space[WrapperObsType]):
        """Set the observation space"""
        self._observation_space = space

    @property
    def metadata(self) -> dict[str, Any]:
        """Returns the metadata of the wrapper class"""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata
    
    @metadata.setter
    def metadata(self, value: dict[str, Any]):
        """Set the metadata for the wrapper class"""
        self._metadata = value

    @property
    def render_mode(self) -> str | None:
        """Returns the base environment's render mode"""
        return self.env.render_mode
    
    def has_wrapper_attr(self, name: str) -> bool:
        """Check whether name attr is either in env or wrapper env"""
        if hasattr(self, name):
            return True
        else:
            return self.env.has_wrapper_attr(name)
        
    def get_wrapper_attr(self, name: str) -> Any:
        """Get name attr from wrapper class or env class"""
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return self.env.get_wrapper_attr(self, name)
        
    def set_wrapper_attr(self, name: str, value: Any, force: bool = True) -> bool:
        """Set the attribute name with the value"""
        if hasattr(self, name):
            setattr(self, name, value)
            return True
        else:
            done = self.env.set_wrapper_attr(name, value, force=False)
            if done:
                return True
            elif force:
                setattr(self, name, value)
                return True
            else:
                return False
    
class ObservationWrapper(Wrapper[WrapperObsType, ActType, ObsType, ActType]):
    def __init__(self, env: Env[ObsType, ActType]):
        """Initiate the Observation Wrapper environment"""
        super().__init__(env=env)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Reset and returns the modified observation"""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info
    
    def step(
            self,
            action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step and returns the modified observation"""
        observation, reward, terminated, truncated, info = self.env.step(action=action)
        return self.observation(observation), reward, terminated, truncated, info
    
    def observation(
            self, 
            observation: ObsType
    ) -> WrapperObsType:
        """Returns a modified observation"""
        raise NotImplementedError
    
class ActionWrapper(Wrapper[ObsType, WrapperActType, ObsType, ActType]):
    def __init__(self, env: Env[ObsType, ActType]):
        """Initialise the action wrapper"""
        super().__init__(env=env)

    def step(
            self,
            action: WrapperActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step and return the modified action"""
        return self.env.step(self.action(action))
    
    def action(
            self,
            action: WrapperActType
    ) -> ActType:
        """Returns the modified action"""
        raise NotImplementedError
    
class RewardWrapper(Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(self, env: Env[ObsType, ActType]):
        """Initialise the reward wrapper"""
        super().__init__(env=env)
    
    def step(
            self,
            action: WrapperActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step and return the modified reward"""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(reward), terminated, truncated, info

    def reward(
            self,
            reward: SupportsFloat
    ) -> SupportsFloat:
        """Return the modified reward"""
        raise NotImplementedError