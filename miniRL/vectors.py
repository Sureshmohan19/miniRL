"""miniRL.vectors"""

from __future__ import annotations

from collections.abc import Iterable, Sequence, Callable
from typing import Any, Generic
from enum import Enum
from copy import deepcopy
import numpy as np

from miniRL.types import ObsType, ActType, ArrayType
from miniRL.envs.registry import EnvSpec
from miniRL.spaces import Space
from miniRL.core import Env
from miniRL.random_utils import np_random
from miniRL.utils import (
    batch_space, 
    batch_different_space, 
    is_space_dtype_shape_equiv,
    create_empty_array,
    concatenate,
    iterate
)

__all__ = ["VectorEnv", 
           "AutoReset",
           "SyncVectorEnv", "AsyncVectorEnv"]

class VectorEnv(Generic[ObsType, ActType, ArrayType]):
    num_envs: int 
    metadata: dict[str, Any] = {}
    spec: EnvSpec | None = None
    render_mode: str | None = None
    closed: bool = False

    observation_space: Space
    single_observation_space: Space
    action_space: Space
    single_action_space: Space

    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """Reset all parallel environments and return a batch of initial observations and info"""
        if seed is not None:
            self._np_random, self._np_random_seed = np_random(seed=seed)
        raise NotImplementedError(f"{self.__str__} reset method is not implememted")

    def step(self, action: ActType) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Take an action in each parallel environment"""
        raise NotImplementedError(f"{self.__str__} step method is not implemented")

    def render(self) -> None:
        """Returns the rendered frame from each parallel environments"""
        raise NotImplementedError(f"{self.__str__} render method is not implemented")

    def close(self, **kwargs:Any) -> None:
        """Close all parallel environments"""
        if self.closed:
            return
        
        self.close_all(**kwargs)
        self.closed = True

    def close_all(self, **kwargs:Any):
        """Clean up the resources"""
        pass

    @property
    def unwrapped(self):
        """Returns the unwrapped base environment"""
        return self

    @property
    def np_random(self) -> np.random.Generator:
        """Return random number"""
        if self._np_random is None:
            self._np_random, self._np_random_seed = np_random()
        return self._np_random
    
    @np_random.setter
    def np_random(self, value: np.random.Generator):
        """Set np_random value"""
        self._np_random = value
        self._np_random_seed = -1

    @property
    def np_random_seed(self) -> int | None:
        """Returns _np_random_seed"""
        if self._np_random_seed is None:
            self._np_random, self._np_random_seed = np_random()
        return self._np_random_seed
     
    def __del__(self):
        """Closes the vector envs"""
        if not getattr(self, "closed", True):
            self.close()

    def __str__(self) -> str:
        """Returns the string representation of this vector environement"""
        if self.spec is None:
            return f"{self.__class__.__name__}(num_envs={self.num_envs})"
        else:
            return f"{self.__class__.__name__}({self.spec.name}, num_envs={self.num_envs})"
        
    def _add_info(
            self,
            vector_info: dict[str, Any], 
            env_info: dict[str, Any],
            env_num: int
    ) -> dict[str, Any]:
        """just a convience method to obtain proper results from vector environments"""
        for key, value in env_info.items():
            if key == "final_obs":
                if "final_obs" in vector_info:
                    array = vector_info["final_obs"]
                else:
                    array = np.full(self.num_envs, fill_value=None, dtype=object)
                array[env_num] = value
            elif isinstance(value, dict):
                array = self._add_info(vector_info.get(key, {}), value, env_num)
            else:
                if key not in vector_info:
                    if type(value) in [int, float, bool] or issubclass (type(value), np.number):
                        array = np.zeros(self.num_envs, dtype=type(value))
                    elif isinstance (value, np.ndarray):
                        array = np.zeros((self.num_envs, *value.shape), dtype=value.dtype)
                    else:
                        array = np.full(self.num_envs, fill_value=None, dtype=object)
                else:
                    array =  vector_info[key]
                array[env_num] = value

            array_mask = vector_info.get(f"{key}", np.zeros(self.num_envs, dtype=np.bool_))
            array_mask[env_num] = True

            vector_info[key], vector_info[f"{key}"] = array, array_mask
        return vector_info

class AutoReset(Enum):
    """Represents different autoreset mode"""
    NEXT_STEP = "nextstep"
    SAME_STEP = "samestep"
    DISABLED = "disabled"

class SyncVectorEnv(VectorEnv):
    """Serially run multiple environments"""
    def __init__(
            self,
            env_fns: Iterable[Callable[[], Env]] | Sequence[Callable[[], Env]],
            copy: bool = True,
            observation_mode: str | Space = "same",
            autoreset: AutoReset = AutoReset.NEXT_STEP
    ):
        """Initiate the sync vector envs"""
        super().__init__()
        self.env_fns = env_fns
        self.copy = copy
        self.observation_mode = observation_mode

        assert isinstance(autoreset, AutoReset), f"You provided autoreset {autoreset} is not in AutoReset class."
        self.autoreset = autoreset

        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        self.metadata =  self.envs[0].metadata
        self.metadata["autoreset"] = self.autoreset
        self.render_mode = self.envs[0].render_mode

        #action space
        self.single_action_space = self.envs[0].action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        #observation space
        if isinstance(observation_mode, tuple) and len(observation_mode)==2:
            assert isinstance(observation_mode[0], Space)
            assert isinstance(observation_mode[1], Space)
            self.observation_space, self.single_observation_space = observation_mode
        else:
            if observation_mode == "same":
                self.single_observation_space = self.envs[0].observation_space
                self.observation_space = batch_space(self.single_observation_space, self.num_envs)
            elif observation_mode == "different":
                self.single_observation_space = self.envs[0].observation_space
                self.observation_space = batch_different_space([env.observation_space for env in self.envs])
            else:
                raise ValueError(f"Invalid observation mode: {observation_mode} provided.")
            
        for env in self.envs:
            if observation_mode == "same":
                assert(env.observation_space == self.single_observation_space
                ), (f"SyncVector(..., observation_mode='same') sub-environments observation spaces are not equal. " 
                    f"single_observation_space={self.single_observation_space}, but sub-environment space={env.observation_space}. " 
                    f"Either use observation_mode='different' or check the spaces")
            else:
                assert is_space_dtype_shape_equiv(env.observation_space, self.single_observation_space
                ), (f"SyncVector(..., observation_mode='different') sub-environments dtype and shape are not equivalent. " 
                    f"single_observation_space={self.single_observation_space} and sub-environment_space={env.observation_space}")
            assert (env.action_space == self.single_action_space
            ), f"Sub-environment action space={env.action_space} is not equal to single_action_space={self.single_action_space}"


        # attributes
        self._env_obs = [None for _ in range(self.num_envs)]
        self._observations = create_empty_array(self.single_observation_space, n=self.num_envs, fn=np.zeros)
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._termination = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncation = np.zeros((self.num_envs,), dtype=np.bool_)
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets each sub envs and concatenate the results together"""

        # seed
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + 1 for _ in range(self.num_envs)]
        
        assert (len(seed) == self.num_envs), f"the length of seeds must match the self.num_envs"

        # options
        if options is not None and "reset_mask" in options:
            reset_mask = options.pop("reset_mask")
            assert isinstance(reset_mask, np.ndarray), f"if reset_mask provided, which is, it should be an np.ndarray"
            assert (reset_mask.shape == (self.num_envs,)), f"reset_mask shape: {reset_mask.shape} must match with self.num_envs:{self.num_envs}"
            assert (reset_mask.dtype == np.bool_), f"reset_mask should be a np.bool_ type instead of the given {reset_mask.dtype}"
            assert np.any(reset_mask), f"reset_mask must atleast one True element in its array"

            # Reset only these specific environments in the vector batch, so
            self._termination[reset_mask] = False
            self._truncation[reset_mask] = False
            self._autoreset_envs[reset_mask] = False

            # Actual reset when reset mask is True
            infos = {}
            for i, (env, single_seed, env_mask) in enumerate(zip(self.envs, seed, reset_mask)):
                if env_mask:
                    self._env_obs[i], env_info = env.reset(seed=single_seed, options=options)
                    infos = self._add_info(vector_info=infos, env_info=env_info, env_num=i)
        
        else: 
            self._termination = np.zeros((self.num_envs,), dtype=np.bool_)
            self._truncation = np.zeros((self.num_envs,), dtype=np.bool_)
            self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

            # reset everything
            infos = {}
            for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
                self._env_obs[i], env_info = env.reset(seed=single_seed, options=options)
                infos = self._add_info(vector_info=infos, env_info=env_info, env_num=i)

        # concatenate into one
        self._observations = concatenate(self.single_observation_space, self._env_obs, self._observations)

        return deepcopy(self._observations) if self.copy else self._observations, infos
    
    def step(self, actions: ActType) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Take a step in each sub-env and return a tuple"""

        actions = iterate(self.action_space, actions)

        infos ={}
        for i, (action, _) in enumerate(zip(actions, self.envs, strict=True)):
            # NEXT STEP (default)
            if self.autoreset == AutoReset.NEXT_STEP:
                if self._autoreset_envs[i]:
                    self._env_obs[i], env_info = self.envs[i].reset()
                    self._rewards[i] = 0.0
                    self._termination[i] = False
                    self._truncation[i] = False
                else:
                    (   
                        self._env_obs[i], 
                        self._rewards[i],
                        self._termination[i],
                        self._truncation[i],
                        env_info
                    ) = self.envs[i].step(action=action)
            
            # DISABLED
            elif self.autoreset ==AutoReset.DISABLED:
                assert not self._autoreset_envs[i], f"{self._autoreset_envs[i]} should not be True"
                (   
                    self._env_obs[i], 
                    self._rewards[i],
                    self._termination[i],
                    self._truncation[i],
                    env_info
                ) = self.envs[i].step(action=action)
            
            # SAME STEP
            elif self.autoreset == AutoReset.SAME_STEP:
                (   
                    self._env_obs[i], 
                    self._rewards[i],
                    self._termination[i],
                    self._truncation[i],
                    env_info
                ) = self.envs[i].step(action=action)

                # check for final obs
                if self._termination[i] or self._truncation[i]:
                    infos = self._add_info(
                        vector_info = infos, 
                        env_info = {"final_obs": self._env_obs[i], "final_info": env_info},
                        env_num=i
                    )
                    # Reset in the same step if this is the final step
                    self._env_obs[i], env_info = self.envs[i].reset()
            else:
                raise ValueError(f"the provided autoreset mode: {self.autoreset} is not acceptable.")
            
            infos = self._add_info(vector_info=infos, env_info=env_info, env_num=i)

        # concatenate into one
        self._observations = concatenate(self.single_observation_space, self._env_obs, self._observations)
        self._autoreset_envs = np.logical_or(self._termination, self._truncation)

        return (
            deepcopy(self._observations) if self.copy else self._observations,
            np.copy(self._rewards),
            np.copy(self._termination),
            np.copy(self._truncation),
            infos
        )

    def render(self) -> None:
        """Return the rendering"""
        return tuple(env.render() for env in self.envs)
    
    def close_all(self, **kwargs: Any):
        """Close the environmetns"""
        if hasattr(self, "envs"):
            [env.close() for env in self.envs]

    @property
    def np_random(self) -> tuple[np.random.Generator, ...]:
        """Returns a tuple of random generators"""
        return self.get_attr("np_random")
    
    @property
    def np_random_seed(self) -> tuple[int, ...]:
        """Returns a tuple of random seeds"""
        return self.get_attr("np_random_seed")
    
    def get_attr(self, name:str) -> tuple[Any, ...]:
        """Find the attribute and return the tuple of values"""
        assert isinstance(name, str), f"miniRL.SyncVector.get_attr needs name params as a valid python str, not {type(name)}"
        return self._call(name)
    
    def set_attr(self, name: str, values: list[Any]| tuple[Any] | Any):
        """Set name to the sub env with values"""
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]

        # important check
        assert (len(values) == self.num_envs), f"Values provided to set_attr must be equal length to number of environments"

        for env, value in zip(self.envs, values):
            env.set_wrapper_attr(name, value)
    
    def _call(self, name:str, *args: Any, **kwargs:Any) -> tuple[Any, ...]:
        """Call sub-envs with name and apply args and kwargs"""
        results = []
        for env in self.envs:
            fns = env.get_wrapper_attr(name)

            if callable(fns):
                results.append(fns(*args, **kwargs))
            else:
                results.append(fns)

        return tuple(results)

class AsyncVectorEnv():
    def __init__(self) -> None:
        raise NotImplementedError(f"AsyncVector hasn't been implemented so far")