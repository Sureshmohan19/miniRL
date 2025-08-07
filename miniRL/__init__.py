"""miniRL"""

from miniRL.spaces import Space, Box, Discrete
from miniRL.types import ObsType, ActType, SpaType, WrapperActType, WrapperObsType
from miniRL.core import Env, Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
from miniRL.wrappers import StepLimit
from miniRL.envs import EnvironmentRegistry
from miniRL.registration import make, register

__all__ = [
    "Space", "Box", "Discrete",
    "ObsType", "ActType", "SpaType", "WrapperObsType", "WrapperActType",
    "Env", "Wrapper", "ObservationWrapper", "ActionWrapper", "RewardWrapper",
    "StepLimit",
    "EnvironmentRegistry",
    "make", "register",
]
