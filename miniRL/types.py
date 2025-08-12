"""miniRL.types"""

from typing import TypeVar

__all__ = ["ObsType", "ActType", 
           "SpaType", "WrapperObsType", "WrapperActType",
           "ArrayType"]

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
SpaType = TypeVar("SpaType", covariant=True)
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")
ArrayType = TypeVar("ArrayType")
