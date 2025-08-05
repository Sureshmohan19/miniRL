"""miniRL.types"""

from typing import TypeVar

__all__ = ["ObsType", "ActType", "SpaType"]

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
SpaType = TypeVar("SpaType", covariant=True)
