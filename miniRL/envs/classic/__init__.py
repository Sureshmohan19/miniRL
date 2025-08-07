"""miniRL.envs.classic"""

from miniRL.registration import register
from miniRL.envs.classic.mountain_car import MountainCar

_all__ = ["MountainCar"]

register(
    name="MountainCar-default",
    entry_point="miniRL.envs.classic:MountainCar"
)