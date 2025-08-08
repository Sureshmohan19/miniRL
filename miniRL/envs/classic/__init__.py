"""miniRL.envs.classic"""

from miniRL.registration import register
from miniRL.envs.classic.mountain_car import MountainCar
from miniRL.envs.classic.cartpole import CartPole

_all__ = ["MountainCar", "CartPole"]

register(
    name="MountainCar-default",
    entry_point="miniRL.envs.classic:MountainCar"
)

register(
    name="CartPole-default",
    entry_point="miniRL.envs.classic:CartPole"
)