"""miniRL.envs.classic"""

from miniRL.registration import register
from miniRL.envs.classic.mountain_car import MountainCar
from miniRL.envs.classic.cartpole import CartPole

_all__ = ["MountainCar", "CartPole"]

register(
    name="MountainCar-default",
    entry_point="miniRL.envs.classic:MountainCar",
    description="A car on a one-dimensional track, between two mountains, must reach the flag on top of the mountain on the right.",
    max_steps=200,
    render_mode= "rgb_array",
    goal_velocity= 0,
    metadata= {"version": "1.0"}
)

register(
    name="CartPole-default",
    entry_point="miniRL.envs.classic:CartPole",
    description="Classic CartPole balancing problem",
    max_steps=500,
    render_mode= "rgb_array",
    sutton_barto_reward= False,
    metadata={"version": "1.0"}
)