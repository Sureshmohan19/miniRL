"""miniRL.tests.test_mountain_car"""

import numpy as np

from miniRL.envs.classic.mountain_car import MountainCar


def test_mountain_car_reset_and_step_basic():
    env = MountainCar()
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert env.observation_space.contains(obs)

    obs2, r, term, trunc, _ = env.step(1)
    assert env.observation_space.contains(obs2)
    assert isinstance(r, float)
    assert trunc is False


