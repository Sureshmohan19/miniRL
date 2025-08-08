"""miniRL.tests.test_dummy_env"""

import numpy as np
from miniRL.core import Env
from tests._dummy_env import DummyEnv

def test_dummy_env_reset():
    env = DummyEnv()
    assert isinstance(env, Env)
    obs, info = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

def test_dummy_env_progress_and_termination():
    env = DummyEnv()
    env.reset()

    # step 1
    obs, reward, terminated, truncated, info = env.step(1)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)
    assert reward in {0.0, 1.0}
    assert terminated is False
    assert truncated is False
    assert isinstance(info, dict)

    # step 2
    obs, reward, terminated, truncated, _ = env.step(0)
    assert terminated is False
    assert truncated is False

    # step 3 -> should terminate
    obs, reward, terminated, truncated, _ = env.step(1)
    assert terminated is True
    assert truncated is False

def test_dummy_env_action_space_and_observation_space():
    env = DummyEnv()
    env.reset()
    assert env.action_space.contains(0)
    assert env.action_space.contains(1)
    assert not env.action_space.contains(2)
    s = env.observation_space.sample()
    assert env.observation_space.contains(s)