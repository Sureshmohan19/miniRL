"""miniRL.tests.test_core_wrappers"""

import numpy as np

from miniRL.core import Env
from miniRL.wrappers import StepLimit
from tests._dummy_env import DummyEnv


def test_env_reset_seeds_rng_and_returns_tuple():
    env = DummyEnv()
    obs, info = env.reset(seed=123)
    assert isinstance(env, Env)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    # accessing np_random should work and be deterministic after seeding
    r1 = env.np_random.integers(0, 1000)
    env2 = DummyEnv()
    env2.reset(seed=123)
    r2 = env2.np_random.integers(0, 1000)
    assert r1 == r2


def test_step_contract_and_state_update():
    env = DummyEnv()
    env.reset()
    o1, r1, term1, trunc1, _ = env.step(1)
    assert o1.shape == (2,)
    assert r1 in {0.0, 1.0}
    assert term1 is False and trunc1 is False
    o2, r2, term2, trunc2, _ = env.step(0)
    assert o2[0] > o1[0]
    assert term2 is False and trunc2 is False


def test_step_limit_wrapper_enforces_max_steps_and_requires_reset():
    env = StepLimit(DummyEnv(), max_steps=2)
    # requires reset before step
    try:
        env.step(0)
        assert False, "expected RuntimeError before reset"
    except RuntimeError:
        pass

    env.reset()
    _, _, term, trunc, _ = env.step(1)
    assert term is False and trunc is False
    _, _, term, trunc, _ = env.step(1)
    assert trunc is True


