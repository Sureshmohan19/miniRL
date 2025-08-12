"""miniRL.tests.test_utils"""

import numpy as np

from miniRL.utils import parse_reset_bound
from miniRL.random_utils import np_random


def test_np_random_returns_generator_and_seed():
    rng, seed = np_random(42)
    assert isinstance(rng, np.random.Generator)
    assert isinstance(seed, int)


def test_np_random_rejects_negative_seed():
    try:
        np_random(-1)
        assert False, "expected ValueError for negative seed"
    except ValueError:
        pass


def test_parse_reset_bound_defaults_and_validation():
    low, high = parse_reset_bound(default_low=-0.6, default_high=-0.4, options=None)
    assert (low, high) == (-0.6, -0.4)

    low, high = parse_reset_bound(default_low=0.0, default_high=1.0, options={"low": 0.2, "high": 0.8})
    assert (low, high) == (0.2, 0.8)

    # coercion from strings
    low, high = parse_reset_bound(default_low=0.0, default_high=1.0, options={"low": "0.1", "high": "0.9"})
    assert (abs(low - 0.1) < 1e-9) and (abs(high - 0.9) < 1e-9)

    # low>high error
    try:
        parse_reset_bound(default_low=0.0, default_high=1.0, options={"low": 0.9, "high": 0.2})
        assert False, "expected ValueError for low>high"
    except ValueError:
        pass


