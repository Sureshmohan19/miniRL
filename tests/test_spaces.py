"""miniRL.tests.test_spaces"""

import numpy as np

from miniRL.spaces import Box, Discrete


def test_box_contains_and_sample_float():
    box = Box(low=np.array([0.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
    assert box.contains(np.array([0.5, 0.0], dtype=np.float32))
    for _ in range(5):
        s = box.sample()
        assert s.shape == (2,) and s.dtype == np.float32
        assert np.all(s >= box.low) and np.all(s <= box.high)


def test_box_contains_and_sample_int():
    box = Box(low=np.array([0, 0], dtype=np.int32), high=np.array([5, 10], dtype=np.int32), dtype=np.int32)
    assert box.contains(np.array([1, 2], dtype=np.int32))
    for _ in range(5):
        s = box.sample()
        assert s.shape == (2,) and s.dtype == np.int32
        assert np.all(s >= box.low) and np.all(s <= box.high)


def test_box_seed_reproducible():
    box1 = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32, seed=123)
    box2 = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32, seed=123)
    s1 = box1.sample()
    s2 = box2.sample()
    assert np.allclose(s1, s2)


def test_discrete_contains_and_sample():
    d = Discrete(3)
    assert d.contains(0)
    assert d.contains(np.int64(2))
    assert not d.contains(3)
    for _ in range(5):
        a = int(d.sample())
        assert 0 <= a < 3


def test_discrete_start_offset_and_equality_repr():
    d1 = Discrete(4, start=1)
    d2 = Discrete(4, start=1)
    d3 = Discrete(4, start=0)
    assert d1 == d2
    assert d1 != d3
    r = repr(d1)
    assert "Discrete(4, start=1)" in r


