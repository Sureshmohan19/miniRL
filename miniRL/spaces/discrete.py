"""miniRL.spaces.discrete"""

from typing import Any
from miniRL.spaces import Space
import numpy as np

__all__ = ["Discrete"]

class Discrete(Space[np.int64]):
    def __init__(
            self,
            n: int | np.integer[Any],
            seed: int | np.random.Generator | None = None,
            start: int | np.integer[Any] = 0,
    ):
        """Initialize the discrete space"""
        assert np.issubdtype(type(n), np.integer), f"n must be an integer, got {type(n)}"
        assert np.issubdtype(type(start), np.integer), f"start must be an integer, got {type(start)}"
        assert start >= 0, f"start must be non-negative, got {start}"

        self.n = np.int64(n)
        self.start = np.int64(start)

        super().__init__(shape=(), dtype=np.int64, seed=seed)

    def sample(self) -> np.int64:
        """Sample a random element from the space."""
        return self.start + self.np_random.integers(self.n)
    
    def contains(self, x: Any) -> bool:
        """Check if a value is a valid element of the space."""
        if isinstance(x, int):
            x_as_int64 = np.int64(x)
        elif isinstance (x, (np.generic, np.ndarray)) and (np.issubdtype(x.dtype, np.integer) and x.shape == ()):
            x_as_int64 = np.int64(x)
        else:
            return False
        
        return bool(self.start <= x_as_int64 < self.start + self.n)
    
    def __repr__(self) -> str:
        """Return a string representation of the space."""
        return f"Discrete({self.n}, start={self.start})" if self.start != 0 else f"Discrete({self.n})"
    
    def __eq__(self, other: Any) -> bool:
        """Check if two spaces are equal. Checking for space, n, and start."""
        return isinstance(other, Discrete) and self.n == other.n and self.start == other.start