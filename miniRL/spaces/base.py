"""miniRL.spaces.base"""

from typing import Any, Sequence, Generic

import numpy as np
from numpy.typing import DTypeLike

from miniRL.utils import np_random
from miniRL.types import SpaType

__all__ = ["Space"]

class Space(Generic[SpaType]):
    def __init__(
            self,
            shape: Sequence[int] | None = None,
            dtype: DTypeLike | None = None,
            seed: int | np.random.Generator | None = None,
    ):
        """Initialize the space"""
        self._shape = None if shape is None else tuple(shape)
        self._dtype = None if dtype is None else np.dtype(dtype)
        self._np_random = None

        if seed is not None:
            if isinstance(seed, np.random.Generator):
                self._np_random = seed
            else:
                self.seed(seed)
    
    @property
    def np_random(self) -> np.random.Generator:
        """Return the PRNG"""
        if self._np_random is None:
            self.seed()

        if self._np_random is None:
            self._np_random, _ = np_random()

        return self._np_random
    
    def seed(self, seed: int | None = None) -> int:
        """Seed the PRNG"""
        self._np_random, np_random_seed = np_random(seed)
        assert isinstance(np_random_seed, int), "seeding did not return a valid python int type"
        return np_random_seed

    def sample(self) -> SpaType:
        """Sample a random element from the space."""
        raise NotImplementedError
    
    @property
    def shape(self) -> tuple[int, ...] | None:
        """Return the shape of the space."""
        return self._shape
    
    @property
    def dtype(self) -> np.dtype[Any] | None:
        """Return the dtype of the space."""
        return self._dtype
    
    def contains(self, x: Any) -> bool:
        """Check if a value is a valid element of the space."""
        raise NotImplementedError
    
    def __contains__(self, x: Any) -> bool:
        """Check if a value is a valid element of the space."""
        return self.contains(x)
    
    def __repr__(self) -> str:
        """Return a string representation of the space."""
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype})"