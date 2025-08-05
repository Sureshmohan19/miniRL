"""miniRL.spaces.box"""

from collections.abc import Iterable
from typing import Any, SupportsFloat, Sequence
from miniRL.spaces import Space
import numpy as np
from numpy.typing import NDArray
import warnings

__all__ = ["Box"]

def float_or_int(x: Any) -> bool:
    """Check if a value is a float or an integer."""
    return np.issubdtype(type(x), np.floating) or np.issubdtype(type(x), np.integer)

class Box(Space[NDArray[Any]]):
    def __init__(
            self,
            low: SupportsFloat | NDArray[Any],
            high: SupportsFloat | NDArray[Any],
            shape: Sequence[int] | None = None,
            seed: int | np.random.Generator | None = None,
            dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32,
    ):
        """Initialize the box space"""
        self._dtype = np.dtype(dtype)
        if not (np.issubdtype(self._dtype, np.floating) or np.issubdtype(self._dtype, np.integer) or self._dtype == np.bool_):
            raise ValueError(f"Box dtype must be one of integer, floating or bool type but got {self._dtype}")
        
        if shape is not None:
            if not isinstance(shape, Iterable):
                raise TypeError(f"Box space shape must be an iterable but got type={type(shape)}")
            elif not all (np.issubdtype(type(dim), np.integer) for dim in shape):
                raise TypeError(f"Box space elements must be integers but got type={tuple(type(dim) for dim in shape)}")
        
            shape = tuple(int(dim) for dim in shape)

        elif isinstance(low, np.ndarray) and isinstance(high, np.ndarray):
            if low.shape != high.shape:
                raise ValueError(f"Box space low and high must have the same shape but got {low.shape} and {high.shape}")
            shape = low.shape # because low and high have the same shape, we can use either one
        
        elif isinstance(low, np.ndarray):
            shape = low.shape
        elif isinstance(high, np.ndarray):
            shape = high.shape
        elif float_or_int(low) and float_or_int(high):
            shape = (1,)
        else:
            raise ValueError(f"Box space low and high must be either both numpy arrays or both floats or integers but got {type(low)} and {type(high)}")

        self._shape: tuple[int, ...] = shape

        if self.dtype == np.bool_:
            dtype_min, dtype_max = 0, 1
        elif np.issubdtype(self.dtype, np.floating):
            dtype_min, dtype_max = float(np.finfo(self.dtype).min), float(np.finfo(self.dtype).max)
        elif np.issubdtype(self.dtype, np.integer):
            dtype_min, dtype_max = int(np.iinfo(self.dtype).min), int(np.iinfo(self.dtype).max)
        else:
            raise ValueError(f"Box space dtype must be one of integer, floating or bool type but got {self.dtype}")
        
        self.low, self.bounded_below = self._validate_low(low, dtype_min)
        self.high, self.bounded_above = self._validate_high(high, dtype_max)

        if self.low.shape != shape:
            raise ValueError(f"Box space low and high must have the same shape but got {self.low.shape} and {shape}")
        
        if self.high.shape != shape:
            raise ValueError(f"Box space low and high must have the same shape but got {self.high.shape} and {shape}")
        
        if np.any(self.low > self.high):
            raise ValueError(f"Box space low must be less than high but got {self.low} and {self.high}")
        
        super().__init__(shape=self.shape, dtype=self.dtype, seed=seed)

    def _validate_low(self, low, dtype_min) -> tuple[np.ndarray, np.ndarray]:
        """Validate the low bound of the box space"""
        
        if float_or_int(low):
            low = np.full(self._shape, low, dtype=self.dtype)

        # now low must be an array
        if not isinstance(low, np.ndarray):
            raise TypeError(f"Box space low must be an array but got type={type(low)}")
        
        elif not (np.issubdtype(low.dtype, np.floating) or np.issubdtype(low.dtype, np.integer) or low.dtype == np.bool_):
            raise ValueError(f"Box space low must be an array of integer, floating or bool type but got {low.dtype}")
        
        elif np.any(np.isnan(low)):
            raise ValueError(f"Box space low must not contain NaN values")
        
        bounded_below = -np.inf < low

        if np.any(np.isneginf(low)):
            if self.dtype.kind == "i":
                low[np.isneginf(low)] = dtype_min # replace negative infinity with the minimum value of the dtype
            elif self.dtype.kind in {"u", "b"}:
                raise ValueError(f"Box space low must not contain negative infinity values for unsigned or boolean dtype")
            
        if low.dtype != self.dtype and np.any(low < dtype_min):    
            raise ValueError(f"Box space low must not contain values less than {dtype_min} for dtype={low.dtype}")
            
        if (np.issubdtype(low.dtype, np.floating) and np.issubdtype(self.dtype, np.floating) and np.finfo(self.dtype).precision < np.finfo(low.dtype).precision):
            warnings.warn(f"Box space low dtype has been downcasted to {self.dtype} from {low.dtype} which will lead to precision loss")

        return low.astype(self.dtype), bounded_below
        
    def _validate_high(self, high, dtype_max) -> tuple[np.ndarray, np.ndarray]:
        """Validate the high bound of the box space"""
        
        if float_or_int(high):
            high = np.full(self._shape, high, dtype=self.dtype)

        # now high must be an array
        if not isinstance(high, np.ndarray):
            raise TypeError(f"Box space high must be an array but got type={type(high)}")
        
        if not (np.issubdtype(high.dtype, np.floating) or np.issubdtype(high.dtype, np.integer) or high.dtype == np.bool_):
            raise ValueError(f"Box space high must be an array of integer, floating or bool type but got {high.dtype}")
        
        elif np.any(np.isnan(high)):
            raise ValueError(f"Box space low must not contain NaN values")
        
        bounded_above = np.inf > high

        if np.any(np.isposinf(high)):
            if self.dtype.kind == "i":
                high[np.isposinf(high)] = dtype_max # replace positive infinity with the maximum value of the dtype
            elif self.dtype.kind in {"u", "b"}:
                raise ValueError(f"Box space high must not contain positive infinity values for unsigned or boolean dtype")
            
        if high.dtype != self.dtype and np.any(high > dtype_max):
            raise ValueError(f"Box space high must not contain values greater than {dtype_max} for dtype={high.dtype}")
            
        if (np.issubdtype(high.dtype, np.floating) and np.issubdtype(self.dtype, np.floating) and np.finfo(self.dtype).precision < np.finfo(high.dtype).precision):
            warnings.warn(f"Box space high dtype has been downcasted to {self.dtype} from {high.dtype} which will lead to precision loss")

        return high.astype(self.dtype), bounded_above
        
    def sample(self) -> NDArray[Any]:
        """Sample a random value from the box space."""
        high = self.high if self.dtype.kind == "f" else self.high.astype("int64") + 1
        sample = np.empty(self.shape)

        unbounded = ~self.bounded_below & ~self.bounded_above
        upper_bounded = ~self.bounded_below & self.bounded_above
        lower_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        sample[unbounded] = self.np_random.normal(size=unbounded[unbounded].shape)
        sample[lower_bounded] = self.np_random.exponential(size=lower_bounded[lower_bounded].shape) + self.low[lower_bounded]
        sample[upper_bounded] = (-self.np_random.exponential(size=upper_bounded[upper_bounded].shape) + high[upper_bounded])
        sample[bounded] = self.np_random.uniform(low=self.low[bounded], high=high[bounded], size=bounded[bounded].shape)

        if self.dtype.kind in ["i", "u", "b"]:
            sample = np.floor(sample)

        if np.issubdtype(self.dtype, np.signedinteger):
            dtype_min = np.iinfo(self.dtype).min + 2
            dtype_max = np.iinfo(self.dtype).max - 2
            sample = sample.clip(min=dtype_min, max=dtype_max)
        elif np.issubdtype(self.dtype, np.unsignedinteger):
            dtype_min = np.iinfo(self.dtype).min
            dtype_max = np.iinfo(self.dtype).max
            sample = sample.clip(min=dtype_min, max=dtype_max)

        sample = sample.astype(self.dtype)

        if self.dtype == np.int64:
            sample = sample.clip(min=self.low, max=self.high)

        return sample
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the box space."""
        return self._shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get the dtype of the box space."""
        return self._dtype
    
    def is_bounded(self) -> tuple[bool, bool]:
        """Check if the box space is bounded below and above."""
        return bool(np.all(self.bounded_below)), bool(np.all(self.bounded_above))
    
    def contains(self, x: Any) -> bool:
        """Check if the box space contains a value."""
        if not isinstance(x, np.ndarray):
            warnings.warn(f"Box space contains method expects a numpy array but got type={type(x)}. Converting to numpy array.")
            try: 
                x = np.array(x, dtype=self.dtype)
            except (ValueError, TypeError) as e:
                return False
            
        return bool(np.can_cast(x.dtype, self.dtype) and x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high))
    
    def __repr__(self) -> str:
        """Return a string representation of the box space."""
        return f"Box(low={self.low}, high={self.high}, shape={self.shape}, dtype={self.dtype})"
    
    def __eq__(self, other: Any) -> bool:
        """Check if the box space is equal to another object."""
        return isinstance(other, Box) and (self.shape == other.shape) and (self.dtype == other.dtype) and np.allclose(self.low, other.low) and np.allclose(self.high, other.high)