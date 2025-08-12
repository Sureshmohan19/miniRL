"""miniRL.utils"""

import numpy as np
from typing import Any, Sequence, Callable, Iterable, Iterator
from functools import singledispatch
from copy import deepcopy

from miniRL.spaces import Space, Box, Discrete
from miniRL.types import SpaType

def parse_reset_bound(
        default_low: float, 
        default_high: float,
        options: dict[str, Any] | None = None, 
) -> tuple[float, float]:
    """for box spaces, parse options params and make sure the low and high are within 
    the pre-defined low and high"""
    if options is None:
        return default_low, default_high

    raw_low: Any = options.get("low") if "low" in options else default_low
    raw_high: Any = options.get("high") if "high" in options else default_high

    # if provided, low and high could be int as well. so need to check this
    try:
        low = float(raw_low)
        high = float(raw_high)
    except (ValueError, TypeError) as e:
        raise ValueError(f"bounds -> low:{low} and high:{high} should be possible to be converted to float")
    
    if low > high:
        raise ValueError(f"bound low:{low} must be smaller than high:{high}")
    
    return low, high

# vector space utils
@singledispatch
def batch_space(space: Space[Any], n: int = 1) -> Space[Any]:
    """Just a dispatch function which routes to matching space type functions"""
    raise TypeError(f"The space {space} is not a miniRL.space instance. Please provide correct one")

@batch_space.register(Box)
def _batch_space_box(space: Box, n: int = 1):
    """Batch space function for Space.Box type"""
    repeats = tuple[[n] + [1] * space.low.ndim]
    low, high = np.tile(space.low, repeats), np.tile(space.high, repeats)
    return Box(low=low, high=high, dtype=space.dtype, seed=deepcopy(space.np_random))

@batch_space.register(Discrete)
def _batch_space_discrete(space: Discrete, n: int = 1):
    """Batch space function for Space.Discrete"""
    # currently miniRL doesn't support MultiDiscrete space type so we are returning a Box here
    low = np.full((n,), space.start, dtype=space.dtype)
    high = low + np.full((n,), space.n, dtype=space.dtype) -1
    return Box(low=space.start, high=high, dtype=space.dtype, seed=deepcopy(space.np_random))

@singledispatch
def batch_different_space(spaces: Sequence[Space]) -> Space:
    """When observation mode in vector env is different"""
    assert len(spaces) > 0, f"The length of spaces should not be empty"
    assert all(isinstance(space, type(space[0])) for space in spaces), f"Expects all space types to be the same"

    return batch_different_space.dispatch(type(spaces[0]))(spaces)

@batch_different_space.register(Box)
def _batch_different_space_Box(spaces: list[Box]):
    """Batch different space for Space.Box"""
    assert all (spaces[0].dtype == space.dtype for space in spaces), f"Expected all dtypes to be equal"
    assert all(spaces[0].low.shape == space.low.shape for space in spaces), f"Expected all low.shape to be equal"
    assert all(spaces[0].high.shape == space.high.shape for space in spaces), f"Expected all high.shpae to be equal"
    
    return Box(
        low= np.array([space.low for space in spaces]),
        high= np.array([space.high for space in spaces]),
        dtype= spaces[0].dtype,
        seed= deepcopy(spaces[0].np_random) 
    )

@batch_different_space.register(Discrete)
def _batch_different_space_Discrete(spaces: list[Discrete]):
    """Batch different space for Space.Discrete"""
    assert all(spaces[0].dtype == space.dtype for space in spaces), f"Expected all dtypes to be equal"
    assert all(spaces[0].shape == space.shape for space in spaces), f"Expected all shapes to be equal"

    return Box(
        low = np.array([space.start for space in spaces]),
        high = np.array([space.start + space.n for space in spaces]) -1,
        dtype = spaces[0].dtype,
        seed = deepcopy(spaces[0].np_random)
    )

@singledispatch
def is_space_dtype_shape_equiv(space_1: Space, space_2: Space) -> bool:
    """For different space, check whether their dtype and shape are equivalent"""
    if isinstance(space_1, Space) and isinstance(space_2, Space):
        raise NotImplementedError(f"need to specify particular miniRL.Space and generic one doesn't work")
    
@is_space_dtype_shape_equiv.register(Box)
@is_space_dtype_shape_equiv.register(Discrete)
def _is_space_dtype_shape_equiv_box_discrete(space_1: Box | Discrete, space_2: Box | Discrete):
    """Check whether two spaces dtypes and shapes are equivalent - for box and discrete"""
    are_they = (type(space_1) is type(space_2) and space_1.shape == space_2.shape and space_1.dtype == space_2.dtype)
    return are_they

@singledispatch
def create_empty_array(space: Space, n: int = 1, fn: Callable = np.zeros) -> np.ndarray:
    """Create an empty array"""
    supported_spaces = [Box, Discrete]
    raise NotImplementedError(f"Space type={space}<{type(space).__name__}> is not implemented.")

@create_empty_array.register(Box)
@create_empty_array.register(Discrete)
def _create_empty_array_box_discrete(space: Space, n: int =1, fn=np.zeros) -> np.ndarray:
    """Create an emtpy array - for box and discrete"""
    return fn((n,) + space.shape, dtype=space.dtype)

@singledispatch
def concatenate(space: Space, items: Iterable, out: np.ndarray) -> np.ndarray:
    """Concatenate multiple obs from space into a single object"""
    if isinstance(space, Space):
        raise TypeError(f"Concatenate needs an instance of miniRL.Space like Box or Discrete. Not {type(space)}")
    
@concatenate.register(Box)
@concatenate.register(Discrete)
def _concatenate_box_discrete(space: Box | Discrete, items: Iterable, out: np.ndarray) -> np.ndarray:
    """Concatenate Box and Discrete space obs"""
    return np.stack(items, axis=0, out=out)

@singledispatch
def iterate(space: Space[SpaType], items: SpaType) -> Iterator:
    """Iterate over the elements of the batched space"""
    if isinstance(space, Space):
        raise TypeError(f"iterate needs an instance of miniRL.Space like Box or Discrete. Not {type(space)}")
    
@iterate.register(Box)
def _iterate_box(space: Box, items: np.ndarray):
    """iterate for miniRL.Box type"""
    try:
        return iter(items)
    except TypeError as e:
        raise TypeError(f"unable to iterate over Box elements: {items}") from e
    
@iterate.register(Discrete)
def _iterate_discrete(space: Discrete, items: Iterable):
    """iterate for miniRL.Discrete type"""
    raise TypeError(f"miniRL.Discrete elements are not iterable")