"""miniRL.utils"""

import numpy as np
from typing import Any

def np_random(seed: int | None = None) -> tuple[np.random.Generator, int]:
    """Create a new PRNG and return it along with the seed"""
    if seed is not None and not (isinstance(seed, int) and seed >= 0):
        raise ValueError("Seed must be a non-negative integer")
    
    seed_seq = np.random.SeedSequence(seed)
    entropy = seed_seq.entropy
    assert isinstance(entropy, int), f"Expected a python integer for seed entropy value, but got {type(entropy)}"
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    
    return rng, entropy

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