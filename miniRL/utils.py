"""miniRL.utils"""

import numpy as np

def np_random(seed: int | None = None) -> tuple[np.random.Generator, int]:
    """Create a new PRNG and return it along with the seed"""
    if seed is not None and not (isinstance(seed, int) and seed >= 0):
        raise ValueError("Seed must be a non-negative integer")
    
    seed_seq = np.random.SeedSequence(seed)
    entropy = seed_seq.entropy
    assert isinstance(entropy, int), f"Expected a python integer for seed entropy value, but got {type(entropy)}"
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    
    return rng, entropy