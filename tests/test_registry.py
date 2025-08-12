"""miniRL.tests.test_registry"""

from miniRL.registration import register, make, registry
from tests._dummy_env import DummyEnv
import numpy as np
from miniRL.registration import (
    make,
    register,
    registry,
    unregister,
    list_environments,
    get_metadata,
    get_required_params,
)

# A simple decorator to clean up registrations before and after a test
def with_cleanup(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if name in registry:
                unregister(name)
            try:
                func(*args, **kwargs)
            finally:
                if name in registry:
                    unregister(name)
        return wrapper
    return decorator

def test_register_and_make_with_max_steps():
    # unique name to avoid clashes across runs
    name = "Dummy-v0"
    try:
        register(name, "tests._dummy_env:DummyEnv")
    except ValueError:
        # already registered in this session; ok for idempotency in tests
        pass

    env = make(name, max_steps=2)
    from miniRL.wrappers import StepLimit
    assert isinstance(env, StepLimit)
    assert isinstance(env.env, DummyEnv)
    obs, info = env.reset()
    _, _, term, trunc, _ = env.step(1)
    assert trunc is False
    _, _, term, trunc, _ = env.step(1)
    assert trunc is True

@with_cleanup("DummyBasic-v0")
def test_basic_registration_and_creation():
    """Tests the core `register` and `make` functionality."""
    print("--- Running: test_basic_registration_and_creation ---")
    name = "DummyBasic-v0"
    
    register(name, "tests._dummy_env:DummyEnv")
    assert name in registry, f"'{name}' was not found in the registry after registration."
    
    env = make(name)
    assert isinstance(env, DummyEnv), f"make('{name}') did not return a DummyEnv instance."
    
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert np.array_equal(obs, np.array([0.0, 0.0]))
    
    obs, reward, term, trunc, info = env.step(1)
    assert reward == 1.0
    assert not term
    assert not trunc
    print("✓ PASSED\n")


def test_classic_environment_auto_registration():
    """Tests that importing the classic envs package registers them."""
    print("--- Running: test_classic_environment_auto_registration ---")
    
    # This import should trigger the `register` calls in its __init__.py
    import miniRL.envs.classic
    
    name = "CartPole-default"
    assert name in registry, f"'{name}' was not auto-registered."
    
    env = make(name)
    from miniRL.envs.classic.cartpole import CartPole
    from miniRL.wrappers import StepLimit
    assert isinstance(env, StepLimit)
    assert isinstance(env.env, CartPole)
    
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    print("✓ PASSED\n")


@with_cleanup("DummyDuplicate-v0")
def test_duplicate_registration_error():
    """Tests that registering the same name twice raises a ValueError."""
    print("--- Running: test_duplicate_registration_error ---")
    name = "DummyDuplicate-v0"
    
    register(name, "_dummy_env:DummyEnv") # First time is fine
    
    try:
        register(name, "_dummy_env:DummyEnv") # Second time should fail
        assert False, "Duplicate registration should have raised a ValueError."
    except ValueError as e:
        assert "already registered" in str(e)
    print("✓ PASSED\n")


def test_make_nonexistent_environment_error():
    """Tests that `make` for an unregistered environment raises a ValueError."""
    print("--- Running: test_make_nonexistent_environment_error ---")
    name = "NonExistent-v0"
    assert name not in registry
    
    try:
        make(name)
        assert False, f"make('{name}') should have raised a ValueError."
    except ValueError as e:
        assert "not found" in str(e)
    print("✓ PASSED\n")


@with_cleanup("DummyBadEntry-v0")
def test_make_with_invalid_entry_point_error():
    """Tests that `make` fails if the entry point is invalid."""
    print("--- Running: test_make_with_invalid_entry_point_error ---")
    name = "DummyBadEntry-v0"
    
    # Registration itself should succeed as it's lazy
    register(name, "_dummy_env:NonExistentClass")
    
    # `make` should fail because it tries to import the class
    try:
        make(name)
        assert False, "make() with a bad entry point should have raised a RuntimeError."
    except RuntimeError as e:
        assert "Could not import the entry point" in str(e)
    print("✓ PASSED\n")


@with_cleanup("DummyMetadata-v0")
def test_registration_with_kwargs_and_metadata():
    """Tests registering with default kwargs and retrieving metadata."""
    print("--- Running: test_registration_with_kwargs_and_metadata ---")
    name = "DummyMetadata-v0"
    
    # Note: DummyEnv doesn't have kwargs, but we test the mechanism.
    # CartPole's registration is a better real-world example.
    register(
        name=name,
        entry_point="_dummy_env:DummyEnv",
        description="A test env with metadata.",
        metadata= {"version": "2.0", "testing": True}
    )
    
    metadata = get_metadata(name)
    assert metadata.get("version") == "2.0"
    assert metadata.get("testing") is True
    
    # Test that `get_metadata` for an env with no metadata returns empty dict
    metadata_cp = get_metadata("CartPole-default")
    assert isinstance(metadata_cp, dict)

    print("✓ PASSED\n")


@with_cleanup("DummyBasic-v0") # Make sure it gets cleaned up
def test_get_required_parameters():
    """Tests the `get_required_params` utility function."""
    print("--- Running: test_get_required_parameters ---")
    
    # Register the environment needed for this specific test
    register("DummyBasic-v0", "_dummy_env:DummyEnv")
    
    # DummyEnv's __init__ takes no arguments (besides self)
    dummy_params = get_required_params("DummyBasic-v0")
    assert isinstance(dummy_params, set)
    assert len(dummy_params) == 0
    
    # CartPole's __init__ has no *required* arguments either
    cp_params = get_required_params("CartPole-default")
    assert isinstance(cp_params, set)
    assert len(cp_params) == 0
    print("✓ PASSED\n")
