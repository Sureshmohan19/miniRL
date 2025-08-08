"""miniRL.tests.test_registry"""

from miniRL.registration import register, make


def test_register_and_make_with_max_steps():
    # unique name to avoid clashes across runs
    name = "Dummy-v0"
    try:
        register(name, "tests._dummy_env:DummyEnv")
    except ValueError:
        # already registered in this session; ok for idempotency in tests
        pass

    env = make(name, max_steps=2)
    obs, info = env.reset()
    _, _, term, trunc, _ = env.step(1)
    assert trunc is False
    _, _, term, trunc, _ = env.step(1)
    assert trunc is True


