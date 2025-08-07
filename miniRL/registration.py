"""miniRL.registration"""

import importlib
from typing import Any

from miniRL.core import Env
from miniRL.envs.registry import EnvironmentRegistry
from miniRL.wrappers import StepLimit

__all__ = ["make", "register"]

registry = EnvironmentRegistry()

#public function for registration
def make(
        name: str,
        max_steps: int | None = None,
        **kwargs: Any
) -> Env[Any, Any]:
    """Look for the environmet and create an instance of that"""
    env_reg = registry.get_environment_class(name=name)
    env = env_reg(**kwargs)

    if max_steps is not None and (max_steps > 0):
        env = StepLimit(env, max_steps=max_steps)

    return env

def register(
        name: str,
        entry_point: str
):
    """Add a new environment to the registry"""
    module_name, class_name = entry_point.split(":")
    module = importlib.import_module(module_name)
    env_reg = getattr(module, class_name)
    registry.register(name=name, env_reg=env_reg)