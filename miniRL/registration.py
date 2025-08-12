"""miniRL.registration"""

import importlib
import inspect
from typing import Any, Set

from miniRL.core import Env
from miniRL.envs.registry import EnvironmentRegistry, EnvSpec, VectorizationMode
from miniRL.wrappers import StepLimit

__all__ = ["make", "make_vec", "register", "registry", 
           "list_environments", "get_metadata", "unregister", "get_environment_class","get_required_params"
]

registry = EnvironmentRegistry()

#public function for registration
def make(name: str, max_steps: int | None = None, **kwargs: Any) -> Env[Any, Any]:
    """Look for the environmet that is registered using miniRL.register and create an instance of that"""
    return registry.make(name=name, max_steps=max_steps, **kwargs)

def make_vec(name: str, num_envs: int = 1, vectorization_mode: VectorizationMode | None = None, vector_kwargs: dict[str, Any] | None = None, **kwargs: Any):
    """Look for the parallel enviornment that is registered using miniRL.register and create an instance of that"""
    return registry.make_vec(name=name, num_envs=num_envs, vectorization_mode=vectorization_mode, vector_kwargs=vector_kwargs, **kwargs)
    
def register(name: str, entry_point: str | None, vector_entry_point: str | None, description: str = "", max_steps: int | None = None, **kwargs: Any):
    """Registers an environment to be able to use it with miniRL.make"""
    registry.register(name=name, entry_point=entry_point, vector_entry_point=vector_entry_point, description=description, max_steps=max_steps, kwargs=kwargs)

# Convenience functions to expose registry functions
def list_environments() -> dict[str, Any]:
    """List all registered environments"""
    return registry.list_environments()

def get_metadata(name:str) -> dict[str, Any]:
    """Get metadata for a registered environment"""
    return registry.get_metadata(name=name)

def unregister(name:str) -> None:
    """Unregister a registered environment"""
    return registry.unregister(name=name)

def get_environment_class(name:str):
    """Get the environment class for a registered env"""
    return registry.get_environment_class(name=name)

def get_required_params(name: str) -> Set[str]:
    """Gets the set of required __init__ parameters for a registered environment."""
    return registry.get_required_params(name=name)