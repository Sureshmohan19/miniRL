"""miniRL.envs.registry"""

import inspect
from typing import Any, Set, Type

from miniRL.core import Env

__all__ = ["EnvironmentRegistry"]

class EnvironmentRegistry():
    """Registry for managing miniRL environments"""
    def __init__(self):
        """Initialise the empty registry"""
        self._environments: dict[str, Type[Env[Any, Any]]] = {}
        self._descriptions: dict[str, str] = {}

    def register(
            self, 
            name: str,
            env_reg: Type[Env[Any, Any]],
            description: str = ""
    ) -> None:
        """Register the environment class to the registry"""
        if not (inspect.isclass(env_reg) and issubclass(env_reg, Env)):
            raise TypeError(f"Registeringi d environment must be a class that inherits from miniRL.Env, but got {env_reg}")

        if name in self._environments:
            raise ValueError(f"Environment {name} already registered with class {self._environments[name].__name__}")
        
        self._environments[name] = env_reg
        self._descriptions[name] = description

        print(f"✓ Registered environment: {name} -> {env_reg.__name__}")

    def make(
            self, 
            name: str,
            **params: Any
    ) -> Env[Any, Any]:
        """Create an environment instance"""
        if name not in self._environments:
            raise ValueError(f"Environment {name} not found. Please register or try listing out available environments")
        
        env_reg = self._environments[name]
        validated_params = self._validate_params(env_reg, params)

        try:
            return env_reg(**validated_params)
        except Exception as e:
            raise RuntimeError(f"Failed to create environment:{name} with params:{params}. Instead got {e}")  

    def _validate_params(
            self, 
            env_reg: Type[Env[Any, Any]],
            params: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate the params passed when make() is used"""
        try:
            sig = inspect.signature(env_reg.__init__)

            valid_param_names = {p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
            given_params = set(params.keys())
            invalid_params = given_params - valid_param_names

            if invalid_params:
                raise TypeError(f"Invalid parameters for {env_reg.__name__}: {sorted(invalid_params)}"
                                f"The valid parameters for this Env are: {sorted(valid_param_names)}"
                )
            
            # check for missing required params as well
            required_params = self._get_required_params(sig)
            missing_params = required_params - given_params

            if missing_params:
                raise TypeError(f"Missing required parameters for the Environment: {env_reg.__name__}: {sorted(missing_params)}")
            
            return params
        
        except Exception as e:
            if isinstance (e, TypeError):
                raise

            print(f"Warning: Could not validate parameters for {env_reg.__name__}: {e}"
                  f"Please make sure the parameters are valid manually")
            return params
        
    def _get_required_params(
            self,
            signature: inspect.Signature
    ) -> Set[str]:
        """Collect all the required params from the env init method"""
        required = set()
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if param.default == inspect.Parameter.empty and param.kind != inspect.Parameter.VAR_KEYWORD:
                required.add(name)

        return required
    
    def get_environment_class(
            self,
            name: str
    ) -> Type[Env[Any, Any]]:
        """Get the environment class for a registered environment"""
        if name not in self._environments:
            raise ValueError(f"Environment {name} is not found."
                             f"The available environments are: {list(self._environments.keys())}") 
        
        return self._environments[name]
        
    def list_environments(self) -> dict[str, str]:
        """List all the registered environments"""
        return self._descriptions.copy()
    
    def unregister(
            self,
            name: str
    ) -> None:
        """Unregister an environment from the registry"""
        if name not in self._environments:
            raise ValueError(f"Environment {name} is not registered at the moment.")
        
        del self._environments[name]
        del self._descriptions[name]

        assert name not in self._environments
        print(f"✓ Unregistered environment: '{name}'")

    def __len__(self) -> int:
        """Return number of registered environments"""
        return len(self._environments)
    
    def __contains__(self, name: str) ->bool:
        """Check if an environment is registered"""
        return name in self._environments
    
    def __repr__(self) -> str:
        """String representation of the registry"""
        env_count = len(self._environments)
        env_names = sorted(self._environments.keys())
        return f"EnvironmentRegistry({env_count}) environments: {env_names}"