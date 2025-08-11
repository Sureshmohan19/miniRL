"""miniRL.envs.registry"""

import inspect
import importlib
import warnings
import copy
from typing import Any, Set, Type, cast

from miniRL.core import Env

__all__ = ["EnvironmentRegistry", "EnvSpec"]

class EnvSpec():
    """Simple class to hold the specification for the environments"""
    def __init__(
            self,
            name: str,
            entry_point: str | None = None,
            description: str = "",
            max_steps: int | None = None,
            kwargs: dict | None = None,
    ):
        """Initialise the EnvSpec class"""
        assert (entry_point is not None), f"entry_point cannot be empty. It should be in the format of module_name:class_name"
        self.name = name
        self.entry_point = entry_point
        self.description = description
        self.max_steps = max_steps
        self.kwargs = {} if kwargs is None else kwargs

    def __repr__(self) -> str:
        """String representation of the EnvSpec"""
        return f"EnvSpec(name='{self.name}', entry_point='{self.entry_point}')"

class EnvironmentRegistry():
    """Registry for managing miniRL environments."""
    def __init__(self):
        """Initialise the empty registry"""
        self._specs: dict[str, EnvSpec] = {}

    def register(
            self, 
            name: str,
            entry_point: str | None = None,
            description: str = "",
            max_steps: int | None = None,
            kwargs: dict | None = None
    ) -> None:
        """Register the environment class to the registry"""
        if name in self._specs:
            raise ValueError(f"Environment {name} already registered.")
        
        new_spec = EnvSpec(
            name=name,
            entry_point=entry_point,
            description=description,
            max_steps=max_steps,
            kwargs=kwargs
        )
        self._specs[name] = new_spec
        print(f"✓ Registered environment: {name}")

    def make(
            self, 
            name: str,
            max_steps: int | None = None,
            **kwargs: Any
    ) -> Env[Any, Any]:
        """Create an environment instance"""
        if isinstance(name, EnvSpec):
            warnings.warn(f"cannot make an environment using EnvSpec at this momemt" 
                          f"Have to use miniRL.register to register the environment")
        
        assert isinstance(name, str)
        if name not in self._specs:
            raise ValueError(f"Environment {name} not found." 
                             f"Please register or try listing out available environments using {list(self._specs.keys())}")
        
        spec = self._specs[name]
        spec_kwargs = copy.deepcopy(spec.kwargs)
        spec_kwargs.update(kwargs)

        # Do not pass registry-only fields like metadata into env __init__
        if "metadata" in spec_kwargs:
            spec_kwargs.pop("metadata", None)

        env_reg = self.get_environment_class(name)

        self._validate_params(env_reg, spec_kwargs, name)

        try:
            env = env_reg(**spec_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create environment:{name} with params:{spec_kwargs}. Instead got {e}")  
        
        # step-limit
        act_max_steps = spec.max_steps if max_steps is None else max_steps

        if act_max_steps is not None and act_max_steps > 0:
            from miniRL.wrappers import StepLimit
            env = StepLimit(env, max_steps=act_max_steps)
        
        return env
    
    def get_metadata(self, name:str) -> dict[str, Any]:
        """Get the metadata from the registered environment from its spec's kwargs"""
        if name not in self._specs:
            raise ValueError(f"Environment {name} not found")
        spec = self._specs[name]
        return spec.kwargs.get("metadata", {})
    
    def _validate_params(
            self, 
            env_reg: Type[Env[Any, Any]],
            params: dict[str, Any],
            env_name: str
    ) -> None:
        """Validate the params passed when make() is used"""
        try:
            sig = inspect.signature(env_reg.__init__)

            valid_params = {p.name for p in sig.parameters.values()}
            invalid_params = set(params.keys()) - valid_params

            if invalid_params:
                raise TypeError(f"Invalid parameters for {env_name}: {sorted(invalid_params)}"
                                f"The valid parameters for this Env are: {sorted(valid_params)}"
                )
            
            # check for missing required params as well
            required_params = self._get_required_params(sig)
            missing_params = required_params - set(params.keys())
            if missing_params:
                raise TypeError(f"Missing required parameters for the Environment: {env_reg}: {sorted(missing_params)}")
            
            return params
        
        except Exception as e:
            if isinstance (e, TypeError):
                raise

            print(f"Warning: Could not validate parameters for {env_reg.__name__}: {e}"
                  f"Please make sure the parameters are valid manually")
        
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
    
    def get_environment_class(self, name: str) -> Type[Env[Any, Any]]:
        """Get the environment class for a registered environment"""
        if name not in self._specs:
            raise ValueError(f"Environment {name} is not found."
                             f"The available environments are: {list(self._specs.keys())}") 
        spec = self._specs[name]
        module_name, class_name = spec.entry_point.split(":")

        try:
            module = importlib.import_module(module_name)
            env_class = getattr(module, class_name)
        except (ModuleNotFoundError, AttributeError) as e:
            raise RuntimeError(f"Could not import the entry point '{spec.entry_point}' for '{name}'. Error: {e}")

        if not (inspect.isclass(env_class) and issubclass(env_class, Env)):
            raise TypeError(f"The entry point '{spec.entry_point}' for '{name}' is not a valid subclass of miniRL.Env.")
        
        return cast(Type[Env[Any, Any]], env_class)
    
    def get_required_params(self, name:str) -> Set[str]:
        """Get the set of required __init__ parameters for a registerd environment"""
        env = self.get_environment_class(name)
        signature = inspect.signature(env.__init__)
        return self._get_required_params(signature)
        
    def list_environments(self) -> dict[str, str]:
        """List all the registered environments"""
        return {spec.name: spec.description for spec in self._specs.values()}
    
    def unregister(self, name: str) -> None:
        """Unregister an environment from the registry"""
        if name not in self._specs:
            raise ValueError(f"Environment {name} is not registered at the moment.")
        
        del self._specs[name]
        print(f"✓ Unregistered environment: '{name}'")

    def __len__(self) -> int:
        """Return number of registered environments"""
        return len(self._specs)
    
    def __contains__(self, name: str) ->bool:
        """Check if an environment is registered"""
        return name in self._specs
    
    def __repr__(self) -> str:
        """String representation of the registry"""
        return f"EnvironmentRegistry({len(self._specs)}) environments: {sorted(self._specs.keys())}"