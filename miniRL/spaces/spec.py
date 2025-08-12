"""miniRL.spaces.spec"""

class EnvSpec():
    """Simple class to hold the specification for the environments"""
    def __init__(
            self,
            name: str,
            entry_point: str | None = None,
            vector_entry_point: str | None = None,
            description: str = "",
            max_steps: int | None = None,
            kwargs: dict | None = None,
    ):
        """Initialise the EnvSpec class"""
        assert (entry_point is not None), f"entry_point cannot be empty. It should be in the format of module_name:class_name"
        self.name = name
        self.entry_point = entry_point
        self.vector_entry_point = vector_entry_point
        self.description = description
        self.max_steps = max_steps
        self.kwargs = {} if kwargs is None else kwargs

    def __repr__(self) -> str:
        """String representation of the EnvSpec"""
        if self.entry_point:
            return f"EnvSpec(name='{self.name}', entry_point='{self.entry_point}')"
        else:
            return f"EnvSpec(name='{self.name}', vector_entry_point='{self.vector_entry_point}')"