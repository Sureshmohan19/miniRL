from miniRL import register

register(
    name="MyCustomEnv-v0",
    entry_point="examples.my_custom_env:MyCustomEnv"
)