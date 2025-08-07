import miniRL
# We must import `examples` here. This tells Python to run the code
# in `examples/__init__.py`, which performs the registration.
import examples 

# --- The main execution block ---
if __name__ == "__main__":
    
    print("--- Creating Environment ---")
    # This now works perfectly. `make` will find "MyCustomEnv-v0" because
    # the line `import examples` already registered it.
    env = miniRL.make("MyCustomEnv-v0", max_steps=100, initial_value=0.2)

    print(f"Successfully created the environment: {env}")

    # Run a simple loop
    obs, info = env.reset(seed=42)
    terminated = False
    truncated = False
    total_reward = 0
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    
    print(f"Episode finished with total reward: {total_reward:.2f}")
    env.close()