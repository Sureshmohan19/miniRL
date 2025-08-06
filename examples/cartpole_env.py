import numpy as np
from miniRL.core import Env
from miniRL.spaces import Box, Discrete

class SimpleCartPoleEnv(Env[np.ndarray, int]):
    """
    A simplified implementation of the CartPole environment.

    Observation: A numpy array of 4 floats:
        [cart_position, cart_velocity, pole_angle, pole_velocity_at_tip]
    
    Actions: Two discrete actions:
        0: Push cart to the left
        1: Push cart to the right
    """

    def __init__(self):
        # Physical constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.theta_threshold_radians = 1000 # Make it impossible for the pole to fall
        self.x_threshold = 10000          # Make it impossible for the cart to go off-screen

        # Define observation space using Box
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = Box(-high, high, dtype=np.float32)

        # Define action space using Discrete
        self.action_space = Discrete(2)

        # Internal state
        self.state: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Reset the state to random values near zero
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        
        return self.state, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.state is None:
            raise RuntimeError("Call reset before using step")

        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # Simplified physics equations
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Update state using Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])

        # Check for termination
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        # For this simple environment, there's no truncation condition
        truncated = False

        # Reward is +1 for every step taken
        reward = 1.0

        return self.state, reward, terminated, truncated, {}

    def render(self) -> None:
        # A simple text-based render
        pos, _, angle, _ = self.state
        print(f"Pos: {pos: .2f}, Angle: {angle: .2f}")