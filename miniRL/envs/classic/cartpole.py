"""miniRL.envs.classic.cartpole"""

from typing import Union, Any
import math
import numpy as np
import warnings

from miniRL.core import Env
from miniRL import spaces, utils

__all__ = ["CartPole"]

class CartPole(Env[np.ndarray, Union[int, np.ndarray]]):
    """CartPoleEnv implementation from gymnasium.envs.classic_control but supports only rgb_array rendering"""
    def __init__(
            self,
            sutton_barto_reward: bool = False,
            render_mode: str = "rgb_array",
    ):
        """Initialise the CartPole env as an instance of Env"""
        self._sutton_barto_reward = sutton_barto_reward
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2, np.inf, self.theta_threshold_radians * 2, np.inf,], dtype=np.float32,)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.isopen = True
        self.state: np.ndarray | None = None
        self.steps_beyond_terminated = None

    def step(self, action):
        """Take a step using the given action"""
        assert self.action_space.contains(action), f"{action}: {type(action)} is invalid"
        assert self.state is not None, "You should use reset() first before attempting to use step()"

        # core concept
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * np.square(theta_dot) * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length* (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)

        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0

            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                warnings.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

            reward = -1.0 if self._sutton_barto_reward else 0.0

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
            self, 
            *, 
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ):
        """Reset the environment"""
        super().reset(seed=seed, options=options)
        low, high = utils.parse_reset_bound(options=options, default_low=-0.05, default_high=0.05)
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        return np.array(self.state, dtype=np.float32), {}
    
    def render(self):
        """Render the environment. Only rgb_array supported at the moment"""
        if self.render_mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {self.render_mode}. Only 'rgb_array' is currently supported.")
                
        try:
            import pygame
            from pygame import gfxdraw
            import numpy as np
            import math
        except ImportError as e:
            raise ValueError(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e
        
        if not pygame.get_init():
            pygame.init()
        
        # Create surface for rendering
        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill((255, 255, 255))
        
        if self.state is None:
            # Convert to RGB array and return blank surface
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
            )
        
        # Calculate scaling and cart/pole dimensions
        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0
        
        x = self.state
        
        # Cart positioning
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        
        # Draw cart
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0))
        
        # Draw pole
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(surf, pole_coords, (202, 152, 101))
        
        # Draw axle
        gfxdraw.aacircle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        
        # Draw ground line
        gfxdraw.hline(surf, 0, self.screen_width, carty, (0, 0, 0))
        
        # Flip the surface (pygame coordinate system is inverted)
        surf = pygame.transform.flip(surf, False, True)
        
        # Convert to RGB array and return
        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

    def close(self):
        """Close the pygame core"""
        try: 
            import pygame
            if pygame.get_init():
                pygame.quit()
        except ImportError:
            pass

        self.isopen = False