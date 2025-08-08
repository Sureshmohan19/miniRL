"""miniRL.envs.classic.mountain_car"""

import math
import numpy as np
from typing import Any

from miniRL.core import Env
from miniRL.spaces import Discrete, Box
from miniRL.utils import parse_reset_bound

__all__ = ["MountainCar"]

class MountainCar(Env):
    """Exact implementation of MountainCarEnv by Gymnasium library. 
    The only difference is human render mode is completely removed and only supports RGB."""

    def __init__(self, render_mode: str = "rgb_array", goal_velocity: int = 0):
        """Initialise the environment"""
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity

        self.force = 0.001
        self.gravity = 0.0025

        # observation space: [position, velocity]
        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.render_mode = render_mode

        self.action_space = Discrete(3) # action space:three actions
        self.observation_space = Box(self.low, self.high, dtype=np.float32)

        self.screen_width = 600
        self.screen_height = 400
        self.isopen = True

    def step(self, action: int):
        """take an action with any of three possible action space"""
        assert self.action_space.contains(action), f"invalid action: {type(action)}"

        # core logic
        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
    
    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55
    
    def reset(
            self, 
            *, 
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ):
        """Reset the mountaincar env"""
        super().reset(seed=seed, options=options)

        # so technically you can provide low and high values as a dict[str, Any] in options params. 
        # so we better parse it and make sure it won't go out of bounds. 
        low, high  = parse_reset_bound(options=options, default_low=-0.6, default_high=-0.4)
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

        return np.array(self.state, dtype=np.float32), {}
    
    def render(self):
        """Render the environment. Only rgb_array supported at the momen"""
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
        
        # Initialize pygame without display
        if not pygame.get_init():
            pygame.init()
        
        # Create surface for rendering
        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill((255, 255, 255))
        
        # Calculate scaling and car dimensions
        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20
        
        pos = self.state[0]
        
        # Draw the mountain/hill
        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))
        pygame.draw.aalines(surf, points=xys, closed=False, color=(0, 0, 0))
        
        # Draw the car
        clearance = 10
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )
        gfxdraw.aapolygon(surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, coords, (0, 0, 0))
        
        # Draw the car wheels
        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )
            gfxdraw.aacircle(
                surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
        
        # Draw the goal flag
        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(surf, flagx, flagy1, flagy2, (0, 0, 0))
        gfxdraw.aapolygon(
            surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        
        # Flip the surface (pygame coordinate system is inverted)
        surf = pygame.transform.flip(surf, False, True)
        
        # Convert to RGB array and return
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
        )
    
    def close(self):
        """Close the pygame core"""
        try: 
            import pygame
            if pygame.get_init():
                pygame.quit()
        except ImportError:
            pass

        self.isopen = False