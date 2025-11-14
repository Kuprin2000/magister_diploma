import pygame
import math
import numpy as np
import typer
import random


COLORS = [
    (217, 237, 146),
    (93, 115, 126),
    (30, 96, 145),
    (62, 63, 63),
    (143, 45, 86),
    (116, 0, 184),
    (56, 4, 14),
]


class Body:
    def __init__(
        self, x, y, mass, velocity, color, rebound_factor, screen_width, screen_height
    ):
        self.x = x
        self.y = y
        self.mass = mass
        self.radius = int(math.sqrt(mass) * 2)
        self.vx, self.vy = velocity
        self.color = color
        self.trace = []
        self.rebound_factor = rebound_factor
        self.screen_width = screen_width
        self.screen_height = screen_height

    def __eq__(self, other) -> bool:

        if self.color == other.color:
            return True
        else:
            return False

    def calculate_grav_force(self, bodies: list, g: float):

        force = (0, 0)
        for body in bodies:
            if body != self:
                force += calculate_gravitational_force(self, body, g)
        force1x, force1y = add_tuples(force)
        self.update(force1x, force1y)

    def update(self, force_x: float, force_y: float):

        ax = force_x / self.mass
        ay = force_y / self.mass
        self.vx += ax
        self.vy += ay
        self.x += self.vx
        self.y += self.vy
        self.update_trace()

    def update_trace(self):

        self.trace.append((self.x, self.y))
        if len(self.trace) == 100:
            self.trace.pop(0)

    def draw(self, screen):

        for point in self.trace:
            pygame.draw.circle(screen, self.color, point, 1)
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


def calculate_gravitational_force(p1: Body, p2: Body, g: float) -> tuple:

    dx = p2.x - p1.x
    dy = p2.y - p1.y
    distance = max(1, math.sqrt(dx**2 + dy**2))
    if distance < 40:
        return (0, 0)
    force = (g * p1.mass * p2.mass) / (distance**2)
    angle = math.atan2(dy, dx)
    force_x = force * math.cos(angle)
    force_y = force * math.sin(angle)
    return (force_x, force_y)


def add_tuples(tuple: tuple) -> tuple:

    even = 0
    odd = 0
    for i in range(len(tuple)):
        if i % 2 == 0:
            even += tuple[i]
        else:
            odd += tuple[i]
    return (even, odd)


def main(
    width: int = 1000,
    height: int = 1000,
    rebound_factor: float = typer.Option(
        0.5,
        help="Factor strength to apply when bodies when bodies bounce off the limits of the screen.",
    ),
    mass: int = typer.Option(10, help="Default mass of the bodies."),
    g: int = typer.Option(9.8, help="The gravitational constant."),
):
    pygame.init()

    # CALCULATE NECESSARY TRIGONOMETRY
    side = 200
    x = np.sqrt(side**2 - (side / 2) ** 2)
    initial_x = width / 2 - side / 2
    initial_y = 500

    for trajectory in range(1000):

        # INITIAL BODIES
        body1 = Body(
            initial_x + 50 * (0.5 - random.random()),
            initial_y + 50 * (0.5 - random.random()),
            mass=mass,
            velocity=(0.1, 0.1),
            color=(116, 148, 196),
            rebound_factor=rebound_factor,
            screen_height=height,
            screen_width=width,
        )
        body2 = Body(
            (initial_x + (initial_x + side)) / 2 + 50 * (0.5 - random.random()),
            initial_y - x + 50 * (0.5 - random.random()),
            mass=mass,
            velocity=(-0.1, 0.1),
            color=(106, 77, 97),
            rebound_factor=rebound_factor,
            screen_height=height,
            screen_width=width,
        )
        body3 = Body(
            initial_x + side + 50 * (0.5 - random.random()),
            initial_y + 50 * (0.5 - random.random()),
            mass=mass,
            velocity=(0.1, -0.1),
            color=(195, 212, 7),
            rebound_factor=rebound_factor,
            screen_height=height,
            screen_width=width,
        )
        bodies = [body1, body2, body3]

        # MAIN LOOP
        positions = np.empty([500, 3, 2], dtype=np.float32)
        directions = np.empty([500, 3, 2], dtype=np.float32)
        counter = 0

        while counter < 500:

            for body in bodies:
                body.calculate_grav_force(bodies, g=g)

            for index, body in enumerate(bodies):
                positions[counter, index, 0] = body.x
                positions[counter, index, 1] = body.y
                directions[counter, index, 0] = body.vx
                directions[counter, index, 1] = body.vy

            counter += 1

        path = (
            ".\\trajectories\\position_"
            + str(trajectory)
            + ".npy"
        )
        positions = positions[[0,499]]
        np.save(path, positions)

        path = (
            ".\\trajectories\\direction_"
            + str(trajectory)
            + ".npy"
        )
        directions = directions[[0,499]]
        np.save(path, directions)

    pygame.quit()


if __name__ == "__main__":
    typer.run(main)
