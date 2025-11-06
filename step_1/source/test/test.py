import pygame
import math
import numpy as np
import typer
import torch
import random
import copy
from egnn.n_body_system.model import EGNN_vel


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

    def update_from_ai(self, new_x: int, new_y: int):

        self.vx = 0.0
        self.vy = 0.0
        self.x = new_x
        self.y = new_y
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


def predict(body1, body2, body3):
    ai_body1 = copy.deepcopy(body1)
    ai_body1.x = (ai_body1.x - 500) / 100
    ai_body1.y = (ai_body1.y - 500) / 100
    ai_body1.vx /= 100
    ai_body1.vy /= 100
    ai_body1.color = (0, 250, 0)
    
    ai_body2 = copy.deepcopy(body2)
    ai_body2.x = (ai_body2.x - 500) / 100
    ai_body2.y = (ai_body2.y - 500) / 100
    ai_body2.vx /= 100
    ai_body2.vy /= 100
    ai_body2.color = (0, 150, 0)

    ai_body3 = copy.deepcopy(body3)
    ai_body3.x = (ai_body3.x - 500) / 100
    ai_body3.y = (ai_body3.y - 500) / 100
    ai_body3.vx /= 100
    ai_body3.vy /= 100
    ai_body3.color = (0, 50, 0)

    model = EGNN_vel(
        in_node_nf=1,
        in_edge_nf=1,
        hidden_nf=64,
        n_layers=4,
        device="cuda",
        recurrent=True,
    )

    model.load_state_dict(
        torch.load(".\\model.pth")
    )
    model.eval()

    start_speeds = torch.tensor(
        [
            math.hypot(ai_body1.vx, ai_body1.vy),
            math.hypot(ai_body2.vx, ai_body2.vy),
            math.hypot(ai_body3.vx, ai_body3.vy),
        ],
        dtype=torch.float32,
    ).reshape([3, 1])

    start_positions = torch.tensor(
        [
            [ai_body1.x, ai_body1.y],
            [ai_body2.x, ai_body2.y],
            [ai_body3.x, ai_body3.y],
        ],
        dtype=torch.float32,
    )

    edges = []
    for i in range(3):
        for j in range(0, 3):
            if i == j:
                continue
            edges.append([i, j])
    edges = torch.tensor(edges, dtype=torch.int32)
    edges = edges.permute(1, 0).reshape(2, -1).cuda()

    start_directions = torch.tensor(
        [
            [ai_body1.vx, ai_body1.vy],
            [ai_body2.vx, ai_body2.vy],
            [ai_body3.vx, ai_body3.vy],
        ],
        dtype=torch.float32,
    )

    start_distances = []
    for elem in torch.unbind(edges, dim=1):
        start_distances.append(
            torch.norm(start_positions[elem[0]] - start_positions[elem[1]])
        )
    start_distances = torch.tensor(start_distances, dtype=torch.float32).reshape([6, 1])

    prediction = model(
        start_speeds.cuda(),
        start_positions.cuda().detach(),
        edges.cuda(),
        start_directions.cuda(),
        start_distances.cuda(),
    ).cpu()

    prediction = prediction * 100 + 500

    ai_body1.update_from_ai(prediction[0, 0].item(), prediction[0, 1].item())
    ai_body2.update_from_ai(prediction[1, 0].item(), prediction[1, 1].item())
    ai_body3.update_from_ai(prediction[2, 0].item(), prediction[2, 1].item())

    return [ai_body1, ai_body2, ai_body3]


def main(
    width: int = 1000,
    height: int = 1000,
    rebound_factor: float = typer.Option(
        0.5,
        help="Factor strength to apply when bodies when bodies bounce off the limits of the screen.",
    ),
    mass: int = typer.Option(10, help="Default mass of the bodies."),
    g: int = typer.Option(9.8, help="The gravitational constant."),
    clock: int = typer.Option(
        60, help="Framerate to delay the game to the given ticks."
    ),
):
    # SETUP
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    # CALCULATE NECESSARY TRIGONOMETRY
    side = 200
    x = np.sqrt(side**2 - (side / 2) ** 2)
    initial_x = width / 2 - side / 2
    initial_y = 500

    # INITIAL BODIES
    body1 = Body(
        initial_x + 50 * (0.5 - random.random()),
        initial_y + 50 * (0.5 - random.random()),
        mass=mass,
        velocity=(0.1, 0.1),
        color=(250, 0, 0),
        rebound_factor=rebound_factor,
        screen_height=height,
        screen_width=width,
    )
    body2 = Body(
        (initial_x + (initial_x + side)) / 2 + 50 * (0.5 - random.random()),
        initial_y - x + 50 * (0.5 - random.random()),
        mass=mass,
        velocity=(-0.1, 0.1),
        color=(150, 0, 0),
        rebound_factor=rebound_factor,
        screen_height=height,
        screen_width=width,
    )
    body3 = Body(
        initial_x + side + 50 * (0.5 - random.random()),
        initial_y + 50 * (0.5 - random.random()),
        mass=mass,
        velocity=(0.1, -0.1),
        color=(50, 0, 0),
        rebound_factor=rebound_factor,
        screen_height=height,
        screen_width=width,
    )

    bodies = [body1, body2, body3]
    predicted_bodies = predict(body1, body2, body3)

    # MAIN LOOP
    running = True
    counter = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if counter > 500:
            continue

        screen.fill((0, 0, 0))

        for body in predicted_bodies:
            body.draw(screen)

        for body in bodies:
            body.calculate_grav_force(bodies, g=g)
            body.draw(screen)

        pygame.display.update()
        pygame.image.save(
            screen,
            f".\\record\\frame_{counter:04d}.png",
        )

        clock.tick(60)

        counter += 1

    gt = np.array([bodies[0].x, bodies[0].y, bodies[1].x, bodies[1].y, bodies[2].x, bodies[2].y], dtype=np.float32)
    predicted = np.array([predicted_bodies[0].x, predicted_bodies[0].y, predicted_bodies[1].x, predicted_bodies[1].y, predicted_bodies[2].x, predicted_bodies[2].y], dtype=np.float32)
    print("Mse", np.square(np.subtract(gt, predicted)).mean())

    pygame.quit()


if __name__ == "__main__":
    typer.run(main)
