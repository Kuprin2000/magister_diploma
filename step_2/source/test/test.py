import time
import warnings
from pathlib import Path
import numpy as np
import torch
from hegnn.models.HEGNN import HEGNN
from egnn.n_body_system.model import EGNN_vel

warnings.filterwarnings("ignore")

trajectories_path = "trajectories\\"
num_samples = 1000
batch_size = 1000


def load_npy(path):
    num_balls = None
    num_steps = None
    all_positions = []
    all_directions = []
    all_speeds = []

    counter = 0
    while counter < num_samples:
        position_path = Path(path + f"position_{counter}.npy")
        direction_path = Path(path + f"direction_{counter}.npy")
        if not position_path.is_file() or not direction_path.is_file():
            break

        positions = np.load(position_path)
        positions = (positions[[0, 1]] - 500) / 100
        positions = torch.from_numpy(positions)

        directions = np.load(direction_path)
        directions = directions[[0, 1]] / 100
        directions = torch.from_numpy(directions)

        if num_balls is None:
            num_balls = positions.shape[1]
        else:
            assert num_balls == positions.shape[1]

        if num_steps is None:
            num_steps = positions.shape[0]
        else:
            assert num_steps == positions.shape[0]

        speeds = torch.hypot(directions[:, :, 0], directions[:, :, 1])
        speeds = torch.reshape(speeds, [num_steps, num_balls, 1])

        all_positions.append(positions)
        all_directions.append(directions)
        all_speeds.append(speeds)

        counter += 1

    all_positions = torch.stack(all_positions, dim=0)
    all_directions = torch.stack(all_directions, dim=0)
    all_speeds = torch.stack(all_speeds, dim=0)

    return all_positions, all_directions, all_speeds


class TrajectoriesDataset(torch.utils.data.Dataset):
    def __init__(self, positions, directions, speeds):
        assert (
            positions.shape == directions.shape
            and directions.shape[0:-1] == speeds.shape[0:-1]
        )
        self.positions = positions
        self.directions = directions
        self.speeds = speeds

        num_vertices = positions.shape[2]
        edges = []
        for i in range(num_vertices):
            for j in range(num_vertices):
                if i == j:
                    continue
                edges.append([i, j])
        self.edges = torch.tensor(edges, dtype=torch.int32)

        num_trajectories = positions.shape[0]
        num_steps = positions.shape[1]

        distances = []
        for trajectory in range(num_trajectories):
            for step in range(num_steps):
                pos_1 = positions[trajectory][step][self.edges[:, 0]]
                pos_2 = positions[trajectory][step][self.edges[:, 1]]
                distance = torch.norm(pos_1 - pos_2, dim=1)
                distances.append(distance)

        self.distances = torch.stack(distances)
        self.distances = torch.reshape(
            self.distances,
            [num_trajectories, num_steps, num_vertices * (num_vertices - 1), 1],
        )

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx):
        return (
            self.positions[idx][0],
            self.directions[idx][0],
            self.speeds[idx][0],
            self.distances[idx][0],
            self.positions[idx][1],
            self.edges,
        )


def test_hegnn(model, dataloader):

    loss = torch.nn.MSELoss()
    time_accumulator = 0
    loss_accumulator = 0
    counter = 0
    for data in dataloader:
        (
            start_positions,
            start_directions,
            start_speeds,
            start_distances,
            end_positions,
            edges,
        ) = data

        num_points = start_positions.shape[1]

        start_speeds = torch.reshape(start_speeds, (-1, 1))
        start_speeds = torch.cat(
            [start_speeds, torch.ones(start_speeds.shape[0], 1)], dim=1
        )
        start_speeds = start_speeds.cuda()

        start_positions = torch.reshape(start_positions, (-1, 2))
        start_positions = torch.cat(
            [start_positions, torch.zeros(start_positions.shape[0], 1)], dim=1
        )
        start_positions = start_positions.cuda()

        start_directions = torch.reshape(start_directions, (-1, 2))
        start_directions = torch.cat(
            [start_directions, torch.zeros(start_directions.shape[0], 1)], dim=1
        )
        start_directions = start_directions.cuda()

        index_offset = 0
        for i in range(edges.shape[0]):
            edges[i] += index_offset
            index_offset += num_points
        edges = edges.permute(2, 0, 1).reshape(2, -1).to(torch.int64).cuda()

        start_distances = torch.reshape(start_distances, (-1, 1)).repeat(1, 2).cuda()

        end_positions = torch.reshape(end_positions, (-1, 2))
        end_positions = torch.cat(
            [end_positions, torch.zeros(end_positions.shape[0], 1)], dim=1
        )
        end_positions = end_positions.cuda()

        start = time.perf_counter()
        prediction = model(
            start_speeds, start_positions, start_directions, edges, start_distances
        )
        torch.cuda.synchronize()
        end = time.perf_counter()
        time_accumulator = time_accumulator + (end - start)

        loss_value = loss(prediction, end_positions)
        loss_accumulator += loss_value.item()
        counter += 1

    return loss_accumulator / counter, time_accumulator


def test_egnn(model, dataloader):

    loss = torch.nn.MSELoss()
    time_accumulator = 0
    loss_accumulator = 0
    counter = 0
    for data in dataloader:

        (
            start_positions,
            start_directions,
            start_speeds,
            start_distances,
            end_positions,
            edges,
        ) = data

        num_points = start_positions.shape[1]
        start_speeds = torch.reshape(start_speeds, (-1, 1)).cuda()
        start_positions = torch.reshape(start_positions, (-1, 2)).cuda()

        index_offset = 0
        for i in range(edges.shape[0]):
            edges[i] += index_offset
            index_offset += num_points
        edges = edges.permute(2, 0, 1).reshape(2, -1).to(torch.int64).cuda()

        start_directions = torch.reshape(start_directions, (-1, 2)).cuda()
        start_distances = torch.reshape(start_distances, (-1, 1)).cuda()

        start = time.perf_counter()
        prediction = model(
            start_speeds, start_positions, edges, start_directions, start_distances
        )
        torch.cuda.synchronize()
        end = time.perf_counter()
        time_accumulator = time_accumulator + (end - start)

        end_positions = torch.reshape(end_positions, (-1, 2)).cuda()
        loss_value = loss(prediction, end_positions)
        loss_accumulator += loss_value.item()
        counter += 1

    return loss_accumulator / counter, time_accumulator


if __name__ == "__main__":

    positions, directions, speeds = load_npy(trajectories_path)

    test_dataset = TrajectoriesDataset(positions, directions, speeds)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model_hegnn = HEGNN(
        num_layer=4,
        node_input_dim=2,
        edge_attr_dim=2,
        hidden_dim=64,
        max_ell=3,
        device="cuda",
    )
    model_hegnn.load_state_dict(torch.load(".\\hegnn.pth"))

    model_egnn = EGNN_vel(
        in_node_nf=1,
        in_edge_nf=1,
        hidden_nf=64,
        n_layers=4,
        device="cuda",
        recurrent=True,
    )
    model_egnn.load_state_dict(torch.load(".\\egnn.pth"))

    model_hegnn.eval()
    model_egnn.eval()
    mse_hegnn, time_hegnn = test_hegnn(model_hegnn, test_dataloader)
    mse_egnn, time_egnn = test_egnn(model_egnn, test_dataloader)

    print(
        "EGNN: MSE", format(mse_egnn, ".3e"), " time", format(time_egnn, ".3e"), "sec"
    )
    print(
        "HEGNN: MSE",
        format(mse_hegnn, ".3e"),
        " time",
        format(time_hegnn, ".3e"),
        "sec",
    )
