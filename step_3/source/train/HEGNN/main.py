import time
import warnings
from collections import deque
from pathlib import Path
import numpy as np
import torch
from hegnn.models.HEGNN import HEGNN

warnings.filterwarnings("ignore")

trajectories_path = "..\\trajectories\\"
num_trajectories = 10000
train_percentage = 0.8
batch_size = 200
lr = 3e-4
num_epoch = 1000
save_every_n_epoch = 50


def load_npy(path):
    print("Loading")
    num_vertices = None
    num_steps = 0

    positions_pairs = deque()
    directions_pairs = deque()
    speeds_pairs = deque()
    distances_pairs = deque()
    edges = None

    for trajectory in range(0, num_trajectories):

        if trajectory % 1000 == 0:
            print("Loaded", trajectory, "trajectories")

        trajectory_positions = []
        trajectory_directions = []
        trajectory_speeds = []
        trajectory_distances = []

        for step in range(0, 501, 50):

            position_path = Path(path + f"{trajectory}\\position_{step}.npy")
            direction_path = Path(path + f"{trajectory}\\direction_{step}.npy")
            edge_path = Path(path + f"{trajectory}\\edges_{step}.npy")

            if (
                not position_path.is_file()
                or not direction_path.is_file()
                or not edge_path.is_file()
            ):
                break

            positions = np.load(position_path)
            positions = torch.from_numpy(positions)
            positions = positions.squeeze(2)

            directions = np.load(direction_path)
            directions = torch.from_numpy(directions)
            directions = directions.squeeze(2)

            if edges is None:
                edges = np.load(edge_path)
                edges = torch.from_numpy(edges).to(dtype=torch.int64)
                edges = edges.squeeze(2)

            if num_vertices is None:
                num_vertices = positions.shape[0]
            else:
                assert num_vertices == positions.shape[0]

            if trajectory == 0:
                num_steps += 1

            speeds = torch.hypot(directions[:, 0], directions[:, 1])
            speeds = torch.reshape(speeds, [speeds.shape[0], 1])

            pos_1 = positions[edges[:, 0]]
            pos_2 = positions[edges[:, 1]]
            distances = torch.norm(pos_1 - pos_2, dim=1).unsqueeze(1)

            trajectory_positions.append(positions)
            trajectory_directions.append(directions)
            trajectory_speeds.append(speeds)
            trajectory_distances.append(distances)

        trajectory_positions = [
            torch.stack([trajectory_positions[i], trajectory_positions[i + 1]])
            for i in range(len(trajectory_positions) - 1)
        ]
        trajectory_directions = [
            torch.stack([trajectory_directions[i], trajectory_directions[i + 1]])
            for i in range(len(trajectory_directions) - 1)
        ]
        trajectory_speeds = [
            torch.stack([trajectory_speeds[i], trajectory_speeds[i + 1]])
            for i in range(len(trajectory_speeds) - 1)
        ]
        trajectory_distances = [
            torch.stack([trajectory_distances[i], trajectory_distances[i + 1]])
            for i in range(len(trajectory_distances) - 1)
        ]

        positions_pairs.extend(trajectory_positions)
        directions_pairs.extend(trajectory_directions)
        speeds_pairs.extend(trajectory_speeds)
        distances_pairs.extend(trajectory_distances)

    positions_pairs = torch.stack(list(positions_pairs))
    directions_pairs = torch.stack(list(directions_pairs))
    speeds_pairs = torch.stack(list(speeds_pairs))
    distances_pairs = torch.stack(list(distances_pairs))

    return positions_pairs, directions_pairs, speeds_pairs, distances_pairs, edges


class TrajectoriesDataset(torch.utils.data.Dataset):
    def __init__(
        self, positions_pairs, directions_pairs, speeds_pairs, distances_pairs, edges
    ):
        assert (
            positions_pairs.shape == directions_pairs.shape
            and directions_pairs.shape[0:-1] == speeds_pairs.shape[0:-1]
        )
        # [num pairs, 2, num vertices, 3 or 1]
        self.positions_pairs = positions_pairs
        self.directions_pairs = directions_pairs
        self.speeds_pairs = speeds_pairs
        # [num pairs, 2, num edges, 1]
        self.distances_pairs = distances_pairs
        # [num edges, 2]
        self.edges = edges

    def __len__(self):
        return self.positions_pairs.shape[0]

    def __getitem__(self, idx):
        return (
            self.positions_pairs[idx][0],
            self.directions_pairs[idx][0],
            torch.cat([self.speeds_pairs[idx][0], torch.ones(self.speeds_pairs[idx][0].shape[0], 1)], dim=1),
            self.distances_pairs[idx][0].repeat(1, 2),
            self.edges,
            self.positions_pairs[idx][1],
        )


if __name__ == "__main__":

    positions_pairs, directions_pairs, speeds_pairs, distances_pairs, edges = load_npy(
        trajectories_path
    )

    indices = torch.arange(0, positions_pairs.shape[0])
    indices = indices[torch.randperm(len(indices))]
    train_examples = int(train_percentage * positions_pairs.shape[0])

    train_dataset = TrajectoriesDataset(
        positions_pairs[0:train_examples],
        directions_pairs[0:train_examples],
        speeds_pairs[0:train_examples],
        distances_pairs[0:train_examples],
        edges,
    )
    validation_dataset = TrajectoriesDataset(
        positions_pairs[train_examples:],
        directions_pairs[train_examples:],
        speeds_pairs[train_examples:],
        distances_pairs[train_examples:],
        edges,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    model = HEGNN(
        num_layer=4,
        node_input_dim=2,
        edge_attr_dim=2,
        hidden_dim=64,
        max_ell=3,
        device="cuda",
    )

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num parameters", num_parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=1, epochs=num_epoch
    )
    loss = torch.nn.MSELoss()

    start_time = time.time()
    for epoch in range(num_epoch):

        model.train()

        loss_accumulator = 0
        counter = 0
        for data in train_dataloader:

            (
                start_positions,
                start_directions,
                start_speeds,
                start_distances,
                edges,
                end_positions,
            ) = data

            with torch.no_grad():
                num_points = start_positions.shape[1]

                start_speeds = torch.reshape(start_speeds, (-1, 2))
                start_speeds = start_speeds.cuda()

                start_positions = torch.reshape(start_positions, (-1, 3))
                start_positions = start_positions.cuda()

                start_directions = torch.reshape(start_directions, (-1, 3))
                start_directions = start_directions.cuda()

                start_distances = (
                    torch.reshape(start_distances, (-1, 2)).cuda()
                )

                end_positions = torch.reshape(end_positions, (-1, 3))
                end_positions = end_positions.cuda()

                index_offset = 0
                for i in range(edges.shape[0]):
                    edges[i] += index_offset
                    index_offset += num_points
                edges = edges.permute(2, 0, 1).reshape(2, -1).to(torch.int64).cuda()

            prediction = model(
                start_speeds, start_positions, start_directions, edges, start_distances
            )

            loss_value = loss(prediction, end_positions)

            with torch.no_grad():
                loss_accumulator += loss_value.item()
                counter += 1

            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % save_every_n_epoch == 0 or epoch == num_epoch - 1:
            mean_loss = loss_accumulator / counter
            print("Epoch", epoch, "mean loss while training", format(mean_loss, ".3e"))

        model.eval()

        loss_accumulator = 0
        counter = 0
        for data in validation_dataloader:

            (
                start_positions,
                start_directions,
                start_speeds,
                start_distances,
                edges,
                end_positions,
            ) = data

            num_points = start_positions.shape[1]

            start_speeds = torch.reshape(start_speeds, (-1, 2))
            start_speeds = start_speeds.cuda()

            start_positions = torch.reshape(start_positions, (-1, 3))
            start_positions = start_positions.cuda()

            start_directions = torch.reshape(start_directions, (-1, 3))
            start_directions = start_directions.cuda()

            index_offset = 0
            for i in range(edges.shape[0]):
                edges[i] += index_offset
                index_offset += num_points
            edges = edges.permute(2, 0, 1).reshape(2, -1).to(torch.int64).cuda()

            start_distances = (
                torch.reshape(start_distances, (-1, 2)).cuda()
            )

            end_positions = torch.reshape(end_positions, (-1, 3))
            end_positions = end_positions.cuda()

            prediction = model(
                start_speeds, start_positions, start_directions, edges, start_distances
            )

            loss_value = loss(prediction, end_positions)

            loss_accumulator += loss_value.item()
            counter += 1

        if epoch % save_every_n_epoch == 0 or epoch == num_epoch - 1:
            mean_loss = loss_accumulator / counter
            print(
                "Epoch", epoch, "mean loss while evaluation", format(mean_loss, ".3e")
            )
            print(
                "Epoch",
                epoch,
                "learning rate",
                format(scheduler.get_last_lr()[0], ".3e"),
            )
            execution_time = time.time() - start_time
            print(f"Time from start: {execution_time:.2f}sec")

        if (epoch % save_every_n_epoch == 0 or epoch == num_epoch - 1) and epoch != 0:
            name = "hegnn_" + str(epoch) + ".pth"
            torch.save(model.state_dict(), name)

        scheduler.step()
