import time
from pathlib import Path
import numpy as np
import torch
from egnn.n_body_system.model import EGNN_vel

trajectories_path = "..\\trajectories\\"
num_samples = 10000
train_percentage = 0.8
batch_size = 100
lr = 7e-4
weight_decay = 1e-12
num_epoch = 10000
save_every_n_epoch = 1000

# def load_txt(path):
#     path = Path(path)

#     num_balls = None
#     num_steps = None
#     positions = []
#     directions = []
#     speeds = []

#     counter = 0
#     for file_path in sorted(path.iterdir()):
#         if counter > num_samples:
#             break

#         if not file_path.is_file() or file_path.suffix != ".txt":
#             continue

#         with open(file_path, "r", encoding="utf-8") as file:
#             num_balls_in_file = int(file.readline())
#             if num_balls is None:
#                 num_balls = num_balls_in_file
#             else:
#                 assert num_balls == num_balls_in_file

#             num_steps_in_file = int(file.readline())
#             if num_steps is None:
#                 num_steps = num_steps_in_file
#             else:
#                 assert num_steps == num_steps_in_file

#             positions_in_file = torch.empty(
#                 [num_steps, num_balls, 2], dtype=torch.float32
#             )
#             directions_in_file = torch.empty(
#                 [num_steps, num_balls, 2], dtype=torch.float32
#             )
#             speeds_in_file = torch.empty([num_steps, num_balls, 1], dtype=torch.float32)

#             for i in range(num_steps):
#                 for j in range(num_balls):
#                     pos_x, pos_y, speed_x, speed_y = [
#                         float(x) for x in file.readline().split(", ")
#                     ]
#                     positions_in_file[i, j, 0] = pos_x
#                     positions_in_file[i, j, 1] = pos_y
#                     directions_in_file[i, j, 0] = speed_x
#                     directions_in_file[i, j, 1] = speed_y
#                     speeds_in_file[i, j, 0] = math.hypot(speed_x, speed_y)

#                 step = int(file.readline())
#                 assert step == i

#             positions.append(positions_in_file)
#             directions.append(directions_in_file)
#             speeds.append(speeds_in_file)

#             counter += 1

#     positions = torch.stack(positions, dim=0)
#     directions = torch.stack(directions, dim=0)
#     speeds = torch.stack(speeds, dim=0)

#     return positions, directions, speeds


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


if __name__ == "__main__":

    positions, directions, speeds = load_npy(trajectories_path)

    indices = torch.arange(0, positions.shape[0])
    indices = indices[torch.randperm(len(indices))]
    train_examples = int(train_percentage * positions.shape[0])

    train_dataset = TrajectoriesDataset(
        positions[0:train_examples],
        directions[0:train_examples],
        speeds[0:train_examples],
    )
    validation_dataset = TrajectoriesDataset(
        positions[train_examples:], directions[train_examples:], speeds[train_examples:]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    model = EGNN_vel(
        in_node_nf=1,
        in_edge_nf=1,
        hidden_nf=64,
        n_layers=4,
        device="cuda",
        recurrent=True,
    )

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num parameters", num_parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
                end_positions,
                edges,
            ) = data

            with torch.no_grad():
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

            prediction = model(
                start_speeds,
                start_positions.detach(),
                edges,
                start_directions,
                start_distances,
            )

            end_positions = torch.reshape(end_positions, (-1, 2)).cuda()
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

            prediction = model(
                start_speeds,
                start_positions,
                edges,
                start_directions,
                start_distances,
            )

            end_positions = torch.reshape(end_positions, (-1, 2)).cuda()
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
            name = "egnn_" + str(epoch) + ".pth"
            torch.save(model.state_dict(), name)

        scheduler.step()
