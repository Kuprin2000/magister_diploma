import time
import argparse
from functools import partial

import torch
from torch import nn
from torch import optim
from torch_geometric.loader import DataLoader

from utils.seed import fix_seed
from utils.train import train
from models.HEGNN import HEGNN
from datasets.nbody.dataset import NBodySystemDataset

parser=argparse.ArgumentParser(description='HEGNN')

# Model
parser.add_argument('--exp_name', type=str, default='simple-exp', help='str type, name of the experiment (default: simple_exp)')
parser.add_argument('--model', type=str, default='HEGNN', help='which model (default: HEGNN)')
parser.add_argument('--ell', type=int, default=3, help='max ell (default: 2)')
parser.add_argument('--dim_hidden', type=int, default=64, help='hiddendim (default: 64)')
parser.add_argument('--num_layer', type=int, default=4, help='number of layers of gnn (default: 4)')
parser.add_argument('--recurrunt_required', action='store_false', help='use recurrunt in the model (default: True)')
parser.add_argument('--attention_required', action='store_true', help='use attention in the model (default: False)')
parser.add_argument('--direction_vector_normalize_required', action='store_true', help='normalize the direction vector (default: False)')
parser.add_argument('--tanh_required', action='store_true', help='use tanh (default: False)')
parser.add_argument('--sigma', type=float, default=1.5, help='sigma in kernel function')
parser.add_argument('--weight', type=float, default=0.01, help='weight of MMD loss')


# Data
parser.add_argument('--data_directory', type=str, default='C:\\projects\\cource\\git\\magister_diploma\\step_2\\source\\train\\HEGNN\\data', help='data directory (required)')
parser.add_argument('--dataset_name', type=str, default='5_0_0', help='name of dataset (required)')
parser.add_argument('--max_train_samples', type=int, default=1e8, help='maximum amount of train samples (default: 1e8)')
parser.add_argument('--max_test_samples', type=int, default=1e8, help='maximum amount of valid and test samples (default: 1e8)')


# Training
parser.add_argument('--seed', type=int, default=43, help='random seed (default: 43)')
parser.add_argument('--epochs', type=int, default=600, help='int type, number of epochs to train (default: 600)')
parser.add_argument('--batch_size', type=int, default=256, help='int type, batch size for training (default: 256)')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate (lr) of optimizer (default: 5e-4)')
parser.add_argument('--weight_decay', type=float, default=1e-12, help='weight decay of optimizer (default: 1e-12)')
parser.add_argument('--times', type=int, default=1, help='experiment repeat times (default: 1)')
parser.add_argument('--early_stop', type=int, default=100, help='early stop (default: 100)')
parser.add_argument('--sample', type=int, default=3, help='how much to sample')


# Log
parser.add_argument('--log_directory', type=str, default='./logs/nbody', help='directory to generate the json log file (default: ./logs/nbody)')
parser.add_argument('--test_interval', type=int, default=5, help='how many epochs to wait before logging test (default: 5)')


# Fast EGNN
parser.add_argument('--cutoff_rate', type=float, default=0, help='cutoff rate of edge_rr')
parser.add_argument('--virtual_channel', type=int, default=1, help='channel count of virtual node')


# Device
parser.add_argument('--device', type=str, default='cpu', help='device (default: cpu)')


args=parser.parse_args()
# print(args)


def get_velocity_attr(loc, vel, rows, cols):
    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff / norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    log_time_suffix = str(time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time())))

    dataset = partial(NBodySystemDataset, dataset_name=args.dataset_name, data_dir=args.data_directory, 
                      cutoff_rate=args.cutoff_rate, virtual_channels=args.virtual_channel, device=args.device)
    dataset_train = dataset(max_samples=args.max_train_samples, partition='train')
    dataset_valid = dataset(max_samples=args.max_test_samples,  partition='valid')
    dataset_test  = dataset(max_samples=args.max_test_samples,  partition='test')

    loader = partial(DataLoader, batch_size=args.batch_size, drop_last=True, num_workers=4)
    loader_train = loader(dataset=dataset_train, shuffle=True)
    loader_valid = loader(dataset=dataset_valid, shuffle=False)
    loader_test  = loader(dataset=dataset_test,  shuffle=False)
    
    fix_seed(seed=args.seed)
    print(args.model)
    # Model
    model = HEGNN(num_layer=args.num_layer, node_input_dim=2, edge_attr_dim=2, hidden_dim=args.dim_hidden, max_ell=args.ell, device=args.device)
 
    print(model)
    print("Number of parameters: %d" % count_parameters(model))
    loss_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    log_directory = args.log_directory
    log_name = f'{args.exp_name}_loss_{log_time_suffix}.json'

    best_log_dict, log_dict = train(model, loader_train, loader_valid, loader_test, optimizer, loss_mse, sigma=args.sigma,
                                    weight=args.weight, device=args.device, test_interval=args.test_interval, config=args,
                                    log_directory=log_directory, log_name=log_name, early_stop=args.early_stop, sample=args.sample)
