import torch
import os
import sys
import types
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from timm import utils
sys.path.append('../')
from data.dataGenerator import DataGeneratorECG
from utils.name_save_path import name_save_path
from utils.load_train_objs import load_train_objs
import numpy as np
torch.seed()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import random
import warnings
from tables import NaturalNameWarning
from utils.trainer import Trainer
warnings.filterwarnings('ignore', category=NaturalNameWarning)
random.seed(32)
np.random.seed(32)

def setup_for_distributed(is_master):
    """
	This function disables printing when not in master process
	"""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.device = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.device = args.rank % torch.cuda.device_count()
        args.world_size = int(os.environ['SLURM_NNODES']) * int(os.environ['SLURM_NTASKS_PER_NODE'])
    else:
        print('Not using distributed mode')
        args.device = 0
        args.rank = 0
        args.distributed = False
        device = torch.device(args.device)
        return device

    args.distributed = True
    args.dist_url = "env://"

    torch.cuda.set_device(args.device)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    device = torch.device(args.device)
    return device

def main(total_epochs: int, save_every: int, batch_size: int, optim_name: str, lr: float,
         dropout: float, path_train_set: str, path_datasetinfo: str, save_path_train: str, prefix: str, loss1: str
         , min_lr: float, sqi: float, ft, path_pretrained_model: str,
         num_channels, weights_mse_inorout_lastsecond, threshold_mse_inside_outside,
        add_FC, norm_a_b_max_min, dataset_FT, input_size_seconds):

    if not ft:
        print("Traning from scratch")
    else:
        print("Fine-tuning")

    distributed_args = types.SimpleNamespace()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = init_distributed_mode(distributed_args)

    if distributed_args.distributed:
        print('Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {distributed_args.rank}, total {distributed_args.world_size}, device {distributed_args.device}.')
    else:
        print("Not distributed mode")

    amp_autocast = partial(torch.autocast, device_type=device.type, dtype=torch.float16)
    utils.random_seed(32, distributed_args.rank)


    tensor_dataset = DataGeneratorECG(path_train_set, batch_size, sqi, ft, path_datasetinfo, num_channels, norm_a_b_max_min, dataset_FT, input_size_seconds, val=False)
    tensor_dataset_val = DataGeneratorECG(path_train_set, batch_size, sqi, ft, path_datasetinfo, num_channels, norm_a_b_max_min, dataset_FT, input_size_seconds, val=True)

    if distributed_args.distributed:
        data_loader = DataLoader(tensor_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                 sampler=DistributedSampler(tensor_dataset), num_workers=8, persistent_workers=True,
                                 prefetch_factor=4)
        data_loader_val = DataLoader(tensor_dataset_val, batch_size=batch_size, pin_memory=True, shuffle=False,
                             sampler=DistributedSampler(tensor_dataset_val), num_workers=8, persistent_workers=True, prefetch_factor=4)


    else:
        data_loader = DataLoader(tensor_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8, persistent_workers=True, prefetch_factor=4)
        data_loader_val = DataLoader(tensor_dataset_val, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8, persistent_workers=True, prefetch_factor=4)

    model, optimizer, lr_scheduler = load_train_objs(optim_name, lr, ft, path_pretrained_model, total_epochs,
                                min_lr, num_channels, add_FC, input_size_seconds, dropout)

    save_path_train = name_save_path(save_path_train, prefix, loss1, optim_name,
                   batch_size, lr, min_lr, dropout, sqi, num_channels, total_epochs,
                   ft, weights_mse_inorout_lastsecond, threshold_mse_inside_outside,
                   add_FC, norm_a_b_max_min,
                   dataset_FT, input_size_seconds)

    if not os.path.exists(save_path_train):
        os.makedirs(save_path_train, exist_ok=True)
        print("The new directory is created!")

    print(f'Output path: {save_path_train}')

    writer = SummaryWriter(save_path_train + '/runs')

    start_epoch = 0
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    trainer = Trainer(model, data_loader, optimizer, device, save_every, save_path_train, loss1, batch_size, writer,
                          lr_scheduler, distributed_args, amp_autocast,
                          ft, num_channels, data_loader_val, weights_mse_inorout_lastsecond,
                          threshold_mse_inside_outside, path_train_set, norm_a_b_max_min, dataset_FT,
                          path_datasetinfo, input_size_seconds)

    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    # Training parameters
    parser.add_argument('--total_epochs', default=300, type=int, help='Total number of training epochs.')
    parser.add_argument('--save_every', default=2, type=int, help='Frequency (in epochs) to save model checkpoints.')
    parser.add_argument('--batch_size', default=1024, type=int, help='Mini-batch size per device used during training.')
    parser.add_argument('--optim_name', type=str, default='SGD', help='Optimizer to use for training (e.g., SGD, Adam, AdamW).')
    parser.add_argument('--lr', type=float, required=False, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--min_lr', type=float, required=False, default=0.1, help='Minimum learning rate (used with scheduler).')
    parser.add_argument('--dropout', type=float, required=False, default=0, help='Dropout rate applied to fusion layers.')
    parser.add_argument('--loss1', type=str, required=False, default="mse_inside_outside_thresholds", help="Loss function name (mse_inside_outside_thresholds).")
    parser.add_argument('--ft', default=False, action='store_true', help="Enable fine-tuning mode (load pretrained weights and continue training).")
    parser.add_argument('--weights_mse_inorout_lastsecond', type=float, nargs='+', required=False, default=[1.0, 1.0], help='Weights applied to inside/outside MSE loss.')
    parser.add_argument('--threshold_mse_inside_outside', type=float, nargs='+', required=False, default=[0.25,-0.25], help='Threshold values used to separate inside and outside regions for MSE loss.')

    # Dataset parameters
    parser.add_argument('--sqi', type=float, required=False, default=-1.0, help="Signal Quality Index (SQI) threshold. Set to -1 to disable filtering.")
    parser.add_argument('--num_channels', type=int, required=False, default=1, help='Number of input channels')
    parser.add_argument('--norm_a_b_max_min', default=False, action='store_true', help="Apply min-max normalization to the input signal.")
    parser.add_argument('--dataset_FT', type=str, required=False, default="None", help="Path or identifier of the dataset used for fine-tuning (if applicable).")
    parser.add_argument('--input_size_seconds', default=4, type=float, help='Duration (in seconds) of each input sample.')

    # Model parameters
    parser.add_argument('--add_FC', default=False, action='store_true', help="Add an additional fully connected (FC) layer at the end of the model.")

    # Paths and prefix
    parser.add_argument('--prefix', type=str, required=False, default="experiments_ssq", help="Prefix string used to name the experiment.")
    parser.add_argument('--path_pretrained_model', default='', type=str, required=False, help="Path to the pretrained model to load.")
    parser.add_argument('--path_train_set', default='', type=str, required=False, help="Path to the training dataset directory.")
    parser.add_argument('--path_datasetinfo', default='', type=str, required=False, help="Path to the dataset metadata file (HDF5 with signal information).")
    parser.add_argument('--save_path_train', type=str, default='', required=False, help="Directory path where training checkpoints and logs will be saved.")
    #################
    args = parser.parse_args()
    total_epochs = args.total_epochs
    save_every = args.save_every
    batch_size = args.batch_size
    optim_name = args.optim_name
    lr = args.lr
    min_lr = args.min_lr
    dropout = args.dropout
    loss1 = args.loss1
    sqi = args.sqi
    ft = args.ft
    path_pretrained_model = args.path_pretrained_model
    num_channels = args.num_channels
    path_train_set = args.path_train_set
    path_datasetinfo = args.path_datasetinfo
    save_path_train = args.save_path_train
    prefix = args.prefix
    weights_mse_inorout_lastsecond= args.weights_mse_inorout_lastsecond
    threshold_mse_inside_outside = args.threshold_mse_inside_outside
    add_FC = args.add_FC
    norm_a_b_max_min = args.norm_a_b_max_min
    dataset_FT = args.dataset_FT
    input_size_seconds = args.input_size_seconds

    main(total_epochs, save_every, batch_size, optim_name, lr, dropout, path_train_set, path_datasetinfo,
        save_path_train, prefix, loss1, min_lr, sqi, ft, path_pretrained_model,
        num_channels, weights_mse_inorout_lastsecond, threshold_mse_inside_outside,
         add_FC, norm_a_b_max_min, dataset_FT, input_size_seconds)


