from datetime import datetime
import numpy as np
import os
from pathlib import Path
import random
import torch
from torch.profiler import record_function
from torch.utils.data import DataLoader

from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import Shard
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from era5_dataset import GetClimateDataset
from profiling_support import maybe_enable_profiling, maybe_enable_memory_snapshot
from SwinIR import SwinIR

def inspect_model(model: FSDPModule):
    '''
    From https://github.com/pytorch/examples/blob/main/distributed/FSDP2/utils.py
    '''
    assert isinstance(model, FSDPModule)

    if torch.distributed.get_rank() == 0:
        print(model)

    for param in model.parameters():
        assert param.placements == (Shard(0),)
        assert param.dtype == torch.float32

def inspect_mixed_precision(model: FSDPModule):
    '''
    From https://github.com/pytorch/examples/blob/main/distributed/FSDP2/utils.py
    '''
    model.unshard()
    for param in model.parameters(recurse=False):
        assert param.dtype == torch.bfloat16
    model.reshard()

def main():
    ### ~~~~~ Set up the multiple GPUs ~~~~~
    init_process_group(backend="nccl")
    world_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    rank = world_mesh.get_rank()
    torch.cuda.set_device(rank)
    print(f"Running on rank {rank}!")
    is_rank0 = (rank==0)

    ### ~~~~~ Parameters (hardcoded for now) ~~~~~
    ## Args for Dataset
    upscale_factor = 2
    transform = torch.from_numpy
    noise = 0.0
    dataset_mean = [6.3024, 278.3945, 18.4262] # hardcoded from get_data_info()
    dataset_std = [3.7376, 21.0588, 16.4687]  # hardcoded from get_data_info()
    downsampling_method = "bicubic"

    ## Args for DataLoader
    batch_size = 1  # Local batch size, not global batch size
    num_dl_workers = 4

    ## Args for model
    in_channels = 3
    window_size = 8
    img_height = (720 // upscale_factor // window_size + 1) * window_size
    img_width = (1440 // upscale_factor // window_size + 1) * window_size

    ## Args for training
    total_epochs = 2
    base_lr = 8e-4
    weight_decay = 1e-6
    gamma_sched = 0.97

    ## Args for profiling and memory snapshot
    enable_profiling = True
    enable_profiling_with_memory = True
    enable_snapshot = False
    logdir = Path('fsdp')
    logdir.mkdir(exist_ok=True)
    break_batch_idx = 9  # WAIT + WARMUP + ACTIVE

    ### ~~~~~~~~~~ Load data ~~~~~~~~~~
    # Training
    dataset_path_train = Path('../datasets/era5/train/')
    dataset_train = GetClimateDataset(location=dataset_path_train,
                                      train=True,
                                      transform=transform,
                                      upscale_factor=upscale_factor,
                                      noise_ratio=noise,
                                      std=dataset_std,
                                      method=downsampling_method)
    sampler_train = DistributedSampler(dataset_train, 
                                       num_replicas=world_size, 
                                       rank=rank, 
                                       shuffle=True)
    dl_train = DataLoader(dataset_train,
                            batch_size = batch_size,
                            num_workers = num_dl_workers,
                            sampler = sampler_train,
                            drop_last = True,
                            pin_memory = True)
    if is_rank0: print(f"Training dataset has {len(dataset_train)} samples, and there are {len(dl_train)} batches")

    ### ~~~~~~~~~~ Initialize model ~~~~~~~~~~
    model = SwinIR(upscale=upscale_factor, 
                   in_chans=in_channels, 
                   img_size=(img_height, img_width),
                   window_size=window_size,
                   img_range=1.,
                   depths=[6, 6, 6, 6, 6, 6],
                   embed_dim=180,
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2,
                   upsampler='pixelshuffle',
                   resi_connection='1conv',
                   mean=dataset_mean,
                   std=dataset_std,
                   use_checkpoint=True)
    fsdp_kwargs = {                         # Replace AMP with FSDP2-specific mixed precision
        # "mp_policy": MixedPrecisionPolicy(
        #     param_dtype=torch.bfloat16,
        #     reduce_dtype=torch.float32
        # ),  # TODO: Resolve dtype bug
        "mesh": world_mesh
    }
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    inspect_model(model)
    inspect_mixed_precision(model)

    # Model summary
    if is_rank0:
        # print(model)
        print('**** Model setup complete ****')

    ### ~~~~~~~~~~ Set optimizer, loss function and learning rate scheduler ~~~~~~~~~~
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)  # AdamW for SwinIR from SuperBench's utils.py
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma_sched)
    loss_fcn = torch.nn.L1Loss().to(rank)

    ### ~~~~~ Train! ~~~~~
    print(f"[Rank {rank}]: Kicking off training!")
    start_epoch = 0

    with maybe_enable_profiling(
        enable_profiling=enable_profiling, with_memory=enable_profiling_with_memory, log_dir=logdir
    ) as torch_profiler, maybe_enable_memory_snapshot(
        enable_snapshot=enable_snapshot, log_dir_parent=logdir
    ) as memory_profiler:
        for epoch in range(start_epoch,total_epochs):
            model.train()
            sampler_train.set_epoch(epoch)

            epoch_train_loss = 0
            for batch_idx, (model_input, target) in enumerate(dl_train): # data shape: [b,c,h,w]
                # Push data to device
                model_input, target = model_input.float().to(rank), target.float().to(rank)

                # Forward
                with record_function("## forward ##"):
                    model_output = model(model_input) 
                    loss = loss_fcn(model_output, target)
                
                # Backward
                with record_function("## backward ##"):
                    loss.backward()
                
                with record_function("## optimizer ##"):
                    optimizer.zero_grad()
                    optimizer.step()

                epoch_train_loss += loss.item() / len(dl_train)

                # Signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()
                
                if batch_idx == break_batch_idx:
                    break
                else:
                    if is_rank0: print(f"Batch index: {batch_idx}...")
            scheduler.step()
            break
    destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"{world_size} GPU devices visible")
    
    main()