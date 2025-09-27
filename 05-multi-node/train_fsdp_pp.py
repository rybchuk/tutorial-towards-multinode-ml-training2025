import copy
from datetime import datetime
import numpy as np
import os
from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.profiler import record_function
from torch.utils.data import DataLoader

from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

from era5_dataset_partitioned import GetClimateDataset
from tt_dataloader import ParallelAwareDataloader
from profiling_support import maybe_enable_profiling, maybe_enable_memory_snapshot
from SwinIR_pp import SwinIR

def prune_modules(modules_to_keep, whole_model):
    '''
    Based off TorchTitan's _build_stage_from_modules
    '''
    model = copy.deepcopy(whole_model)

    new_layers = []
    for module_name, module_value in model.named_children():
        # Allow coarse grained paritioning of layer-like structures (e.g., "layers.0", "layers.1")
        if isinstance(module_value, (nn.ModuleDict, nn.ModuleList)):
            layers_to_keep = {
                name.split(".", 1)[1]
                for name in modules_to_keep
                if name.startswith(f"{module_name}.")
            }
            if layers_to_keep:
                # Keep only specified layers
                if isinstance(module_value, nn.ModuleDict):
                    for layer_name in list(module_value.keys()):
                        if layer_name not in layers_to_keep:
                            del module_value[layer_name]
                elif isinstance(module_value, nn.ModuleList):
                    indices_to_keep = {
                        int(idx) for idx in layers_to_keep if idx.isdigit()
                    }
                    new_layers = nn.ModuleList(
                        [
                            layer
                            for i, layer in enumerate(module_value)
                            if i in indices_to_keep
                        ]
                    )
                    setattr(model, module_name, new_layers)
            else:
                # No layers from this structure needed, set to empty structure
                if isinstance(module_value, nn.ModuleDict):
                    setattr(model, module_name, nn.ModuleDict())
                elif isinstance(module_value, nn.ModuleList):
                    setattr(model, module_name, nn.ModuleList())
        # Handle simple module attributes (e.g., "linear", "norm")
        elif module_name not in modules_to_keep:
            # Replace with None
            setattr(model, module_name, None)

    return model

def main():
    ### ~~~~~ Set up the multiple GPUs ~~~~~
    init_process_group(backend="nccl")
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 8, "This script has been hardcoded for two GPU nodes, each with 4 GPUs. Stopping."
    world_mesh = init_device_mesh("cuda", mesh_shape=(2,4), mesh_dim_names=("fsdp","pp"))
    pp_mesh = world_mesh['pp']
    fsdp_mesh = world_mesh['fsdp']

    device_type = 'cuda'
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    node_rank = int(os.environ['GROUP_RANK'])
    fsdp_degree = fsdp_mesh.size()
    fsdp_rank = fsdp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()
    pp_rank = pp_mesh.get_local_rank()
    device = torch.device(f"{device_type}:{local_rank}")
    is_local_device0 = (local_rank==0)
    is_last_device = (local_rank==3)
    if is_local_device0:
        print(f"[node{node_rank}] pp_mesh: {pp_mesh}")
        print(f"[node{node_rank}] fsdp_mesh: {fsdp_mesh}")

    # Set different random seeds on different nodes
    torch.manual_seed(node_rank)
    os.environ["PYTHONHASHSEED"] = str(node_rank % 2**32)

    ### ~~~~~ Parameters (hardcoded for now) ~~~~~
    ## Args for Dataset
    upscale_factor = 2
    transform = torch.from_numpy
    noise = 0.0
    dataset_mean = [6.3024, 278.3945, 18.4262] # hardcoded from get_data_info()
    dataset_std = [3.7376, 21.0588, 16.4687]  # hardcoded from get_data_info()
    downsampling_method = "bicubic"

    ## Args for DataLoader
    batch_size = 4  # Local batch size, not global batch size
    num_dl_workers = 4

    ## Args for model
    in_channels = 3
    window_size = 8
    img_height = (720 // upscale_factor // window_size + 1) * window_size
    img_width = (1440 // upscale_factor // window_size + 1) * window_size

    ## Args for training
    total_epochs = 5
    base_lr = 8e-4
    weight_decay = 1e-6
    gamma_sched = 0.97

    ## Args for profiling and memory snapshot
    enable_profiling = True
    enable_profiling_with_memory = True
    enable_snapshot = False
    logdir = Path('fsdp_pp')
    logdir.mkdir(exist_ok=True)
    break_batch_idx = 9  # WAIT + WARMUP + ACTIVE

    ### ~~~~~~~~~~ Load data ~~~~~~~~~~
    # Training
    if node_rank == 0:
        years_to_load = [2008, 2010]
    elif node_rank == 1:
        years_to_load = [2011, 2013]
    else:
        raise ValueError("This script is hardcoded to work with two nodes")
    dataset_path_train = Path('../datasets/era5/train/')
    dataset_train = GetClimateDataset(location=dataset_path_train,
                                    train=True,
                                    transform=transform,
                                    upscale_factor=upscale_factor,
                                    noise_ratio=noise,
                                    std=dataset_std,
                                    method=downsampling_method,
                                    years=years_to_load)
    dl_train = DataLoader(dataset_train,
                            batch_size = batch_size,
                            num_workers = num_dl_workers,
                            sampler = None,
                            drop_last = True,
                            pin_memory = True)
    if is_local_device0: print(f"Training dataset has {len(dataset_train)} samples, and there are {len(dl_train)} batches")

    ### ~~~~~~~~~~ Initialize model ~~~~~~~~~~
    ## Global model
    input_size = [360, 720]  # Should be the same size as img_size, but isn't; I think the discrepancy comes from padding?
    whole_model = SwinIR(upscale=upscale_factor, 
                in_chans=in_channels, 
                img_size=(img_height, img_width),
                input_size=input_size,
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
    whole_model = whole_model.to(device)
    if is_local_device0: print(f"[GPU{node_rank}] Initialized global model")

    ## Partition the model to prepare for PP
    # Partition the model and form the stages
    if pp_rank == 0:
        modules_to_keep = ['shift_mean_sub', 'conv_first', 'patch_embed', 'pos_drop', 'layers.0']
    elif pp_rank == 1:
        modules_to_keep = ['layers.1', 'layers.2']
    elif pp_rank == 2:
        modules_to_keep = ['layers.3', 'layers.4']
    elif pp_rank == 3:
        modules_to_keep = ['layers.5', 'norm_after_ff', 'patch_unembed', 'conv_after_body', 'conv_before_upsample', 'upsample', 'conv_last', 'shift_mean_add']
    else:
        raise ValueError("Pruning is currently hardcoded to 4 GPUs")
    local_model = prune_modules(modules_to_keep, whole_model)

    stage_idx = pp_rank
    num_stages = pp_degree
    local_stage = PipelineStage(
        local_model,
        stage_idx,
        num_stages,
        device,
        group=pp_mesh.get_group("pp"),
    )
    if is_local_device0: print(f"[GPU{node_rank}] Partitioned model and created stages")
    trainable_params = sum(p.numel() for p in local_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in local_model.parameters())
    print(f"[{device}] Trainable parameters: {trainable_params:,} / {total_params:,}")

    ## Apply data parallelism
    fsdp_kwargs = {
        "mesh": fsdp_mesh
    }
    for layer in local_model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(local_model, **fsdp_kwargs)
    if is_local_device0: print(f"[GPU{node_rank}] Applied FSDP")

    del whole_model

    ## Create the pipeline schedule
    n_microbatches = 1 * batch_size
    loss_fn = torch.nn.L1Loss()
    pp_schedule = ScheduleGPipe(local_stage, n_microbatches=n_microbatches, loss_fn=loss_fn)
    if is_local_device0: print(f"[GPU{node_rank}] Created pipeline schedule")

    ## Model summary
    if is_local_device0:
        # print(model)
        print('**** Model setup complete ****')

    ## Set model parts to train mode
    for _, m in local_model.named_modules():
        m.train()
    if is_local_device0: print(f"[GPU{node_rank}] Set model parts to train mode")

    ### ~~~~~~~~~~ Set optimizer, loss function and learning rate scheduler ~~~~~~~~~~
    optimizer = torch.optim.AdamW(local_model.parameters(), lr=base_lr, weight_decay=weight_decay)  # AdamW for SwinIR from SuperBench's utils.py
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma_sched)
    loss_fcn = torch.nn.L1Loss().to(device)

    ### ~~~~~ Train! ~~~~~
    print(f"[{device}]: Kicking off training!")
    start_epoch = 0
    with maybe_enable_profiling(
        enable_profiling=enable_profiling, with_memory=enable_profiling_with_memory, log_dir=logdir
    ) as torch_profiler, maybe_enable_memory_snapshot(
        enable_snapshot=enable_snapshot, log_dir_parent=logdir
    ) as memory_profiler:
        for epoch in range(start_epoch,total_epochs):
            epoch_train_loss = 0

            for batch_idx, (model_input, target) in enumerate(dl_train): # data shape: [b,c,h,w]
                if is_local_device0:
                    model_input = model_input.to(device)
                    target = None
                    losses = None
                elif is_last_device:
                    model_input = None
                    target = target.to(device)
                    losses = []
                else:
                    model_input = None
                    target = None
                    losses = None

                # Reset gradients
                optimizer.zero_grad()

                # Run forward+backward
                if is_local_device0:
                    pp_schedule.step(model_input, target=target, losses=losses)
                else:
                    pp_schedule.step(target=target, losses=losses)

                # Calculate losses
                if is_last_device:
                    loss = torch.mean(torch.stack(losses))
                    epoch_train_loss += loss.item() / len(dl_train)
                else:
                    loss = torch.tensor([-1.0], device=device)

                # Run optimizer and update LR scheduler
                optimizer.step()

                # Signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                if batch_idx == break_batch_idx:
                    break
            break
            current_lr = optimizer.param_groups[0]['lr']
            if is_last_device: print(f"[node{node_rank}] Loss: {epoch_train_loss}\tLR: {current_lr}")
            lr_scheduler.step()

    destroy_process_group()


if __name__ == "__main__":
    main()
