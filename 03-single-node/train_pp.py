import copy
from datetime import datetime
import numpy as np
import os
from pathlib import Path
import random
import torch
from torch.profiler import record_function
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from custom_pipeline import create_pipeline_stage

from era5_dataset import GetClimateDataset
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
    pp_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("pp",))
    seed = 42
    torch.manual_seed(seed)  # Based of TT's set_determinism, it's helpful to set a common seed across PP devices

    device_type = 'cuda'
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"{device_type}:{local_rank}")
    is_device0 = (local_rank==0)
    is_last_device = (local_rank==3)
    if is_device0: print(f'Using {pp_mesh["pp"]} GPUs for Pipeline Parallel')

    ### ~~~~~ Parameters (hardcoded for now) ~~~~~
    ## Args for Dataset
    upscale_factor = 2
    transform = torch.from_numpy
    noise = 0.0
    dataset_mean = [6.3024, 278.3945, 18.4262] # hardcoded from get_data_info()
    dataset_std = [3.7376, 21.0588, 16.4687]  # hardcoded from get_data_info()
    downsampling_method = "bicubic"

    ## Args for DataLoader
    # batch_size = 1
    batch_size = 1 * world_size
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
    logdir = Path('pp')
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
    dl_train = DataLoader(dataset_train,
                            batch_size = batch_size,
                            num_workers = num_dl_workers,
                            sampler = None,
                            drop_last = True,
                            pin_memory = True)
    if is_device0: print(f"Training dataset has {len(dataset_train)} samples, and there are {len(dl_train)} batches")

    ### ~~~~~~~~~~ Initialize model ~~~~~~~~~~
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
    if is_device0: print(f"Total number of layers in the whole model: {len(list(whole_model.modules()))}")

    # ### ~~~~~~~~~~ Apply Pipeline Parallelism ~~~~~~~~~~
    ## Partition the model and form the stages
    if local_rank == 0:
        modules_to_keep = ['shift_mean_sub', 'conv_first', 'patch_embed', 'pos_drop', 'layers.0']
    elif local_rank == 1:
        modules_to_keep = ['layers.1', 'layers.2']
    elif local_rank == 2:
        modules_to_keep = ['layers.3', 'layers.4']
    elif local_rank == 3:
        modules_to_keep = ['layers.5', 'norm_after_ff', 'patch_unembed', 'conv_after_body', 'conv_before_upsample', 'upsample', 'conv_last', 'shift_mean_add']
    else:
        raise ValueError("Pruning is currently hardcoded to 4 GPUs")
    local_model = prune_modules(modules_to_keep, whole_model)

    for _, m in local_model.named_modules():
        m.train()
    trainable_params = sum(p.numel() for p in local_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in local_model.parameters())
    print(f"[{device}] Trainable parameters: {trainable_params:,} / {total_params:,}")

    del whole_model

    ## Create the pipeline / pipeline schedule
    stage_idx = local_rank
    num_stages = world_size
    local_stage = PipelineStage(
        local_model,
        stage_idx,
        num_stages,
        device,
        group=pp_mesh.get_group("pp"),
    )

    n_microbatches = 1 * world_size
    loss_fn = torch.nn.L1Loss()
    pp_schedule = ScheduleGPipe(local_stage, n_microbatches=n_microbatches, loss_fn=loss_fn)

    ### ~~~~~~~~~~ Set optimizer and learning rate scheduler ~~~~~~~~~~
    optimizer = torch.optim.AdamW(local_model.parameters(), lr=base_lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma_sched)

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
                if is_device0:
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
                if is_device0:
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
            lr_scheduler.step()

    destroy_process_group()


if __name__ == "__main__":
    main()
