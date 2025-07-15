from datetime import datetime
import numpy as np
import os
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader

from era5_dataset_cropped import GetClimateDataset
from SwinIR import SwinIR

def main():
    ### ~~~~~ Parameters (hardcoded for simplicity) ~~~~~
    device = 'cuda:0'

    ## Args for Dataset
    upscale_factor = 2
    transform = torch.from_numpy
    noise = 0.0
    dataset_mean = [6.3024, 278.3945, 18.4262] # hardcoded from get_data_info()
    dataset_std = [3.7376, 21.0588, 16.4687]  # hardcoded from get_data_info()
    crop_size = 128  # images are always cropped into squares in the original SuperBench; height of high resolution image
    n_patches = 8
    downsampling_method = "bicubic"

    ## Args for DataLoader
    batch_size = 16
    num_dl_workers = 4

    ## Args for model
    in_channels = 3
    hidden_channels = 64
    window_size = 8
    img_height = (crop_size // upscale_factor // window_size + 1) * window_size  # Height of low resolution image
    img_width = (crop_size // upscale_factor // window_size + 1) * window_size

    ## Args for training
    total_epochs = 2
    base_lr = 8e-4
    weight_decay = 1e-6
    gamma_sched = 0.97
    print_epoch_frequency = 1

    ### ~~~~~~~~~~ Load data ~~~~~~~~~~
    # Training
    dataset_path_train = Path('../datasets/era5/train/')
    dataset_train = GetClimateDataset(location=dataset_path_train,
                                      train=True,
                                      transform=transform,
                                      upscale_factor=upscale_factor,
                                      noise_ratio=noise,
                                      std=dataset_std,
                                      crop_size=crop_size,
                                      n_patches=n_patches,
                                      method=downsampling_method)

    dl_train = DataLoader(dataset_train,
                            batch_size = batch_size,
                            num_workers = num_dl_workers,
                            shuffle = True,
                            sampler = None,
                            drop_last = True,
                            pin_memory = True)
    print(f"Training dataset has {len(dataset_train)} samples, and there are {len(dl_train)} batches")

    ### ~~~~~~~~~~ Initialize model ~~~~~~~~~~
    model = SwinIR(upscale=upscale_factor, in_chans=in_channels, img_size=(img_height, img_width),
                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv',mean=dataset_mean,std=dataset_std)
    model = model.to(device)

    # Model summary
    # print(model)
    print('**** Setup ****')
    print('Total params Generator: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('************')

    ### ~~~~~~~~~~ Set optimizer, loss function and learning rate scheduler ~~~~~~~~~~
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)  # AdamW for SwinIR from SuperBench's utils.py
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma_sched)
    loss_fcn = torch.nn.L1Loss().to(device)

    ### ~~~~~ Train! ~~~~~
    print("Kicking off training!")
    start_epoch = 0
    for epoch in range(start_epoch,total_epochs):
        model.train()
        epoch_train_loss = 0
        for batch_idx, (model_input, target) in enumerate(dl_train): # data shape: [b,c,h,w]
            # Push data to device
            model_input, target = model_input.float().to(device), target.float().to(device)

            # Forward
            model_output = model(model_input)
            loss = loss_fcn(model_output, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() / len(dl_train)

        current_lr = optimizer.param_groups[0]['lr']
        if epoch % print_epoch_frequency == 0:
            print(
                f"Epoch : {epoch+1} - datetime {datetime.now()} - training loss : {epoch_train_loss:.4f} - current_lr: {current_lr}\n"
            )

        scheduler.step()

if __name__ == "__main__":
    main()
