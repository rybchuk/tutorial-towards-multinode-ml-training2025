# Step 1: Increase the size of the network and break it on a single GPU
In the previous example, we were training our network to generate high-resolution images of cropped ERA5 regions sized (128, 128). Let's now train a network to generate full horizontal slices of ERA5 sized (720, 1440). 

Run `train.py`. This script loads full ERA5 slices using `era5_dataset` instead of smaller regions like with `era5_dataset_cropped`. This script is basically otherwise identical to `train_with_profiling.py`. As provided, the script hits an OOM error.

One simple way to decrease memory requirements is to decrease the batch size. However, even if we drop the batch size from 16 to 1, the network still OOMs. Using what you learned in the last section, you may be inclined to look at a memory profile now. However, a profile doesn't get generated when an OOM error happens. You can look at a Memory Snapshot however, though I don't find them particularly insightful in this current situation.