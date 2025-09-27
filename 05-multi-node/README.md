# Multi-node training
Now that we've laid the groundwork, we are ready to train on multiple nodes.

### Pure data parallelism
The simplest multi-node parallelization strategy is to adopt pure data parallelism, as demonstrated in `train_ddp.py` and `train_fsdp.py`. These scripts are almost identical to their counterparts in `03-single-node/`, except for one change to `DistributedSampler`: change `rank=local_rank` to `rank=global_rank`, otherwise you will train on duplicate data across nodes.

To run these training scripts, run `run_pure_ddp.sh` or `run_pure_fsdp.sh`. These scripts set the appropriate environment variables, as discussed in `04-multi-node-sanity-check/`. These bash scripts also show the correct way to set `RDZV_ENDPOINT` for an HPC system and the way to pair SLURM and `torchrun`, e.g., `srun -N 2 --ntasks-per-node 1 torchrun --nnodes 2 --nproc-per-node 4 --rdzv-backend c10d --rdzv-endpoint $RDZV_ENDPOINT --rdzv-id 42 train_ddp.py`.

The PyTorch docs talk about something called [Hybrid Sharding Data Parallel (HSDP)](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html#how-to-use-devicemesh-with-hsdp). This is a pure data parallelism strategy in which you use FSDP within a node and DDP across nodes, as FSDP has higher communication costs. I don't demo this strategy, but I mention it here for completeness. I'll also note that DDP and FSDP processed one epoch of data in approximately the same amount of time---roughly 15 min across two nodes---which implies that the extra communication costs from FSDP are minimal in this configuration.

### Combining model parallelism and data parallelism (2D/3D parallelism)
For the final training example in this tutorial, I demonstrate how to run Pipeline Parallelism on the 4 GPUs within a node and Data Parallelism (DP) across two nodes. This concept is an example of 2D parallelism. If we added in Tensor Parallelism, it would be 3D parallelism. To run 2D parallelism, we need to make 3 main changes.

First, our DeviceMesh must now specify sub-meshes for PP and DP. The order of the names matters. The below sequence sets up the DDP mesh between two nodes and the PP mesh within a node.
```
world_mesh = init_device_mesh("cuda", mesh_shape=(2,4), mesh_dim_names=("ddp","pp"))
pp_mesh = world_mesh['pp']
ddp_mesh = world_mesh['ddp']
```

Second, special care must be given to the dataset and data loading. You want the different GPUs connected by PP to see the same input/output pair, but you want the different GPUs connected by DP to see different input/output pairs. For the sake of simplicity, I ensured this process by creating two unique datasets: years 2008 and 2010 go to node0 and years 2011 and 2013 go to node1. This feels hacky, so be cautious if you choose to adopt this strategy. At the same time, I think HuggingFace's `datasets` library basically does the same thing with their [split_dataset_by_node](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.distributed.split_dataset_by_node) function. Within each node, I use the same data handling process as `03-single-node/train_pp.py`.

Third, you need to actually apply PP and DP to your `whole_model`. The order of operations here was tricky, and I was able to figure out something that didn't crash by looking at [TorchTitan's train.py](https://github.com/pytorch/torchtitan/blob/main/torchtitan/train.py). (Full disclosure: I'm not sure that I set this up correctly, as one epoch of running DDP+PP took the same amount of time as running one epoch of running pure DDP on a single node. This could indicate that something about the pipeline is broken or that there is just more substantially more communication overhead than I anticipated.) First, create your local models and your stages for PP. Next, apply DP. For FSDP, this process looked the same as for the single node case, but for DDP, this involved a whole different approach:
```
from torch.distributed._composable.replicate import replicate
...
replicate(local_model, device_mesh=ddp_mesh, bucket_cap_mb=100)
```
I think this function gets around some weirdness about copying PyTorch `nn.Module` objects. After applying DP, create the pipeline schedule. From there, proceed to the training loop as was done in the single-node PP example.

### Wrapping up
To conclude, in this tutorial we started with a pre-existing SciML algorithm and incrementally scaled it up so that it can run on multiple nodes. This tutorial showcased several tools along the way, ranging from activation checkpointing, to `torchrun`, to memory profiling code. These tools can help you scale up your ML algorithm of interest, and I'll note a few more useful concepts that I didn't have time to touch on
* Checkpointing and Stateful DataLoaders
* Meta devices for when the whole model parameters are too big to fit onto one GPU
* Recovering from hardware failures

Hopefully, I gave a realistic sense of the different pitfalls and challenges that you can encounter in this scaling process. If you find any bugs, please submit an issue.

Thanks for reading!
