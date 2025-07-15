# Step 3: Optimizations on a single GPU node with 4 GPUs
The README and scripts for this section are still under development. I am happy with the Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP) demos. I want to be more confident in my Tensor Parallel (TP) and Pipeline Parallel (PP) experiments before saying anything more definitive about them. 

Now that we have reduced memory demand on a single GPU, we can move on to training on multiple GPUs within a node. When training on multiple GPUs, you need to inform PyTorch of the existence of the different GPUs, and you need to set up communications between them. 
* In recent years, PyTorch has encouraged the use of `torchrun` to run jobs on multiple GPUs. `torchrun` gets installed along side the `torch` library, so you should already have access to it. In the previous section, we launched jobs with `python my_script.py`. Now, we will launch jobs with `torchrun --nnodes 1 --nprocs-per-node 2 my_script.py`, assuming you want to run on `2` GPUs.
* Additionally, PyTorch has encouraged the use of `DeviceMesh`. This abstraction is especially helpful if you want to apply multiple types of parallelism at once, and you want to specify how parallelism is configured across GPUs. For example, in a multi-node training setup, you may want to apply Tensor Parallelism within a node and Data Parallelism between nodes. This is a relatively new feature and the official documentation is a bit confusing, but I have had success with the following code:
```
def main():
    ### ~~~~~ Set up the multiple GPUs ~~~~~
    init_process_group(backend="nccl")
    world_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    rank = world_mesh.get_rank()
    torch.cuda.set_device(rank)

...  # Rest of code

    destroy_process_group()
```

Once the devices are configured, we can start to apply parallel training techniques. The simplest way to do multi-GPU training is to do DDP, which is demonstrated in `train_ddp.py`. DDP distributes a local batch of data to each GPU, and then training information (gradient info) is synced between GPUs during the backward pass. At its core, DDP lets you have a larger global batch size. Fundamentally, DDP gets deployed with this snippet of code:
```
    model = model.to(rank)
    ddp_model = DDP(model, device_mesh=world_mesh)
```

Here is a memory trace from DDP. This trace is just collected from GPU0, not all the GPUs. As such, you don't see any meaningful differences from the single-GPU memory trace from the past section. 

![Memory trace with DDP](../figs/03_memory_trace_ddp.png?raw=true "Memory trace with DDP")

In recent years, a different type of data parallelism called Fully Shareded Data Parallel (FSDP) has grown in popularity and in ease of usage. FSDP has gone through heavy development recently, and PyTorch (as of 2.7 I believe) now automatically uses something that was called FSDP2. The best explanation of FSDP that I've come across is by [Ahmed Taha](https://www.youtube.com/watch?v=By_O0k102PY). Fundamentally, FSDP allows you to increase your global batch size, and it also reduces memory requirements from parameters, gradients, and the optimizer state. I demonstrate FSDP in `train_fsdp.py`. As of PyTorch 2.7, this is how FSDP is deployed:
```
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy

fsdp_kwargs = {                         # Option to replace AMP with FSDP2-specific mixed precision
    # "mp_policy": MixedPrecisionPolicy(
    #     param_dtype=torch.bfloat16,
    #     reduce_dtype=torch.float32
    # ),  # TODO: Resolve dtype bug
    "mesh": world_mesh
}
for layer in model.layers:
    fully_shard(layer, **fsdp_kwargs)
fully_shard(model, **fsdp_kwargs)
```

The PyTorch docs encourage swapping out Automatic Mixed Precision with a prescribed `mp_policy`, though I haven't had success with it. Below, you can see a Memory Trace from FSDP. The main difference is that the (already small) memory demand from the parameter state and optimizer state falls even further with FSDP. You may notice that FSDP has a slightly higher memory usage than DDP, but that is because I didn't run the FSDP script with any sort of mixed precision policy (due to running out of time).

![Memory trace with FSDP](../figs/03_memory_trace_fsdp.png?raw=true "Memory trace with FSDP")


In the near future, I plan to update this tutorial to discuss Tensor Parallelism and Pipeline Parallelism.
