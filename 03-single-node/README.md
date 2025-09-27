# Step 3: Optimizations on a single GPU node with 4 GPUs
Now that we have reduced memory demands on a single GPU, we can move on to training on multiple GPUs within a node. When training on multiple GPUs, you need to inform PyTorch of the existence of the different GPUs, and you need to set up communications between them. 
* In recent years, PyTorch has encouraged the use of `torchrun` to run jobs on multiple GPUs. `torchrun` gets installed alongside the `torch` library, so you should already have access to it. In the previous section, we launched jobs with `python my_script.py`. Now, we will launch jobs with `torchrun --nnodes 1 --nproc-per-node 4 my_script.py`, assuming you want to run on `4` GPUs.
* Additionally, PyTorch has encouraged the use of `DeviceMesh`. This abstraction is especially helpful if you want to apply multiple types of parallelism at once, and you want to specify how parallelism is configured across GPUs. For example, in a multi-node training setup, you may want to apply Tensor Parallelism within a node and Data Parallelism between nodes. This is a relatively new feature and the official documentation is a bit confusing, but I have had success with the following code:
```
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh

def main():
    ### ~~~~~ Set up the multiple GPUs ~~~~~
    init_process_group(backend="nccl")
    world_size = int(os.environ["WORLD_SIZE"])
    ddp_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("ddp",))

    device_type = 'cuda'
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"{device_type}:{local_rank}")
    is_device0 = (local_rank==0)

...  # Rest of code

    destroy_process_group()
```

### Distributed Data Parallel (DDP)
Once the devices are configured, we can start to apply parallel training techniques. The simplest way to do multi-GPU training is to do DDP, which is demonstrated in `train_ddp.py`. DDP distributes a local batch of data to each GPU, and then training information (gradient info) is synced between GPUs during the backward pass. At its core, DDP lets you have a larger global batch size. Fundamentally, DDP gets deployed with this snippet of code:
```
model = model.to(device)
ddp_model = DDP(model, device_mesh=ddp_mesh)
```

DDP also uses a `DistributedSampler` to coordinate the different inputs across the different GPUs:
```
sampler_train = DistributedSampler(dataset_train, 
                                   num_replicas=world_size, 
                                   rank=local_rank, 
                                   shuffle=True)
dl_train = DataLoader(dataset_train,
                        batch_size = batch_size,
                        num_workers = num_dl_workers,
                        sampler = sampler_train,
                        drop_last = True,
                        pin_memory = True)
```

Here is a memory trace from DDP. This trace is just collected from GPU0, not all the GPUs. As such, you don't see any meaningful differences from the single-GPU memory trace from the past section. But DDP enables you to increase your global batch size while keeping your memory demands constant on each GPU.

![Memory trace with DDP](../figs/03_memory_trace_ddp.png?raw=true "Memory trace with DDP")

### Fully Sharded Data Parallel (FSDP)
In recent years, a different type of data parallelism called Fully Sharded Data Parallel (FSDP) has grown in popularity and has become easier to use. FSDP has undergone heavy development recently, and PyTorch (as of 2.7 I believe) now automatically uses something that was called FSDP2. The best explanation of FSDP that I've come across is by [Ahmed Taha](https://www.youtube.com/watch?v=By_O0k102PY). Just like DDP, FSDP allows you to increase your global batch size. Unlike DDP, it reduces memory requirements from parameters, gradients, and the optimizer state by taking a GPU-to-GPU communication penalty. I demonstrate FSDP in `train_fsdp.py`. As of PyTorch 2.7, this is how FSDP is deployed:
```
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy

fsdp_kwargs = {                         # Option to replace AMP with FSDP2-specific mixed precision
    # "mp_policy": MixedPrecisionPolicy(
    #     param_dtype=torch.bfloat16,
    #     reduce_dtype=torch.float32
    # ),  # TODO: Resolve dtype bug
    "mesh": fsdp_mesh
}
for layer in model.layers:
    fully_shard(layer, **fsdp_kwargs)
fully_shard(model, **fsdp_kwargs)
```

The PyTorch docs encourage swapping out Automatic Mixed Precision with a prescribed `mp_policy`, though I haven't had success with it. This should be possible, and TorchTitan [mentions it](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md). Below, you can see a memory trace from FSDP. The main benefit is that the (already small) memory demand from the parameter state and optimizer state falls even further with FSDP. You may notice that FSDP has a slightly higher memory usage than DDP, but that is because I didn't run the FSDP script with any sort of mixed precision policy.

![Memory trace with FSDP](../figs/03_memory_trace_fsdp.png?raw=true "Memory trace with FSDP")


### Pipeline Parallelism (PP)
Now, let's look at our first model parallelization strategy: Pipeline Parallelism (PP). I demo this technique in `train_pp.py` and `SwinIR_pp.py`. 

Let's say your network consists of 6 layers. When using DDP, each GPU stores the entirety of each of those 6 layers. When using FSDP, each GPU also stores each of those 6 layers, but only a portion of each of them. If you use PP with four GPUs, you can store `layer0` on GPU0, `layer1` and `layer2` on GPU1, `layer3` and `layer4` on GPU2, and `layer5` on GPU3. PP can substantially reduce the memory demands on an individual GPU. However, to enable this, PP requires communication across all GPUs during each forward and backward pass.

As of summer 2025, PyTorch's [PP code](https://docs.pytorch.org/docs/stable/distributed.pipelining.html) `torch.distributed.pipelining` is in alpha state and under heavy development. [TorchTitan](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/pipeline_parallel.py) has a working version of PP, and I base my code off their code.

Before I demo how to implement PP, I wanted to urge some caution:
* Before putting this tutorial together, I was under the impression that I could run with a global batch size of 1, but unfortunately I think the minimum batch size is `n_gpus`.
   * I believe this is enforced so that the pipeline can split the data into microbatches, which enable each GPU to run compute operations in parallel with minimal downtime ("bubbles")
* I believe that some architectures are fundamentally incompatible with PP, particularly those with many complex components or connection patterns.
   * I haven't dug into the [FLUX architecture](https://discuss.pytorch.org/t/distributed-w-torchtitan-flux-is-here-experience-diffusion-model-training-on-torchtitan/221119), but notably, the TorchTitan team doesn't showcase PP usage for it.
   * I had to make some modifications to the basic SwinIR code (see below)
* As I understand it, it is best practice to scale up as much as possible using data parallelism first before you hit sufficiently diminishing returns. After this point, you should add in model parallelism on top of the data parallelism.

To set up PP, start by assigning different layers of your global model to different GPUs. You can either do this manually or you can try to [do this automatically](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html) with a `split_spec`. I wasn't able to get the automatic process to work, so I did this manually. I started by making a deepcopy of the global model on a given GPU, and then removing the layers that shouldn't be retained on that GPU. As discussed in `00-run-baseline/`, most of the memory demands come from the 6 RSTB blocks, so I split these up (unevenly) across the four GPUs. I also assigned the modules before RSTB0 to GPU0 and the modules after RSTB5 to GPU3. After this pruning, you can delete the original global model to free up space. Once each GPU has its model layers, set up the sections of the pipeline by assigning each local model with `PipelineStage()`. Finally, define the execution pattern for the pipeline with `pp_schedule = ScheduleGPipe(local_stage, n_microbatches=n_microbatches, loss_fn=loss_fn)`. I set `n_microbatches` to match the global batch size.

In order to get PP to work with the provided SwinIR code, I made two main modifications. 
* I removed the skip connection that started before all 6 Residual Swin Transformer Blocks (RSTBs) and ended after all these blocks. This skip connection was preventing a clean separation that would allow unidirectional flow from GPU0 to GPU3.
   * Unfortunately, I think the model performs worse after removing this skip connection. But nonetheless I decided to remove it for the sake of demoing PP and the potential challenges that come with it
* I split the `self.shift_mean` module into `self.shift_mean_add` and `self.shift_mean_sub`. `self.shift_mean` was being used both before and after the RSTBs, and I couldn't figure out how to specifically assign `self.shift_mean(x, mode='sub')` to GPU0 and `self.shift_mean(x, mode='add')` to GPU3.

To execute PP training, you need to make several changes relative to single-GPU training. For starters, model input/output handling needs to be more explicit to ensure each GPU sees what it should:
```
for batch_idx, (model_input, target) in enumerate(dl_train): # data shape: [b,c,h,w]
    if is_device0 == 0:
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
```

Also, the forward and backward statements now look like:
```
# Run forward+backward
if local_rank == 0:
    pp_schedule.step(model_input, target=target, losses=losses)
else:
    pp_schedule.step(target=target, losses=losses)
```

Finally, the optimizer statement gets modified too:
```
# Calculate losses
if is_last_device:
    loss = torch.mean(torch.stack(losses))
else:
    loss = torch.tensor([-1.0], device=device)

# Run optimizer and update LR scheduler
optimizer.step()
```

With these modifications, we can run PP training and profile the code. Here's the memory usage of GPU0: <img width="1600" height="960" alt="03_memory_trace_pp" src="https://github.com/user-attachments/assets/4fc4ee2a-173b-4355-821f-f1df54fc61ce" />

As a reminder, we're using a global batch size of 4, whereas in the `02-single-gpu/` cases we used a global batch size of 1. The above trace shows that a similar amount of memory is ultimately used, though the shape of the activation memory is different here. The use of microbatches flattens out the activation memory demand, relative to the sharp pyramid seen in the previous examples. Interestingly, new types of memory demands now pop up: `TEMPORARY`, `AUTOGRAD_DETAIL`, and `Unknown`. I'll also mention that GPU1 and GPU2 probably use twice as much memory as GPU0 and GPU3, as these two GPUs hold twice as many RSTB layers. 

### Tensor Parallelism (TP)
For the sake of completeness, I will briefly demo a different model parallelism strategy called [Tensor Parallelism](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) (TP, read more [here](https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/) and [here](https://www.determined.ai/blog/tp)). I demo TP in `train_tp.py`. In PP, different layers were partitioned to different GPUs. TP works differently. Imagine `layer1` in our model involves a very large matrix multiplication, like with classic self-attention or an MLP layer. With TP, that matrix multiplication operation gets split up across multiple GPUs. This parallelism strategy uses even more communication than PP, so as a rule of thumb, use TP between GPUs that have a fast interconnect.

Just like PP, TP has its limitations. I think it is supported in PyTorch for self-attention and MLP layers, but [not things like convolutions and group norms](https://github.com/pytorch/pytorch/issues/133221). Our SwinIR model uses a special type of attention, so TP can't be applied to that attention layer without some custom modifications. However, TP can still be applied to the MLP layers in the Swin Transformer Blocks. Also, in this example, I was unable to produce any meaningful savings in memory if I was also using Activation Checkpointing. So for the sake of demonstrating TP, I removed AC here. I was also having problems with Automatic Mixed Precision.

To set up TP, add this code after defining your model:
```
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel
)

...

    for layer_id, rstb_layer in enumerate(model.layers):  # Iterate over RSTBs
        for block_id, transformer_block in enumerate(rstb_layer.residual_group.blocks):  # Iterate over SwinTransformerBlocks
            # Only apply TP
            layer_tp_plan = {
                "mlp.fc1": ColwiseParallel(),
                "mlp.fc2": RowwiseParallel(),
            } 

            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan
            )
```

I set up this plan by opening up `SwinIR.py` and finding the names of the `Linear` layers within `Mlp`. 

To wrap up the TP section, I'll mention Sequence Parallelism. I didn't have time to dig into this more, but it often gets mentioned alongside TP. SP splits data along the `hidden_state` dimension, so it should be able to deal with training for a global batch size of 1. I believe SP only works for Transformer-style networks.
