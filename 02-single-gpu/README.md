# Step 2: Optimizations on a single GPU
To decrease our memory demands, we are going to employ two strategies.  

First, we will use [Automatic Mixed Precision (AMP)](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html). This tool conveniently reduces the precision of certain operations from `float32` to `float16` or `bfloat16` while keeping the training process stable. I'll note that there's been a recent push for float8 ([1](https://pytorch.org/blog/training-using-float8-fsdp2/), [2](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/)), though at present, this precision involves substantially more manual intervention and I'd say it is still an experimental feature. It's relatively simple to implement AMP. Import some components, initialize the loss gradient scaler, and then wrap a few components:
```
from torch.amp import autocast, GradScaler
...
scaler = GradScaler()
...
# Forward
with record_function("## forward ##"):
    with autocast('cuda'):
        model_output = model(model_input) 
        loss = loss_fcn(model_output, target)

# Backward
with record_function("## backward ##"):
    optimizer.zero_grad()
    scaler.scale(loss).backward()

with record_function("## optimizer ##"):
    scaler.step(optimizer)
    scaler.update()
```

If we run with AMP, our network successfully trains! If you look at `logs_amp/rank0_memory.html`, you will see the network uses about 70 GB of VRAM.

![Memory trace with AMP](../figs/02_memory_trace_amp.png?raw=true "Memory trace with AMP")

The second strategy is known as [Activation Checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html) (AC). You can see it in action along with Activation Offloading and Parameter Offloading [here](https://medium.com/pytorch/pytorch-data-parallel-best-practices-on-google-cloud-6c8da2be180d). AC is a neat tool that lets you decrease memory demands of you activations so long as you are okay increasing compute time. How does it work? Imagine feeding a neural network an input during training. During the forward pass, activations get saved at every layer so they can be used later in combination with gradient values in the backward pass. As we saw earlier, these activations can represent the majority of the memory requirements, especially for Transformer-based networks. If you use AC, you only save activations for a fraction of the total layers. Then, when you need activations in the backward pass, you recompute the activations in the unstored layers by recomputing them forward from a nearby saved layer. 

AC usually requires more thought to implement than AMP, but not necessarily a ton of work. In this tutorial, we got lucky because SuperBench's SwinIR already has activation checkpointing built into it. All we need to do is set `model = SwinIR(upscale=upscale_factor, ..., use_checkpoint=True)`. If you're building off of someone else's machine learning code, there's a good chance that activation checkpointing may already be built into your architecture as well. If you need to implement AC from scratch for your network, you can see that SwinIR's implementation is simple. Inside their `BasicLayer` class, you see the line where it gets deployed:
```
def forward(self, x, x_size):
    for blk in self.blocks:
        if self.use_checkpoint:
            x = checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)
        else:
            x = blk(x, x_size)
    if self.downsample is not None:
        x = self.downsample(x)
    return x
```

If we include both AMP and AC, the training process only uses 12 GB of VRAM! This is a reduction of more than 5x from before. However, the training takes about twice as long in this case. Ultimately, you need to make the call on if this tradeoff is worth it for you. For more info, there has been some recent work to allow [a more fine grained tradeoff](https://www.youtube.com/watch?v=v3gsrJtGLiA) between memory savings and compute penalty.

![Memory trace with AMP and AC](../figs/02_memory_trace_amp_ac.png?raw=true "Memory trace with AMP and AC")

To close out, I will mention [torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html). This tool can speed up your training code for relatively little effort. However, it does not result in memory savings, which is why I don't discuss it in more detail here. 