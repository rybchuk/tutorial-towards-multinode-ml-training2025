# Step 0: Run and benchmark SuperBench training for a "small" network configuration on a single GPU
Before we scale up our algorithm, we need to get the training code working for a baseline case.

### Install the necessary Python libraries
I recommend using [conda](https://nrel.github.io/HPC/Documentation/Environment/Customization/conda/) to create an environment for this tutorial. In additional to the usual PyTorch libraries (`torch`, `torchvision`), you will also need less common libraries (e.g., `timm`, `h5py`). You can see my exact conda environment in `conda-env.yml`. I'll note that I will be using PyTorch 2.7.1 unless explicitly stated.

### `train.py`
[Download the curated ERA5 dataset](https://github.com/erichson/SuperBench?tab=readme-ov-file#usage) and untar it. Move the `train/` folder so your directory has this structure: `tutorial-towards-multinode-ml-training2025/datasets/era5/train`.

First, we will run `train.py`. This script leverages code from the SuperBench repo that has been modified. This script has the following structure:
* Define important parameters, including `crop_size=128`, the size of the high-resolution output of the network
* Set up the data loader
* Initialize the model, optimizer, loss function, and learning rate
* Kick off the training loop
    * To keep us focused on the problem of scaling to large samples, I keep the number of training epochs small, omit model verification code, and omit wandb/Tensorboard logging

To run this script, launch a SLURM batch job that looks like:
```
# ...
# <SLURM ARGUMENTS>
# ...

module load conda
conda activate <environment name>

export CUDA_VISIBLE_DEVICES=0  # This might not be necessary on Kestrel

python train.py
```

If all goes well, you will see something like:
```
Getting file stats from ../datasets/era5/train/2008.h5
Number of samples per year: 365
Training dataset has 11680 samples, and there are 730 batches
/projects/ai4wind/orybchuk/conda/sciml_2412_pt250/lib/python3.12/site-packages/torch/functional.py:534: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /opt/conda/conda-bld/pytorch_1728945388038/work/aten/src/ATen/native/TensorShape.cpp:3595.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
**** Setup ****
Total params Generator: 11.752M
************
Kicking off training!
Epoch : 1 - datetime 2025-07-07 16:11:38.789383 - training loss : 1.2190 - current_lr: 0.0008

Epoch : 2 - datetime 2025-07-07 16:15:56.399834 - training loss : 0.4999 - current_lr: 0.000776
```

Side note: If I happen to be at my computer when the SLURM job starts, I like to monitor the state of the GPU at a coarse level. To do this, I (1) identify the name of the GPU node that my job is running on, (2) SSH into it in a new terminal session, e.g., `ssh x3112c0s41b0n0`, and then (3) run `nvtop`. In the below image, you can see that we're only running on GPU0, which is our intended behavior at the moment.

![nvtop](../figs/00_nvtop.png?raw=true "nvtop")

### `train_with_profiling.py`: Memory and speed profiling
Now that we've established that we can run the training loop, let's get a greater understanding of how the GPUs are being used. We're going to do some profiling, which will give us a better sense of how fast certain parts of the training code are running and how memory is being used. If you want to make your code use memory more efficiently or run faster, it's usually helpful to better understand what your code is currently doing before trying to blindly improve it.

As far as I am aware, there are two different profiling tools that are popular today for our problem of interest: the native PyTorch profiler and NVIDIA Nsight. These tools have different strengths and weaknesses. In my opinion, the PyTorch profiler is simpler and can better access details inside of the main Python process, whereas Nsight is better at dealing with multiple processes (e.g., like with a multi-worker DataLoader) and with inter-node communication. I'll will only demonstrate native PyTorch tools, but I wanted to mention Nsight for the sake of completeness. If you want to learn more about Nsight, check out [this video](https://www.youtube.com/watch?v=K27rLXkOiqo&t=95s) on Argonne's YouTube channel.

We're now going to run `train_with_profiling.py`. This script is identical to `train.py` everywhere except in the training loop.
* We wrap the entire training loop inside of two functions. The first one is `maybe_enable_profiling()` and it will give us a profiler with speed info. The second one is `maybe_enable_memory_snapshot()` and it will give us information about memory usage.
    * This code is based off a [TorchTitan](https://github.com/pytorch/torchtitan) script
    * The first function deploys the PyTorch profiler `torch.profiler.profile`. This saves out information about performance (speed). It also conveniently shows how much memory is being used by different aspects of the network (parameters, activations, optimizer state, etc.)
    * The second function saves out a "Memory Snapshot" that shows the live tensors over time. I personally have found this tool less useful than the memory profiler.
    * You can read more about these PyTorch profiling tools [here](https://pytorch.org/blog/understanding-gpu-memory-1/) and [here](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
* We also now add annotations that can be used to flag sections of code: `with record_function("## forward ##"):`
* Profiling saves out data at a very high frequency, leading to large files. As such, it is common practice to run profilers for short periods (<1 min) and just a few batches of data


##### Performance profiling
Run `train_with_profiling.py` on a GPU node. This script will generate a few files. Download the contents of `logs/` to your local machine. We will start by opening the performance trace `logs/rank0_trace.json`. Download the file and go to https://ui.perfetto.dev/ . This is a tool made by Google to conveniently analyze performance traces using the brower as the interface. An older generation of this tool called `chrome://tracing` lived within the Chrome browser. Drag the performance trace into Perfetto. Click on the `python <number>` row, and you should see a trace that looks like this:

![Performance trace](../figs/00_perfetto_zoomed_out.png?raw=true "Performance trace")

There's a ton of information in this trace and it can be overwhelming. Our goal with this performance trace is to see how long different sections of the code run for, so that way we better understand how to speed up our script. Like I mentioned earlier, I am moreso focused on memory optimization than speed optimization in this tutorial, so I will discuss performances trace very briefly. I haven't found any simple and comprehensive guides on how to parse these traces, but I have found it helpful to paste screenshots of these traces into an LLM and to talk to it about the trace. The GPU Mode YouTube channel also has some in depth videos on performance traces. 

 In the above image, the `3606751` row shows the code processing 5 batches of data, each corresponding to a downward spike. Zoom into one of these batches (Ctrl + scroll on a Mac).

 ![Performance trace, zoomed in](../figs/00_perfetto_zoomed_in.png?raw=true "Performance trace, zoomed in")

You can see the forward pass section (cyan) and the backward pass section (blue). After the backward section, you can see a very short optimization section. You can zoom in further to see even more granular detail. You can further dig through the performance trace file to understand the timing of data loading. The data loader is often a bottleneck, leading to projects like [NVIDIA DALI](https://developer.nvidia.com/dali).

##### Memory profiling
Drag `logs/rank_memoty0.html` into your web browser. You will see a chart of memory usage over time. Again, you see the profiler process 5 batches of data, with maximum memory allocation occupying around 24 GB of VRAM.

![Memory trace](../figs/00_memory_trace.png?raw=true "Memory trace")

The memory usage is helpfully binned into different categories. You can see that the vast majority of memory goes to the training activations, which is typical of Transformer-based networks. As a result, we will target a strategy to minimize the memory demands of these activations later in the tutorial.

##### Memory Snapshot
Grab `logs/iteration_5/rank0_memory_snapshot.pickle` and upload it to https://docs.pytorch.org/memory_viz. 

![Memory snapsho](../figs/00_memory_snapshot.png?raw=true "Memory snapshot")

Here, you also get a plot of memory usage versus time, as well as a few other tabs with very detailed information about the memory state. My impression is that this tool is moreso targeted at programmers doing low-level optimizations, which I don't have experience with. 

We have now demonstrated that we can train a baseline network, and we have explored some profiling tools. Next, let's try to train on larger input samples.