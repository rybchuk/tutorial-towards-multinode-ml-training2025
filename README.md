# A journey towards multi-node ML training
### A motivating problem
Let's say you want to train a "big" neural network, but you're running out of GPU memory or the network is training too slowly. In this tutorial, I walk through a few strategies to train big neural networks in Pytorch, with a particular emphasis on Out Of Memory (OOM) errors. I became particularly interested in this problem when I started trying to apply neural networks [to large three-dimensional arrays](https://onlinelibrary.wiley.com/doi/full/10.1002/we.70020), on which I quickly ran out of memory. I think neural network OOMs are an increasingly important problem in the scientific machine learning world as we move away from toy problems and towards larger scale, real-world problems. LLM researchers have devoted a lot of attention to this problem in recent years, and this tutorial leverages several strategies that have been deployed by that community. I'll also emphasize that I am fairly new to the world of multi-node ML, so if you find any bugs, please submit an issue and let me know.

The tutorial is structured in the following manner: 
* I start with strategies involving only a single GPU. I introduce a tool for profiling memory and speed here.
* I then discuss strategies for a handful of GPUs that are all located on a single node and linked with fast interconnects.
* I wrap up by talking about multi-node strategies, in which the nodes have relatively slow interconnects between them. As part of this discussion, I build PyTorch from source to deal with problems that arise on compute systems that use Slingshot.

To make things more concrete, I'll use a reference problem that is quite popular today: super-resolution of weather data. In this problem, you start with weather data over some region at a coarse resolution, and then you use a neural network to upsample that data to a finer resolution. There are a lot of public resources on this problem, and we're going to use the [SuperBench](https://github.com/erichson/SuperBench) repo as our starting point. This repo has both a nice weather dataset as well as a bunch of super-resolution algorithms ready to go. SuperBench was initially developed to work on cropped regions from the [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) dataset sized (128, 128). We will scale the algorithm to read in full horizontal slices of ERA5 sized (720, 1440). After we successfully scale to inputs of this size, we then add in more GPUs to increase the global batch size and reduce single-GPU memory demands.

To keep things simple, we're only going to use SwinIR, an algorithm based off Swin Transformers. Many of the techniques I talk about here are architecture agnostic (e.g., Transformer-based, convolution-based, etc.), but not all of them are.

I'll also note that this tutorial is particularly targeted at my coworkers at NREL, but I think this tutorial is useful to a wider audience. NREL's supercomputer is called Kestrel. Each GPU node on Kestrel has 4 GPUs, and they are connected via NVLink. Communication between nodes happens over Slingshot.

Helpful resources:
* The [TorchTitan](https://github.com/pytorch/torchtitan) project. I largely built this tutorial by picking apart their codebase, so a huge shout out to the TorchTitan team.
* A [2022 PyTorch blog post](https://medium.com/pytorch/pytorch-data-parallel-best-practices-on-google-cloud-6c8da2be180d) showcasing several techniques discussed in this tutorial
* [Ahmed Taha's explanation of FSDP](https://www.youtube.com/watch?v=By_O0k102PY)
* "[Slaying OOMs with PyTorch FSDP and torchao](https://www.youtube.com/watch?v=UvRl4ansfCg)" by two PyTorch devs
* The [GPU Mode](https://www.youtube.com/@GPUMODE/videos) group
* "[From bare metal to a 70B model](https://imbue.com/research/70b-infrastructure/)" by the Imbue team
* "[AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions](https://arxiv.org/abs/2509.13523)" - a Swin-based weather forecasting model that uses advanced model parallelism techniques (code should be open sourced in the coming months)
