# A journey towards multi-node ML training
### A motivating problem
Let's say you want to train a "big" neural network, but you're running out of GPU memory or the network is training too slowly. In this tutorial, I walk through a few strategies to train big neural networks in Pytorch, with a particular emphasis on Out Of Memory (OOM) errors. I became particularly interested in this problem when I started trying to apply neural networks [to large three-dimensional arrays](https://onlinelibrary.wiley.com/doi/full/10.1002/we.70020) and quickly ran out of memory. I think neural network OOMs are an increasingly important problem in the scientific machine learning world as we move away from toy problems and towards problems on more realistic scales. LLM researchers have devoted a lot of attention to this problem in recent years, and this tutorial leverages several strategies that have been deployed by that community. I'll also emphasize that I am fairly new to the world of multi-node ML, and this tutorial is a collection of useful tools and ideas that I have picked up while learning about this topic. 

The tutorial is structured in the following manner: 
* I start with strategies involving only a single GPU. I'll introduce two tools for profiling memory and speed here.
* I then discuss strategies for a handful of GPUs that are all located on a single node and linked with fast interconnects.
* I wrap up by talking about multi-node strategies, in which the nodes have relatively slow interconnects between them.

To make things more concrete, I'll use a reference problem that is quite popular today: super-resolution of weather data. In this problem, you start with weather data over some region at a coarse resolution, and then you use a neural network to upsample that data to a finer resolution. There are a lot of public resources on this problem, and we're going to use the [SuperBench](https://github.com/erichson/SuperBench) repo as our starting point. This repo has both a nice weather dataset as well as a bunch of super-resolution algorithms ready to go. SuperBench was initially developed to work on cropped regions from the [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) dataset sized (128, 128). We will scale the algorithm to read in full horizontal slices of ERA5 sized (720, 1440). After we successfully scale to inputs of this size, we then deal with a synthetic dataset with even larger samples. This specific problem is somewhat contrived for the tutorial, but it's a useful learning tool to demonstrate one approach to scale an algorithm to a larger problem.

To keep things simple, we're only going to use SwinIR, an algorithm based off Swin Transformers. I'll note that the ML community has put in a lot of work towards scaling Transformer architectures, largely motivated by improvements for LLMs. Many of the techniques I talk about here are architecture agnostic, but it's unclear to me how effective some of these methods are for e.g., convolutional neural networks.

I'll also note that this tutorial is particularly targeted at my coworkers at NREL, but people outside the lab may find this useful. Our supercomputer is called Kestrel. Each GPU node on Kestrel has 4 GPUs, and they are connected via NVLink. Communication between nodes happens over Slingshot.

Helpful resources:
* A [2022 PyTorch blog post](https://medium.com/pytorch/pytorch-data-parallel-best-practices-on-google-cloud-6c8da2be180d) showcasing several techniques discussed in this tutorial
* [Ahmed Taha's explanation of FSDP](https://www.youtube.com/watch?v=By_O0k102PY)
* "[Slaying OOMs with PyTorch FSDP and torchao](https://www.youtube.com/watch?v=UvRl4ansfCg)" by two PyTorch devs
* The [torchtitan](https://github.com/pytorch/torchtitan) project
* The [GPU Mode](https://www.youtube.com/@GPUMODE/videos) group
* "[From bare metal to a 70B model](https://imbue.com/research/70b-infrastructure/)" by the Imbue team
