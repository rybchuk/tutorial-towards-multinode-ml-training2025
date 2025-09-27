# Building PyTorch from source for multi-node training on a Slingshot system
In the last section, we saw different parallelism strategies deployed on a single GPU node. Using that foundation, it is typically pretty straightforward to start training on a multi-node cluster. However, we are going to take a quick detour. The nodes on NREL's supercomputer Kestrel are connected via Slingshot, and NCCL doesn't natively support Slingshot. We could technically do multi-node training by communicating over a different network on Kestrel, but it is substantially slower. Instead, we will need to build NCCL and then PyTorch using the [aws-ofi-nccl plugin](https://github.com/aws/aws-ofi-nccl) If you're not at NREL, you can probably skip this section and move on to the next one. Speaking of which, a huge shout out to Matt Selensky and Walid Arsalane for figuring all of this out.

### Building and testing NCCL
To build NCCL, run `build_nccl.sh`, but change `INSTALL_DIR` to a directory that you can access. Make sure you are logged in on a GPU node.

To confirm that NCCL was built successfully, run `run_standalone_nccl.sh` in an interactive session with two GPU nodes or as a batch job. This runs all-reduce and all-gather operations. If everything went well, you'll see in-place busbw cap out around 80 GB/s.
```
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum	  -1    18.56    0.00    0.00	   0    17.42    0.00    0.00	   0
          16             4     float     sum	  -1    16.55    0.00    0.00	   0    16.57    0.00    0.00	   0
          32             8     float     sum	  -1    16.85    0.00    0.00	   0    17.29    0.00    0.00	   0
       	8192          2048     float     sum      -1    31.43    0.26    0.46      0    24.99    0.33    0.57      0
...
  1073741824     268435456     float     sum      -1    23803   45.11   78.94      0    23788   45.14   78.99      0
  2147483648     536870912     float     sum      -1    47402   45.30   79.28      0    47352   45.35   79.37      0
  4294967296    1073741824     float     sum      -1    94534   45.43   79.51      0    94484   45.46   79.55      0
```

### Building PyTorch
Next, we'll build PyTorch and torchvision from source. I modified [nersc-pytorch-build](https://github.com/sparticlesteve/nersc-pytorch-build/tree/main) (shout out to @sparticlesteve) for Kestrel to build PyTorch 2.8.0 using CUDA 12.9. Prepare to run `install.sh`, but before you do:
* Modify `SCRIPT_DIR` in `install.sh`
* Modify `NCCL_INCLUDE_DIR` and `NCCL_LIB_DIR` in `base_config.sh` to point to your version of NCCL. You may want to change `INSTALL_BASE` as well.

If all goes well, afer a few hours you'll see a message like:
```
Successfully installed pillow-11.3.0 torchvision-0.23.0a0+824e8c8
/scratch/orybchuk/pytorch-build/pytorch/2.8.0
[2025-08-18 12:11:37] INFO:  Successfully completed build of pytorch
[2025-08-18 12:11:37] INFO:  Installation completed successfully
```

### Running PyTorch
To run PyTorch going forward, load the correct libraries and modify `LD_PRELOAD`
```
# Load modules in correct order
module unload PrgEnv-gnu
module load gcc-stdalone/13.1.0
module load cuda/12.9
module load cudnn/9.2.0.82-12
module unload craype-x86-genoa
module load conda

# Activate your PyTorch environment
conda activate /scratch/$USER/conda/pytorch/2.8.0 

# Force the correct libstdc++ to load first
export LD_PRELOAD="/nopt/nrel/apps/gpu_stack/compilers/06-24/linux-rhel8-zen4/gcc-12.3.0/gcc-13.1.0-2gnfzy5425yehx7zzh237h5jouktucg4/lib64/libstdc++.so.6:$LD_PRELOAD"

# Now PyTorch should import successfully
python -c "import torch; print('PyTorch imported successfully!')"
```

If you don't update `LD_PRELOAD`, you'll see an error like this:
```
(/scratch/orybchuk/conda/pytorch/2.5.1) orybchuk@x3113c0s9b0n0 ~ $ python
Python 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/scratch/orybchuk/conda/pytorch/2.5.1/lib/python3.12/site-packages/torch/__init__.py", line 367, in <module>
    from torch._C import *  # noqa: F403
    ^^^^^^^^^^^^^^^^^^^^^^
ImportError: /scratch/orybchuk/conda/pytorch/2.5.1/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: _cray$mt_kmpc_fork_call_with_flags
>>> exit()
```

### Testing the PyTorch build
To confirm that PyTorch was built correctly, run `test_python_nccl.sh` in an interactive session with two GPU nodes. One unfortunate situation is that we needed to set the runtime variable `export NCCL_NET_GDR_LEVEL=LOC`, which has the effect of throttling busbw to a cap of around 55 GB/s. If this shell script runs well, you'll see something like:
```
Size (elements) Data Type  Op       Time (ms)    Alg BW (GB/s)   Bus BW (GB/s)   Correct
------------------------------------------------------------------------------------------
1024            float32    sum      0.038        0.11            0.19            ✓
1024            float16    sum      0.029        0.07            0.12            ✓
...
134217728	float32    sum      17.143	 31.32           54.81           ✓
134217728	float16    sum      8.763        30.63           53.61           ✓
```
