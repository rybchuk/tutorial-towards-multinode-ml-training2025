#!/bin/bash
source /projects/hpcapps/orybchuk/superbench/compile_on_kestrel/pytorch-2.8.0-cuda-12.9/kestrel-pytorch-build/config/base_config.sh
module list

# Activate your PyTorch environment
conda activate /scratch/orybchuk/conda/pytorch/2.8.0
export LD_PRELOAD="/nopt/nrel/apps/gpu_stack/compilers/06-24/linux-rhel8-zen4/gcc-12.3.0/gcc-13.1.0-2gnfzy5425yehx7zzh237h5jouktucg4/lib64/libstdc++.so.6:$LD_PRELOAD"  # technically not necessary for this script

# NCCL runtime variables
export NCCL_DEBUG=INFO
export NCCL_HOME=/projects/hpcapps/orybchuk/superbench/compile_on_kestrel/nccl
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$NCCL_HOME/plugin/lib:$LD_LIBRARY_PATH
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_OFI_CXI_COUNTER_REPORT=2
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=LOC
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
echo "NCCL_HOME is set to $NCCL_HOME"

srun -N 2 --ntasks=8 --ntasks-per-node=4 python test_pytorch_nccl.py
