#!/bin/bash

module load cuda/12.3

export NCCL_HOME=/projects/hpcapps/orybchuk/superbench/compile_on_kestrel/nccl
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_OFI_CXI_COUNTER_REPORT=2

export LD_LIBRARY_PATH=$NCCL_HOME/lib:$NCCL_HOME/plugin/lib:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=NODE

echo "NCCL_HOME is set to $NCCL_HOME"
echo ========== RUNNING NCCL TESTS ==========
srun -N 2 --ntasks=8 --ntasks-per-node=4 $PWD/nccl-tests/build/all_reduce_perf -b 8 -e 4G -f 2

srun -N 2 --ntasks=8 --ntasks-per-node=4 $PWD/nccl-tests/build/all_gather_perf -b 8 -e 4G -f 2
