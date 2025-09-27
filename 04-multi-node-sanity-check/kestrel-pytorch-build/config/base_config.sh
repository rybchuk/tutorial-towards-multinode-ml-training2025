# Source me to setup config for the installations

source "$(dirname "${BASH_SOURCE[0]}")/../utils/logging.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../utils/validation.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../utils/env_utils.sh"

INSTALL_BASE=/scratch/$USER/conda

# Configure the installation
export INSTALL_NAME="pytorch"
export PYTHON_VERSION=3.12
export PYTORCH_VERSION="2.8.0"
export PYTORCH_BRANCH="v${PYTORCH_VERSION}"
export PYTORCH_URL=https://github.com/pytorch/pytorch.git
export VISION_VERSION="0.23.0"
export VISION_BRANCH="v${VISION_VERSION}"
export RESUME_PYTORCH_BUILD=${RESUME_PYTORCH_BUILD:-false}
export BUILD_DIR=/scratch/$USER/pytorch-build/$INSTALL_NAME/$PYTORCH_VERSION
export INSTALL_DIR=$INSTALL_BASE/$INSTALL_NAME/$PYTORCH_VERSION
export CMAKE_PREFIX_PATH=$INSTALL_DIR:${CMAKE_PREFIX_PATH:-}
# export RESUME_PYTORCH_BUILD="true"  # Added for debugging
# export BUILD_ENV=false  # Added for debugging

# Setup programming environment
module load cmake

# gcc 13 + cuda 12.9
module unload PrgEnv-gnu
module load gcc-stdalone/13.1.0
module load cuda/12.9
module load cudnn/9.2.0.82-12

# Ending loads/unloads
module load conda
module unload craype-x86-genoa
module list

export LD_PRELOAD="/nopt/nrel/apps/gpu_stack/compilers/06-24/linux-rhel8-zen4/gcc-12.3.0/gcc-13.1.0-2gnfzy5425yehx7zzh237h5jouktucg4/lib64/libstdc++.so.6:$LD_PRELOAD"
export CUDA_HOME=/nopt/cuda/12.9
export NCCL_ROOT=/projects/hpcapps/orybchuk/superbench/compile_on_kestrel/nccl
export NCCL_INCLUDE_DIR=$NCCL_ROOT/include
export NCCL_LIB_DIR=$NCCL_ROOT/lib
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=0
export MAX_JOBS=4  # Build on single GPU node

export NVCC_PREPEND_FLAGS='-allow-unsupported-compiler'


export CXX=g++
export CC=gcc

# Validate configuration
validate_env_vars #|| exit 1
validate_dependencies #|| exit 1

# Print some stuff
module list
echo "Configuring on $(hostname) as $USER"
echo "  Build directory $BUILD_DIR"
echo "  Install directory $INSTALL_DIR"
module list
