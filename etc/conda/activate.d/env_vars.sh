#!/usr/bin/env bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CONDA_PREFIX} --xla_gpu_autotune_level=0"
# export JULIA_SYSIMAGE_DIFFEQTORCH="${CONDA_PREFIX}/.julia_sysimage_diffeqtorch.so"
export JULIA_SYSIMAGE_DIFFEQTORCH="${HOME}/.julia_sysimage_diffeqtorch.so"


# Don't prealloacte 90% of GPU memory as it recently led to memory leaks
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
# export $XLA_PYTHON_CLIENT_MEM_FRACTION=".5"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
