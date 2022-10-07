#!/usr/bin/env bash

# SETUP
# setup gpu env
conda env create -f environment.yml
conda activate unle

cp etc/conda/activate.d/env_vars.sh $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
cp etc/conda/deactivate.d/env_vars.sh $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh


python -m pip install blackjax
python -m pip install abcpy
python -m pip install git+https://github.com/mackelab/pyloric.git

# forked snvi in order to record intermediary results
pythom -m pip install pck3/

# custom dependencies
# helpers for plotting densities
python -m pip install -e density-utils/
# helpers for launching/loading experiments
python -m pip install -e experiments_utils/
# helpers to test GPU+jax style setups
python -m pip install -e jax-utils/
# forked version of sbibm that returns posterior objects.
python -m pip install -e sbibm/
# sbibm bridge for SMNLE
python -m pip install -e smnle/

# # test gpu env is well installed
python -c 'from jax_utils import test_jax_installation; test_jax_installation.test_jax_installation()'

# to make lotka_volterra simulations faster
# export JULIA_SYSIMAGE_DIFFEQTORCH="$HOME/.julia_sysimage_diffeqtorch.so"

# TEARDOWN
# unlink custom activation/deactivation scripts
# rm $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# rm $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
# 
# # uninstall custom dependencies
# # python -m pip uninstall -y density-utils
# # python -m pip uninstall -y experiments_utils
# # python -m pip uninstall -y jax-utils
# # python -m pip uninstall -y sbibm
# # python -m pip uninstall -y smnle
