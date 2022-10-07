This repository contains the code to reproduce the results of the ICLR 2023 submission: "Maximum Likelihood Learning of Energy-Based Models for Simulation-Based Inference"


Requirements:

- Conda
- Optionally (recommended) A GPU, along with cuda driver libraries.

To reproduce the experiments, please first set up an environment (which will be named `unle`) containing all the necessary dependencies by running the lines present in the `setup.sh` file provided at the top-level of this repository.
Additionally, we strongly recommend that you use a GPU when running the experiments; using a GPU yields considerable speedups for training and inference. All necessary GPU libraries
should be installed when running the `setup.sh` script, apart from the driver libraries which should already be available on your system.
`CUDA`-specific environment variables will be set upon activation of this environment, which are necessary to make `jax` work when using user-installed (in this case, via `conda`) 
`CUDA` runtime libraries.

All the experiment submission / visualisation scripts take the form of a `jupyter` notebook. **No `jupyter` notebook engine is not provided** as part of the environment. You can either install 
`jupyter-notebook`/`jupyterlab` in this environment directly (by running the bash command `conda install -n unle jupyterlab`), or register the
`python` executable of the `unle` environment to an external `jupyterlab` engine. In the
latter case, the aforementioned CUDA environment variables need to be specified in the `share/jupyter/kernels/unle/kernel.json` (this file being relative to the jupyterlab environment root folder).
Here is an example `kernel.json` file that does so. You will need to change the placeholder paths indicated using </the/following/convention>:

```json
{
 "argv": [
  "</path/to/unle/env>/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "unle",
 "language": "python",
 "metadata": {
  "debugger": true
 },
 "env": {
   "XLA_FLAGS":"--xla_gpu_cuda_data_dir=</path/to/unle/env> --xla_gpu_autotune_level=0",
   "LD_LIBRARY_PATH":"</path/to/unle/env>/lib",
   "PATH":"</path/to/unle/env>/bin:$PATH",
   "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
   "XLA_PYTHON_CLIENT_ALLOCATOR":"platform"
  }
}
```

The results used to plot the figure of the submission are provided in the following [google drive location](https://drive.google.com/drive/folders/1f3MCjNZUE5BhIYcEYc9U4rsHlPMyMQOJ?usp=sharing)
an external google drive location. To plot the results, either re-run the experiments, or download the contents present at the google drive link and 

- move `iclr_experiments_2` into `sbi_ebm/examples/experiments/experiment_2/results/iclr_experiments_2`
- move `iclr_experiments_3` into `sbi_ebm/examples/experiments/experiment_3/results/iclr_experiments_3`
