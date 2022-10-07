#!/usr/bin/env python3
import os
import argparse
import datetime
import inspect
import pickle
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional, Union

import cloudpickle
import pandas as pd


def _get_defaults(func: Callable) -> dict:
    defs = {}
    for k, v in dict(inspect.signature(func).parameters).items():
        if v.default is inspect._empty:
            continue
        defs[k] = v.default
    return defs


from dataclasses import dataclass

@dataclass
class SLURMJobMetadata:
    sbatch_script_filepath: Optional[Union[str, Path]]
    slurm_jobid: Optional[str]
    ran_in_slurm: bool


def run_and_save(
    func: Callable,
    kwargs: dict,
    folder_path: Path,
    experience_name: str = "experience",
    sbatch_script_filepath: Optional[str] = None,
):
    # XXX: currently,`run_and_save` requires a different folder for each different function `func`
    # TODO: create subfolder within a folder for each function.

    defaults = _get_defaults(func)
    all_kwargs = {**kwargs, "experience_name": experience_name}
    defaults.update(all_kwargs)
    all_kwargs = defaults

    print(all_kwargs)

    # CHECK IF TASK ALREADY DONE
    results_dir = folder_path

    if not results_dir.exists():
        results_dir.mkdir()

    results_metadata_path = results_dir / "results.pkl"

    if not results_metadata_path.exists():
        _mi = pd.MultiIndex.from_tuples(tuple(), names=tuple(all_kwargs.keys()))
        results_metadata = pd.Series([], index=_mi, dtype=object)
    else:
        with open(results_metadata_path, "rb") as f:
            results_metadata = pickle.load(f)

    new_task_mi = pd.MultiIndex.from_tuples(
        (tuple(all_kwargs.values()),), names=tuple(all_kwargs.keys())
    )
    print(f"currently {len(results_metadata)} results fouund in {results_metadata_path}")

    try:
        cached_results = results_metadata.loc[new_task_mi].values
        assert len(cached_results) == 1
        print("task already done, returning early")
        return
    except KeyError:
        pass

    print(
        f"results not already in disk, running {func.__name__}({', '.join([f'{k}={v}' for k, v in kwargs.items()])})"
    )

    ret = func(**kwargs)

    print("saving results..")
    # save results
    filename = f"{hash(time.time())}.pkl"

    mi = pd.MultiIndex.from_tuples(
        (tuple(all_kwargs.values()),), names=tuple(all_kwargs.keys())
    )
    s = pd.Series([filename], mi)

    # Re-load the results metadata as experiments take a while, and new entries may have been written to:
    if not results_metadata_path.exists():
        _mi = pd.MultiIndex.from_tuples(tuple(), names=tuple(all_kwargs.keys()))
        results_metadata = pd.Series([], index=_mi, dtype=object)
    else:
        with open(results_metadata_path, "rb") as f:
            results_metadata = pickle.load(f)

    # Append new results
    new_results_metadata = results_metadata.append(s)

    with open(results_metadata_path, "wb") as f:
        cloudpickle.dump(new_results_metadata, f)


    this_job_slurm_metadata = SLURMJobMetadata(
        sbatch_script_filepath=sbatch_script_filepath,
        slurm_jobid=os.environ.get("SLURM_JOBID", None),
        ran_in_slurm=os.environ.get("SLURM_JOBID", None) is not None,
    )

    with open(f"{results_dir}/{filename}", "wb") as f:
        cloudpickle.dump((all_kwargs, ret, this_job_slurm_metadata), f)

    print("done, exiting")


def _get_default_sbatch_directives() -> dict:
    cwd = os.getcwd()
    return {
        "ntasks": "1",
        "cpus-per-task": "4",
        "partition": "gpu",
        "gres": "gpu:1",
        "mem": "32G",
        "exclude": "gpu-350-01,gpu-350-02,gpu-350-03,gpu-350-04,gpu-350-05,gpu-380-10,gpu-380-11,gpu-380-12,gpu-380-13,gpu-380-14",  # slow nodes
        "output": f"{cwd}/slurm-logs/%x-%j.out",
        "error": f"{cwd}/slurm-logs/%x-%j.err",
    }


def run_maybe_remotely(
    func: Callable,
    folder_name: str,
    experience_name: str,
    use_slurm: bool = True,
    slurm_kwargs: Optional[Dict] = None,
    **kwargs,
):

    import os
    folder_abspath = Path("results").resolve() / folder_name
    folder_abspath.mkdir(exist_ok=True, parents=True)
    (Path(os.getcwd()) / "slurm-logs").mkdir(exist_ok=True)

    if not use_slurm:
        run_and_save(
            func,
            kwargs=kwargs,
            folder_path=folder_abspath,
            experience_name=experience_name,
        )

    sbatch_directives = _get_default_sbatch_directives()
    if slurm_kwargs is not None:
        sbatch_directives.update(slurm_kwargs)
        if sbatch_directives["partition"] != "gpu":
            sbatch_directives.pop("gres")


    tmp_dir = Path(f"/ceph/scratch/{os.environ['USER']}/tmp")
    tmp_dir.mkdir(exist_ok=True)

    sbatch_script_filename = f'{func.__name__}-{experience_name}-{datetime.datetime.utcnow().strftime("%H%M%S%f")}.sbatch'
    func_and_args_filename = f'{func.__name__}-{experience_name}-{datetime.datetime.utcnow().strftime("%H%M%S%f")}_args.pkl'

    sbatch_script_filepath = tmp_dir / sbatch_script_filename
    func_and_args_filepath = tmp_dir / func_and_args_filename

    srun_cmd = [
        str(Path(__file__)),
        "--func_and_args_path",
        str(func_and_args_filepath),
    ]

    with open(func_and_args_filepath, "wb") as f:
        pickle.dump(
            (
                func,
                {
                    **kwargs,
                    "folder_name": folder_abspath,
                    "experience_name": experience_name,
                    "sbatch_script_filepath": sbatch_script_filepath,
                },
            ),
            f,
        )

    sbatch_content = [
        f"#!/bin/bash\n",
        *[f"#SBATCH --{k}={v}\n" for k, v in sbatch_directives.items()],
        "which python\n",
        "hostname\n",
        "echo $XLA_FLAGS\n" "echo 'START TIME'\n",
        "date\n",
        "echo \n",
        "echo \n",
        "echo \n",
        f"echo running {' '.join(srun_cmd)}\n",
        f"{' '.join(srun_cmd)}\n",
        "echo 1\n",
        "echo 'END TIME'\n",
        "date\n",
        "echo \n",
    ]
    with open(sbatch_script_filepath, "w") as f:
        f.writelines(sbatch_content)

    p = subprocess.Popen(
        # ["srun", "-p", "gpu", "--gres", "gpu:1", str(cli_path), *processed_args], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ["sbatch", str(sbatch_script_filepath)],
    )
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--func_and_args_path", required=True)

    args = vars(parser.parse_args()).copy()

    func_and_args_path = args.pop("func_and_args_path")
    with open(func_and_args_path, "rb") as f:

        func, kwargs = pickle.load(f)

    experience_name = kwargs.pop("experience_name")
    folder_name = kwargs.pop("folder_name")

    sbatch_script_filepath = kwargs.pop("sbatch_script_filepath")
    run_and_save(
        func,
        kwargs,
        experience_name=experience_name,
        folder_path=folder_name,
        sbatch_script_filepath=sbatch_script_filepath,
    )
