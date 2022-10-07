import os
import subprocess


def show_environment_variables():
    # Make sure jax knows where to look for cuda runtime libraries
    print(f"{os.environ.get('XLA_FLAGS')=}")
    print(f"{os.environ.get('LD_LIBRARY_PATH')=}")
    print(f"{os.environ.get('PATH')=}")
    print(f"{os.environ.get('CUDA_VISIBLE_DEVICES')=}")
    print(f"{os.environ.get('CUDA_VISIBLE_DEVICES')=}")
    print(f"{os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')=}")
    print(f"{os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR')=}")


def test_jax_installation():
    try:
        import jax
        import jax.numpy as jnp
        from jax import random
        from jaxlib.xla_extension import GpuDevice
    except ModuleNotFoundError:
        raise ValueError("jax is not installed")

    # Check access to cuda compiler
    print("test access to a cuda compiler...", end="")
    try:
        subprocess.check_output(["which", "ptxas"])
        # os.system("ptxas --version")
    except subprocess.CalledProcessError as e:
        raise ValueError("No cuda compiler found in $PATH") from e
    print(" OK.")

    # tell which device jax uses
    print("checking if jax can detect a GPU device...", end="")
    assert any(isinstance(d, GpuDevice) for d in jax.local_devices())
    print(" OK.")

    # create a simple jax array
    print("testing array creation on GPU...", end="")
    key = random.PRNGKey(0)
    x = random.normal(key, (10,))
    print(" OK.")

    # Use specialized cuda lib such as linear algebra solvers
    print(
        "testing use of specialized cuda libraries such as linear algebra solvers...",
        end="",
    )
    A = jnp.array([[0, 1], [1, 1], [1, 1], [2, 1]])
    _, _ = jnp.linalg.qr(A)

    A = jnp.eye(10)
    _, _ = jnp.linalg.eigh(A)

    print(" OK.")

    # Use cudnn primitives such as convolutions
    # (cudnn has to be installed separately)
    print("testing use of cudnn primitives...", end="")
    key = random.PRNGKey(0)
    x = jnp.linspace(0, 10, 500)
    y = jnp.sin(x) + 0.2 * random.normal(key, shape=(500,))

    window = jnp.ones(10) / 10
    _ = jnp.convolve(y, window, mode="same")
    print(" OK.")

    print("Test done, everything seems well installed.")


if __name__ == "__main__":
    test_jax_installation()
