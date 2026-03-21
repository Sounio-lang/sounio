"""Sounio Jupyter Kernel - Epistemic computing in notebooks."""

__version__ = "0.1.0"

from sounio_kernel.kernel import SounioKernel

__all__ = ["SounioKernel"]


def kernelspec():
    """Jupyter kernelspec entry point."""
    return {
        "display_name": "Sounio",
        "language": "sounio",
        "argv": [
            "python",
            "-m",
            "sounio_kernel",
            "-f",
            "{connection_file}",
        ],
    }


if __name__ == "__main__":
    from ipykernel.kernelapp import IPKernelApp
    app = IPKernelApp.instance(kernel_class=SounioKernel)
    app.initialize()
    app.start()
