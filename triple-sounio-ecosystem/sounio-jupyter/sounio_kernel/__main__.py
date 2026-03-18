"""Sounio kernel main entry point - enables `python -m sounio_kernel`."""

from ipykernel.kernelapp import IPKernelApp
from sounio_kernel.kernel import SounioKernel

if __name__ == "__main__":
    app = IPKernelApp.instance(kernel_class=SounioKernel)
    app.initialize()
    app.start()
