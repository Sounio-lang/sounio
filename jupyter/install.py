#!/usr/bin/env python3
"""
Sounio Jupyter Kernel Installer

This script installs the Sounio Jupyter kernel so it can be used with
Jupyter Notebook, JupyterLab, or any other Jupyter-compatible frontend.

Usage:
    python install.py [--user] [--sys-prefix] [--prefix PREFIX]

Options:
    --user          Install to the per-user kernel directory
    --sys-prefix    Install to sys.prefix (for virtualenvs)
    --prefix PREFIX Install to the specified prefix

After installation, start Jupyter and select "Sounio" from the kernel list.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_kernel_dir(user=False, sys_prefix=False, prefix=None):
    """Get the kernel installation directory."""
    if prefix:
        return Path(prefix) / "share" / "jupyter" / "kernels" / "sounio"
    elif sys_prefix:
        return Path(sys.prefix) / "share" / "jupyter" / "kernels" / "sounio"
    elif user:
        # User kernel directory
        if sys.platform == "win32":
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Jupyter"
        else:
            base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "jupyter"
        return base / "kernels" / "sounio"
    else:
        # Try jupyter-client to get the default location
        try:
            from jupyter_client.kernelspec import KernelSpecManager
            ksm = KernelSpecManager()
            # For system-wide installation, use the first writable path
            for path in ksm.kernel_dirs:
                if os.access(path, os.W_OK):
                    return Path(path) / "sounio"
            # Fall back to user directory
            return get_kernel_dir(user=True)
        except ImportError:
            # jupyter-client not installed, fall back to user
            return get_kernel_dir(user=True)


def find_kernel_binary():
    """Find the sounio-jupyter binary."""
    # Check if it's in PATH
    which = shutil.which("sounio-jupyter")
    if which:
        return Path(which).resolve()

    # Check in common build locations relative to this script
    script_dir = Path(__file__).parent.resolve()

    # Check target/debug and target/release
    for build_type in ["release", "debug"]:
        binary = script_dir / "target" / build_type / "sounio-jupyter"
        if binary.exists():
            return binary
        # Also check parent directories
        binary = script_dir.parent / "target" / build_type / "sounio-jupyter"
        if binary.exists():
            return binary

    # Check workspace root target
    workspace_root = script_dir.parent
    for build_type in ["release", "debug"]:
        binary = workspace_root / "target" / build_type / "sounio-jupyter"
        if binary.exists():
            return binary

    return None


def build_kernel():
    """Build the kernel binary."""
    script_dir = Path(__file__).parent.resolve()

    print("Building sounio-jupyter kernel...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=script_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Build failed:")
        print(result.stderr)
        return False

    print("Build successful!")
    return True


def install_kernel(user=False, sys_prefix=False, prefix=None, binary_path=None):
    """Install the Sounio Jupyter kernel."""
    kernel_dir = get_kernel_dir(user, sys_prefix, prefix)

    print(f"Installing Sounio kernel to: {kernel_dir}")

    # Create kernel directory
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # Determine binary path
    if binary_path:
        binary = Path(binary_path).resolve()
    else:
        binary = find_kernel_binary()
        if not binary:
            print("sounio-jupyter binary not found. Building...")
            if not build_kernel():
                print("Failed to build kernel. Please build manually with:")
                print("  cd jupyter && cargo build --release")
                sys.exit(1)
            binary = find_kernel_binary()
            if not binary:
                print("Binary still not found after build. Please check the build output.")
                sys.exit(1)

    if not binary.exists():
        print(f"Binary not found at: {binary}")
        sys.exit(1)

    print(f"Using binary: {binary}")

    # Create kernel.json
    kernel_spec = {
        "argv": [
            str(binary),
            "{connection_file}"
        ],
        "display_name": "Sounio",
        "language": "sounio",
        "metadata": {
            "debugger": False
        },
        "interrupt_mode": "signal",
        "env": {}
    }

    kernel_json_path = kernel_dir / "kernel.json"
    with open(kernel_json_path, "w") as f:
        json.dump(kernel_spec, f, indent=4)

    print(f"Wrote kernel spec to: {kernel_json_path}")

    # Copy logo files if they exist
    script_dir = Path(__file__).parent.resolve()
    for logo in ["logo-32x32.png", "logo-64x64.png"]:
        logo_src = script_dir / logo
        if logo_src.exists():
            shutil.copy(logo_src, kernel_dir / logo)
            print(f"Copied {logo}")

    print()
    print("Sounio Jupyter kernel installed successfully!")
    print()
    print("To use the kernel:")
    print("  1. Start Jupyter: jupyter notebook  or  jupyter lab")
    print("  2. Create a new notebook and select 'Sounio' as the kernel")
    print()
    print("Example Sounio code to try:")
    print("  let x = 42")
    print("  let y = x * 2")
    print("  y + 10")
    print()


def uninstall_kernel(user=False, sys_prefix=False, prefix=None):
    """Uninstall the Sounio Jupyter kernel."""
    kernel_dir = get_kernel_dir(user, sys_prefix, prefix)

    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
        print(f"Removed kernel from: {kernel_dir}")
    else:
        print(f"Kernel not found at: {kernel_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Install the Sounio Jupyter kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Install for current user (default)
    python install.py --user

    # Install system-wide (requires admin/root)
    python install.py

    # Install to a virtualenv
    python install.py --sys-prefix

    # Install to a specific location
    python install.py --prefix /opt/jupyter

    # Uninstall
    python install.py --uninstall
        """
    )

    parser.add_argument(
        "--user",
        action="store_true",
        help="Install to the per-user kernel directory"
    )
    parser.add_argument(
        "--sys-prefix",
        action="store_true",
        help="Install to sys.prefix (for virtualenvs)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Install to the specified prefix"
    )
    parser.add_argument(
        "--binary",
        type=str,
        help="Path to the sounio-jupyter binary"
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Uninstall the kernel"
    )

    args = parser.parse_args()

    # Check for conflicting options
    install_options = [args.user, args.sys_prefix, args.prefix is not None]
    if sum(install_options) > 1:
        print("Error: --user, --sys-prefix, and --prefix are mutually exclusive")
        sys.exit(1)

    if args.uninstall:
        uninstall_kernel(args.user, args.sys_prefix, args.prefix)
    else:
        # Default to user install if no option specified
        if not any(install_options):
            args.user = True
        install_kernel(args.user, args.sys_prefix, args.prefix, args.binary)


if __name__ == "__main__":
    main()
