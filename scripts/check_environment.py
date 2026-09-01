#!/usr/bin/env python3
"""Report prerequisites without importing or launching Isaac Sim."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys


def main() -> int:
    libc_name, libc_version = platform.libc_ver()
    libc_tuple = tuple(int(part) for part in libc_version.split(".")[:2])
    python_ok = sys.version_info[:2] == (3, 12)
    libc_ok = libc_name == "glibc" and libc_tuple >= (2, 35)
    isaac_installed = importlib.util.find_spec("isaaclab") is not None
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("libc:", libc_name, libc_version)
    print("Isaac Lab installed:", isaac_installed)
    print("uv installed:", shutil.which("uv") is not None)
    print("Docker installed:", shutil.which("docker") is not None)
    if shutil.which("nvidia-smi"):
        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
        print("GPU(s):")
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
        print((result.stdout or result.stderr).strip())
    if not python_ok:
        print("ACTION: run Isaac Lab in a dedicated Python 3.12 environment.")
    if not libc_ok:
        print("ACTION: use Ubuntu 22.04+ or an official Isaac Lab GPU container.")
    if not isaac_installed:
        print("ACTION: install Isaac Lab after the host/container prerequisites pass.")
    return 0 if python_ok and libc_ok and isaac_installed else 1


if __name__ == "__main__":
    raise SystemExit(main())
