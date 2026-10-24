"""Capture the machine that will run the sample. Values come from the host, not the plan."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip()


def _windows_computer() -> dict:
    script = (
        "$sys = Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory | ConvertTo-Json -Compress; "
        "$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 Name | ConvertTo-Json -Compress; "
        "Write-Output $sys; Write-Output '---'; Write-Output $cpu"
    )
    raw = _run(["powershell", "-NoProfile", "-Command", script])
    if "---" not in raw:
        return {"raw": raw}
    system_raw, cpu_raw = raw.split("---", 1)
    try:
        system = json.loads(system_raw.strip())
        cpu = json.loads(cpu_raw.strip())
    except json.JSONDecodeError:
        return {"raw": raw}
    return {
        "manufacturer": system.get("Manufacturer"),
        "model": system.get("Model"),
        "ram_bytes": system.get("TotalPhysicalMemory"),
        "cpu": cpu.get("Name"),
    }


def _gpu() -> dict:
    raw = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    header = _run(["nvidia-smi"])
    cuda_version = None
    for line in header.splitlines():
        if "CUDA UMD Version:" in line:
            cuda_version = line.split("CUDA UMD Version:", 1)[-1].strip().split()[0]
        elif "CUDA Version:" in line:
            cuda_version = line.split("CUDA Version:", 1)[-1].strip().split()[0]
            break
    if not raw:
        return {"available_to_nvidia_smi": False, "driver_cuda_version": cuda_version}
    parts = [part.strip() for part in raw.split(",")]
    return {
        "available_to_nvidia_smi": True,
        "name": parts[0] if parts else raw,
        "memory_total": parts[1] if len(parts) > 1 else None,
        "driver_version": parts[2] if len(parts) > 2 else None,
        "driver_cuda_version": cuda_version,
    }


def collect() -> dict:
    computer = _windows_computer() if sys.platform.startswith("win") else {}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": computer,
        "gpu": _gpu(),
        "torch": {
            "version": torch.__version__,
            "cuda_compiled": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = collect()
    destination = root / "reports" / "environment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
