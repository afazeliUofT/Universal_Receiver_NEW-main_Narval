from __future__ import annotations

import argparse
import ctypes.util
import importlib
import importlib.util
import json
import os
import platform
import shutil
import site
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SCRATCH_ROOT = Path("/home/rsadve1/scratch")
DEFAULT_PROJECT_ROOT = DEFAULT_SCRATCH_ROOT / "Universal_Receiver_NEW-main_Narval"
DEFAULT_VENV_PATH = DEFAULT_SCRATCH_ROOT / ".venvUPAIR"


def _run(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {
            "cmd": cmd,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _bash(command: str, timeout: int = 30) -> dict[str, Any]:
    return _run(["bash", "-lc", command], timeout=timeout)


def _module_bash(command: str, timeout: int = 30) -> dict[str, Any]:
    init = "source /etc/profile.d/modules.sh 2>/dev/null || true"
    return _bash(f"{init}; {command}", timeout=timeout)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _path_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "is_file": path.is_file(),
    }
    try:
        status["resolved"] = str(path.resolve())
    except Exception as exc:
        status["resolve_error"] = f"{type(exc).__name__}: {exc}"
    try:
        probe_target = path if path.exists() else path.parent
        usage = shutil.disk_usage(probe_target)
        status["disk_usage_bytes"] = {
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
        }
    except Exception as exc:
        status["disk_usage_error"] = f"{type(exc).__name__}: {exc}"
    status["access"] = {
        "read": os.access(path if path.exists() else path.parent, os.R_OK),
        "write": os.access(path if path.exists() else path.parent, os.W_OK),
        "execute": os.access(path if path.exists() else path.parent, os.X_OK),
    }
    return status


def _module_info(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    info: dict[str, Any] = {"available": spec is not None}
    if spec is None:
        return info

    info["origin"] = spec.origin
    try:
        mod = importlib.import_module(module_name)
        info["version"] = getattr(mod, "__version__", None)
        info["file"] = getattr(mod, "__file__", None)
    except Exception as exc:
        info["import_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _attr_probe(module_name: str, attr_names: list[str]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "module": module_name,
        "available": False,
        "attrs": {},
    }
    try:
        mod = importlib.import_module(module_name)
        info["available"] = True
        info["file"] = getattr(mod, "__file__", None)
        for attr_name in attr_names:
            value = getattr(mod, attr_name, None)
            info["attrs"][attr_name] = {
                "available": value is not None,
                "repr": repr(value)[:500] if value is not None else None,
            }
    except Exception as exc:
        info["import_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _tensorflow_runtime_probe() -> dict[str, Any]:
    spec = importlib.util.find_spec("tensorflow")
    if spec is None:
        return {"skipped": "tensorflow not importable in current Python"}
    code = """
import json
import tensorflow as tf

payload = {
    "tf_version": tf.__version__,
    "built_with_cuda": bool(tf.test.is_built_with_cuda()),
    "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
    "cpus": [d.name for d in tf.config.list_physical_devices("CPU")],
    "build_info": tf.sysconfig.get_build_info(),
    "compile_flags": tf.sysconfig.get_compile_flags(),
    "link_flags": tf.sysconfig.get_link_flags(),
}
print(json.dumps(payload, indent=2, sort_keys=True))
"""
    return _run([sys.executable, "-c", code], timeout=90)


def _sionna_runtime_probe() -> dict[str, Any]:
    spec = importlib.util.find_spec("sionna")
    if spec is None:
        return {"skipped": "sionna not importable in current Python"}
    modules = {
        "sionna.phy.nr": ["PUSCHConfig", "PUSCHTransmitter", "PUSCHReceiver", "PUSCHLSChannelEstimator"],
        "sionna.nr": ["PUSCHConfig", "PUSCHTransmitter", "PUSCHReceiver", "PUSCHLSChannelEstimator"],
        "sionna.phy.channel": ["OFDMChannel"],
        "sionna.channel": ["OFDMChannel"],
        "sionna.phy.channel.tr38901": ["Antenna", "AntennaArray", "CDL"],
        "sionna.channel.tr38901": ["Antenna", "AntennaArray", "CDL"],
        "sionna.phy.ofdm": ["LinearDetector"],
        "sionna.ofdm": ["LinearDetector"],
        "sionna.phy.mimo": ["StreamManagement"],
        "sionna.mimo": ["StreamManagement"],
    }
    return {module_name: _attr_probe(module_name, attrs) for module_name, attrs in modules.items()}


def _library_probe() -> dict[str, str | None]:
    return {
        name: ctypes.util.find_library(name)
        for name in [
            "cuda",
            "cudart",
            "cublas",
            "cublasLt",
            "cudnn",
            "cufft",
            "curand",
            "cusolver",
            "cusparse",
            "nccl",
            "stdc++",
        ]
    }


def _read_text(path: Path, max_chars: int = 12000) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        info["size_bytes"] = path.stat().st_size
        info["text"] = text[:max_chars]
        if len(text) > max_chars:
            info["truncated"] = True
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    scratch_root = Path(args.scratch_root)
    project_root = Path(args.project_root)
    venv_path = Path(args.venv_path)
    output_path = Path(args.output)

    interesting_env = [
        "CC_CLUSTER",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_SUBMIT_DIR",
        "CUDA_HOME",
        "CUDA_PATH",
        "EBROOTCUDA",
        "CUDNN_ROOT",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_CONFIG_FILE",
        "XDG_CACHE_HOME",
        "TMPDIR",
    ]

    commands = {
        "date_utc": _bash("date -u +%Y-%m-%dT%H:%M:%SZ"),
        "hostname": _run(["hostname"]),
        "uname": _run(["uname", "-a"]),
        "id": _run(["id"]),
        "pwd": _run(["pwd"]),
        "df_home_scratch": _bash("df -h /home/rsadve1 /home/rsadve1/scratch 2>&1 || true"),
        "quota": _bash("quota -s 2>&1 || true", timeout=60),
        "python_version": _run([sys.executable, "--version"]),
        "python_venv_help": _run([sys.executable, "-m", "venv", "--help"], timeout=30),
        "pip_version": _run([sys.executable, "-m", "pip", "--version"]),
        "pip_debug": _run([sys.executable, "-m", "pip", "debug", "--verbose"], timeout=90),
        "gcc_version": _run(["gcc", "--version"], timeout=30) if _which("gcc") else {"skipped": "gcc not found"},
        "gpp_version": _run(["g++", "--version"], timeout=30) if _which("g++") else {"skipped": "g++ not found"},
        "nvidia_smi": _run(["nvidia-smi"], timeout=45) if _which("nvidia-smi") else {"skipped": "nvidia-smi not found"},
        "nvcc_version": _run(["nvcc", "--version"], timeout=30) if _which("nvcc") else {"skipped": "nvcc not found"},
        "module_version": _module_bash("module --version 2>&1 || true"),
        "module_list": _module_bash("module list 2>&1 || true"),
        "module_avail_stdenv": _module_bash("module avail StdEnv 2>&1 | head -200", timeout=60),
        "module_avail_gcc": _module_bash("module avail gcc 2>&1 | head -250", timeout=60),
        "module_avail_python": _module_bash("module avail python 2>&1 | head -250", timeout=60),
        "module_avail_cuda": _module_bash("module avail cuda 2>&1 | head -250", timeout=60),
        "module_avail_cudnn": _module_bash("module avail cudnn 2>&1 | head -250", timeout=60),
        "module_avail_tensorflow": _module_bash("module avail tensorflow 2>&1 | head -250", timeout=60),
        "module_avail_scipy_stack": _module_bash("module avail scipy-stack 2>&1 | head -250", timeout=60),
        "module_spider_python": _module_bash("module spider python 2>&1 | head -300", timeout=90),
        "module_spider_cuda": _module_bash("module spider cuda 2>&1 | head -300", timeout=90),
        "module_spider_tensorflow": _module_bash("module spider tensorflow 2>&1 | head -300", timeout=90),
        "slurm_sinfo_gpu": _bash("sinfo -N -o '%N %c %m %G %f %t' 2>&1 | head -150 || true", timeout=60),
        "slurm_scontrol_config": _bash("scontrol show config 2>&1 | head -180 || true", timeout=60),
        "tensorflow_runtime_probe": _tensorflow_runtime_probe(),
    }

    payload: dict[str, Any] = {
        "label": args.label,
        "notes": [
            "Generated by probe_narval_environment.py.",
            "Use this to choose Narval-compatible modules and Python package pins for /home/rsadve1/scratch/.venvUPAIR.",
        ],
        "paths": {
            "cwd": str(Path.cwd()),
            "home": str(Path.home()),
            "scratch_root": _path_status(scratch_root),
            "project_root": _path_status(project_root),
            "target_venv": _path_status(venv_path),
            "output": str(output_path),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "site_packages": site.getsitepackages() if hasattr(site, "getsitepackages") else [],
            "user_site": site.getusersitepackages() if hasattr(site, "getusersitepackages") else None,
        },
        "env": {key: os.environ.get(key) for key in interesting_env if os.environ.get(key) is not None},
        "executables": {
            name: _which(name)
            for name in [
                "python",
                "python3",
                "pip",
                "pip3",
                "virtualenv",
                "module",
                "ml",
                "nvidia-smi",
                "nvcc",
                "gcc",
                "g++",
                "cc",
                "c++",
                "mpicc",
                "git",
                "sbatch",
                "srun",
                "salloc",
                "sinfo",
                "squeue",
            ]
        },
        "shared_libraries": _library_probe(),
        "commands": commands,
        "python_modules": {
            name: _module_info(name)
            for name in [
                "numpy",
                "scipy",
                "pandas",
                "matplotlib",
                "yaml",
                "tensorflow",
                "keras",
                "sionna",
                "importlib_resources",
                "packaging",
                "setuptools",
                "wheel",
                "pip",
            ]
        },
        "sionna_required_api": _sionna_runtime_probe(),
        "project_files": {
            "requirements": _read_text(project_root / "requirements.txt"),
            "pyproject": _read_text(project_root / "pyproject.toml"),
            "base_config": _read_text(project_root / "configs" / "twc_comprehensive_mu32_base.yaml"),
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the Narval software stack for UPAIR venv installation.")
    parser.add_argument("--label", default="default", help="Short label for this probe run.")
    parser.add_argument("--scratch-root", default=str(DEFAULT_SCRATCH_ROOT))
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--venv-path", default=str(DEFAULT_VENV_PATH))
    parser.add_argument("--output", default=str(DEFAULT_SCRATCH_ROOT / "narval_env_probe_default.json"))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(json.dumps({
        "label": args.label,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "project_root_exists": payload["paths"]["project_root"]["exists"],
        "target_venv_exists": payload["paths"]["target_venv"]["exists"],
        "tensorflow": payload["python_modules"]["tensorflow"],
        "sionna": payload["python_modules"]["sionna"],
        "nvidia_smi_found": bool(_which("nvidia-smi")),
        "nvcc_found": bool(_which("nvcc")),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
