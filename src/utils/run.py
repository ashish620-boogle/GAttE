from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import psutil


def get_env_info() -> Dict[str, Any]:
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    try:
        import tensorflow as tf

        info["tensorflow_version"] = tf.__version__
    except Exception:
        info["tensorflow_version"] = None
    return info


def create_run_dir(outputs_dir: str | Path, tag: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{tag}" if tag else ts
    run_dir = Path(outputs_dir) / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_env_info(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    info = get_env_info()
    (run_dir / "env.json").write_text(json.dumps(info, indent=2), encoding="utf-8")