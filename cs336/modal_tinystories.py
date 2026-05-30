from __future__ import annotations

import shlex
from datetime import UTC, datetime
from pathlib import Path

import modal


APP_NAME = "cs336-tinystories"
VOLUME_NAME = "cs336-tinystories"
REMOTE_ROOT = Path("/vol")
REMOTE_DATA_DIR = REMOTE_ROOT / "data"
REMOTE_EXPERIMENTS_DIR = REMOTE_ROOT / "cs336" / "experiments"
VOLUME_EXPERIMENTS_DIR = Path("/cs336") / "experiments"
REMOTE_SOURCE_ROOT = Path("/root")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch~=2.11.0",
        "numpy>=2.4",
        "einops>=0.8",
        "jaxtyping>=0.3",
        "regex>=2026.3.32",
        "tiktoken>=0.12.0",
        "psutil>=7",
        "tqdm>=4.67",
    )
    .add_local_dir(
        "cs336/ass1/assignment1-basics/cs336_basics",
        remote_path=str(REMOTE_SOURCE_ROOT / "cs336_basics"),
        copy=True,
    )
)

app = modal.App(APP_NAME)


def _check_remote_environment() -> dict[str, object]:
    import os
    import subprocess

    import numpy as np
    import torch

    print(f"cwd={Path.cwd()}")
    print(f"python={os.sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count={torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"cuda_device[{idx}]={torch.cuda.get_device_name(idx)}")

    subprocess.run(["nvidia-smi"], check=False)

    data_summary: dict[str, object] = {}
    for name in ("tinystories_train_ids.npy", "tinystories_valid_ids.npy"):
        path = REMOTE_DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing uploaded data file: {path}")
        arr = np.load(path, mmap_mode="r")
        data_summary[name] = {"shape": arr.shape, "dtype": str(arr.dtype)}
        print(f"{path}: shape={arr.shape}, dtype={arr.dtype}")

    probe_dir = REMOTE_ROOT / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    marker = probe_dir / "last_probe.txt"
    marker.write_text(
        f"ok {datetime.now(UTC).isoformat()}\n"
        f"torch={torch.__version__}\n"
        f"cuda_available={torch.cuda.is_available()}\n",
        encoding="utf-8",
    )
    volume.commit()
    print(f"wrote {marker}")
    return data_summary


def _run_training(config: dict[str, object]) -> str:
    import csv
    import os
    import subprocess
    import sys
    import threading
    import time

    experiment_name = str(config["experiment_name"])
    run_dir = REMOTE_EXPERIMENTS_DIR / experiment_name
    volume_run_dir = VOLUME_EXPERIMENTS_DIR / experiment_name
    checkpoint_out = run_dir / "checkpoint.pt"

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_SOURCE_ROOT}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "cs336_basics.run_training",
        "--experiment-name",
        experiment_name,
        "--random-seed",
        str(config["random_seed"]),
        "--train-data",
        str(REMOTE_DATA_DIR / "tinystories_train_ids.npy"),
        "--valid-data",
        str(REMOTE_DATA_DIR / "tinystories_valid_ids.npy"),
        "--vocab-size",
        str(config["vocab_size"]),
        "--context-length",
        str(config["context_length"]),
        "--d-model",
        str(config["d_model"]),
        "--num-layers",
        str(config["num_layers"]),
        "--num-heads",
        str(config["num_heads"]),
        "--d-ff",
        str(config["d_ff"]),
        "--rope-theta",
        str(config["rope_theta"]),
        "--lr",
        str(config["lr"]),
        "--min-lr",
        str(config["min_lr"]),
        "--batch-size",
        str(config["batch_size"]),
        "--num-iters",
        str(config["num_iters"]),
        "--warmup-iters",
        str(config["warmup_iters"]),
        "--eval-interval",
        str(config["eval_interval"]),
        "--save-interval",
        str(config["save_interval"]),
        "--eval-iters",
        str(config["eval_iters"]),
        "--max-grad-norm",
        str(config["max_grad_norm"]),
        "--device",
        "cuda",
        "--checkpoint-out",
        str(checkpoint_out),
    ]

    print("running:")
    print(" ".join(shlex.quote(part) for part in cmd))
    REMOTE_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    telemetry_interval = float(config["telemetry_interval"])
    stop_telemetry = threading.Event()
    telemetry_thread = None

    def telemetry_loop() -> None:
        query_fields = [
            "timestamp",
            "index",
            "name",
            "utilization.gpu",
            "utilization.memory",
            "memory.used",
            "memory.total",
            "power.draw",
            "temperature.gpu",
        ]
        header = [
            "elapsed_seconds",
            "timestamp",
            "index",
            "name",
            "gpu_util_percent",
            "mem_util_percent",
            "mem_used_mib",
            "mem_total_mib",
            "power_w",
            "temp_c",
        ]
        gpu_csv = run_dir / "gpu.csv"
        start = time.perf_counter()
        while not run_dir.exists() and not stop_telemetry.is_set():
            time.sleep(0.05)
        if stop_telemetry.is_set():
            return
        with gpu_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            while not stop_telemetry.is_set():
                sample = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={','.join(query_fields)}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                elapsed = time.perf_counter() - start
                for line in sample.stdout.splitlines():
                    values = [part.strip() for part in line.split(",")]
                    writer.writerow([f"{elapsed:.6f}", *values])
                f.flush()
                stop_telemetry.wait(telemetry_interval)

    if telemetry_interval > 0:
        telemetry_thread = threading.Thread(target=telemetry_loop, name="gpu-telemetry", daemon=True)
        telemetry_thread.start()

    try:
        subprocess.run(cmd, cwd=str(REMOTE_ROOT), env=env, check=True)
    finally:
        stop_telemetry.set()
        if telemetry_thread is not None:
            telemetry_thread.join(timeout=max(2.0, telemetry_interval + 1.0))

    volume.commit()
    print(f"committed run artifacts under {run_dir}")
    return str(volume_run_dir)


@app.function(image=image, volumes={str(REMOTE_ROOT): volume}, gpu="L4", timeout=20 * 60)
def probe_l4() -> dict[str, object]:
    return _check_remote_environment()


@app.function(image=image, volumes={str(REMOTE_ROOT): volume}, gpu="B200", timeout=20 * 60)
def probe_b200() -> dict[str, object]:
    return _check_remote_environment()


@app.function(image=image, volumes={str(REMOTE_ROOT): volume}, gpu="L4", timeout=6 * 60 * 60)
def train_l4(config: dict[str, object]) -> str:
    return _run_training(config)


@app.function(image=image, volumes={str(REMOTE_ROOT): volume}, gpu="B200", timeout=12 * 60 * 60)
def train_b200(config: dict[str, object]) -> str:
    return _run_training(config)


def _default_experiment_name(mode: str, gpu: str, lr: float, num_iters: int, batch_size: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr_tag = f"{lr:.2e}".replace("+", "").replace(".", "p")
    return f"modal_{mode}_{gpu}_lr_{lr_tag}_bs{batch_size}_it{num_iters}_{ts}"


def _training_config(
    experiment_name: str,
    random_seed: int,
    lr: float,
    min_lr: float,
    batch_size: int,
    num_iters: int,
    warmup_iters: int,
    eval_interval: int,
    save_interval: int,
    eval_iters: int,
    telemetry_interval: float,
) -> dict[str, object]:
    return {
        "experiment_name": experiment_name,
        "random_seed": random_seed,
        "vocab_size": 10000,
        "context_length": 256,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10000.0,
        "lr": lr,
        "min_lr": min_lr,
        "batch_size": batch_size,
        "num_iters": num_iters,
        "warmup_iters": warmup_iters,
        "eval_interval": eval_interval,
        "save_interval": save_interval,
        "eval_iters": eval_iters,
        "max_grad_norm": 1.0,
        "telemetry_interval": telemetry_interval,
    }


@app.local_entrypoint()
def main(
    mode: str = "probe",
    gpu: str = "l4",
    experiment_name: str = "",
    random_seed: int = 42,
    lr: float = 1.25e-3,
    min_lr: float = 1.25e-4,
    batch_size: int = 16,
    num_iters: int = 20,
    warmup_iters: int = 2,
    eval_interval: int = 10,
    save_interval: int = 20,
    eval_iters: int = 2,
    telemetry_interval: float = 1.0,
) -> None:
    gpu_key = gpu.lower()
    if gpu_key not in {"l4", "b200"}:
        raise ValueError("gpu must be 'l4' or 'b200'")

    if mode == "probe":
        result = probe_b200.remote() if gpu_key == "b200" else probe_l4.remote()
        print(result)
        return

    if mode not in {"smoke_train", "train"}:
        raise ValueError("mode must be 'probe', 'smoke_train', or 'train'")

    if not experiment_name:
        experiment_name = _default_experiment_name(mode, gpu_key, lr, num_iters, batch_size)

    config = _training_config(
        experiment_name=experiment_name,
        random_seed=random_seed,
        lr=lr,
        min_lr=min_lr,
        batch_size=batch_size,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        eval_interval=eval_interval,
        save_interval=save_interval,
        eval_iters=eval_iters,
        telemetry_interval=telemetry_interval,
    )
    remote_path = train_b200.remote(config) if gpu_key == "b200" else train_l4.remote(config)
    print(f"remote run path: {remote_path}")
    print(f"download with: python -m modal volume get {VOLUME_NAME} {remote_path} cs336/experiments/{experiment_name}")
