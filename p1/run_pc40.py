"""Run the six fixed PC40 experiments sequentially, with checks before training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from checkpoint_io import RunInterrupted, run_monitored


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = (
    ("adamw-default", "config/pc40-adamw-default.py"),
    ("adamw-half", "config/pc40-adamw-half.py"),
    ("muon-default", "config/pc40-muon-default.py"),
    ("muon-half", "config/pc40-muon-half.py"),
    ("sgdm-default", "config/pc40-sgdm-default.py"),
    ("sgdm-half", "config/pc40-sgdm-half.py"),
)


def command(args, *, cwd=ROOT):
    print("+", " ".join(map(str, args)), flush=True)
    result = run_monitored([str(part) for part in args], cwd=cwd)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, args)


def check_environment():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; PC40 experiments require the classroom GPU")
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("Native torch.optim.Muon is unavailable; use the course PyTorch >=2.9 environment")
    print(json.dumps({
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "native_muon": True,
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="run environment/checker/CPU/GPU gates, then stop")
    parser.add_argument("--eager", action="store_true", help="disable torch.compile equally for all six training runs")
    parser.add_argument("--only", choices=("default", "half"), help="run only one learning-rate group")
    args = parser.parse_args()
    os.chdir(ROOT)

    check_environment()
    command([sys.executable, "check_solution.py"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    smoke_dir = ROOT / "checks" / f"pc40-smoke-{stamp}"
    command([sys.executable, "train.py", "config/smoke.py", "--device=cpu", f"--out_dir={smoke_dir}"])
    command([sys.executable, "setup/verify_env.py"])
    if args.check_only:
        print("All preflight checks passed; no full training runs were started.", flush=True)
        return 0

    results = ROOT / "runs" / f"pc40-{stamp}"
    if results.exists():
        raise FileExistsError(f"Results path already exists: {results}")
    results.mkdir(parents=True)
    selected = [(name, config) for name, config in EXPERIMENTS
                if args.only is None or name.endswith(args.only)]
    completed = []
    try:
        for index, (name, config) in enumerate(selected):
            output = results / name
            train_args = [sys.executable, "train.py", config,
                          f"--out_dir={output}", "--device=cuda"]
            if args.eager:
                train_args.append("--compile=False")
            command(train_args)
            completed.append(name)
            command([sys.executable, "plot.py", f"--run-dir={output}"])
            if index == 0 and (output / "last.pt").is_file():
                command([sys.executable, "sample.py", f"--checkpoint={output / 'last.pt'}",
                         "--device=cuda", "--tokens=400"])
        command([sys.executable, "plot.py", f"--runs-dir={results}"])
        archive = results / "results.zip"
        command([sys.executable, "export_results.py", f"--runs-dir={results}", f"--output={archive}"])
    except (KeyboardInterrupt, RunInterrupted):
        print(f"Interrupted. Finished runs remain in {results}: {', '.join(completed) or 'none'}; no complete sweep archive was created.",
              file=sys.stderr, flush=True)
        return 130

    print(f"Completed {len(completed)} run(s). Results: {results}\nSubmission archive: {archive}\nKeep checkpoints separately in private storage.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
