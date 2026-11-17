"""Create lossless model-only endpoint artifacts for a completed PC40 session."""
import argparse
import json
import os
from pathlib import Path
import sys
import uuid

import torch


RUNS = {
    "adamw-default", "adamw-half", "muon-default", "muon-half",
    "sgdm-default", "sgdm-half",
}
KEEP = ("model", "model_args", "config", "tokenizer", "step", "tokens",
        "train_seconds", "best_loss", "muon_implementation")
REQUIRED = set(KEEP) - {"muon_implementation"}


def same_metadata(left, right):
    return all(left.get(key) == right.get(key)
               for key in ("step", "tokens", "model_args", "config", "artifact_kind"))


def same_model(left, right):
    a, b = left.get("model"), right.get("model")
    if not isinstance(a, dict) or not isinstance(b, dict) or a.keys() != b.keys():
        return False
    for name, tensor in a.items():
        other = b[name]
        if not isinstance(tensor, torch.Tensor) or not isinstance(other, torch.Tensor):
            return False
        if tensor.dtype != other.dtype or tensor.shape != other.shape or not torch.equal(tensor, other):
            return False
    return True


def check_session(session):
    if session.is_symlink():
        raise ValueError("Session path must not be a symlink")
    session = session.resolve()
    if not session.is_dir():
        raise ValueError(f"Session directory does not exist: {session}")
    folders = {p.name for p in session.iterdir() if p.is_dir() and not p.is_symlink()}
    if folders != RUNS:
        raise ValueError(f"Expected exactly six PC40 run folders; found {sorted(folders)}")
    for name in sorted(RUNS):
        folder = session / name
        summary_path = folder / "summary.json"
        checkpoint = folder / "last.pt"
        if (not summary_path.is_file() or summary_path.is_symlink() or not checkpoint.is_file()
                or checkpoint.is_symlink() or checkpoint.stat().st_size == 0):
            raise ValueError(f"{name}: completed summary and nonempty last.pt are required")
        if any(p.name.endswith(".tmp") for p in folder.iterdir()):
            raise ValueError(f"{name}: temporary file exists; wait for training/save to finish")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            raise ValueError(f"{name}: summary status is {summary.get('status')!r}, not completed")
    return session


def compact(session, output):
    session = check_session(session)
    if output.expanduser().is_symlink():
        raise ValueError("Output path must not be a symlink")
    output = output.expanduser().resolve()
    if output == session or output.is_relative_to(session) or session.is_relative_to(output):
        raise ValueError("Output must be a separate directory, outside the source session")
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise ValueError("Output path already exists and is not a normal directory")
    output.mkdir(parents=True, exist_ok=True)

    for name in sorted(RUNS):
        src = session / name / "last.pt"
        run_out = output / name
        if run_out.exists() and (not run_out.is_dir() or run_out.is_symlink()):
            raise ValueError(f"Output path is not a normal run directory: {run_out}")
        run_out.mkdir(exist_ok=True)

        print(f"Loading {name}/last.pt on CPU", flush=True)
        source = torch.load(src, map_location="cpu", weights_only=True)
        missing = REQUIRED - source.keys()
        if missing:
            raise ValueError(f"{name}: source checkpoint is missing fields {sorted(missing)}")
        artifact = {key: source[key] for key in KEEP if key in source}
        artifact["artifact_kind"] = "endpoint-model-only"
        destination = run_out / "last.pt"
        if destination.is_symlink():
            raise ValueError(f"{name}: output last.pt must not be a symlink")
        if destination.exists():
            existing = torch.load(destination, map_location="cpu", weights_only=True)
            if not same_metadata(artifact, existing) or not same_model(artifact, existing):
                raise ValueError(f"{name}: existing output differs from source; refusing to reuse it")
            print(f"Reused verified {name}/last.pt", flush=True)
            continue

        temporary = run_out / f".last.pt.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                torch.save(artifact, stream)
                stream.flush()
                os.fsync(stream.fileno())
            # Hard-linking publishes atomically and cannot overwrite another file.
            os.link(temporary, destination)
            print(f"Wrote {name}/last.pt ({destination.stat().st_size / (1024 ** 3):.2f} GiB)", flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Lossless model-only artifacts are ready: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path, help="completed pc40-* source folder")
    parser.add_argument("--output", required=True, type=Path, help="new/previously validated artifact folder")
    args = parser.parse_args()
    compact(args.session, args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Compaction stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
