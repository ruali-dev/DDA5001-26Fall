"""Privately back up one completed six-run PC40 session to the Hugging Face Hub."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
REPO_ID = "ruali-dev/DDA5001-p1-checkpoints"
EXPECTED = {
    "adamw-default": "adamw", "adamw-half": "adamw",
    "muon-default": "muon", "muon-half": "muon",
    "sgdm-default": "sgdm", "sgdm-half": "sgdm",
}
ROOT_FILES = {"results.zip", "comparison.json", "comparison_tokens_vs_tokens.png",
              "comparison_tokens_vs_train_seconds.png"}
RUN_FILES = {"config.json", "environment.json", "metrics.csv", "summary.json",
             "run.log", "samples.txt", "loss_vs_tokens.png", "loss_vs_train_seconds.png",
             "last.pt", "best.pt"}


class BackupError(RuntimeError):
    pass


def select_session(run_dir):
    if run_dir:
        session = Path(run_dir).expanduser().resolve()
        if not session.is_relative_to(RUNS.resolve()):
            raise BackupError("--run-dir must select a session under p1/runs")
    else:
        candidates = sorted((p for p in RUNS.glob("pc40-*") if p.is_dir() and not p.is_symlink()),
                            key=lambda p: p.name, reverse=True)
        if not candidates:
            raise BackupError("No p1/runs/pc40-* session was found")
        session = candidates[0]
    if session.is_symlink() or not re.fullmatch(r"pc40-\d{8}T\d{12}Z", session.name):
        raise BackupError("Selected path is not a normal pc40-timestamp session directory")
    if not session.is_dir():
        raise BackupError(f"Selected session does not exist: {session}")
    return session


def validate_session(session):
    for current, dirs, files in os.walk(session, followlinks=False):
        here = Path(current)
        if any((here / name).is_symlink() for name in dirs + files):
            raise BackupError("Session contains a symlink; refusing to upload it")
        if any(name.endswith(".tmp") or name == ".tmp" for name in files + dirs):
            raise BackupError("Session contains a temporary .tmp file or directory")

    root_entries = {p.name for p in session.iterdir()}
    if root_entries != set(EXPECTED) | ROOT_FILES:
        missing = sorted((set(EXPECTED) | ROOT_FILES) - root_entries)
        extra = sorted(root_entries - (set(EXPECTED) | ROOT_FILES))
        raise BackupError(f"Session contents do not match one complete six-run export (missing={missing}, extra={extra})")

    selected = [session / name for name in sorted(ROOT_FILES)]
    for run_name, optimizer in EXPECTED.items():
        folder = session / run_name
        if not folder.is_dir() or folder.is_symlink():
            raise BackupError(f"Missing expected run folder: {run_name}")
        names = {p.name for p in folder.iterdir()}
        required = {"config.json", "environment.json", "metrics.csv", "summary.json",
                    "run.log", "loss_vs_tokens.png", "loss_vs_train_seconds.png", "last.pt"}
        if not required <= names or names - RUN_FILES:
            raise BackupError(f"Run {run_name} has missing or unexpected files")
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        json.loads((folder / "environment.json").read_text(encoding="utf-8"))
        if summary.get("status") != "completed" or config.get("optimizer_name") != optimizer:
            raise BackupError(f"Run {run_name} is incomplete or has the wrong optimizer")
        if (folder / "last.pt").stat().st_size == 0:
            raise BackupError(f"Run {run_name} has an empty endpoint checkpoint")
        selected.extend(folder / name for name in sorted(names))

    with (session / "comparison.json").open(encoding="utf-8") as stream:
        comparison = json.load(stream)
    if set(comparison.get("images", [])) != {"comparison_tokens_vs_tokens.png", "comparison_tokens_vs_train_seconds.png"}:
        raise BackupError("Comparison report does not name both expected aggregate plots")
    archive = session / "results.zip"
    if archive.stat().st_size == 0:
        raise BackupError("results.zip is empty")
    try:
        with zipfile.ZipFile(archive) as bundle:
            if not bundle.namelist() or bundle.testzip() is not None:
                raise BackupError("results.zip is empty or failed its integrity check")
    except (OSError, zipfile.BadZipFile) as exc:
        raise BackupError("results.zip is not a valid archive") from exc
    if not all(path.is_file() and not path.is_symlink() for path in selected):
        raise BackupError("An upload candidate is missing or is not a regular file")
    return sorted(selected, key=lambda p: p.relative_to(session).as_posix())


def local_manifest(files, session):
    entries = []
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        sha256 = hashlib.sha256()
        git_sha1 = hashlib.sha1()
        if size < 10_000_000:
            git_sha1.update(f"blob {size}\0".encode("ascii"))
        else:
            git_sha1 = None
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                sha256.update(chunk)
                if git_sha1 is not None:
                    git_sha1.update(chunk)
        entry = {"path": path.relative_to(session).as_posix(), "size": size,
                 "sha256": sha256.hexdigest()}
        if git_sha1 is not None:
            entry["git_blob_sha1"] = git_sha1.hexdigest()
        entries.append(entry)
    return {"session": session.name, "repo_id": REPO_ID, "files": entries}, total


def remote_verify(api, session_name, entries):
    remote_paths = [f"{session_name}/{item['path']}" for item in entries]
    remote = api.get_paths_info(REPO_ID, remote_paths, repo_type="model")
    by_path = {item.path: item for item in remote}
    missing = sorted(set(remote_paths) - set(by_path))
    if missing:
        raise BackupError(f"Hub verification is missing {len(missing)} uploaded files")
    for item in entries:
        path = f"{session_name}/{item['path']}"
        info = by_path[path]
        if info.size != item["size"]:
            raise BackupError(f"Remote size verification failed for {item['path']}")
        if info.lfs is not None:
            remote_sha256 = info.lfs.sha256 if not isinstance(info.lfs, dict) else info.lfs.get("sha256")
            if remote_sha256 != item["sha256"]:
                raise BackupError(f"Remote LFS SHA-256 verification failed for {item['path']}")
        elif item.get("git_blob_sha1") != info.blob_id:
            raise BackupError(f"Remote Git blob verification failed for {item['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="explicit pc40 session folder (default: newest pc40-* session)")
    parser.add_argument("--dry-run", action="store_true", help="validate and list files without authentication or hashing")
    args = parser.parse_args()
    session = select_session(args.run_dir)
    files = validate_session(session)
    if args.dry_run:
        total = sum(path.stat().st_size for path in files)
        print(f"Dry run passed: {session.name}; {len(files)} files, {total / (1024 ** 3):.2f} GiB")
        for path in files:
            print(f"{path.relative_to(session).as_posix()}\t{path.stat().st_size}")
        return 0
    total = sum(path.stat().st_size for path in files)
    print(f"Validated {session.name}: {len(files)} files, {total / (1024 ** 3):.2f} GiB", flush=True)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise BackupError("huggingface_hub is missing; run p1/backup_hf.sh to prepare the isolated backup environment") from exc
    if not sys.stdin.isatty():
        raise BackupError("Run this command in the PC40 terminal so token input can be hidden")
    token = getpass.getpass("Hugging Face WRITE token (hidden input): ").strip()
    if not token:
        raise BackupError("No token was entered")
    api = HfApi(token=token)
    identity = api.whoami()
    if identity.get("name") != "ruali-dev":
        raise BackupError("The entered token does not belong to ruali-dev")
    api.create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(REPO_ID, repo_type="model")
    if info.private is not True:
        raise BackupError("Target repository is not private; no files were uploaded")

    session_name = session.name
    zip_path = session / "results.zip"
    print(f"Uploading archive first to https://huggingface.co/{REPO_ID}/tree/main/{session_name}", flush=True)
    api.upload_file(path_or_fileobj=str(zip_path), path_in_repo=f"{session_name}/results.zip",
                    repo_id=REPO_ID, repo_type="model", commit_message=f"Back up {session_name} archive")
    archive_manifest, _ = local_manifest([zip_path], session)
    remote_verify(api, session_name, archive_manifest["files"])
    print("Archive uploaded and verified. Now hashing checkpoints for integrity verification.", flush=True)
    manifest, _ = local_manifest(files, session)
    allowed = [path.relative_to(session).as_posix() for path in files]
    print(f"Uploading {len(files)} allowlisted files; Hub transfer uses Xet/resume where available.", flush=True)
    api.upload_folder(folder_path=str(session), path_in_repo=session_name, repo_id=REPO_ID,
                      repo_type="model", allow_patterns=allowed,
                      commit_message=f"Back up {session_name} experiment files")

    cache = ROOT / ".cache" / "hf-backup"
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / f"{session_name}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_entry = {"path": "manifest.json", "size": manifest_path.stat().st_size,
                      "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      "git_blob_sha1": hashlib.sha1(b"blob " + str(manifest_path.stat().st_size).encode() + b"\0" + manifest_path.read_bytes()).hexdigest()}
    print("Uploading local SHA-256/size manifest and verifying all remote file metadata.", flush=True)
    api.upload_file(path_or_fileobj=str(manifest_path), path_in_repo=f"{session_name}/manifest.json",
                    repo_id=REPO_ID, repo_type="model", commit_message=f"Add {session_name} integrity manifest")
    remote_verify(api, session_name, manifest["files"] + [manifest_entry])
    print(f"Backup verified: https://huggingface.co/{REPO_ID}/tree/main/{session_name}\nKeep local checkpoints until TX3 verification.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BackupError as exc:
        print(f"Backup stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f", HTTP {status}" if status is not None else ""
        print(f"Backup stopped ({type(exc).__name__}{detail}); no successful full-backup verification was reported. Keep the local files and rerun to resume.", file=sys.stderr)
        raise SystemExit(1)
