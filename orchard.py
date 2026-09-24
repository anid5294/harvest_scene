"""Run the pinned OrchardBench environment beside this repository."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
WORK_ROOT = ROOT.parent
DEFAULT_ORCHARD_ROOT = WORK_ROOT / "orchardbench"
ORCHARDBENCH_COMMIT = "6313313db8b1a7d23fb2cc3afd67cac46f29399a"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def verify_orchardbench(repo: Path) -> dict:
    if not (repo / "scripts/grow_tree.py").is_file():
        raise SystemExit(f"OrchardBench not found at {repo}")
    actual = git(repo, "rev-parse", "HEAD")
    dirty = bool(git(repo, "status", "--short"))
    if actual != ORCHARDBENCH_COMMIT:
        raise SystemExit(
            f"OrchardBench revision mismatch: expected {ORCHARDBENCH_COMMIT}, got {actual}"
        )
    if dirty:
        raise SystemExit("OrchardBench checkout is dirty; commit or discard its changes")
    return {"expected_commit": ORCHARDBENCH_COMMIT, "actual_commit": actual, "clean": True}


def contained_environment() -> dict[str, str]:
    env = os.environ.copy()
    paths = {
        "PIXI_HOME": WORK_ROOT / ".pixi-home",
        "PIXI_CACHE_DIR": WORK_ROOT / ".pixi-cache",
        "XDG_CACHE_HOME": WORK_ROOT / ".cache",
        "XDG_CONFIG_HOME": WORK_ROOT / ".config",
        "XDG_DATA_HOME": WORK_ROOT / ".local/share",
        "WARP_CACHE_PATH": WORK_ROOT / ".cache/warp",
        "CUDA_CACHE_PATH": WORK_ROOT / ".cache/cuda",
        "TMPDIR": WORK_ROOT / "tmp",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env["PIXI_NO_CONFIG"] = "1"
    return env


def pixi_binary() -> str:
    local = WORK_ROOT / ".tools/bin/pixi"
    if local.is_file():
        return str(local)
    found = shutil.which("pixi")
    if found:
        return found
    raise SystemExit("pixi not found; expected it at ../.tools/bin/pixi")


def command_for(args: argparse.Namespace) -> list[str]:
    if args.command == "g1":
        command = [
            pixi_binary(), "run", "python", str(ROOT / "g1_orchard.py"),
            "--orchard-root", str(args.orchard_root.resolve()),
            "--seed", str(args.seed), "--frames", str(args.frames),
            "--output", str(args.output.resolve()),
        ]
        if args.usd_only:
            command.append("--usd-only")
        return command
    script = [pixi_binary(), "run", "python", "scripts/grow_tree.py", "--seed", str(args.seed)]
    if args.command == "tree":
        return script + ["--apples", "--break", "--viewer", "null", "--frames", str(args.frames)]
    if args.command == "view":
        return script + ["--apples", "--foliage", "--break", "--viewer", "gl"]
    if args.command == "usd":
        return script + ["--apples", "--foliage", "--break", "--viewer", "usd", "--output", str(args.output), "--frames", str(args.frames)]
    if args.command == "harvest":
        return script + ["--auto", "--foliage", "--break", "--viewer", args.viewer, "--frames", str(args.frames), "--metrics", str(args.metrics)]
    raise AssertionError(args.command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchard-root", type=Path, default=DEFAULT_ORCHARD_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    for name in ("tree", "view", "usd", "harvest", "g1"):
        item = sub.add_parser(name)
        item.add_argument("--seed", type=int, default=42)
        item.add_argument("--frames", type=int, default=300)
        if name == "usd":
            item.add_argument("--output", type=Path, default=WORK_ROOT / "artifacts/tree_seed42.usda")
        if name == "harvest":
            item.add_argument("--viewer", choices=("gl", "null"), default="gl")
            item.add_argument("--metrics", type=Path, default=WORK_ROOT / "artifacts/harvest_seed42.json")
        if name == "g1":
            item.add_argument("--output", type=Path, default=ROOT / "artifacts/g1_orchard_seed42.gif")
            item.add_argument(
                "--usd-only",
                action="store_true",
                help="skip OpenGL and write a USD scene directly",
            )
    args = parser.parse_args()
    repo = args.orchard_root.resolve()
    check = verify_orchardbench(repo)
    if args.command == "check":
        print(json.dumps(check, indent=2))
        return 0

    command = command_for(args)
    started = dt.datetime.now(dt.timezone.utc)
    result = subprocess.run(command, cwd=repo, env=contained_environment())
    ended = dt.datetime.now(dt.timezone.utc)
    run_dir = ROOT / "artifacts/runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "exit_code": result.returncode,
        "harvest_scene_commit": git(ROOT, "rev-parse", "HEAD"),
        "harvest_scene_dirty": bool(git(ROOT, "status", "--short")),
        "orchardbench_commit": check["actual_commit"],
        "command": command,
    }
    path = run_dir / f"{started.strftime('%Y%m%dT%H%M%SZ')}_{args.command}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"run manifest: {path}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
