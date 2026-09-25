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
    if args.command == "g1-harvest":
        xvfb = shutil.which("xvfb-run")
        if not xvfb:
            raise SystemExit("g1-harvest requires xvfb-run for stable headless video")
        inner = [
            pixi_binary(), "run", "python", str(ROOT / "g1_harvest.py"),
            "--orchard-root", str(args.orchard_root.resolve()),
            "--seed", str(args.seed), "--max-frames", str(args.frames),
            "--width", str(args.width), "--height", str(args.height),
            "--video", str(args.video.resolve()),
            "--report", str(args.report.resolve()),
            "--trace", str(args.trace.resolve()),
        ]
        screen = f"-screen 0 {args.width}x{args.height}x24 +extension GLX"
        return [xvfb, "-a", "-s", screen, *inner]
    if args.command == "g1":
        command = [
            pixi_binary(), "run", "python", str(ROOT / "g1_orchard.py"),
            "--orchard-root", str(args.orchard_root.resolve()),
            "--seed", str(args.seed), "--frames", str(args.frames),
            "--width", str(args.width), "--height", str(args.height),
            "--output", str(args.output.resolve()),
        ]
        xvfb = shutil.which("xvfb-run")
        use_xvfb = args.renderer == "xvfb" or (
            args.renderer == "auto" and not os.environ.get("DISPLAY") and xvfb is not None
        )
        if use_xvfb:
            if not xvfb:
                raise SystemExit(
                    "--renderer xvfb requested, but xvfb-run is not installed; "
                    "ask the workstation administrator to install the Ubuntu xvfb package"
                )
            screen = f"-screen 0 {args.width}x{args.height}x24 +extension GLX"
            command = [xvfb, "-a", "-s", screen, *command]
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
    for name in ("tree", "view", "usd", "harvest", "g1", "g1-harvest"):
        item = sub.add_parser(name)
        item.add_argument("--seed", type=int, default=42)
        item.add_argument("--frames", type=int, default=1000 if name == "g1-harvest" else 300)
        if name == "usd":
            item.add_argument("--output", type=Path, default=WORK_ROOT / "artifacts/tree_seed42.usda")
        if name == "harvest":
            item.add_argument("--viewer", choices=("gl", "null"), default="gl")
            item.add_argument("--metrics", type=Path, default=WORK_ROOT / "artifacts/harvest_seed42.json")
        if name == "g1":
            item.add_argument("--output", type=Path, default=ROOT / "artifacts/g1_orchard_seed42.gif")
            item.add_argument("--width", type=int, default=640)
            item.add_argument("--height", type=int, default=360)
            item.add_argument(
                "--renderer",
                choices=("auto", "egl", "xvfb"),
                default="auto",
                help="headless rendering backend (auto prefers Xvfb when available)",
            )
        if name == "g1-harvest":
            item.add_argument("--video", type=Path, default=ROOT / "artifacts/g1_harvest_seed42.mp4")
            item.add_argument("--report", type=Path, default=ROOT / "artifacts/g1_harvest_seed42.json")
            item.add_argument("--trace", type=Path, default=ROOT / "artifacts/g1_harvest_seed42.npz")
            item.add_argument("--width", type=int, default=960)
            item.add_argument("--height", type=int, default=540)
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
