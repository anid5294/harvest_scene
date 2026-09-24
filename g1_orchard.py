"""Assemble a fixed-base G1/Dex3 and a fruit tree, then render a headless GIF."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from contract import SIM_JOINT_NAMES
from orchard import DEFAULT_ORCHARD_ROOT, ORCHARDBENCH_COMMIT, verify_orchardbench

ROOT = Path(__file__).resolve().parent
G1_MODEL = ROOT / "assets/unitree_g1/model.xml"


def _index_ending(labels, name: str, start: int = 0) -> int:
    matches = [i for i in range(start, len(labels)) if labels[i].rsplit("/", 1)[-1] == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one imported {name!r}, found {len(matches)}")
    return matches[0]


def add_g1(builder, _robot_params):
    """OrchardBench robot hook: import the vendored G1 with a fixed base."""
    import warp as wp

    body_start = builder.body_count
    joint_start = builder.joint_count
    facing_tree = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi)
    builder.add_mjcf(
        G1_MODEL,
        xform=wp.transform(wp.vec3(2.2, 0.0, 0.0), facing_tree),
        floating=False,
        enable_self_collisions=False,
        parse_mujoco_options=False,
    )

    pelvis = _index_ending(builder.body_label, "pelvis", body_start)
    wrist = _index_ending(builder.body_label, "right_wrist_yaw_link", body_start)
    joint_ids = [
        _index_ending(builder.joint_label, name, joint_start)
        for name in SIM_JOINT_NAMES
    ]
    home = [float(builder.joint_q[builder.joint_q_start[j]]) for j in joint_ids]
    add_g1.report = {
        "body_count": builder.body_count - body_start,
        "joint_count": builder.joint_count - joint_start,
        "contract_joint_count": len(joint_ids),
        "contract_joint_names": list(SIM_JOINT_NAMES),
        "fixed_base": True,
        "position_m": [2.2, 0.0, 0.0],
        "faces_tree": True,
    }
    # OrchardBench converts this map after finalization. Its Ridgeback driver is
    # deliberately not created by this standalone scene-assembly check.
    return {
        "chassis": pelvis,
        "wrist": wrist,
        "planar_joint": joint_ids[0],
        "arm_jids": joint_ids,
        "arm_home": home,
        "nbody": builder.body_count - body_start,
    }


add_g1.report = {}


def build_scene(orchard_root: Path, seed: int):
    sys.path.insert(0, str(orchard_root))
    import newton
    import warp as wp
    from treesim import builder, robot as orchard_robot
    from treesim.config import StiffnessModel, TreeConfig, preset

    cfg = TreeConfig(lsystem=preset("apple"))
    cfg.seed = seed
    cfg.device = "cuda"
    cfg.deformable = True
    cfg.physics.model = StiffnessModel.BEAM
    cfg.physics.mj_solver = "cg"
    cfg.fruit.enabled = True
    cfg.fruit.max_count = 40
    cfg.foliage.set_density(0.6)
    cfg.robot.enabled = True
    cfg.robot.camera = False

    original = orchard_robot.build_robot
    orchard_robot.build_robot = add_g1
    try:
        tree = builder.generate_and_build(cfg, num_envs=1)
    finally:
        orchard_robot.build_robot = original

    state, _ = tree.state_pair()
    return newton, wp, tree, state


def render_gif(newton, wp, tree, state, output: Path, frames: int, fps: int,
               width: int, height: int) -> None:
    from PIL import Image

    viewer = newton.viewer.ViewerGL(headless=True, width=width, height=height)
    viewer.set_model(tree.model)
    lo, hi = tree.skeleton.bounds()
    tree_height = float(hi[2] - lo[2])
    target = np.array([0.85, 0.0, max(0.9, float(lo[2] + 0.48 * tree_height))])
    distance = max(4.2, 1.8 * tree_height)
    images = []
    try:
        for frame in range(frames):
            phase = frame / max(frames - 1, 1)
            angle = math.radians(-72.0 + 34.0 * phase)
            pos = target + np.array([
                distance * math.cos(angle),
                distance * math.sin(angle),
                0.42 * distance,
            ])
            direction = target - pos
            direction /= np.linalg.norm(direction) + 1e-9
            viewer.set_camera(
                pos=wp.vec3(*map(float, pos)),
                pitch=math.degrees(math.asin(float(direction[2]))),
                yaw=math.degrees(math.atan2(float(direction[1]), float(direction[0]))),
            )
            viewer.begin_frame(frame / fps)
            viewer.log_state(state)
            viewer.end_frame()
            wp.synchronize()
            pixels = viewer.get_frame().numpy()
            if pixels.shape[-1] == 4:
                pixels = pixels[:, :, :3]
            images.append(Image.fromarray(pixels.astype(np.uint8)))
    finally:
        viewer.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, round(1000 / fps))
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def render_usd(newton, tree, state, output: Path, frames: int, fps: int) -> None:
    """Persistent fallback for servers without a usable off-screen GL context."""
    output.parent.mkdir(parents=True, exist_ok=True)
    viewer = newton.viewer.ViewerUSD(output_path=str(output), num_frames=frames)
    viewer.set_model(tree.model)
    try:
        for frame in range(frames):
            viewer.begin_frame(frame / fps)
            viewer.log_state(state)
            viewer.end_frame()
    finally:
        viewer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchard-root", type=Path, default=DEFAULT_ORCHARD_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/g1_orchard_seed42.gif")
    args = parser.parse_args()

    # With SSH X forwarding or a workstation desktop, pyglet should use that
    # display. With no DISPLAY, request its EGL/surfaceless backend.
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("PYGLET_HEADLESS", "1")
    orchard_root = args.orchard_root.resolve()
    verify_orchardbench(orchard_root)
    newton, wp, tree, state = build_scene(orchard_root, args.seed)
    actual_output = args.output
    render_error = None
    try:
        render_gif(newton, wp, tree, state, actual_output, args.frames, args.fps,
                   args.width, args.height)
    except Exception as exc:
        render_error = f"{type(exc).__name__}: {exc}"
        actual_output = args.output.with_suffix(".usda")
        print(f"headless OpenGL unavailable ({render_error}); writing {actual_output}")
        render_usd(newton, tree, state, actual_output, args.frames, args.fps)
    report = {
        "passed": add_g1.report.get("contract_joint_count") == 43 and actual_output.is_file(),
        "purpose": "scene_assembly_only",
        "orchardbench_commit": ORCHARDBENCH_COMMIT,
        "seed": args.seed,
        "tree_bodies": tree.n_bodies,
        "apples": len(tree.apple_bodies),
        "g1": add_g1.report,
        "render_requested": str(args.output.resolve()),
        "render_actual": str(actual_output.resolve()),
        "render_fallback": actual_output != args.output,
        "render_error": render_error,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"rendered {actual_output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
