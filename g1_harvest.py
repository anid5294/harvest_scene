"""Run the first fixed-base G1 apple-harvest integration test."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from contract import FPS, SIM_JOINT_NAMES, validate_state_action_trace
from orchard import DEFAULT_ORCHARD_ROOT, ORCHARDBENCH_COMMIT, verify_orchardbench

ROOT = Path(__file__).resolve().parent
G1_MODEL = ROOT / "assets/unitree_g1/model.xml"
TCP_OFFSET = np.array([0.135, 0.045, -0.015], dtype=float)
APPLE_LOCAL = np.array([0.50, -0.20], dtype=float)
TRAY_LOCAL = np.array([0.35, -0.35, 0.88], dtype=float)
TRAY_HALF = np.array([0.13, 0.11], dtype=float)
RIGHT_ARM = tuple(f"right_{part}_joint" for part in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
))


def default_home() -> np.ndarray:
    home = np.zeros(len(SIM_JOINT_NAMES), dtype=float)
    values = {
        "left_shoulder_pitch_joint": 0.5,
        "left_shoulder_roll_joint": 0.18,
        "left_elbow_joint": 0.25,
        "right_shoulder_roll_joint": -0.18,
        "right_elbow_joint": 0.25,
    }
    for name, value in values.items():
        home[SIM_JOINT_NAMES.index(name)] = value
    return home


def _ending(labels, name: str, start: int = 0) -> int:
    matches = [i for i in range(start, len(labels)) if labels[i].rsplit("/", 1)[-1] == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one imported {name!r}, found {len(matches)}")
    return matches[0]


def choose_apple(positions: np.ndarray) -> int:
    """Choose an outer apple in the G1's comfortable standing reach band."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("apple positions must have shape [n, 3]")
    radius = np.linalg.norm(positions[:, :2], axis=1)
    height_cost = np.abs(positions[:, 2] - 1.12)
    score = height_cost + 0.20 / np.maximum(radius, 0.05)
    reachable = (positions[:, 2] >= 0.88) & (positions[:, 2] <= 1.38)
    if reachable.any():
        score = np.where(reachable, score, np.inf)
    return int(np.argmin(score))


def robot_placement(apple_world: np.ndarray) -> tuple[np.ndarray, float]:
    """Place the robot outside the canopy with the chosen apple at APPLE_LOCAL."""
    apple_world = np.asarray(apple_world, dtype=float)
    outward = apple_world[:2] / max(float(np.linalg.norm(apple_world[:2])), 1e-6)
    desired_angle = math.atan2(-outward[1], -outward[0])
    local_angle = math.atan2(APPLE_LOCAL[1], APPLE_LOCAL[0])
    yaw = desired_angle - local_angle
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    base_xy = apple_world[:2] - rotation @ APPLE_LOCAL
    return np.array([base_xy[0], base_xy[1], 0.0]), yaw


def local_to_world(point: np.ndarray, base: np.ndarray, yaw: float) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    return base + np.array([c * point[0] - s * point[1], s * point[0] + c * point[1], point[2]])


def world_to_local(point: np.ndarray, base: np.ndarray, yaw: float) -> np.ndarray:
    delta = np.asarray(point, dtype=float) - base
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]])


def apple_in_tray(apple: np.ndarray, tray: np.ndarray, speed: float) -> bool:
    delta = np.asarray(apple) - np.asarray(tray)
    return bool(
        abs(delta[0]) < TRAY_HALF[0] - 0.04
        and abs(delta[1]) < TRAY_HALF[1] - 0.04
        and 0.015 < delta[2] < 0.16
        and speed < 0.10
    )


def add_g1(builder, _robot_params):
    """OrchardBench hook that adds the G1, a tray, and position servos."""
    import newton
    import warp as wp

    apple_ids = [i for i, label in enumerate(builder.body_label) if label.rsplit("/", 1)[-1].startswith("apple")]
    if not apple_ids:
        raise RuntimeError("the generated tree contains no apples")
    body_q = np.asarray(builder.body_q, dtype=float).reshape(-1, 7)
    apple_positions = body_q[apple_ids, :3]
    target_index = choose_apple(apple_positions)
    target_body = apple_ids[target_index]
    target_position = apple_positions[target_index]
    base, yaw = robot_placement(target_position)

    tray = local_to_world(TRAY_LOCAL, base, yaw)
    tray_cfg = builder.ShapeConfig(density=0.0, mu=1.0, collision_group=3)
    color = (0.08, 0.34, 0.58)
    builder.add_shape_box(-1, xform=wp.transform(p=wp.vec3(*tray), q=wp.quat_identity()),
                          hx=TRAY_HALF[0], hy=TRAY_HALF[1], hz=0.012,
                          cfg=tray_cfg, color=color)
    wall_z = tray[2] + 0.055
    for dx, dy, hx, hy in (
        (TRAY_HALF[0], 0.0, 0.012, TRAY_HALF[1]),
        (-TRAY_HALF[0], 0.0, 0.012, TRAY_HALF[1]),
        (0.0, TRAY_HALF[1], TRAY_HALF[0], 0.012),
        (0.0, -TRAY_HALF[1], TRAY_HALF[0], 0.012),
    ):
        builder.add_shape_box(-1, xform=wp.transform(
            p=wp.vec3(float(tray[0] + dx), float(tray[1] + dy), float(wall_z)),
            q=wp.quat_identity()), hx=hx, hy=hy, hz=0.055, cfg=tray_cfg, color=color)

    body_start = builder.body_count
    joint_start = builder.joint_count
    facing = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    builder.add_mjcf(G1_MODEL, xform=wp.transform(wp.vec3(*base), facing),
                     floating=False, enable_self_collisions=False,
                     parse_mujoco_options=False)

    pelvis = _ending(builder.body_label, "pelvis", body_start)
    wrist = _ending(builder.body_label, "right_wrist_yaw_link", body_start)
    joint_ids = [_ending(builder.joint_label, name, joint_start) for name in SIM_JOINT_NAMES]
    home = default_home()
    for index, joint in enumerate(joint_ids):
        qs = builder.joint_q_start[joint]
        ds = builder.joint_qd_start[joint]
        builder.joint_q[qs] = float(home[index])
        name = SIM_JOINT_NAMES[index]
        if "hand" in name:
            kp, kd = 45.0, 2.0
        elif name in RIGHT_ARM:
            kp, kd = 180.0, 12.0
        else:
            kp, kd = 420.0, 32.0
        builder.joint_target_ke[ds] = kp
        builder.joint_target_kd[ds] = kd
        builder.joint_target_mode[ds] = newton.JointTargetMode.POSITION

    add_g1.scene = {
        "apple_index": target_index,
        "apple_body": target_body,
        "apple_initial_world": target_position.tolist(),
        "robot_base_world": base.tolist(),
        "robot_yaw_rad": yaw,
        "tray_center_world": tray.tolist(),
        "joint_ids": joint_ids,
        "home": home.tolist(),
        "wrist": wrist,
    }
    return {"chassis": pelvis, "wrist": wrist, "planar_joint": joint_ids[0],
            "arm_jids": joint_ids, "arm_home": home.tolist(),
            "nbody": builder.body_count - body_start}


add_g1.scene: dict = {}


class G1Kinematics:
    """MuJoCo inverse kinematics in the G1 base frame."""

    def __init__(self):
        import mujoco
        import xml.etree.ElementTree as ET

        self.mujoco = mujoco
        root = ET.parse(G1_MODEL).getroot()
        root.find("compiler").set("meshdir", str(G1_MODEL.parent / "meshes"))
        pelvis = root.find(".//body[@name='pelvis']")
        free = pelvis.find("joint[@type='free']")
        if free is not None:
            pelvis.remove(free)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array([self.model.joint(name).id for name in SIM_JOINT_NAMES])
        self.qpos = self.model.jnt_qposadr[self.joint_ids]
        self.arm_indices = np.array([SIM_JOINT_NAMES.index(name) for name in RIGHT_ARM])
        self.arm_joints = self.joint_ids[self.arm_indices]
        self.arm_dofs = self.model.jnt_dofadr[self.arm_joints]
        self.wrist = self.model.body("right_wrist_yaw_link").id

    def solve(self, seed: np.ndarray, tcp_local: np.ndarray) -> np.ndarray:
        q = np.asarray(seed, dtype=float).copy()
        self.data.qpos[self.qpos] = q
        target = np.asarray(tcp_local, dtype=float) - TCP_OFFSET
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        for _ in range(500):
            self.mujoco.mj_forward(self.model, self.data)
            rotation = self.data.xmat[self.wrist].reshape(3, 3)
            rotation_error = sum(np.cross(rotation[:, i], np.eye(3)[:, i]) for i in range(3)) / 2
            error = np.r_[target - self.data.xpos[self.wrist], rotation_error]
            if np.linalg.norm(error) < 2e-3:
                return self.data.qpos[self.qpos].copy()
            self.mujoco.mj_jacBody(self.model, self.data, jp, jr, self.wrist)
            jac = np.vstack((jp[:, self.arm_dofs], jr[:, self.arm_dofs]))
            step = jac.T @ np.linalg.solve(jac @ jac.T + 0.002 * np.eye(6), error)
            adr = self.qpos[self.arm_indices]
            self.data.qpos[adr] += np.clip(step, -0.07, 0.07)
            limits = self.model.jnt_range[self.arm_joints]
            self.data.qpos[adr] = np.clip(self.data.qpos[adr], limits[:, 0] + 1e-3, limits[:, 1] - 1e-3)
        raise RuntimeError(f"G1 right-arm IK failed with residual {np.linalg.norm(error):.4f}")


def _close_hand(action: np.ndarray) -> np.ndarray:
    action = action.copy()
    values = {
        "right_hand_thumb_0_joint": 0.0,
        "right_hand_thumb_1_joint": -0.60,
        "right_hand_thumb_2_joint": -1.10,
        "right_hand_index_0_joint": 1.10,
        "right_hand_index_1_joint": 1.20,
        "right_hand_middle_0_joint": 1.10,
        "right_hand_middle_1_joint": 1.20,
    }
    for name, value in values.items():
        action[SIM_JOINT_NAMES.index(name)] = value
    return action


@dataclass
class Phase:
    name: str
    target: np.ndarray
    frames: int


class HarvestPolicy:
    """Feedback-gated joint-position policy for one selected apple."""

    def __init__(self, home: np.ndarray, apple_local: np.ndarray):
        ik = G1Kinematics()
        pre = ik.solve(home, apple_local + np.array([-0.10, 0.0, 0.10]))
        grasp = ik.solve(pre, apple_local)
        closed = _close_hand(grasp)
        pull = ik.solve(closed, apple_local + np.array([-0.16, 0.0, 0.08]))
        pull[29:] = closed[29:]
        above_tray = ik.solve(pull, TRAY_LOCAL + np.array([0.0, 0.0, 0.20]))
        above_tray[29:] = closed[29:]
        lower = ik.solve(above_tray, TRAY_LOCAL + np.array([0.0, 0.0, 0.075]))
        lower[29:] = closed[29:]
        release = lower.copy()
        release[29:] = 0.0
        retreat = ik.solve(release, TRAY_LOCAL + np.array([-0.12, 0.0, 0.25]))
        self.phases = [
            Phase("HOME", home.copy(), 25), Phase("PREGRASP", pre, 75),
            Phase("GRASP", grasp, 45), Phase("CLOSE", closed, 35),
            Phase("PULL", pull, 65), Phase("TRANSPORT", above_tray, 90),
            Phase("LOWER", lower, 50), Phase("RELEASE", release, 35),
            Phase("RETREAT", retreat, 55), Phase("SETTLE", retreat, 120),
        ]
        self.index = 0
        self.frame = 0
        self.start = home.copy()
        self.transitions: list[dict] = []
        self.failed_reason: str | None = None
        self.hold_started = False
        self.released = False
        self.stable_frames = 0

    @property
    def name(self) -> str:
        return self.phases[self.index].name if self.index < len(self.phases) else "SUCCESS"

    @staticmethod
    def _blend(value: float) -> float:
        x = float(np.clip(value, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _advance(self, joint_state: np.ndarray, apple: np.ndarray) -> None:
        old = self.name
        self.index += 1
        self.frame = 0
        self.start = joint_state.copy()
        self.transitions.append({"from": old, "to": self.name,
                                 "apple_world": apple.tolist()})

    def act(self, joint_state: np.ndarray, apple: np.ndarray, detached: bool,
            speed: float, tray: np.ndarray) -> tuple[np.ndarray, str | None]:
        if self.name == "SUCCESS":
            return joint_state.copy(), None
        phase = self.phases[self.index]
        self.frame += 1
        command: str | None = None
        if phase.name == "CLOSE" and self.frame == 18:
            command = "hold"
            self.hold_started = True
        if phase.name == "PULL" and self.frame >= phase.frames and not detached:
            if self.frame > phase.frames + 90:
                self.failed_reason = "apple did not detach under the commanded pull"
            return phase.target.copy(), command
        if phase.name == "RELEASE" and self.frame == 15:
            command = "release"
            self.released = True
        if phase.name == "SETTLE":
            self.stable_frames = self.stable_frames + 1 if apple_in_tray(apple, tray, speed) else 0
            if self.stable_frames >= 20:
                self._advance(joint_state, apple)
                return joint_state.copy(), command
        if self.failed_reason is None and self.frame >= phase.frames and phase.name != "SETTLE":
            self._advance(joint_state, apple)
        blend = self._blend(self.frame / max(phase.frames, 1))
        return self.start + blend * (phase.target - self.start), command


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.process = subprocess.Popen([
            "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(path),
        ], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        code = self.process.wait()
        if code:
            raise RuntimeError(f"ffmpeg exited with status {code}")


def build_scene(orchard_root: Path, seed: int):
    sys.path.insert(0, str(orchard_root))
    import warp as wp
    from treesim import builder, robot as orchard_robot
    from treesim.config import StiffnessModel, TreeConfig, preset
    from treesim.sim import Sim

    cfg = TreeConfig(lsystem=preset("apple"))
    cfg.seed = seed
    cfg.device = "cuda"
    cfg.deformable = True
    cfg.physics.model = StiffnessModel.BEAM
    cfg.physics.mj_solver = "cg"
    cfg.fruit.enabled = True
    cfg.fruit.max_count = 24
    cfg.foliage.set_density(0.35)
    cfg.robot.enabled = True
    cfg.robot.camera = False
    original = orchard_robot.build_robot
    orchard_robot.build_robot = add_g1
    try:
        tree = builder.generate_and_build(cfg, num_envs=1)
    finally:
        orchard_robot.build_robot = original
    sim = Sim(tree, solver="mujoco", fps=FPS, substeps=3,
              enable_breaking=False, collisions=True)
    sim.apples.hold_off = wp.vec3(*TCP_OFFSET)
    return wp, tree, sim


def run(args: argparse.Namespace) -> dict:
    import newton

    wp, tree, sim = build_scene(args.orchard_root, args.seed)
    scene = add_g1.scene
    target = int(scene["apple_index"])
    apple_body = int(tree.apple_bodies[target])
    wrist = int(tree.robot_data["wrist"][0])
    tq = np.asarray(tree.robot_data["arm_tq"][0], dtype=int)
    joint_ids = np.asarray(scene["joint_ids"], dtype=int)
    q_indices = tree.model.joint_q_start.numpy()[joint_ids].astype(int)
    home = np.asarray(scene["home"], dtype=float)
    target_host = sim.control.joint_target_q.numpy()
    target_host[tq] = home
    sim.control.joint_target_q.assign(target_host)

    base = np.asarray(scene["robot_base_world"])
    yaw = float(scene["robot_yaw_rad"])
    apple_initial = sim.body_q_np()[apple_body, :3].copy()
    policy = HarvestPolicy(home, world_to_local(apple_initial, base, yaw))
    tray = np.asarray(scene["tray_center_world"])

    viewer = newton.viewer.ViewerGL(headless=True, width=args.width, height=args.height)
    sim.set_viewer(viewer)
    center = (apple_initial + tray) * 0.5
    radial = base[:2] / max(float(np.linalg.norm(base[:2])), 1e-6)
    camera = np.r_[base[:2] + 2.8 * radial + np.array([-radial[1], radial[0]]) * 1.6, 1.9]
    direction = center - camera
    direction /= np.linalg.norm(direction) + 1e-9
    viewer.set_camera(pos=wp.vec3(*map(float, camera)),
                      pitch=math.degrees(math.asin(float(direction[2]))),
                      yaw=math.degrees(math.atan2(float(direction[1]), float(direction[0]))))
    writer = VideoWriter(args.video, args.width, args.height, args.video_fps)
    states, actions, timestamps, apple_xyz, apple_speed, phase_ids = [], [], [], [], [], []
    phase_names = [p.name for p in policy.phases] + ["SUCCESS", "FAILED"]
    error = None
    try:
        for frame in range(args.max_frames):
            q = sim.joint_q_np()[q_indices].astype(np.float32)
            bq = sim.body_q_np()
            pos = bq[apple_body, :3].copy()
            velocity = sim.state_0.body_qd.numpy()[apple_body, :3]
            speed = float(np.linalg.norm(velocity))
            detached = bool(sim.apples.detached[target])
            action, command = policy.act(q, pos, detached, speed, tray)
            if command == "hold":
                tcp = bq[wrist, :3]
                if float(np.linalg.norm(tcp - pos)) > 0.20:
                    policy.failed_reason = "wrist did not reach the selected apple"
                else:
                    sim.apples.hold(target, wrist)
            elif command == "release":
                sim.apples.release(target)
            target_host[tq] = action
            sim.control.joint_target_q.assign(target_host)
            states.append(q.copy())
            actions.append(np.asarray(action, dtype=np.float32))
            timestamps.append(np.float32(frame / FPS))
            apple_xyz.append(pos.astype(np.float32))
            apple_speed.append(np.float32(speed))
            phase_ids.append(np.int16(phase_names.index(policy.name)))
            sim.step()
            if frame % max(FPS // args.video_fps, 1) == 0:
                # Two swaps per captured state avoid Xvfb front/back-buffer parity flicker.
                sim.render()
                sim.render()
                wp.synchronize()
                writer.write(viewer.get_frame().numpy())
            if policy.name == "SUCCESS" or policy.failed_reason:
                break
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            writer.close()
        finally:
            viewer.close()

    states_array = np.asarray(states, dtype=np.float32)
    actions_array = np.asarray(actions, dtype=np.float32)
    times_array = np.asarray(timestamps, dtype=np.float32)
    validate_state_action_trace(states_array, actions_array, times_array)
    final_pos = sim.body_q_np()[apple_body, :3].copy()
    final_speed = float(np.linalg.norm(sim.state_0.body_qd.numpy()[apple_body, :3]))
    placed = apple_in_tray(final_pos, tray, final_speed)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.trace, observation_state=states_array,
                        action=actions_array, timestamp=times_array,
                        apple_position=np.asarray(apple_xyz, dtype=np.float32),
                        apple_speed=np.asarray(apple_speed, dtype=np.float32),
                        phase=np.asarray(phase_ids, dtype=np.int16))
    artifacts_written = (
        args.video.is_file() and args.video.stat().st_size > 0
        and args.trace.is_file() and args.trace.stat().st_size > 0
    )
    passed = bool(
        len(tq) == 43 and policy.name == "SUCCESS"
        and sim.apples.detached[target] and placed
        and artifacts_written and error is None
    )
    return {
        "passed": passed,
        "test": "fixed_base_g1_single_apple_harvest",
        "seed": args.seed,
        "orchardbench_commit": ORCHARDBENCH_COMMIT,
        "policy": "feedback-gated scripted joint-position policy",
        "learned": False,
        "fixed_base": True,
        "contract_joint_count": len(tq),
        "selected_apple": target,
        "apple_detached": bool(sim.apples.detached[target]),
        "apple_in_tray": placed,
        "final_apple_world": final_pos.tolist(),
        "final_apple_speed_m_s": final_speed,
        "final_phase": "FAILED" if policy.failed_reason or error else policy.name,
        "failure_reason": policy.failed_reason or error,
        "transitions": policy.transitions,
        "frames": len(states),
        "video": str(args.video.resolve()),
        "trace": str(args.trace.resolve()),
        "data_contract": {"profile": "g1_29body_dex3_43d_v1",
                          "state_action_valid": True,
                          "training_eligible": False,
                          "missing": ["cam_left_high", "cam_left_wrist", "cam_right_wrist"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchard-root", type=Path, default=DEFAULT_ORCHARD_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--video", type=Path, default=ROOT / "artifacts/g1_harvest_seed42.mp4")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/g1_harvest_seed42.json")
    parser.add_argument("--trace", type=Path, default=ROOT / "artifacts/g1_harvest_seed42.npz")
    args = parser.parse_args()
    args.orchard_root = args.orchard_root.resolve()
    if not os.environ.get("DISPLAY"):
        raise SystemExit("g1_harvest requires DISPLAY; launch it through orchard.py and Xvfb")
    verify_orchardbench(args.orchard_root)
    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
