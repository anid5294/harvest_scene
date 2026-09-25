"""Unitree G1 tabletop apple pick-and-place demo.

The policy uses MuJoCo state as an oracle observation. It is not learned and it
does not use a camera. Task transitions are triggered by observed pose,
contact, lift, and placement conditions; time is used only for
smooth motion, brief stability windows, and failure timeouts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from recording.recorder import (
    HarvestRecorder,
    CameraConfig,
)

ROOT = Path(__file__).resolve().parent
ROBOT_MODEL = ROOT / "assets/unitree_g1/model.xml"
APPLE_DIR = ROOT / "assets/ycb/013_apple/google_16k"
APPLE_CENTER = np.array([0.000859, -0.0037845, 0.0355515])
TRAY_CENTER = np.array([0.43, 0.0])
TRAY_HALF = np.array([0.12, 0.09])


def build_model():
    """Build the fixed-pelvis tabletop scene from vendored assets."""
    root = ET.parse(ROBOT_MODEL).getroot()
    root.find("visual/global").set("offwidth", "960")
    root.find("visual/global").set("offheight", "720")
    root.find("compiler").set("meshdir", str(ROBOT_MODEL.parent / "meshes"))
    ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast")

    # anchoring pelvis
    pelvis = root.find(".//body[@name='pelvis']")
    pelvis.remove(pelvis.find("joint[@type='free']"))
    for joint in root.findall(".//joint"):
        joint.set("armature", "0.002" if "hand" in joint.get("name", "") else "0.02")
    for motor in root.find("actuator"):
        hand = "hand" in motor.get("name", "")
        motor.tag = "position"
        motor.set("kp", "12" if hand else "150")
        motor.set("kv", "0.3" if hand else "12")

    asset = root.find("asset")
    ET.SubElement(asset, "mesh", name="ycb_apple", file=str(APPLE_DIR / "textured.obj"))
    ET.SubElement(asset, "texture", name="apple_texture", type="2d", file=str(APPLE_DIR / "texture_map.png"))
    ET.SubElement(asset, "material", name="apple_material", texture="apple_texture", specular="0.15", shininess="0.15")

    world = root.findall("worldbody")[-1]

    ## Added cameras
    ET.SubElement(
        world,
        "camera",
        name="front",
        pos="1.45 -1.45 1.45",
        xyaxes="0.707 0.707 0 -0.25 0.25 0.935",
        fovy="55",
    )

    left_wrist = root.find(".//body[@name='left_wrist_yaw_link']")

    ET.SubElement(
        left_wrist,
        "camera",
        name="left_wrist",
        pos="0.04 0 0.03",
        quat="1 0 0 0",
        fovy="70",
    )


    right_wrist = root.find(".//body[@name='right_wrist_yaw_link']")

    ET.SubElement(
        right_wrist,
        "camera",
        name="right_wrist",
        pos="0.04 0 0.03",
        quat="1 0 0 0",
        fovy="70",
    )

    ET.SubElement(world, "geom", name="table", type="box", pos="0.48 -0.12 0.89",
                  size="0.21 0.40 0.025", rgba="0.93 0.93 0.93 1")
    for x in (0.34, 0.62):
        for y in (-0.45, 0.21):
            ET.SubElement(world, "geom", type="box", pos=f"{x} {y} 0.445",
                          size="0.018 0.018 0.445", rgba="0.2 0.23 0.26 1")

    ET.SubElement(world, "geom", name="tray_floor", type="box",
                  pos="0.43 0.0 0.925", size="0.13 0.10 0.01", rgba="0.13 0.42 0.62 1")
    for name, pos, size in (
        ("left", "0.30 0.0 0.96", "0.01 0.11 0.025"),
        ("right", "0.56 0.0 0.96", "0.01 0.11 0.025"),
        ("front", "0.43 -0.10 0.96", "0.12 0.01 0.025"),
        ("back", "0.43 0.10 0.96", "0.12 0.01 0.025"),
    ):
        ET.SubElement(world, "geom", name=f"tray_{name}", type="box",
                      pos=pos, size=size, rgba="0.13 0.42 0.62 1")

    fruit = ET.SubElement(world, "body", name="fruit", pos="0.43 -0.25 0.9509445")
    ET.SubElement(fruit, "freejoint", name="fruit_free")
    ET.SubElement(fruit, "geom", name="fruit_geom", type="mesh", mesh="ycb_apple",
                  pos=" ".join(map(str, -APPLE_CENTER)), material="apple_material",
                  mass="0.068", friction="1.5 0.02 0.002", condim="4",
                  solref="0.004 1", solimp="0.99 0.999 0.001", priority="1")
    for geom in root.findall(".//geom"):
        if geom.get("mesh", "").startswith("right_hand"):
            geom.set("friction", "1.5 0.02 0.002")
            geom.set("condim", "4")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def home_pose(model):
    pose = np.zeros(model.nu)
    ids = {model.actuator(i).name: i for i in range(model.nu)}
    pose[ids["left_shoulder_pitch_joint"]] = 0.5
    pose[ids["left_shoulder_roll_joint"]] = 0.18
    pose[ids["right_shoulder_roll_joint"]] = -0.18
    pose[ids["left_elbow_joint"]] = 0.25
    pose[ids["right_elbow_joint"]] = 0.25
    return pose


def ik(model, seed, position):
    """Solve a damped least-squares right-wrist pose target."""
    data = mujoco.MjData(model)
    joints = model.actuator_trnid[:, 0]
    qpos_addresses = model.jnt_qposadr[joints]
    arm = np.array([
        i for i in range(model.nu)
        if model.actuator(i).name.startswith("right_")
        and "hand" not in model.actuator(i).name
        and any(part in model.actuator(i).name for part in ("shoulder", "elbow", "wrist"))
    ])
    dofs = model.jnt_dofadr[joints[arm]]
    data.qpos[qpos_addresses] = seed
    body = model.body("right_wrist_yaw_link").id
    position_jacobian = np.zeros((3, model.nv))
    rotation_jacobian = np.zeros((3, model.nv))
    for _ in range(400):
        mujoco.mj_forward(model, data)
        rotation = data.xmat[body].reshape(3, 3)
        rotation_error = sum(
            np.cross(rotation[:, i], np.eye(3)[:, i]) for i in range(3)
        ) / 2
        error = np.r_[position - data.xpos[body], rotation_error]
        if np.linalg.norm(error) < 1e-5:
            return data.qpos[qpos_addresses].copy()
        mujoco.mj_jacBody(model, data, position_jacobian, rotation_jacobian, body)
        jacobian = np.vstack([position_jacobian[:, dofs], rotation_jacobian[:, dofs]])
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.001 * np.eye(6), error
        )
        data.qpos[qpos_addresses[arm]] += np.clip(delta, -0.08, 0.08)
        data.qpos[qpos_addresses[arm]] = np.clip(
            data.qpos[qpos_addresses[arm]],
            model.jnt_range[joints[arm], 0] + 0.001,
            model.jnt_range[joints[arm], 1] - 0.001,
        )
    raise RuntimeError(f"IK failed: {error}")


def contacts(model, data):
    fruit_geom = model.geom("fruit_geom").id
    hand, support = set(), False
    for contact in data.contact:
        if fruit_geom not in (contact.geom1, contact.geom2) or contact.dist > 0.001:
            continue
        other = contact.geom2 if contact.geom1 == fruit_geom else contact.geom1
        body = model.body(model.geom_bodyid[other]).name
        if body.startswith("right_hand") or body == "right_wrist_yaw_link":
            hand.add(body)
        else:
            support = True
    return sorted(hand), support


def placement_state(model, data):
    fruit_geom = model.geom("fruit_geom").id
    mesh = model.geom_dataid[fruit_geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = (
        model.mesh_vert[start:start + count] @ data.geom_xmat[fruit_geom].reshape(3, 3).T
        + data.geom_xpos[fruit_geom]
    )
    low, high = vertices.min(0), vertices.max(0)
    tray_floor = model.geom("tray_floor").id
    floor_height = float(data.geom_xpos[tray_floor, 2] + model.geom("tray_floor").size[2])
    inside = bool(
        np.all(low[:2] > TRAY_CENTER - TRAY_HALF)
        and np.all(high[:2] < TRAY_CENTER + TRAY_HALF)
        and low[2] > floor_height - 0.002
        and high[2] < floor_height + 0.12
    )
    floor_contact = any(
        fruit_geom in (c.geom1, c.geom2)
        and tray_floor in (c.geom1, c.geom2)
        and c.dist <= 0.001
        for c in data.contact
    )
    hand, _ = contacts(model, data)
    dof = model.joint("fruit_free").dofadr[0]
    speed = float(np.linalg.norm(data.qvel[dof:dof + 3]))
    angular_speed = float(np.linalg.norm(data.qvel[dof + 3:dof + 6]))
    passed = inside and floor_contact and not hand and speed < 0.01 and angular_speed < 0.1
    return {
        "inside_tray": inside,
        "tray_floor_contact": floor_contact,
        "hand_contact": bool(hand),
        "speed_m_s": speed,
        "angular_speed_rad_s": angular_speed,
        "passed": passed,
    }


@dataclass
class Observation:
    joint_position: np.ndarray
    wrist_position: np.ndarray
    fruit_position: np.ndarray
    hand_contacts: tuple[str, ...]
    support_contact: bool
    placement: dict


class PickPlacePolicy:
    """Feedback state machine mapping observations to joint-position actions."""

    ORDER = (
        "READY", "APPROACH", "ALIGN", "GRASP", "LIFT", "HOLD",
        "TRANSFER", "LOWER", "RELEASE", "RETREAT", "SETTLE", "SUCCESS",
    )

    # Nominal interpolation time and hard timeout for each active state.
    TIMING = {
        "READY": (0.5, 2.0), "APPROACH": (3.0, 5.0),
        "ALIGN": (2.0, 4.0), "GRASP": (2.0, 4.0),
        "LIFT": (3.0, 5.0), "HOLD": (0.5, 3.0),
        "TRANSFER": (3.0, 5.0), "LOWER": (2.0, 4.0),
        "RELEASE": (2.0, 4.0), "RETREAT": (2.0, 4.0),
        "SETTLE": (0.5, 4.0),
    }

    def __init__(self, model, initial_joint_position, initial_fruit_position):
        self.model = model
        self.state = "READY"
        self.state_time = 0.0
        self.total_time = 0.0
        self.failed_reason = None
        self.initial_fruit_height = float(initial_fruit_position[2])
        self.hand_ids = np.array([
            i for i in range(model.nu) if "right_hand" in model.actuator(i).name
        ])
        self.motion_ids = np.array([
            i for i in range(model.nu) if "right_hand" not in model.actuator(i).name
        ])
        self.targets = self._initial_targets(initial_joint_position, initial_fruit_position)
        self.start_action = initial_joint_position.copy()
        self.hold_time = 0.0
        self.grasp_contact_time = 0.0
        self.placement_time = 0.0
        self.transitions = []

    def _initial_targets(self, seed, initial_fruit_position):
        fruit = np.asarray(initial_fruit_position)
        grasp_offset = np.array([0.135, 0.045, 0.0]) - [0, 0, 0.015]
        grasp = ik(self.model, seed, fruit - grasp_offset)
        approach = ik(self.model, grasp, fruit - grasp_offset + [0, 0, 0.18])
        ready = ik(self.model, approach, fruit - grasp_offset + [-0.06, 0, 0.18])
        closed = grasp.copy()
        for name, value in {
            "thumb_0": 0.0, "thumb_1": -0.6, "thumb_2": -1.1,
            "index_0": 1.1, "index_1": 1.2,
            "middle_0": 1.1, "middle_1": 1.2,
        }.items():
            closed[self.model.actuator(f"right_hand_{name}_joint").id] = value
        lifted = ik(self.model, grasp, fruit - grasp_offset + [0, 0, 0.16])
        lifted[self.hand_ids] = closed[self.hand_ids]
        return {
            "READY": ready, "APPROACH": approach, "ALIGN": grasp,
            "GRASP": closed, "LIFT": lifted, "HOLD": lifted,
        }

    @staticmethod
    def _smoothstep(value):
        value = np.clip(value, 0.0, 1.0)
        return value**3 * (10 - 15 * value + 6 * value**2)

    def _joint_close(self, observation, tolerance=0.035):
        error = observation.joint_position - self.targets[self.state]
        return float(np.max(np.abs(error[self.motion_ids]))) < tolerance

    def _transition(self, next_state, observation):
        self.transitions.append({
            "from": self.state,
            "to": next_state,
            "time_s": self.total_time,
            "fruit_xyz": observation.fruit_position.tolist(),
            "hand_contacts": list(observation.hand_contacts),
        })
        # Continue from the previously commanded target
        self.start_action = self.targets[self.state].copy()
        self.state = next_state
        self.state_time = 0.0

    def _plan_placement(self, observation):
        # Re-plan from the observed fruit-to-wrist offset after the grasp
        offset = observation.fruit_position - observation.wrist_position
        transfer = ik(
            self.model, self.targets["HOLD"],
            np.r_[TRAY_CENTER, observation.fruit_position[2]] - offset,
        )
        transfer[self.hand_ids] = self.targets["GRASP"][self.hand_ids]
        lower = ik(self.model, transfer, np.r_[TRAY_CENTER, 1.0] - offset)
        lower[self.hand_ids] = self.targets["GRASP"][self.hand_ids]
        released = lower.copy()
        released[self.hand_ids] = 0.0
        retreat = ik(self.model, released, np.r_[TRAY_CENTER, 1.20] - offset)
        self.targets.update({
            "TRANSFER": transfer, "LOWER": lower, "RELEASE": released,
            "RETREAT": retreat, "SETTLE": retreat,
        })

    def act(self, observation, dt):
        """Return the next joint-position action from the current observation."""
        if self.state in ("SUCCESS", "FAILED"):
            return observation.joint_position.copy()

        self.state_time += dt
        self.total_time += dt
        move_time, timeout = self.TIMING[self.state]

        if self.state == "READY" and self._joint_close(observation):
            self._transition("APPROACH", observation)
        elif self.state == "APPROACH" and self.state_time >= move_time and self._joint_close(observation):
            self._transition("ALIGN", observation)
        elif self.state == "ALIGN" and self.state_time >= move_time and self._joint_close(observation, 0.05):
            self._transition("GRASP", observation)
        elif self.state == "GRASP":
            grasped = len(observation.hand_contacts) >= 2
            self.grasp_contact_time = self.grasp_contact_time + dt if grasped else 0.0
            # Contact is the completion signal; dwell to let fingers finish wrapping
            if self.grasp_contact_time >= 0.25 and self.state_time >= move_time:
                self._transition("LIFT", observation)
        elif self.state == "LIFT" and self.state_time >= move_time and observation.fruit_position[2] > self.initial_fruit_height + 0.10 and not observation.support_contact:
            self._transition("HOLD", observation)
        elif self.state == "HOLD":
            stable = len(observation.hand_contacts) >= 2 and not observation.support_contact
            self.hold_time = self.hold_time + dt if stable else 0.0
            if self.hold_time >= 0.75:
                self._plan_placement(observation)
                self._transition("TRANSFER", observation)
        elif self.state == "TRANSFER" and self.state_time >= move_time and self._joint_close(observation, 0.05):
            self._transition("LOWER", observation)
        elif self.state == "LOWER" and self.state_time >= move_time and self._joint_close(observation, 0.05):
            self._transition("RELEASE", observation)
        elif self.state == "RELEASE" and self.state_time >= move_time:
            supported_in_tray = (
                observation.placement["inside_tray"]
                and observation.placement["tray_floor_contact"]
            )
            if supported_in_tray:
                self._transition("RETREAT", observation)
        elif self.state == "RETREAT" and self.state_time >= move_time and self._joint_close(observation, 0.05):
            self._transition("SETTLE", observation)
        elif self.state == "SETTLE":
            self.placement_time = self.placement_time + dt if observation.placement["passed"] else 0.0
            if self.placement_time >= 1.0:
                self._transition("SUCCESS", observation)

        if self.state not in ("SUCCESS", "FAILED") and self.state_time > timeout:
            self.failed_reason = f"timeout waiting for feedback condition in {self.state}"
            self._transition("FAILED", observation)
            return observation.joint_position.copy()

        if self.state in ("SUCCESS", "FAILED"):
            return observation.joint_position.copy()
        move_time = self.TIMING[self.state][0]
        blend = self._smoothstep(self.state_time / move_time)
        return self.start_action + blend * (self.targets[self.state] - self.start_action)


def observe(model, data, joint_qpos_addresses):
    hand, support = contacts(model, data)
    return Observation(
        joint_position=data.qpos[joint_qpos_addresses].copy(),
        wrist_position=data.body("right_wrist_yaw_link").xpos.copy(),
        fruit_position=data.body("fruit").xpos.copy(),
        hand_contacts=tuple(hand),
        support_contact=support,
        placement=placement_state(model, data),
    )


def run(args):
    model = build_model()
    data = mujoco.MjData(model)
    joints = model.actuator_trnid[:, 0]
    qpos_addresses = model.jnt_qposadr[joints]
    dof_addresses = model.jnt_dofadr[joints]

    seed = home_pose(model)
    data.qpos[qpos_addresses] = seed
    data.ctrl[:] = seed
    mujoco.mj_forward(model, data)
    policy = PickPlacePolicy(model, seed, data.body("fruit").xpos.copy())
    data.qpos[qpos_addresses] = policy.targets["READY"]
    data.ctrl[:] = policy.targets["READY"]
    mujoco.mj_forward(model, data)
    policy.start_action = data.qpos[qpos_addresses].copy()
    policy.initial_fruit_height = float(data.body("fruit").xpos[2])

    recorder = None

    if args.record:
        cameras = {
            "front": CameraConfig(
                name="front",
                width=640,
                height=480,
            ),
            "left_wrist": CameraConfig(
                name="left_wrist",
                width=640,
                height=480,
            ),
            "right_wrist": CameraConfig(
                name="right_wrist",
                width=640,
                height=480,
            ),
        }

        recorder = HarvestRecorder(
            model=model,
            root=args.dataset_root,
            repo_id=args.repo_id,
            fps=args.record_fps,
            cameras=cameras,
            task="Pick up the apple and place it in the tray.",
        )

    viewer = None
    if not args.headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.22, -0.12, 0.85]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 135, -18

    max_time = 40.0
    last_state = None
    wall_start = time.monotonic()
    try:
        while policy.state not in ("SUCCESS", "FAILED") and data.time < max_time:
            observation = observe(model, data, qpos_addresses)
            if policy.state != last_state:
                print(policy.state, flush=True)
                last_state = policy.state
            # data.ctrl[:] = policy.act(observation, model.opt.timestep)
            action = policy.act(observation, model.opt.timestep)
            data.ctrl[:] = action

            if recorder is not None:
                recorder.add_frame(
                    data=data,
                    action=action,
                )
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[dof_addresses] = data.qfrc_bias[dof_addresses]
            mujoco.mj_step(model, data)
            if viewer is not None and int(data.time / model.opt.timestep) % 16 == 0:
                if not viewer.is_running():
                    policy.failed_reason = "viewer closed"
                    break
                viewer.sync()
                time.sleep(max(0.0, wall_start + data.time - time.monotonic()))

        final = observe(model, data, qpos_addresses)
        passed = policy.state == "SUCCESS" and final.placement["passed"]
        if recorder is not None:

            if passed:
                print("Saving successful LeRobot episode...")
                recorder.save_episode()

            else:
                print("Discarding failed episode...")
                recorder.discard_episode()
        report = {
            "demo": "rule_based_policy_pick_place",
            "policy": "closed-loop feedback state machine",
            "learned": False,
            "observation_source": "oracle MuJoCo state (no camera)",
            "fixed_pelvis": True,
            "visible_fixture": False,
            "final_state": policy.state,
            "failure_reason": policy.failed_reason,
            "simulation_time_s": float(data.time),
            "transitions": policy.transitions,
            "final_placement": final.placement,
            "warnings": int(sum(w.number for w in data.warning)),
            "passed": bool(passed and sum(w.number for w in data.warning) == 0),
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    # finally:
    #     if viewer is not None:
    #         viewer.close()
    finally:

        if recorder is not None:
            recorder.finalize()

        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    parser.add_argument("--record", action="store_true", help="Record a LeRobot v3 demonstration")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "outputs" / "lerobot")
    parser.add_argument("--repo-id", type=str, default="local/harvest")
    parser.add_argument("--record-fps", type=int, default=30)
    raise SystemExit(run(parser.parse_args()))
