"""Executable G1 + dual Dex3 training-data contract.

Keep the simulator and hardware adapters behind this interface. Dataset writers
must use this order and validate every episode before it is admitted for training.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping

import numpy as np

PROFILE = "g1_29body_dex3_43d_v1"
VERSION = "1.0"
FPS = 30
SYNC_TOLERANCE_S = 0.015
STATE_SIZE = ACTION_SIZE = 43
IMAGE_SHAPE = (480, 640, 3)
REQUIRED_CAMERAS = (
    "cam_left_high",
    "cam_left_wrist",
    "cam_right_wrist",
)
OPTIONAL_CAMERAS = ("cam_right_high",)

JOINT_NAMES = (
    "kLeftHipPitch", "kLeftHipRoll", "kLeftHipYaw", "kLeftKnee",
    "kLeftAnklePitch", "kLeftAnkleRoll",
    "kRightHipPitch", "kRightHipRoll", "kRightHipYaw", "kRightKnee",
    "kRightAnklePitch", "kRightAnkleRoll",
    "kWaistYaw", "kWaistRoll", "kWaistPitch",
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw",
    "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
    "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1",
    "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1",
    "kRightHandMiddle0", "kRightHandMiddle1",
)

# Explicit adapter names for the vendored G1/Dex3 MJCF. Never infer this order
# from the simulator's internal joint enumeration.
SIM_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "left_hand_middle_0_joint",
    "left_hand_middle_1_joint", "left_hand_index_0_joint",
    "left_hand_index_1_joint", "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
)

if len(JOINT_NAMES) != STATE_SIZE or len(SIM_JOINT_NAMES) != STATE_SIZE:
    raise RuntimeError("the contract must contain exactly 43 canonical and simulator joints")


class ContractError(ValueError):
    """Raised when a sample or episode cannot satisfy the training contract."""


def ordered_joint_vector(values: Mapping[str, float]) -> np.ndarray:
    """Convert a name-keyed reading into the canonical float32[43] order."""
    missing = [name for name in JOINT_NAMES if name not in values]
    extra = sorted(set(values) - set(JOINT_NAMES))
    if missing or extra:
        raise ContractError(f"joint map mismatch: missing={missing}, extra={extra}")
    result = np.asarray([values[name] for name in JOINT_NAMES], dtype=np.float32)
    _validate_numeric("joint vector", result, (STATE_SIZE,))
    return result


def _validate_numeric(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if value.dtype != np.float32:
        raise ContractError(f"{name} must be float32, got {value.dtype}")
    if value.shape != shape:
        raise ContractError(f"{name} must have shape {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ContractError(f"{name} contains non-finite values")


def validate_step(
    state: np.ndarray,
    action: np.ndarray,
    images: Mapping[str, np.ndarray],
) -> None:
    """Validate one synchronized sample after command clipping/filtering."""
    _validate_numeric("observation.state", state, (STATE_SIZE,))
    _validate_numeric("action", action, (ACTION_SIZE,))
    missing = [name for name in REQUIRED_CAMERAS if name not in images]
    if missing:
        raise ContractError(f"missing required cameras: {missing}")
    unknown = set(images) - set(REQUIRED_CAMERAS) - set(OPTIONAL_CAMERAS)
    if unknown:
        raise ContractError(f"unknown camera keys: {sorted(unknown)}")
    for name, frame in images.items():
        if frame.dtype != np.uint8 or frame.shape != IMAGE_SHAPE:
            raise ContractError(
                f"{name} must be uint8{IMAGE_SHAPE}, got {frame.dtype}{frame.shape}"
            )


def validate_episode(
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    images: Mapping[str, np.ndarray],
) -> None:
    """Validate the core synchronized arrays before LeRobot v2.1 encoding."""
    validate_state_action_trace(states, actions, timestamps)
    frames = states.shape[0]
    missing = [name for name in REQUIRED_CAMERAS if name not in images]
    if missing:
        raise ContractError(f"missing required cameras: {missing}")
    unknown = set(images) - set(REQUIRED_CAMERAS) - set(OPTIONAL_CAMERAS)
    if unknown:
        raise ContractError(f"unknown camera keys: {sorted(unknown)}")
    for name, video in images.items():
        if video.dtype != np.uint8 or video.shape != (frames, *IMAGE_SHAPE):
            raise ContractError(
                f"{name} must be uint8[frames,480,640,3], got {video.dtype}{video.shape}"
            )


def validate_state_action_trace(
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    """Validate a 30 Hz control trace before robot cameras are attached."""
    if states.ndim != 2:
        raise ContractError("states must have shape [frames, 43]")
    frames = states.shape[0]
    _validate_numeric("states", states, (frames, STATE_SIZE))
    _validate_numeric("actions", actions, (frames, ACTION_SIZE))
    if timestamps.dtype != np.float32 or timestamps.shape != (frames,):
        raise ContractError("timestamps must be float32[frames]")
    if not np.isfinite(timestamps).all():
        raise ContractError("timestamps contain non-finite values")
    expected = np.arange(frames, dtype=np.float32) / np.float32(FPS)
    if not np.allclose(timestamps, expected, atol=1e-6, rtol=0.0):
        raise ContractError("timestamps are not a contiguous 30 Hz sample grid")


def schema() -> dict:
    return {
        "profile": PROFILE,
        "version": VERSION,
        "fps": FPS,
        "sync_tolerance_ms": int(SYNC_TOLERANCE_S * 1000),
        "state": {"dtype": "float32", "shape": [STATE_SIZE], "joints": JOINT_NAMES},
        "sim_joint_names": SIM_JOINT_NAMES,
        "action": {"dtype": "float32", "shape": [ACTION_SIZE], "semantics": "absolute_post_filter_joint_position_target_rad"},
        "images": {"required": REQUIRED_CAMERAS, "optional": OPTIONAL_CAMERAS, "dtype": "uint8", "shape_hwc": IMAGE_SHAPE},
    }


def self_test() -> None:
    state = np.zeros(STATE_SIZE, dtype=np.float32)
    action = np.ones(ACTION_SIZE, dtype=np.float32)
    images = {name: np.zeros(IMAGE_SHAPE, dtype=np.uint8) for name in REQUIRED_CAMERAS}
    validate_step(state, action, images)
    assert ordered_joint_vector(dict(zip(JOINT_NAMES, state, strict=True))).shape == (43,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("contract self-test passed")
    else:
        print(json.dumps(schema(), indent=2))
