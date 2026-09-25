from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class CameraConfig:
    name: str
    width: int = 640
    height: int = 480
    enabled: bool = True


DEFAULT_CAMERAS = {
    "front": CameraConfig("front"),
    "left_wrist": CameraConfig("left_wrist"),
    "right_wrist": CameraConfig("right_wrist"),
}


class HarvestRecorder:
    """Record MuJoCo harvesting demonstrations as LeRobot v3."""

    def __init__(
        self,
        model: mujoco.MjModel,
        root: Path,
        repo_id: str = "local/harvest",
        fps: int = 30,
        cameras: dict[str, CameraConfig] | None = None,
        task: str = "Pick up the apple and place it in the tray.",
    ):
        self.model = model
        self.fps = fps
        self.task = task

        self.cameras = {
            name: cfg
            for name, cfg in (cameras or DEFAULT_CAMERAS).items()
            if cfg.enabled
        }

        self.renderers = {
            name: mujoco.Renderer(
                model,
                height=cfg.height,
                width=cfg.width,
            )
            for name, cfg in self.cameras.items()
        }

        # State/action correspond to the actuated joints.
        self.joints = model.actuator_trnid[:, 0]
        self.qpos_addresses = model.jnt_qposadr[self.joints]

        state_dim = len(self.qpos_addresses)
        action_dim = model.nu

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": None,
            },
        }

        for name, cfg in self.cameras.items():
            features[f"observation.images.{name}"] = {
                "dtype": "video",
                "shape": (cfg.height, cfg.width, 3),
                "names": ["height", "width", "channels"],
            }

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=root,
            fps=fps,
            robot_type="unitree_g1_mujoco",
            features=features,
            use_videos=True,
        )

        self.next_record_time = 0.0

    def should_record(self, sim_time: float) -> bool:
        """Downsample MuJoCo simulation to dataset FPS."""
        return sim_time + 1e-9 >= self.next_record_time

    def _render_camera(
        self,
        name: str,
        data: mujoco.MjData,
    ) -> np.ndarray:

        renderer = self.renderers[name]

        renderer.update_scene(
            data,
            camera=name,
        )

        image = renderer.render()

        return np.asarray(image, dtype=np.uint8)

    def add_frame(
        self,
        data: mujoco.MjData,
        action: np.ndarray,
    ):
        if not self.should_record(float(data.time)):
            return

        frame = {
            "observation.state":
                data.qpos[self.qpos_addresses]
                .astype(np.float32)
                .copy(),

            "action":
                np.asarray(action, dtype=np.float32).copy(),

            "task": self.task,
        }

        for name in self.cameras:
            frame[f"observation.images.{name}"] = (
                self._render_camera(name, data)
            )

        self.dataset.add_frame(frame)

        # Avoid cumulative timing drift.
        while self.next_record_time <= data.time + 1e-9:
            self.next_record_time += 1.0 / self.fps

    def save_episode(self):
        self.dataset.save_episode()
        self.next_record_time = 0.0

    def discard_episode(self):
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()

        self.next_record_time = 0.0

    def finalize(self):
        self.dataset.finalize()

        for renderer in self.renderers.values():
            renderer.close()