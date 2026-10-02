# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path

# import mujoco
# import numpy as np

# from lerobot.datasets.lerobot_dataset import LeRobotDataset


# @dataclass
# class CameraConfig:
#     name: str
#     width: int = 640
#     height: int = 480
#     enabled: bool = True


# DEFAULT_CAMERAS = {
#     "front": CameraConfig("front"),
#     "left_wrist": CameraConfig("left_wrist"),
#     "right_wrist": CameraConfig("right_wrist"),
# }


# class HarvestRecorder:
#     """Record MuJoCo harvesting demonstrations as LeRobot v3."""

#     def __init__(
#         self,
#         model: mujoco.MjModel,
#         root: Path,
#         repo_id: str = "local/harvest",
#         fps: int = 30,
#         cameras: dict[str, CameraConfig] | None = None,
#         task: str = "Pick up the apple and place it in the tray.",
#     ):
#         self.model = model
#         self.fps = fps
#         self.task = task

#         self.cameras = {
#             name: cfg
#             for name, cfg in (cameras or DEFAULT_CAMERAS).items()
#             if cfg.enabled
#         }

#         self.renderers = {
#             name: mujoco.Renderer(
#                 model,
#                 height=cfg.height,
#                 width=cfg.width,
#             )
#             for name, cfg in self.cameras.items()
#         }

#         # State/action correspond to the actuated joints.
#         self.joints = model.actuator_trnid[:, 0]
#         self.qpos_addresses = model.jnt_qposadr[self.joints]

#         state_dim = len(self.qpos_addresses)
#         action_dim = model.nu

#         features = {
#             "observation.state": {
#                 "dtype": "float32",
#                 "shape": (state_dim,),
#                 "names": None,
#             },
#             "action": {
#                 "dtype": "float32",
#                 "shape": (action_dim,),
#                 "names": None,
#             },
#         }

#         for name, cfg in self.cameras.items():
#             features[f"observation.images.{name}"] = {
#                 "dtype": "video",
#                 "shape": (cfg.height, cfg.width, 3),
#                 "names": ["height", "width", "channels"],
#             }

#         self.dataset = LeRobotDataset.create(
#             repo_id=repo_id,
#             root=root,
#             fps=fps,
#             robot_type="unitree_g1_mujoco",
#             features=features,
#             use_videos=True,
#         )

#         self.next_record_time = 0.0

#     def should_record(self, sim_time: float) -> bool:
#         """Downsample MuJoCo simulation to dataset FPS."""
#         return sim_time + 1e-9 >= self.next_record_time

#     def _render_camera(
#         self,
#         name: str,
#         data: mujoco.MjData,
#     ) -> np.ndarray:

#         renderer = self.renderers[name]

#         renderer.update_scene(
#             data,
#             camera=name,
#         )

#         image = renderer.render()

#         return np.asarray(image, dtype=np.uint8)

#     def add_frame(
#         self,
#         data: mujoco.MjData,
#         action: np.ndarray,
#     ):
#         if not self.should_record(float(data.time)):
#             return

#         frame = {
#             "observation.state":
#                 data.qpos[self.qpos_addresses]
#                 .astype(np.float32)
#                 .copy(),

#             "action":
#                 np.asarray(action, dtype=np.float32).copy(),

#             "task": self.task,
#         }

#         for name in self.cameras:
#             frame[f"observation.images.{name}"] = (
#                 self._render_camera(name, data)
#             )

#         self.dataset.add_frame(frame)

#         # Avoid cumulative timing drift.
#         while self.next_record_time <= data.time + 1e-9:
#             self.next_record_time += 1.0 / self.fps

#     def save_episode(self):
#         self.dataset.save_episode()
#         self.next_record_time = 0.0

#     def discard_episode(self):
#         if self.dataset.has_pending_frames():
#             self.dataset.clear_episode_buffer()

#         self.next_record_time = 0.0

#     def finalize(self):
#         self.dataset.finalize()

#         for renderer in self.renderers.values():
#             renderer.close()

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    from .cameras import CAMERA_CONFIG
except ImportError:
    # Allows the file to still be imported directly during quick local tests.
    from cameras import CAMERA_CONFIG


@dataclass
class CameraConfig:
    name: str
    width: int = 640
    height: int = 480
    enabled: bool = True


def _default_cameras() -> dict[str, CameraConfig]:
    cameras: dict[str, CameraConfig] = {}

    for name, cfg in CAMERA_CONFIG.items():
        cameras[name] = CameraConfig(
            name=name,
            width=int(cfg.get("width", 640)),
            height=int(cfg.get("height", 480)),
            enabled=bool(cfg.get("enabled", True)),
        )

    return cameras


DEFAULT_CAMERAS = _default_cameras()


def _resolve_dataset_root(root: Path) -> Path:
    """
    Return a path that does not already exist, as required by
    LeRobotDataset.create().

    If the requested root is unused, keep it unchanged.
    If it already exists, create a unique session path inside it.
    """
    root = Path(root).expanduser()

    if not root.exists():
        return root

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"session_{timestamp}"

    counter = 1
    while candidate.exists():
        candidate = root / f"session_{timestamp}_{counter:02d}"
        counter += 1

    return candidate


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

        requested = dict(cameras) if cameras is not None else dict(DEFAULT_CAMERAS)

        # We are intentionally not changing strawberry_load.py.  Add any camera
        # marked always_record here (currently the new head camera).
        for name, mount_cfg in CAMERA_CONFIG.items():
            if not mount_cfg.get("always_record", False):
                continue
            if not mount_cfg.get("enabled", True):
                continue

            requested.setdefault(
                name,
                CameraConfig(
                    name=name,
                    width=int(mount_cfg.get("width", 640)),
                    height=int(mount_cfg.get("height", 480)),
                    enabled=True,
                ),
            )

        self.cameras = {
            name: cfg
            for name, cfg in requested.items()
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

        # Resolve body IDs once.  The pose itself is recomputed every frame.
        self._camera_body_ids: dict[str, int] = {}

        for name in self.cameras:
            mount_cfg = CAMERA_CONFIG.get(name, {})

            if mount_cfg.get("mode", "model") != "body":
                continue

            body_name = mount_cfg.get("body")
            if not body_name:
                raise ValueError(
                    f'Body-mounted camera "{name}" is missing a body name.'
                )

            body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )

            if body_id == -1:
                raise ValueError(
                    f'Camera "{name}" body "{body_name}" does not exist.'
                )

            self._camera_body_ids[name] = body_id

            template_name = mount_cfg.get("template_camera")
            if template_name:
                template_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    template_name,
                )

                if template_id == -1:
                    raise ValueError(
                        f'Camera "{name}" template camera '
                        f'"{template_name}" does not exist.'
                    )

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

        # LeRobotDataset.create() requires `root` to NOT already exist.
        # Keep the requested path on the first run; if it already exists,
        # record into a unique timestamped session directory instead.
        self.root = _resolve_dataset_root(Path(root))

        if self.root != Path(root).expanduser():
            print(
                f'Dataset root already exists: {Path(root).expanduser()}\n'
                f'Recording this run to: {self.root}'
            )

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=self.root,
            fps=fps,
            robot_type="unitree_g1_mujoco",
            features=features,
            use_videos=True,
        )

        self.next_record_time = 0.0

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm < 1e-9:
            raise ValueError("Camera direction vector has near-zero length.")
        return vector / norm

    def _apply_body_camera_pose(
        self,
        name: str,
        data: mujoco.MjData,
    ) -> None:
        """Attach a rendered camera pose to a moving MuJoCo body."""

        cfg = CAMERA_CONFIG[name]
        body_id = self._camera_body_ids[name]

        # Body pose in world coordinates.
        body_position = np.asarray(data.xpos[body_id], dtype=np.float64)
        body_rotation = np.asarray(
            data.xmat[body_id],
            dtype=np.float64,
        ).reshape(3, 3)

        local_position = np.asarray(
            cfg["position"],
            dtype=np.float64,
        )
        local_forward = np.asarray(
            cfg["forward"],
            dtype=np.float64,
        )
        local_up = np.asarray(
            cfg["up"],
            dtype=np.float64,
        )

        # Mount position and viewing axes follow the body every frame.
        camera_position = (
            body_position
            + body_rotation @ local_position
        )

        forward = self._normalized(
            body_rotation @ local_forward
        )

        up = body_rotation @ local_up

        # Ensure 'up' is exactly orthogonal to the view direction.
        up = up - forward * np.dot(up, forward)
        up = self._normalized(up)

        renderer = self.renderers[name]

        # MuJoCo keeps two OpenGL camera entries (left/right eye).  For our
        # monocular dataset both receive the exact same pose.
        for gl_camera in renderer.scene.camera:
            gl_camera.pos[:] = camera_position
            gl_camera.forward[:] = forward
            gl_camera.up[:] = up

    def should_record(self, sim_time: float) -> bool:
        """Downsample MuJoCo simulation to dataset FPS."""
        return sim_time + 1e-9 >= self.next_record_time

    def _render_camera(
        self,
        name: str,
        data: mujoco.MjData,
    ) -> np.ndarray:
        renderer = self.renderers[name]
        mount_cfg = CAMERA_CONFIG.get(name, {})
        mode = mount_cfg.get("mode", "model")

        if mode == "body":
            # First populate geoms/lights and inherit projection/FOV from an
            # existing model camera.  Then replace only the world camera pose.
            template_camera = mount_cfg.get(
                "template_camera",
                "front",
            )

            renderer.update_scene(
                data,
                camera=template_camera,
            )

            self._apply_body_camera_pose(
                name,
                data,
            )

        else:
            # Normal named MuJoCo camera, e.g. the external front view.
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
