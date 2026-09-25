import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from robot_env import ARM_JOINTS, CAMERA_GEOM_GROUPS, HEAD_CAMERA, LIDAR_RAYS, WRIST_CAMERA, build_model

CAMERAS = (HEAD_CAMERA, WRIST_CAMERA)
# get_depth normalises depth by this and clamps to [0, 1]; a miss renders as 0.
DEPTH_SCALE_M = 10.0


class MarsWarpEnv:
    """`nworld` MARS robots simulated in parallel on the GPU. Commands take a scalar (all
    worlds) or one value per world; each `step()` advances one control period."""

    def __init__(self, nworld: int = 1, control_hz: float = 50.0, cameras: bool = False) -> None:
        self.mj_model = build_model()
        self.model = mjw.put_model(self.mj_model)
        self.data = mjw.make_data(self.mj_model, nworld=nworld)
        self.nworld = nworld
        self.control_period = 1.0 / control_hz
        self.substeps = round(self.control_period / self.mj_model.opt.timestep)

        self.ctrl = np.zeros((nworld, self.mj_model.nu), dtype=np.float32)
        self._arm_ids = [self.mj_model.actuator(name).id for name in ARM_JOINTS]
        self._head_id = self.mj_model.actuator("joint_head").id
        self._forward_id = self.mj_model.actuator("base_forward").id
        self._yaw_id = self.mj_model.actuator("base_yaw").id
        first_beam = self.mj_model.sensor("lidar_0").adr[0]
        self.lidar_slice = slice(first_beam, first_beam + LIDAR_RAYS)

        self._step_graph: wp.Graph | None = None
        self._snapshot = mujoco.MjData(self.mj_model)
        self._render_context: mjw.RenderContext | None = None
        self._images: dict[str, tuple[wp.array, wp.array]] = {}
        if cameras:
            self._init_cameras()
        self.reset()

    def _init_cameras(self) -> None:
        self._render_context = mjw.create_render_context(
            self.mj_model, self.nworld, render_rgb=True, render_depth=True, enabled_geom_groups=CAMERA_GEOM_GROUPS
        )
        for name in CAMERAS:
            width, height = self.mj_model.cam_resolution[self.mj_model.camera(name).id]
            self._images[name] = (
                wp.zeros((self.nworld, height, width), dtype=wp.vec3),
                wp.zeros((self.nworld, height, width), dtype=float),
            )

    def reset(self) -> None:
        home = self.mj_model.key("home")
        mjw.reset_data_keyframe(self.model, self.data, home.id)
        self.ctrl[:] = home.ctrl
        mjw.forward(self.model, self.data)

    def drive(self, forward: float | np.ndarray, turn: float | np.ndarray) -> None:
        """Base velocity in m/s and rad/s, like /cmd_vel."""
        self.ctrl[:, self._forward_id] = forward
        self.ctrl[:, self._yaw_id] = turn

    def set_arm(self, angles: np.ndarray) -> None:
        """Joint targets (rad) for joint1..joint6, shape (6,) or (nworld, 6)."""
        self.ctrl[:, self._arm_ids] = angles

    def set_head(self, angle: float | np.ndarray) -> None:
        self.ctrl[:, self._head_id] = angle

    def step(self) -> None:
        self.data.ctrl.assign(self.ctrl)
        if self._step_graph is None:
            self._step_graph = self._capture_control_step()
        wp.capture_launch(self._step_graph)

    def _capture_control_step(self) -> wp.Graph:
        """Records one control period as a CUDA graph, removing per-kernel launch overhead.
        The lidar's rays cost several times the physics, so sensors run only on the last
        substep. The flag is read while recording, so the graph keeps both settings."""
        sensors_on = self.model.opt.disableflags
        with wp.ScopedCapture() as capture:
            self.model.opt.disableflags = sensors_on | mjw.DisableBit.SENSOR
            for _ in range(self.substeps - 1):
                mjw.step(self.model, self.data)
            self.model.opt.disableflags = sensors_on
            mjw.step(self.model, self.data)
        return capture.graph

    def snapshot(self, world_id: int = 0) -> mujoco.MjData:
        """Copy one world to the host, for MuJoCo's named accessors and the viewer. The
        returned MjData is reused by the next call."""
        mjw.get_data_into(self._snapshot, self.mj_model, self.data, world_id)
        return self._snapshot

    def render(self) -> None:
        if self._render_context is None:
            raise RuntimeError("MarsWarpEnv was created with cameras=False")
        mjw.render(self.model, self.data, self._render_context)
        for name, (rgb, depth) in self._images.items():
            camera_id = self.mj_model.camera(name).id
            mjw.get_rgb(self._render_context, camera_id, rgb)
            mjw.get_depth(self._render_context, camera_id, DEPTH_SCALE_M, depth)

    def rgb(self, camera: str) -> np.ndarray:
        """uint8 images from the last `render()`, shape (nworld, height, width, 3)."""
        return (self._images[camera][0].numpy() * 255).astype(np.uint8)

    def depth(self, camera: str) -> np.ndarray:
        """Depth along the optical axis (m) from the last `render()`; NaN where nothing
        was hit within DEPTH_SCALE_M."""
        depth = self._images[camera][1].numpy()
        return np.where((depth > 0) & (depth < 1), depth * DEPTH_SCALE_M, np.nan)
