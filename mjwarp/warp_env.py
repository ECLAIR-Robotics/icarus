import mujoco
import mujoco_warp as mjw
import numpy as np

from robot_env import build_model

_RESET_TILT = 0.05
# 5 steps of the 2 ms integrator: one 10 ms drive command.
_N_SUBSTEPS = 5
# Pole past this angle from vertical has fallen.
_POLE_DONE = 0.5

# obs: pole angular speed, angle from vertical
_OBS_DIM = 2


class MarsWarpEnv:
    """Batched cartpole. The action is the normalized drive force in x and y."""

    def __init__(self, nworld: int, seed: int | None = None) -> None:
        self.mj_model = build_model()
        self.model = mjw.put_model(self.mj_model)
        self.data = mjw.make_data(self.mj_model, nworld=nworld)
        self.nworld = nworld
        self._rng = np.random.default_rng(seed)
        self._pole = _joint_addrs(self.mj_model, "robot_pole")
        self.reset()

    def reset(self) -> np.ndarray:
        mjw.reset_data(self.model, self.data)
        qpos = self.data.qpos.numpy()
        qvel = self.data.qvel.numpy()
        self._tilt_pole(qpos, np.ones(self.nworld, dtype=bool))
        qvel[:] = 0.0
        self.data.qpos.assign(qpos)
        self.data.qvel.assign(qvel)
        return self._observe(qpos, qvel)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hold one drive command for 10 ms. Returns obs, reward, and which worlds fell."""
        ctrl = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if ctrl.shape != (self.nworld, 2):
            raise ValueError(
                f"action must have shape ({self.nworld}, 2), got {ctrl.shape}"
            )
        self.data.ctrl.assign(ctrl)
        for _ in range(_N_SUBSTEPS):
            mjw.step(self.model, self.data)
        qpos = self.data.qpos.numpy()
        qvel = self.data.qvel.numpy()
        obs = self._observe(qpos, qvel)
        done = obs[:, 1] > _POLE_DONE
        reward = np.where(done, np.float32(0.0), np.float32(1.0))
        if np.any(done):
            qpos[done] = self.mj_model.qpos0
            self._tilt_pole(qpos, done)
            qvel[done] = 0.0
            self.data.qpos.assign(qpos)
            self.data.qvel.assign(qvel)
            obs = self._observe(qpos, qvel)
        return obs, reward, done

    def _tilt_pole(self, qpos: np.ndarray, mask: np.ndarray) -> None:
        idx = np.flatnonzero(mask)
        tilt = self._rng.uniform(0.0, _RESET_TILT, size=idx.size)
        phi = self._rng.uniform(0.0, 2.0 * np.pi, size=idx.size)
        half = 0.5 * tilt
        qadr = self._pole[0]
        qpos[idx, qadr] = np.cos(half)
        qpos[idx, qadr + 1] = np.sin(half) * np.cos(phi)
        qpos[idx, qadr + 2] = np.sin(half) * np.sin(phi)
        qpos[idx, qadr + 3] = 0.0

    def _observe(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        qadr, dadr = self._pole
        axis = _pole_axis(qpos[:, qadr : qadr + 4])
        # Ball joint has no single signed rate, so this is the speed of the pole.
        rate = np.linalg.norm(qvel[:, dadr : dadr + 3], axis=1)
        obs = np.empty((self.nworld, _OBS_DIM), dtype=np.float32)
        obs[:, 0] = rate
        obs[:, 1] = _tilt_angle(axis)
        return obs


def _joint_addrs(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])


def _pole_axis(quat: np.ndarray) -> np.ndarray:
    """Pole +z axis in the base frame. Upright is (0, 0, 1)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.stack(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=1,
    )


def _tilt_angle(axis: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(axis[:, 2], -1.0, 1.0))
