import mujoco
import mujoco_warp as mjw
import numpy as np

from robot_env import build_model

_RESET_TILT = 0.05
_FALL_COST = 20.0
_ALIVE_BONUS = 1.0

# obs: pole axis (3), pole angular velocity (3), x, y, vx, vy, drive xy
_OBS_DIM = 12
_ACTION_DIM = 2


class MarsWarpEnv:
    """Batched cartpole. The action is the normalized drive force in x and y."""

    def __init__(self, nworld: int) -> None:
        self.mj_model = build_model()
        self.model = mjw.put_model(self.mj_model)
        self.data = mjw.make_data(self.mj_model, nworld=nworld)
        self.nworld = nworld
        self._rng = np.random.default_rng()
        self._x = _joint_addrs(self.mj_model, "robot_base_x")
        self._y = _joint_addrs(self.mj_model, "robot_base_y")
        self._pole = _joint_addrs(self.mj_model, "robot_pole")
        self._prev_action = np.zeros((nworld, _ACTION_DIM), dtype=np.float32)
        self.reset()

    def reset(self) -> np.ndarray:
        mjw.reset_data(self.model, self.data)
        qpos = self.data.qpos.numpy()
        qvel = self.data.qvel.numpy()
        self._tilt_pole(qpos, np.ones(self.nworld, dtype=bool))
        qvel[:] = 0.0
        self.data.qpos.assign(qpos)
        self.data.qvel.assign(qvel)
        self._prev_action[:] = 0.0
        return self._observe(qpos, qvel, self._prev_action)

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

    def _observe(self, qpos: np.ndarray, qvel: np.ndarray, action: np.ndarray) -> np.ndarray:
        obs = np.empty((self.nworld, _OBS_DIM), dtype=np.float32)
        qadr, dadr = self._pole
        obs[:, 0:3] = _pole_axis(qpos[:, qadr : qadr + 4])
        obs[:, 3:6] = qvel[:, dadr : dadr + 3]
        obs[:, 6] = qpos[:, self._x[0]]
        obs[:, 7] = qpos[:, self._y[0]]
        obs[:, 8] = qvel[:, self._x[1]]
        obs[:, 9] = qvel[:, self._y[1]]
        obs[:, 10:12] = action
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


def _reward(
    tilt: np.ndarray,
    pole_rate: np.ndarray,
    cart: np.ndarray,
    action: np.ndarray,
    prev_action: np.ndarray,
    done: np.ndarray,
) -> np.ndarray:
    cost = (
        4.0 * tilt**2
        + 0.05 * np.sum(pole_rate**2, axis=1)
        + 0.1 * np.sum(cart**2, axis=1)
        + 0.01 * np.sum(action**2, axis=1)
        + 0.05 * np.sum((action - prev_action) ** 2, axis=1)
    )
    bonus = np.where(done, -_FALL_COST, _ALIVE_BONUS)
    return -cost + bonus
