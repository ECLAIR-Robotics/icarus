"""The balancing pole: an unactuated rod on a ball joint above the robot's head, plus the
numpy helpers for observing and scoring it. Upright is the unstable equilibrium; the base
drive (see MarsWarpEnv.drive) is what's meant to keep it there. policy.py is the learned
controller's network; it consumes OBS_DIM observations and emits ACTION_DIM actions."""

import mujoco
import numpy as np

POLE_BODY = "pole"
# Above the head (head origin is z=0.26). The capsule runs POLE_LENGTH up from the pivot.
POLE_PIVOT_Z = 0.32
POLE_LENGTH = 0.7
POLE_RADIUS = 0.015

_RESET_TILT = 0.05
# 5 steps of the 2 ms integrator: one 10 ms drive command.
N_SUBSTEPS = 5
# Pole past this angle from vertical has fallen.
_POLE_DONE = 0.5

# obs: pole angular speed, angle from vertical
OBS_DIM = 2
# action: normalized (forward, turn), scaled to the base's speed limits by MarsWarpEnv
ACTION_DIM = 2


def add_pole(robot_spec: mujoco.MjSpec) -> None:
    pole = robot_spec.body("base_link").add_body(name=POLE_BODY, pos=[0.0, 0.0, POLE_PIVOT_Z])
    joint = pole.add_joint()
    joint.name = POLE_BODY
    joint.type = mujoco.mjtJoint.mjJNT_BALL
    joint.damping[:] = [0.01, 0.01, 0.01]
    geom = pole.add_geom()
    geom.name = POLE_BODY
    geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    geom.fromto[:] = [0.0, 0.0, 0.0, 0.0, 0.0, POLE_LENGTH]
    geom.size[0] = POLE_RADIUS
    geom.mass = 0.15
    geom.rgba[:] = [0.95, 0.45, 0.1, 1.0]
    geom.contype = 0
    geom.conaffinity = 0


def hide_pole_from_cameras(model: mujoco.MjModel, hidden_group: int) -> None:
    model.geom_group[model.geom("robot_pole").id] = hidden_group


def joint_addrs(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    """(qpos address, dof address) of a joint."""
    joint = model.joint(name)
    return int(joint.qposadr[0]), int(joint.dofadr[0])


def tilt_pole(qpos: np.ndarray, qadr: int, rng: np.random.Generator, mask: np.ndarray) -> None:
    """Start the masked worlds' poles a random few degrees off vertical, in place."""
    idx = np.flatnonzero(mask)
    tilt = rng.uniform(0.0, _RESET_TILT, size=idx.size)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=idx.size)
    half = 0.5 * tilt
    qpos[idx, qadr] = np.cos(half)
    qpos[idx, qadr + 1] = np.sin(half) * np.cos(phi)
    qpos[idx, qadr + 2] = np.sin(half) * np.sin(phi)
    qpos[idx, qadr + 3] = 0.0


def pole_axis(quat: np.ndarray) -> np.ndarray:
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


def tilt_angle(axis: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(axis[:, 2], -1.0, 1.0))


def observe(qpos: np.ndarray, qvel: np.ndarray, qadr: int, dadr: int) -> np.ndarray:
    axis = pole_axis(qpos[:, qadr : qadr + 4])
    # Ball joint has no single signed rate, so this is the speed of the pole.
    rate = np.linalg.norm(qvel[:, dadr : dadr + 3], axis=1)
    obs = np.empty((qpos.shape[0], OBS_DIM), dtype=np.float32)
    obs[:, 0] = rate
    obs[:, 1] = tilt_angle(axis)
    return obs


def score(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(reward, done): 1 per step while the pole is up, 0 once it has fallen."""
    done = obs[:, 1] > _POLE_DONE
    return np.where(done, np.float32(0.0), np.float32(1.0)), done
