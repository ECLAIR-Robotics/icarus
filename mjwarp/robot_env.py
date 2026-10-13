import math

import mujoco

import _driver_pkg
from mars_sim_driver import world

# Plane at z=0, which is the chassis contact height. The robot drives on it.
_SCENE_XML = """
<mujoco model="mjwarp_cartpole">
    <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10"/>
    <worldbody>
        <light type="directional" pos="0 0 3" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
        <geom name="ground" type="plane" size="20 20 0.1" friction="0.9 0.01 0.001"/>
    </worldbody>
</mujoco>
"""

# Hold the arm and head so the only free motion is the pole.
_HOLD_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint6M", "joint_head")
_DRIVE_FORCE_N = 20.0
# Above the head (head origin is z=0.26). The capsule runs 0.7 m up from the pivot.
_POLE_PIVOT_Z = 0.32
_POLE_LENGTH = 0.7


def add_drive_base(robot_spec: mujoco.MjSpec) -> None:
    """Planar base (x, y, yaw) with force motors on x and y, so the chassis can drive on the plane."""
    base_body = robot_spec.body("base_link")
    for name, jtype, axis in (
        ("base_x", mujoco.mjtJoint.mjJNT_SLIDE, (1.0, 0.0, 0.0)),
        ("base_y", mujoco.mjtJoint.mjJNT_SLIDE, (0.0, 1.0, 0.0)),
        ("base_yaw", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 0.0, 1.0)),
    ):
        joint = base_body.add_joint()
        joint.name = name
        joint.type = jtype
        joint.axis = axis
        joint.damping[0] = 1.0 if jtype == mujoco.mjtJoint.mjJNT_SLIDE else 0.2
    for name, target in (("drive_x", "base_x"), ("drive_y", "base_y")):
        actuator = robot_spec.add_actuator(name=name, target=target)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.set_to_motor()
        actuator.gear[0] = _DRIVE_FORCE_N
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        actuator.ctrlrange[:] = [-1.0, 1.0]


def add_pole(robot_spec: mujoco.MjSpec) -> None:
    """Unactuated pole on a ball joint. Upright is the unstable cartpole equilibrium."""
    pole = robot_spec.body("base_link").add_body(name="pole", pos=[0.0, 0.0, _POLE_PIVOT_Z])
    joint = pole.add_joint()
    joint.name = "pole"
    joint.type = mujoco.mjtJoint.mjJNT_BALL
    joint.damping[:] = [0.01, 0.01, 0.01]
    geom = pole.add_geom()
    geom.name = "pole"
    geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    geom.fromto[:] = [0.0, 0.0, 0.0, 0.0, 0.0, _POLE_LENGTH]
    geom.size[0] = 0.015
    geom.mass = 0.15
    geom.rgba[:] = [0.85, 0.15, 0.1, 1.0]
    geom.contype = 0
    geom.conaffinity = 0


def hold_arm(robot_spec: mujoco.MjSpec) -> None:
    for name in _HOLD_JOINTS:
        robot_spec.joint(name).stiffness[0] = 40.0


def build_model() -> mujoco.MjModel:
    world_spec = mujoco.MjSpec.from_string(_SCENE_XML)
    robot_spec = world.load_robot_spec(world.default_urdf_path())
    add_drive_base(robot_spec)
    hold_arm(robot_spec)
    add_pole(robot_spec)
    world.tune_contacts(robot_spec)
    world_spec.attach(robot_spec, frame=world_spec.worldbody.add_frame(), prefix="robot_")
    model = world_spec.compile()
    world.style_robot_geoms(model)
    _mark_pole(model)
    return model


def _mark_pole(model: mujoco.MjModel) -> None:
    """Bright pole, starting a few degrees off vertical so it falls unless driven."""
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_pole")
    model.geom_rgba[geom] = [0.95, 0.45, 0.1, 1.0]
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_pole")
    qadr = int(model.jnt_qposadr[joint])
    tilt = 0.12
    model.qpos0[qadr : qadr + 4] = [math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0]
