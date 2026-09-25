import mujoco
import numpy as np

import _driver_pkg  # noqa: F401 -- puts mars_sim_driver on sys.path
from mars_sim_driver import world
from mars_sim_driver.constants import (
    CAMERA_FX,
    CAMERA_FY,
    CAMERA_HEIGHT,
    CAMERA_PRINCIPAL_PIXEL,
    CAMERA_WIDTH,
    WRIST_CAMERA_FOVY,
)

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
DRIVEN_JOINTS = [*ARM_JOINTS, "joint_head"]
GRIPPER_JOINT = "joint6"

# Innate's arm servo (mars_sim_driver/core.py).
KP_JOINT = 50.0
KD_JOINT = 1.0
EFFORT_LIMIT = 50.0
GRIPPER_EFFORT_LIMIT = 2.0

LIDAR_RAYS = 360
HEAD_CAMERA = "head"
WRIST_CAMERA = "wrist"
# The viewer draws groups 0-2; the robot's cameras draw only these (see _hide_head_from_cameras).
CAMERA_GEOM_GROUPS = [0, 1]
_CAMERA_HIDDEN_GROUP = 2

# A walled 6x6 m room with a few obstacles, so the lidar and SLAM have something to see.
_SCENE_XML = """
<mujoco model="mars_arena">
    <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10"/>
    <default>
        <geom rgba="0.7 0.7 0.75 1"/>
    </default>
    <worldbody>
        <light type="directional" pos="0 0 3" dir="0.3 0.5 -1" diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35"/>
        <geom name="ground" type="plane" size="20 20 0.1" friction="0.9 0.01 0.001" rgba="0.35 0.35 0.35 1"/>
        <geom name="wall_north" type="box" pos="0 3 0.5" size="3 0.05 0.5"/>
        <geom name="wall_south" type="box" pos="0 -3 0.5" size="3 0.05 0.5"/>
        <geom name="wall_east" type="box" pos="3 0 0.5" size="0.05 3 0.5"/>
        <geom name="wall_west" type="box" pos="-3 0 0.5" size="0.05 3 0.5"/>
        <geom name="crate" type="box" pos="1.5 1.2 0.25" size="0.25 0.25 0.25" rgba="0.6 0.4 0.2 1"/>
        <geom name="pillar" type="cylinder" pos="-1.4 -1.0 0.5" size="0.15 0.5" rgba="0.3 0.5 0.7 1"/>
        <geom name="divider" type="box" pos="-1.0 1.8 0.4" size="0.8 0.05 0.4" euler="0 0 30"/>
    </worldbody>
</mujoco>
"""


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_string(_SCENE_XML)
    robot_spec = world.load_robot_spec(world.default_urdf_path())
    world.add_planar_base(robot_spec)
    world.tune_contacts(robot_spec)
    spec.attach(robot_spec, frame=spec.worldbody.add_frame(), prefix="robot_")

    _add_joint_servos(spec)
    _add_base_drive(spec)
    _add_encoders(spec)
    _add_odometry(spec)
    _add_lidar(spec)
    _add_cameras(spec)
    spec.add_key(name="home")

    model = spec.compile()
    world.style_robot_geoms(model)
    _hide_head_from_cameras(model)
    _set_home_pose(model)
    return model


def _add_joint_servos(spec: mujoco.MjSpec) -> None:
    for name in DRIVEN_JOINTS:
        is_gripper = name == GRIPPER_JOINT
        limit = GRIPPER_EFFORT_LIMIT if is_gripper else EFFORT_LIMIT
        servo = spec.add_actuator(name=name, target=f"robot_{name}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        # The finger's tiny inertia makes a kv term unstable; world.FINGER_DAMPING damps it instead.
        servo.set_to_position(kp=KP_JOINT, kv=0.0 if is_gripper else KD_JOINT)
        servo.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        servo.forcerange = [-limit, limit]
        servo.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        servo.ctrlrange = spec.joint(f"robot_{name}").range


def _add_base_drive(spec: mujoco.MjSpec) -> None:
    """Body-frame velocity servos, equivalent to Innate's cmd_vel controller. The lateral
    one always holds 0: it stands in for the wheels' sideways grip."""
    spec.body("robot_base_link").add_site(name="base_drive")
    max_linear, max_angular = world.MAX_BASE_LINEAR_SPEED, world.MAX_BASE_ANGULAR_SPEED
    for name, gear, gain, limit in (
        ("base_forward", [1, 0, 0, 0, 0, 0], world.KP_FORWARD, max_linear),
        ("base_lateral", [0, 1, 0, 0, 0, 0], world.KP_LATERAL, max_linear),
        ("base_yaw", [0, 0, 0, 0, 0, 1], world.KP_YAW, max_angular),
    ):
        drive = spec.add_actuator(name=name, target="base_drive", trntype=mujoco.mjtTrn.mjTRN_SITE)
        drive.set_to_velocity(kv=gain)
        drive.gear = gear
        drive.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        drive.ctrlrange = [-limit, limit]


def _add_encoders(spec: mujoco.MjSpec) -> None:
    for name in DRIVEN_JOINTS:
        for suffix, sensor_type in (
            ("pos", mujoco.mjtSensor.mjSENS_JOINTPOS),
            ("vel", mujoco.mjtSensor.mjSENS_JOINTVEL),
        ):
            spec.add_sensor(
                name=f"{name}_{suffix}", type=sensor_type, objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=f"robot_{name}"
            )


def _add_odometry(spec: mujoco.MjSpec) -> None:
    """Ground-truth base pose (world frame) and twist (base frame), as Innate's sim reports it."""
    spec.body("robot_base_link").add_site(name="odom")
    for name, sensor_type in (
        ("odom_pos", mujoco.mjtSensor.mjSENS_FRAMEPOS),
        ("odom_quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT),
        ("odom_linear_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
        ("odom_angular_vel", mujoco.mjtSensor.mjSENS_GYRO),
    ):
        spec.add_sensor(name=name, type=sensor_type, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="odom")


def _add_lidar(spec: mujoco.MjSpec) -> None:
    """One rangefinder per beam, CCW from -pi like a LaserScan. The beams sit on base_link
    (at base_laser's offset) because a rangefinder ignores its own body: that keeps the
    chassis out of the scan, as Innate's lidar does."""
    base = spec.body("robot_base_link")
    laser_pos = spec.body("robot_base_laser").pos
    for i, angle in enumerate(np.linspace(-np.pi, np.pi, LIDAR_RAYS, endpoint=False)):
        beam = base.add_site(name=f"lidar_{i}", pos=laser_pos)
        beam.alt.type = mujoco.mjtOrientation.mjORIENTATION_ZAXIS
        beam.alt.zaxis = [np.cos(angle), np.sin(angle), 0.0]
        rangefinder = spec.add_sensor(
            name=f"lidar_{i}",
            type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=beam.name,
        )
        rangefinder.intprm[0] = 1 << int(mujoco.mjtRayDataField.mjRAYDATA_DIST)


def _add_cameras(spec: mujoco.MjSpec) -> None:
    # MuJoCo cameras look down -z with +y up; xyaxes gives the image's right and up axes in the body frame.
    head = spec.body("robot_camera_optical_frame").add_camera(name=HEAD_CAMERA)
    head.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    head.alt.xyaxes = [1, 0, 0, 0, -1, 0]
    # The real head lens: anisotropic focal length and an off-centre principal point.
    head.resolution = [CAMERA_WIDTH, CAMERA_HEIGHT]
    head.focal_pixel = [CAMERA_FX, CAMERA_FY]
    head.principal_pixel = list(CAMERA_PRINCIPAL_PIXEL)
    head.sensor_size = [0.0064, 0.0048]  # any size: focal_pixel overrides it

    wrist = spec.body("robot_arm_camera_link").add_camera(name=WRIST_CAMERA, fovy=WRIST_CAMERA_FOVY)
    wrist.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    wrist.alt.xyaxes = [0, -1, 0, 0, 0, 1]
    wrist.resolution = [CAMERA_WIDTH, CAMERA_HEIGHT]


def _hide_head_from_cameras(model: mujoco.MjModel) -> None:
    """mujoco_warp's ray-traced cameras have no near clip, so the head camera would see
    the inside of its own housing."""
    for body in ("robot_head", "robot_head_camera_left", "robot_head_camera_right"):
        visual = (model.geom_bodyid == model.body(body).id) & (model.geom_group == world.VISUAL_GROUP)
        model.geom_group[visual] = _CAMERA_HIDDEN_GROUP


def _set_home_pose(model: mujoco.MjModel) -> None:
    home = model.key("home")
    for name, angle in world.ARM_HOME.items():
        home.qpos[model.joint(f"robot_{name}").qposadr] = angle
        home.ctrl[model.actuator(name).id] = angle
    mimic, source, ratio = world.MIMIC_JOINT
    home.qpos[model.joint(f"robot_{mimic}").qposadr] = ratio * world.ARM_HOME[source]
