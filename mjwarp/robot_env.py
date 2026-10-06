import mujoco

import _driver_pkg
from mars_sim_driver import world

_GROUND_XML = """
<mujoco model="mjwarp_ground">
    <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10"/>
    <worldbody>
        <light type="directional" pos="0 0 3" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
        <geom name="ground" type="plane" size="20 20 0.1" friction="0.9 0.01 0.001"/>
    </worldbody>
</mujoco>
"""

# Matches sim/sandbox/drive_mars.py's own placeholder servo feel (world.py's DRIVEN_JOINTS).
ARM_HEAD_JOINTS =  [f"joint{i}" for i in range(1, 7)] + ["joint_head"]
KP_JOINT = 8.0
KD_JOINT = 0.6

BASE_JOINTS = ["base_x", "base_y", "base_yaw"]


def _add_actuators(robot_spec: mujoco.MjSpec) -> None:
    """Position actuators on the arm/head (the real hardware is position-servoed) and
    motor actuators on the planar base (force-driven, PD-tracked in ros_bridge.py the
    same way apply_drive_force does for the sandbox's test robot)."""
    for name in ARM_HEAD_JOINTS:
        act = robot_spec.add_actuator()
        act.name = f"{name}_act"
        act.target = name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_position(kp=KP_JOINT, kv=KD_JOINT)

    for name in BASE_JOINTS:
        act = robot_spec.add_actuator()
        act.name = f"{name}_act"
        act.target = name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_motor()


def build_model() -> mujoco.MjModel:
    world_spec = mujoco.MjSpec.from_string(_GROUND_XML)
    robot_spec = world.load_robot_spec(world.default_urdf_path())
    world.add_planar_base(robot_spec)
    world.tune_contacts(robot_spec)
    _add_actuators(robot_spec)
    world_spec.attach(
        robot_spec, frame=world_spec.worldbody.add_frame(), prefix="robot_")
    model = world_spec.compile()
    world.style_robot_geoms(model)
    return model
