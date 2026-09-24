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

def build_model() -> mujoco.MjModel:
    world_spec = mujoco.MjSpec.from_string(_GROUND_XML)
    robot_spec = world.load_robot_spec(world.default_urdf_path())
    world.add_planar_base(robot_spec)
    world.tune_contacts(robot_spec)
    world_spec.attach(robot_spec, frame=world_spec.worldbody.add_frame(), prefix="robot_")
    model = world_spec.compile()
    world.style_robot_geoms(model)
    return model