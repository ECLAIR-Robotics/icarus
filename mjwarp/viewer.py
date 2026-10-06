import time
import mujoco
import mujoco.viewer

from warp_env import MarsWarpEnv

mwe = MarsWarpEnv(1024)

mirror_data = mujoco.MjData(mwe.mj_model)
dt = mwe.mj_model.opt.timestep

with mujoco.viewer.launch_passive(mwe.mj_model, mirror_data) as viewer:
    i = 0
    while viewer.is_running():
        i += 1
        print(f"Step {i}")
        mwe.step()

        mirror_data.qpos[:] = mwe.data.qpos.numpy()[0]
        mirror_data.qvel[:] = mwe.data.qvel.numpy()[0]
        mujoco.mj_forward(mwe.mj_model, mirror_data)

        viewer.sync()
        time.sleep(dt)