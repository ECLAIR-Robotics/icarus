import warp as wp
import mujoco_warp as mjw

from robot_env import build_model

# put_model/make_data/step all run on Warp's current *global* device, not a per-call
# argument -- so this has to run once, before any MarsWarpEnv exists, not inside __init__.
if not wp.is_cuda_available():
    print("No CUDA-capable GPU found -- falling back to Warp's CPU backend (correct, much slower).")
    wp.set_device("cpu")


class MarsWarpEnv:
    def __init__(self, nworld: int):
        self.mj_model = build_model()
        self.model = mjw.put_model(self.mj_model)
        self.data = mjw.make_data(self.mj_model, nworld=nworld)
        self.nworld = nworld

    def step(self):
        mjw.step(self.model, self.data)
          