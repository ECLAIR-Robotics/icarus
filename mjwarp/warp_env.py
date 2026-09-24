import mujoco_warp as mjw

from robot_env import build_model

class MarsWarpEnv: 
    def __init__(self, nworld: int):
        self.mj_model = build_model()
        self.model = mjw.put_model(self.mj_model)
        self.data = mjw.make_data(self.mj_model, nworld=nworld)
        self.nworld = nworld

    def step(self):
        mjw.step(self.model, self.data)
          