from warp_env import MarsWarpEnv


mwe = MarsWarpEnv(nworld=512)

for i in range(100):
    print(f"step {i}")
    mwe.step()

