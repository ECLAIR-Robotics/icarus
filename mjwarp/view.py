"""Drive one simulated MARS in the MuJoCo viewer: arrow keys change the base velocity,
space stops it."""

import time

import glfw
import mujoco.viewer

from warp_env import MarsWarpEnv

SPEED_STEP = 0.1  # m/s per key press
TURN_STEP = 0.3  # rad/s per key press


class KeyboardTeleop:
    def __init__(self) -> None:
        self.forward = 0.0
        self.turn = 0.0

    def on_key(self, key: int) -> None:
        match key:
            case glfw.KEY_UP:
                self.forward += SPEED_STEP
            case glfw.KEY_DOWN:
                self.forward -= SPEED_STEP
            case glfw.KEY_LEFT:
                self.turn += TURN_STEP
            case glfw.KEY_RIGHT:
                self.turn -= TURN_STEP
            case glfw.KEY_SPACE:
                self.forward = self.turn = 0.0


def main() -> None:
    env = MarsWarpEnv()
    teleop = KeyboardTeleop()
    with mujoco.viewer.launch_passive(env.mj_model, env.snapshot(), key_callback=teleop.on_key) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            env.drive(teleop.forward, teleop.turn)
            env.step()
            env.snapshot()
            viewer.sync()
            time.sleep(max(0.0, env.control_period - (time.perf_counter() - start)))


if __name__ == "__main__":
    main()
