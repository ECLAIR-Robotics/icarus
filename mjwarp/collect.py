"""Headless, ROS-free batched stepping for data collection: N parallel worlds, no
visualization, no user control. Run via `mjwarp/run.sh -n <N> [-p]`, not directly."""

import sys

from warp_env import MarsWarpEnv


def main() -> None:
    args = [arg for arg in sys.argv[1:] if arg != "--pole"]
    pole = "--pole" in sys.argv[1:]
    nworld = int(args[0]) if args else 1024
    mwe = MarsWarpEnv(nworld, pole=pole)
    print(f"Running {nworld} worlds headlessly{' with the pole' if pole else ''}. Ctrl+C to stop.")
    while True:
        mwe.step()
        # TODO: data collection/storage goes here once the task and reward are defined.


if __name__ == "__main__":
    main()
