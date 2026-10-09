"""Headless, ROS-free batched stepping for data collection: N parallel worlds, no
visualization, no user control. Run via `mjwarp/run.sh -n <N>`, not directly."""

import sys

from warp_env import MarsWarpEnv


def main() -> None:
    nworld = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    mwe = MarsWarpEnv(nworld)
    print(f"Running {nworld} worlds headlessly. Ctrl+C to stop.")
    while True:
        mwe.step()
        # TODO: data collection/storage goes here once the task and reward are defined.


if __name__ == "__main__":
    main()
