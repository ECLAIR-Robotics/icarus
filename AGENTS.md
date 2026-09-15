# ICARUS — Agent Reference

ECLAIR Robotics' ICARUS project: software for Innate's **MARS** robot (a small mobile
base with an arm, running an NVIDIA Jetson Orin Nano 8GB). ICARUS is built **on top of**
Innate OS, not as a fork of it.

The project is at an early stage: the only code so far is a skeleton ROS 2 node.

## Repository layout

```
icarus/                  Python package: ROS 2 nodes (currently icarus_node.py, a stub)
launch/icarus.launch.py  Launches icarus_node
resource/icarus          ament index marker (required by setup.py; keep it)
package.xml, setup.py,   The repo root IS the `icarus` ament_python ROS 2 package
setup.cfg
scripts/setup.sh         One-time setup: laptop (Docker/sim/dev image) or `--robot`
scripts/build.sh         Builds icarus as a colcon overlay; same command everywhere
scripts/docker-build.sh  Builds the icarus-dev image (host)
scripts/docker-shell.sh  Starts/joins the dev container (host)
docker/Dockerfile        icarus-dev image, FROM Innate's sim dependency image
docker/docker-compose.yml  Runs the dev container (+ attaches to a running sim)
innate-os/               Git submodule: Innate's OS (third-party, pinned; see below)
```

## Stack

- **ROS 2 Humble** on Ubuntu 22.04, Python 3.10. `setuptools==59.6.0` is pinned on
  purpose: ament_python on Humble breaks on newer setuptools.
- **Zenoh** (`rmw_zenoh_cpp`) is the ROS middleware, not FastDDS/Cyclone. Environment
  comes from `innate-os/config/dds/setup_dds.zsh`; `ROS_DOMAIN_ID=0`. A process can
  only see the robot's topics if it reaches the same Zenoh router.
- **Innate OS** provides the robot runtime: drivers, Nav2, arm IK, cameras, the
  `brain_client` agent loop, skills, and the `innate` CLI.

## innate-os submodule

`innate-os/` is ECLAIR's fork (https://github.com/ECLAIR-Robotics/innate-os) of Innate
Inc's repository (https://github.com/innate-inc/innate-os), pinned to a specific commit. **Treat it as read-only**: don't edit files in it; put ICARUS
code in this repo. Update the pin deliberately (`git -C innate-os fetch && git -C
innate-os checkout <ref>`, then commit the submodule bump) and re-run `scripts/docker-build.sh`,
because the dev image's base is derived from files in the submodule.

Read these before working with Innate APIs: `innate-os/AGENTS.md` (the `innate` CLI, ROS
package map, skill cancellation rules), `innate-os/CLAUDE.md` (their code style),
`innate-os/README.md` (skills, agents, inputs), `innate-os/sim/README.md` (simulator),
`innate-os/docs/SYSTEM_OVERVIEW.md` (architecture).

Things from Innate that matter when writing ICARUS code:

- **Skills** (`Skill` subclasses) and **agents** are the intended extension points, and
  they are plain Python loaded from `innate-os/workspace/`. In skill code use
  `self.sleep()`, never `time.sleep()`: only `self.sleep()` can be cancelled by Stop.
- **Avoid adding ROS nodes.** The Jetson already runs 20+ and has little RAM. Prefer adding
  a timer or callback to an existing node, or loading into a composable container.
- Useful ROS packages: `mars_control`, `mars_bringup`, `mars_arm`, `mars_cam`, `mars_nav`
  (Nav2 + SLAM), `brain_client`, `manipulation`.

## Environments

| Where | How | innate-os comes from |
|---|---|---|
| Dev container | `scripts/docker-shell.sh` | the submodule, at `/root/innate-os` |
| Simulator | `cd innate-os && ./innate-sim up` (host) | the submodule, in Innate's own container |
| Robot | native, after `scripts/setup.sh --robot` | the robot's `~/innate-os` |

### Dev container

- `scripts/docker-build.sh` builds the image (after a submodule bump or Dockerfile edit).
  `scripts/docker-shell.sh` starts one long-lived container and every terminal joins it;
  `scripts/docker-shell.sh <cmd>` runs a single command; `--restart` recreates it and
  `--stop` removes it (build volumes survive). The shell opens in the directory the script
  was run from: inside this repo that is the matching path under `/root/icarus`; outside
  it, the folder is mounted at its host path (recreating the container if it isn't yet).
- The image is `docker/Dockerfile`, built on `ghcr.io/innate-inc/innate-os-sim-deps:deps-<hash>`.
  That is the same apt/pip stack as the simulator, and the hash comes from
  `python3 innate-os/sim/launcher/config.py deps-image-hash`. `scripts/docker-build.sh` resolves it
  and writes it to `docker/.env`, which is gitignored and machine-specific.
- The container mounts: this repo at `/root/icarus`, and `/root/innate-os` as a symlink to
  the submodule. Build output goes to named volumes: `/root/icarus_ws` for the ICARUS
  overlay, and `innate-os/ros2_ws/{build,install,log}` for the Innate underlay. Source is
  bind-mounted, so editing code never requires rebuilding the image.
- Tools included: colcon, rosdep, ccache, gdb, clangd, ruff, basedpyright, pytest, and
  ipython. There are no GUI tools; use Foxglove at `ws://localhost:8765` while the sim runs.

### Simulator

The sim is Innate's MuJoCo digital twin. It runs the real robot software in Docker, and
the physics world server runs natively on the host. It is started only through Innate's
launcher (`./innate-sim up|status|logs|sh|down`), and its web UI is at https://localhost.

**Limitation:** the launcher's container mounts only paths inside `innate-os/`, so it
cannot see ICARUS code. To run ICARUS nodes against the sim, start the dev container in the
sim container's network namespace:

```bash
scripts/docker-shell.sh --sim
```

This joins the network of `innate-dev-<id>` (the name is in `docker/.env`). If the sim was
restarted, the script recreates the dev container so it attaches to the new one.

This works for ROS nodes and topics. Skills and agents run inside `brain_client` in the sim
container. To try them in sim today, copy them into `innate-os/workspace/custom_skills/`,
which is gitignored in the submodule. They load natively on the robot (see below).

### Robot

`scripts/setup.sh --robot` runs rosdep for this package, builds the overlay into
`~/icarus_ws` against `~/innate-os/ros2_ws/install`, and adds a line to `~/.zshrc` that
sources the overlay. It warns if the robot's innate-os commit differs from the submodule
pin. After pulling changes, run `scripts/build.sh`. The robot has no Docker; the robot's
shell already sets up Zenoh.

## Build, run, verify

```bash
scripts/build.sh               # colcon build --symlink-install --paths <repo> into $ICARUS_WS
scripts/build.sh --underlay    # also build innate-os/ros2_ws first (slow; once per container volume)
scripts/build.sh --clean       # wipe the overlay first
source $ICARUS_WS/install/setup.zsh
ros2 launch icarus icarus.launch.py
```

- Colcon uses `--paths`, never `--base-paths`. `--base-paths` would recurse into the
  submodule and build all of Innate OS as part of the overlay.
- `--symlink-install` means Python edits take effect without rebuilding. After adding a
  file, entry point, or launch file, rebuild.
- New console scripts go in `setup.py` `entry_points`; new launch files go in `launch/`,
  which `setup.py` installs. Dependencies go in `package.xml` so rosdep can resolve them on
  the robot.
- Lint and test inside the dev container: `ruff check . && ruff format --check .` and
  `python3 -m pytest`. A `test/` directory does not exist yet; `package.xml` already
  declares the ament flake8/pep257/pytest test dependencies.

## Conventions

- Follow `innate-os/CLAUDE.md` style for Python that uses Innate APIs: full type hints,
  early returns, comments only for why, and narrow `try`/`except`.
- Never commit `docker/.env`, `.env`, or colcon `build/ install/ log/` (all gitignored).
- Package maintainer (in `package.xml`): Manas. GitHub: `ECLAIR-Robotics/icarus`.
