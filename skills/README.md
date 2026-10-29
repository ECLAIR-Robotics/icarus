# ICARUS skills (robot side)

A skill pack for Innate OS. On the robot it is symlinked into the workspace, so brain_client
loads it, hot-reloads it on edit, and lists its skills as `icarus_skills/<name>`.

| File | What |
|---|---|
| `rl_nav.py` | `RlNavigate`: drives to a goal given relative to the robot, avoiding obstacles with the lidar policy trained in `mjwarp/` |
| `rl_nav_utils.py` | No ROS, no torch: the policy runners (onnxruntime / numpy), lidar binning, observation builder (checked against the trainer by `mjwarp/verify_deploy.py`), and `NavController`, the per-tick policy + safety loop shared with `icarus/rl_nav_node.py` |
| `rl_nav_policy/` | `policy.onnx`, `policy.npz` + `policy_io.json`, written by `mjwarp/play.py --export --install` |

The same policy also runs as a ROS node with a web UI (map, live scan, click-to-send goals); see
[Running it as a node with the web UI](#running-it-as-a-node-with-the-web-ui) below.

## Install on the robot (once)

```bash
cd ~/icarus && git pull                       # or copy the repo / just skills/ over
ln -s ~/icarus/skills ~/innate-os/workspace/icarus_skills
innate restart                                # or wait: new packs are picked up on brain_client start
innate skill list | grep rl_navigate
```

Needs only `numpy` and `pydantic`, which brain_client already has. `pip install onnxruntime` on the
robot makes the policy ~35% faster (it is picked automatically when importable); without it the
numpy runner is used. Both give the same clamped actions as the trainer.

**Why CPU and not the GPU / TensorRT:** the actor is a ~110k-weight MLP run once every 100 ms at
batch size 1. One forward pass takes ~10 µs on onnxruntime and ~15 µs on numpy (one x86 core;
expect a few times that on the Orin's A78s), and a whole control tick including scan binning
~80 µs. A GPU path would add kernel-launch and copy latency larger than the compute, and a CUDA
context costs hundreds of MB of the Jetson's 8 GB shared RAM (TorchScript on CPU alone peaks at
~450 MB vs ~33 MB for numpy and ~46 MB for onnxruntime). Revisit TensorRT only if the policy
grows to a CNN/transformer over images.

To update the policy, run `python play.py --checkpoint <model_N.pt> --export --install` in
`mjwarp/` on the training machine, then copy `skills/rl_nav_policy/` to the robot. The skill
re-reads the files on each run; the node reads them at start (restart it).

## First tests on the robot: do them in this order

1. **Dry run, robot on blocks or E-stop in hand.** `rl_navigate(forward_m=2, left_m=0, dry_run=True)`
   publishes nothing and reports the goal in the robot frame, the command it would send, and
   the nearest lidar return, every 0.1 s.
   - Put a box about 1 m **to the robot's left**. The nearest return should be about 1 m, and
     the commanded turn should change as you move it. If a box on the left reads as if it were on
     the right, the scan's angle convention differs from training (CCW, 0 = forward); fix that
     before driving.
   - With nothing within 1 m, the nearest return should not be below 0.2 m. Returns closer
     than 0.2 m are dropped as self-hits, so anything just above that in a fixed direction is
     part of the robot that the sim didn't model (cable, bracket). Note the angle.
2. **Open floor, short goal.** `rl_navigate(forward_m=1.5, left_m=0)`. It should arrive within
   0.3 m.
3. **One obstacle in the way.** Put a box or chair leg on the line to a 3 m goal.
4. **Clutter.**

## Safety behaviour (independent of the policy)

- **Hard stop:** forward or backward speed is zeroed whenever there is a lidar return within
  0.25 m in the ±60° sector in the direction of travel. The robot can still turn. The output
  data counts these stops.
- **Stale sensors:** the base gets zero commands if the scan stops updating for 0.6 s or
  odometry for 0.3 s.
- **cmd_vel deadman:** each command expires after 0.3 s, so the base stops if the loop dies.
  Stop or cancel is handled by `self.sleep` and brakes the base.
- **Limits:** the goal must be at most 5 m away, and the time limit is 10 s per metre
  (at least 20 s). Speeds are capped at 0.4 m/s and 1.0 rad/s, the limits from
  innate-os `motion_control.yaml` that the policy trained with.

Velocity comes from successive odometry poses, the way the trainer measures it, not from
`Odometry.twist`: the sim driver publishes zero twist, so before this the policy saw zero
velocity in the sim.

## Running it as a node with the web UI

`icarus/rl_nav_node.py` runs the same `NavController` as a ROS 2 node and serves a page to drive it.

```bash
scripts/build.sh                                  # installs the node, the UI and skills/rl_nav_policy
ros2 launch icarus rl_nav.launch.py               # args: backend:=auto|onnxruntime|numpy http_port:=8080 dry_run:=false
```

Then open **http://<robot>:8080/** (e.g. `http://mars.local:8080/`) from any browser on the robot's
network. The page connects to the robot's rosbridge on port 9090, the same socket the innate-os
webapp uses (override with `?ws=ws://host:port`). It shows the map (`/map`), the live scan, what
the policy sees (its 40 lidar bins), the robot's trail, the command and run status. Click the map
to pick a goal, then **Drive** (Enter); **Stop** (Space or Esc) cancels. **Dry run** runs the policy
without publishing commands.

| Interface | Type | |
|---|---|---|
| `/scan`, `/odom`, TF | in | goal frame -> odom every tick, base_link -> laser, map -> laser at scan time (display) |
| `/icarus/rl_nav/goal` | `geometry_msgs/PoseStamped` | goal in `map` or `odom` (orientation ignored); also works from RViz or other nodes |
| `/icarus/rl_nav/cancel` | `std_msgs/Bool` | `true` stops the run |
| `/icarus/rl_nav/status` | `std_msgs/String` (JSON) | state, poses in the display frame, scan, bins, command, timings (5 Hz) |
| `/cmd_vel_skills` | `geometry_msgs/Twist` | the skills input of the cmd_vel mux; teleop overrides it, and nothing is published while idle |
| `dry_run` | parameter | read when a goal arrives |

A goal in `map` is re-expressed in `odom` every tick, so AMCL's corrections of odometry drift move
it back where it belongs; with no `map -> odom` (map-free mode) the page works in `odom`. The
limits and safety checks are the skill's (5 m goals, 10 s/m, hard stop, stale feeds); goals
beyond 5 m are refused, so far goals need waypoints for now.

Tested in ROS 2 Humble (numpy 1.21, Python 3.10, as on the Jetson) against a fake robot, not yet
on hardware. First bring-up on the robot: follow [`RL_NAV_ROBOT_CHECKLIST.md`](../RL_NAV_ROBOT_CHECKLIST.md)
(gated phases, uses `scripts/rl_nav_cli.py`).

## What the policy was trained to handle (see mjwarp/nav_env.py)

Gaussian command latency (70 ± 25 ms, plus 10 ms jitter per step), a randomized motor model
(gain ±15%, turn bias, deadband, 50–250 ms lag, acceleration limits), lidar at 5–10 Hz with
noise, dropouts, spurious returns and ±2° mount yaw, wheel odometry with ±5% scale error,
drift and no slip sensing, random pushes, rooms of 4–8 m with missing walls, and cylinders and
rotated boxes from 8 cm poles to 2 m wall segments.

**It has not seen:**
- obstacles below the lidar plane (about 17 cm) or overhangs, which it cannot detect;
- glass and mirrors;
- moving people;
- rooms larger than 8 m, or goals beyond 5 m;
- floors where wheels slip much more than the ±5% odometry error.

The goal is tracked in the **odometry frame**, so odometry drift moves the goal. Expect
about 10 cm of error at the goal in training conditions.
