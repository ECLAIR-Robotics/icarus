# RL nav policy on the real MARS: bring-up checklist (for an agent on the robot)

You are an agent with a shell on the MARS robot (Jetson Orin Nano, ROS 2 Humble, Zenoh, zsh).
Your job is to get the MJWarp-trained navigation policy driving the real robot through
`icarus/rl_nav_node.py` and its web UI, one gated phase at a time, and to report what you saw.

The policy was trained and verified only in simulation, and the node was tested in ROS 2 Humble
against a fake robot. **This is its first contact with hardware.** Expect at least one convention
to be wrong; the phases are ordered so that a wrong convention shows up before anything moves.

## Rules (read first, follow throughout)

1. **A human operator with the E-stop must be present and must say "go" before every step marked
   🛑 MOTION.** Ask, wait for an explicit yes, then run it. Never chain two motion steps without asking.
2. **Stop at the first failed check.** Don't work around it, don't retune anything, don't continue
   to later phases. Report the failure (exact output) and your diagnosis, then wait.
3. **Never** edit anything in `~/innate-os` (third-party, read-only), raise speed limits, widen
   goal limits, or weaken the safety parameters (`safety_stop_range`, `stale_*`, `max_goal_distance`).
4. **Never upgrade numpy** or other system Python packages. ROS Humble and innate-os are built
   against the system numpy (1.21.x). If a `pip` command would change numpy, don't run it.
5. Don't run the `RlNavigate` skill and `rl_nav_node` at the same time: both drive `/cmd_vel_skills`.
   Make sure no agent or skill is active in the Innate app while testing.
6. To stop the robot at any time, in this order of preference: the UI's **Stop**, then
   `rlnav cancel`, then `pkill -INT -f rl_nav_node`, then the E-stop. The base also stops by itself
   within 0.5 s of the node going silent (the cmd_vel mux's freshness window).
7. Keep a log of every command you ran and its key output; the report at the end needs it.

### Facts the steps rely on

| | |
|---|---|
| Inputs | `/scan` (LaserScan, frame `base_laser`, ~6 Hz: `/scan_fast` throttled), `/odom` (~30 Hz), TF |
| Output | `/cmd_vel_skills` (Twist, 10 Hz while driving, silent when idle) -> innate-os `cmd_vel_mux` -> `/cmd_vel`. Teleop overrides it. |
| Goals | `/icarus/rl_nav/goal` (PoseStamped, frame `map` or `odom`; re-expressed in odom every tick) |
| Cancel | `/icarus/rl_nav/cancel` (Bool `true`) |
| Status | `/icarus/rl_nav/status` (String, JSON, 5 Hz) |
| Dry run | node parameter `dry_run`: the policy runs, nothing is published to the base |
| UI | `http://<robot>:8080/`, which talks to rosbridge (`rws_server`, started by `mars_control app.launch.py`) on port 9090 |
| Limits | 0.4 m/s, 1.0 rad/s; goals <= 5 m; 10 s per metre (min 20 s); forward speed zeroed if a return is < 0.25 m in the +-60 deg sector of travel |
| Angles | robot frame: 0 deg ahead, **+90 deg = left**, -90 deg = right, +-180 deg behind |

### The CLI you'll use

`~/icarus/scripts/rl_nav_cli.py` wraps every check. Define this in each new shell (after Phase 2.1):

```zsh
source ~/icarus_ws/install/setup.zsh
rlnav() { python3 ~/icarus/scripts/rl_nav_cli.py "$@"; }
```

| Command | What it does |
|---|---|
| `rlnav status` / `rlnav status --watch` | One line (or a stream) from the node: `state`, `cmd=[m/s, rad/s]`, `nearest=<m> at <deg>` (what the policy sees), `blocked`, `stale`, `remaining`, `backend`, `step_us`, feed ages |
| `rlnav goal FWD LEFT` | Goal FWD m ahead and LEFT m to the left of where the robot is now, sent in the odom frame |
| `rlnav cancel` | Stop the run (errors clearly if the node isn't running) |
| `rlnav scan` | `/scan` frame, beam count, rate, angle range, and the nearest raw return with its angle (works without the node) |
| `rlnav pose` | Odometry x, y, yaw |
| `rlnav drive LIN ANG SECS` | **Moves the base.** Publishes to `/cmd_vel_skills` at 10 Hz, capped at 0.2 m/s, 0.5 rad/s, 2 s; then zero. Prints how far odometry moved and turned. |

Background: `skills/README.md` (section "Running it as a node with the web UI") and `mjwarp/TRAINING.md`.

---

## Phase 0: Code and policy are on the robot

- [ ] 0.1 The ICARUS repo is at `~/icarus`. The RL nav code may be on a branch the operator pushed:
      ask which one, then `cd ~/icarus && git fetch && git checkout <branch> && git pull`.
- [ ] 0.2 All of these exist (if any is missing, the code wasn't pushed: stop and tell the operator):
      ```zsh
      ls ~/icarus/icarus/rl_nav_node.py ~/icarus/launch/rl_nav.launch.py ~/icarus/web/rl_nav/index.html \
         ~/icarus/scripts/rl_nav_cli.py ~/icarus/skills/rl_nav_utils.py \
         ~/icarus/skills/rl_nav_policy/{policy.onnx,policy.npz,policy_io.json}
      ```
- [ ] 0.3 The policy files are the verified export (model_2999 of run `2026-10-26_22-48-06`):
      ```zsh
      md5sum ~/icarus/skills/rl_nav_policy/*
      # a0dbaebac4208c13f318a2b928e27a35  policy_io.json
      # fbd95035e7c0d1b62ebba8ee799cc686  policy.npz
      # 89e3120ec47ed7b30b83025234011d7a  policy.onnx
      ```
      A mismatch is OK only if the operator says a newer policy was installed on purpose; record which.
- [ ] 0.4 The innate-os stack is up: `innate` prints a healthy status, and `tmux ls` shows the
      `ros_nodes` session.

## Phase 1: Build and check the robot's interfaces (no motion)

- [ ] 1.1 Build: `cd ~/icarus && scripts/build.sh`. It must end with `1 package finished`. Then define
      `rlnav` as above. Check the installed copy has the policy:
      `ls $(ros2 pkg prefix icarus)/share/icarus/rl_nav/rl_nav_policy/` lists all three files.
- [ ] 1.2 `rlnav scan`. Record the whole line. Pass: `frame=base_laser`, rate ~6 Hz, several hundred
      beams, angles spanning roughly -180..+180 deg.
- [ ] 1.3 `ros2 topic hz /odom` for ~5 s, then Ctrl-C. Expect ~30 Hz.
- [ ] 1.4 The cmd_vel mux exists and listens on the skills input:
      `ros2 node list | grep cmd_vel_mux` and `ros2 topic info /cmd_vel_skills` (Subscription count >= 1).
      No mux: stop. Commands would never reach the base.
- [ ] 1.5 The lidar's axes match base_link (the policy assumes they do):
      `timeout 5 ros2 run tf2_ros tf2_echo base_link base_laser`. The rotation must be ~identity
      (yaw within a few degrees of 0). **If the yaw is ~180 or ~+-90 deg, stop**: every lidar bin
      would be rotated and the policy would steer into obstacles.
- [ ] 1.6 Nav mode: `ros2 topic echo --once /nav/current_mode`. In `navigation` mode with a map,
      `timeout 5 ros2 run tf2_ros tf2_echo map odom` must print a transform. Map-free mode is fine
      too (the node and UI then work in the `odom` frame); note which.
- [ ] 1.7 Resources before starting the node: `free -m` and `uptime`. Record them.

## Phase 2: Start the node in dry run (no motion)

- [ ] 2.1 Port 8080 is free: `ss -ltn | grep -w 8080` prints nothing. If it's taken, add
      `http_port:=8090` to every launch command below and use that port in the URL.
- [ ] 2.2 Optional, faster backend (onnxruntime). First see what pip would do:
      `pip3 install --user --dry-run onnxruntime 2>&1 | tail -5`.
      Install only if that output does **not** mention numpy: `pip3 install --user onnxruntime`, then
      `python3 -c "import onnxruntime, numpy; print(onnxruntime.__version__, numpy.__version__)"`
      (numpy must be the same version as before). If pip wants to touch numpy, skip this step: the
      numpy backend gives the same clamped actions, it's just a little slower.
- [ ] 2.3 Backend self-check (both backends must agree after the clamp to [-1, 1]):
      ```zsh
      python3 - <<'EOF'
      import importlib.util, sys
      from pathlib import Path
      import numpy as np
      d = Path.home() / "icarus/skills"
      spec = importlib.util.spec_from_file_location("rl_nav_utils", d / "rl_nav_utils.py")
      u = importlib.util.module_from_spec(spec); sys.modules["rl_nav_utils"] = u; spec.loader.exec_module(u)
      s = u.NavSpec.load(d / "rl_nav_policy/policy_io.json")
      npy = u.load_policy(d / "rl_nav_policy", "numpy")
      rng = np.random.default_rng(0)
      obs = [u.build_observation(s, rng.uniform(0.05, 1, s.lidar_bins).astype(np.float32), tuple(rng.uniform(-4, 4, 2)),
                                 rng.uniform(-0.4, 0.4), rng.uniform(-1, 1), rng.uniform(-1, 1, (s.action_history, 2)))
             for _ in range(200)]
      a = np.clip([npy(o) for o in obs], -1, 1)
      print("numpy ok, obs size", npy.obs_size)
      try:
          onx = u.load_policy(d / "rl_nav_policy", "onnxruntime")
          print("onnxruntime vs numpy, max clamped diff:", np.abs(np.clip([onx(o) for o in obs], -1, 1) - a).max())
      except ImportError:
          print("onnxruntime not installed: the node will use numpy")
      EOF
      ```
      Pass: `obs size 53`, and the diff is `0.0` (anything < 1e-3 is fine) if onnxruntime is installed.
- [ ] 2.4 Start the node **in dry run**, in its own tmux window so it outlives your shell:
      ```zsh
      tmux new-window -t ros_nodes -n icarus-rlnav "zsh -ic 'source ~/icarus_ws/install/setup.zsh && ros2 launch icarus rl_nav.launch.py dry_run:=true; exec zsh'"
      sleep 6; tmux capture-pane -p -t ros_nodes:icarus-rlnav | tail -5
      ```
      Pass: `RL nav ready: <onnxruntime|numpy> policy from ..., driving /cmd_vel_skills; UI on http://<robot>:8080/`.
      Record the backend. Any traceback: stop and report it.
      (To restart the node later: `tmux kill-window -t ros_nodes:icarus-rlnav`, then run this again,
      without `dry_run:=true` once Phase 5 says so.)
- [ ] 2.5 `rlnav status`. Pass: `state=idle`, `dry_run=True`, `scan_age` < 0.4, `odom_age` < 0.1, `robot`
      not None, `frame=map` (or `odom` in map-free mode).
- [ ] 2.6 Idle means silent: `timeout 3 ros2 topic echo /cmd_vel_skills` prints **nothing**.
- [ ] 2.7 UI: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/` prints 200, and
      `curl -s http://localhost:8080/config.json` lists the topics. Give the operator the URL
      `http://<robot-ip or mars.local>:8080/` to open on a laptop or phone on the same network. They
      should see a green dot ("connected to ws://...:9090"), the map if there is one, red scan dots
      lying on walls, and the robot as a white circle. A red dot means rosbridge on 9090 isn't
      reachable from their device: report it (the node itself is still fine).
- [ ] 2.8 Resources with the node running: `ps -o rss=,pcpu= -p $(pgrep -f rl_nav_node | head -1)`.
      Expect well under 150 MB RSS and a few % CPU. Record them.

## Phase 3: Perception and decision checks (dry run, nothing moves)

The operator places objects; you read `rlnav scan` and `rlnav status`. The arm must be in its
home/rest pose: training only saw it at home, and an arm hanging in front would read as an obstacle.

- [ ] 3.1 **Left/right convention (the most important check).** Ask the operator to clear ~1.5 m
      around the robot, then put a box ~1 m to the robot's **left**. Both `rlnav scan` and `rlnav status`
      must report the nearest return at ~1.0 m, **~+90 deg**. Move it ~1 m to the **right**: ~-90 deg.
      Then ~1 m in **front**: ~0 deg. The UI's "What the policy sees" panel (forward = up) should agree.
      **If left reads negative, or front reads +-180, stop**: the scan's angle convention differs from
      training, and driving would steer into obstacles.
- [ ] 3.2 Self-hits: with nothing within 1 m, `rlnav scan`'s "nearest return" should be > ~0.3 m.
      A short return (< 0.3 m) at a fixed angle that stays when objects are removed is part of the
      robot (cable, bracket, arm). Record its angle and range. **If it's in the front +-60 deg sector,
      stop**: it would hold the hard stop on permanently.
- [ ] 3.3 Dry-run decisions. With ~2.5 m clear ahead and to both sides:
      - `rlnav goal 2.0 1.0`, then `rlnav status` three times ~0.5 s apart: `state=dry_run`,
        `remaining=~2.2`, and `cmd` angular **positive** (turn left), forward ~+0.4.
      - `rlnav goal 2.0 -1.0`: angular **negative** (turn right).
      - `timeout 2 ros2 topic echo /cmd_vel_skills` prints **nothing** during dry run.

      The policy is bang-bang, so expect `cmd` to read exactly +-0.4 / +-1.0. If a sign is reversed,
      first check that nothing is within ~1 m on that side (the policy turns away from close
      obstacles); if the side is clear and it's still reversed, stop and report.
- [ ] 3.4 `rlnav cancel` gives `state=cancelled`.
- [ ] 3.5 Refusal: `rlnav goal 5.5 0` gives `state=error` with "the policy handles up to 5.0 m".

## Phase 4: Command path sign checks (🛑 MOTION, tiny, operator present)

These check the base's conventions independently of the policy. Clear 1 m around the robot.
`rlnav drive` publishes on `/cmd_vel_skills` like the node does; the node is idle, so they don't conflict.

- [ ] 4.1 🛑 MOTION. Ask for "go". `rlnav drive 0 0.4 1`.
      Pass: the operator sees it turn **left (counter-clockwise seen from above)**, the output says
      `yaw` of roughly +10 to +25 deg (motor lag eats some), and it stops by itself.
- [ ] 4.2 🛑 MOTION. Ask for "go". `rlnav drive 0.15 0 1`.
      Pass: it moves **forward**, and the output says roughly +0.07 to +0.15 m forward and ~0 deg yaw.
- [ ] 4.3 If either sign is wrong, stop: the policy's commands would be mirrored on the real base.

## Phase 5: First policy drives (🛑 MOTION, operator with the E-stop)

Turn dry run off: `ros2 param set /rl_nav dry_run false` (or untick it in the UI). Check with
`ros2 param get /rl_nav dry_run` (False). During every run keep `rlnav status --watch` running in
another pane and watch for `stale=True` or long streaks of `blocked=True`.

- [ ] 5.1 🛑 MOTION. Open floor, nothing within 1 m of the path. `rlnav goal 1.5 0`.
      Pass: `state=reached`, "Reached the goal (<0.3 m off)", within ~10 s, and the operator calls the
      motion acceptable. Record the time, safety stops, and the operator's measured distance from the
      intended spot.
- [ ] 5.2 Control rate: during a run, `ros2 topic hz /cmd_vel_skills` reads ~10 Hz. Record `step_us`
      from the status (expect well under 10000).
- [ ] 5.3 🛑 MOTION. `rlnav goal 1.5 1.0`, then `rlnav goal 1.5 -1.0` (after the first is reached). Both reached.
- [ ] 5.4 🛑 MOTION. Stopping mid-run:
      - `rlnav goal 3 0`, and after ~1 s press **Stop** in the UI (or `rlnav cancel`). The robot stops within ~0.5 s.
      - Again `rlnav goal 3 0`, and after ~1 s run `pkill -INT -f rl_nav_node`. It stops within ~0.5 s (mux deadman).
        Then restart the node (see 2.4) **without** `dry_run:=true`.
- [ ] 5.5 🛑 MOTION. Hard stop: `rlnav goal 2 0` with a box ~0.8 m ahead on the path; the operator slides
      it to ~0.4 m in front as the robot approaches. The robot must never touch it: `blocked=True`
      appears and forward speed is zeroed while it can still turn. Record `safety_stops`.
- [ ] 5.6 Teleop override (optional): during a run the operator drives with the app's joystick. Teleop
      must win; when they let go, the policy resumes.

## Phase 6: Obstacle avoidance (🛑 MOTION)

- [ ] 6.1 🛑 MOTION. One obstacle >= 20 cm tall (above the lidar plane at ~17 cm), centred on the line
      to `rlnav goal 3 0`. Pass: goes around it and reaches the goal. 3 runs; record reached, time,
      safety stops, and the closest approach the operator saw.
- [ ] 6.2 🛑 MOTION. Clutter: 3-5 obstacles, goals 3-4 m away. 3 runs, same records.
- [ ] 6.3 🛑 MOTION. Navigation mode with a map only: the operator clicks a point 3-4 m away on the UI's
      map and presses **Drive**. Pass: the robot ends within ~0.3-0.5 m of the clicked point on the
      map (map goals follow AMCL's corrections).

Known limits, not failures: obstacles lower than ~17 cm and overhangs are invisible to it; glass and
mirrors; moving people; goals beyond 5 m are refused; the policy is bang-bang (nearly always full
speed and full turn rate), so motion can look jerky.

## Phase 7: Leave it in a known state

- [ ] 7.1 Decide with the operator whether the node stays running. If not:
      `tmux kill-window -t ros_nodes:icarus-rlnav`. Either way, `timeout 3 ros2 topic echo /cmd_vel_skills`
      must print nothing while idle.
- [ ] 7.2 `git -C ~/icarus status --short`: you should not have changed tracked files. If you did,
      list them in the report; don't commit unless asked.

## Report (send this back)

```
Robot / innate-os:   <innate version>, nav mode <navigation|mapfree>, map <name or none>
Code:                branch <..> @ <commit>; policy md5s match? <yes/no>
Backend:             <onnxruntime x.y | numpy>; step_us <..>; node RSS <..> MB, CPU <..> %; free -m before/after
Interfaces:          rlnav scan line: <paste>; /odom <..> Hz; base_link->base_laser yaw <..> deg; mux present <yes/no>
Perception:          box left -> <..> deg, right -> <..> deg, front -> <..> deg; self-hits <none | range @ angle>
Decisions (dry run): goal 2,+1 -> cmd <..>; goal 2,-1 -> cmd <..>
Sign checks:         drive 0 0.4 1 -> yaw <..> deg, seen turning <left/right>; drive 0.15 0 1 -> <..> m <forward/back>
Stops:               UI stop <..> s; node kill <..> s; hard stop held <yes/no> (safety_stops <..>)
Drives (each run):   <goal>, <reached|timeout|error>, <time s>, <safety stops>, <operator-measured final error m>, <notes>
Failed / stopped at: <phase.step and exact output, or "none">
Observations:        <anything odd: jerky motion, oscillation near the goal, getting stuck, latency>
```
