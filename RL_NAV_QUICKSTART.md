# RL nav on the robot: quick start

The short version of [RL_NAV_ROBOT_CHECKLIST.md](RL_NAV_ROBOT_CHECKLIST.md). Use the full checklist
for first bring-up on new hardware or after retraining the policy.

**Someone with the E-stop must be present for every step that moves the robot.**

## 1. Build (after pulling)

```zsh
cd ~/icarus && git checkout robot && git pull
scripts/build.sh            # stale build errors: scripts/build.sh --clean
```

## 2. Shell setup (every new terminal)

```zsh
source ~/icarus_ws/install/setup.zsh
rlnav() { python3 ~/icarus/scripts/rl_nav_cli.py "$@"; }
```

## 3. Start the node (dry run: the policy runs, nothing moves)

```zsh
tmux new-window -t ros_nodes -n icarus-rlnav "zsh -ic 'source ~/icarus_ws/install/setup.zsh && ros2 launch icarus rl_nav.launch.py dry_run:=true; exec zsh'"
rlnav status --watch        # keep running in its own pane
```

Healthy: `state=idle dry_run=True`, `scan_age` < 0.4, `odom_age` < 0.1, `camera_age` < 1.0,
`camera_pitch_deg` within ±5. The web UI is at `http://<robot-ip>:8080/`.

Before driving, keep the arm at home and the head level, and keep anything off the floor within
~0.5 m in front (`camera_nearest` should read `-` on a clear floor).

## 4. Check decisions (still dry run)

```zsh
rlnav goal 2.0 1.0          # cmd angular > 0 (turns left)
rlnav goal 2.0 -1.0         # cmd angular < 0 (turns right)
rlnav cancel
```

## 5. Drive

```zsh
ros2 param set /rl_nav dry_run false
rlnav goal 1.5 0            # FWD m ahead, LEFT m left of the robot; max 5 m
```

You can also click a point on the map in the UI and press **Drive**.

## Stop

Best option first: the UI's **Stop**, then `rlnav cancel`, then `pkill -INT -f rl_nav_node`, then the E-stop.
The base also stops by itself within 0.5 s of the node going quiet.

## Shut down

```zsh
tmux kill-window -t ros_nodes:icarus-rlnav
```

## Limits

- Speed is capped at 0.4 m/s and 1.0 rad/s, and goals at 5 m.
- Forward motion stops if the lidar sees anything within 0.25 m in the ±60° sector ahead.
- The camera sees low obstacles only in a 112° cone ahead, from 0.3 to 2 m.
- Don't run the `RlNavigate` skill at the same time: both drive `/cmd_vel_skills`.
