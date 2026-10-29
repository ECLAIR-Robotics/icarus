# MJWarp RL training: nav policy (agent runbook)

How to train, evaluate, record, export and deploy the MARS goal-reaching / obstacle-avoidance
policy. Written for an agent working on a TACC Stampede3 GPU node; every command is run
from `mjwarp/` unless stated otherwise. The robot side is `../skills/` (see its README).

## What it is

- **Task** (`nav_env.py`, `MarsNavEnv`, an rsl_rl `VecEnv`): the robot spawns at a random pose in
  a walled room of random size (4-8 m per side; each wall is missing with 15% chance). It must
  drive to a random goal (>= 1 m away, reached within 0.3 m) without hitting a random subset of
  16 obstacles: cylinders of radius 8-50 cm and rotated boxes from thin panels to 2 m walls.
  Episodes end on goal, collision, or 45 s.
- **Realism**: everything the actor sees is noisy and delayed, with parameters drawn per
  episode. See the module docstring and constants of `nav_env.py`.
  - Gaussian command latency (70 +/- 25 ms, plus 10 ms jitter per step), simulated in 10 ms
    physics chunks.
  - A randomized motor model: gain, turn bias, deadband, first-order lag, acceleration limit,
    command noise.
  - Lidar at 5-10 Hz with range noise, dropouts, spurious returns and mount-yaw offset.
  - Wheel odometry with scale error, random-walk drift and no slip sensing. The goal and
    velocity observations come from it.
  - Random pushes.
  - The critic gets the clean state plus these hidden parameters (asymmetric actor-critic).
- **Policy I/O**: 53 actor observations (40 lidar bins, goal from odometry in the base frame,
  odometry velocity, last 3 actions) and 2 actions (forward, turn) in [-1, 1], scaled to
  0.4 m/s and 1.0 rad/s (innate-os `motion_control.yaml`), at 10 Hz. The arm and head hold
  their home pose. `nav_env.deploy_spec()` is the robot-side contract; it is exported as
  `policy_io.json`.
- **Trainer** (`train.py`): rsl_rl PPO (`rsl-rl-lib` 5.x), with MLP actor and critic of size
  256-256-128 and observation normalization. The actor reads obs group `policy` and the critic
  reads `critic`. It runs one process per GPU through `torchrun`, and rsl_rl all-reduces the
  gradients.
- **Files**:
  - `nav_env.py`: the task, rewards and scene.
  - `train.py`: training, with `TRAIN_CFG` holding the PPO hyperparameters.
  - `train.sh`: the launcher.
  - `play.py`: evaluation, video, export (`--export`) and install into the skill pack (`--install`).
  - `verify_deploy.py`: checks the robot-side code against the trainer, then runs the real skill
    closed-loop in the sim.
  - `watch_videos.sh`: renders a video every N iterations while a run trains.
  - `warp_env.py` / `robot_env.py`: the shared simulator. `MarsWarpEnv` and `build_model`
    are also used by the ROS bridge, so keep them backward compatible.

## 1. Check the node

```bash
nvidia-smi -L                                  # expect GPUs; training uses all of them
echo $SLURM_JOB_ID $SLURM_JOB_PARTITION        # must be inside a job (idev or sbatch), not a login node
echo "hours left: $(( (SLURM_JOB_END_TIME - $(date +%s)) / 3600 ))"
pgrep -af train.py                             # is a run already going? don't start a second one on the same GPUs
```

Training processes die when the Slurm job ends. Use `sbatch` for runs longer than the
time the current job has left.

## 2. One-time environment setup

The venv lives at `mjwarp/.venv` (gitignored). Skip this step if `.venv/bin/activate` exists.

```bash
module load python/3.12.11                     # TACC default module set already has it
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install "mujoco>=3.8" mujoco-warp warp-lang rsl-rl-lib tensorboard imageio-ffmpeg onnx onnxscript onnxruntime pydantic ruff
python test.py                                 # smoke test; first run JIT-compiles Warp kernels (~1 min, cached in ~/.cache/warp)
```

- `rsl-rl-lib` installs its own torch (2.14 / cu130) into the venv, which takes precedence
  over the module's torch 2.9. That is expected; the "torchvision requires torch==2.9.1"
  warnings are harmless.
- Blackwell GPUs (sm_120) need a recent `warp-lang` (1.18 works).

## 3. Launch training

```bash
# Inside an interactive job: detach so it survives the shell
nohup ./train.sh --num-envs 4096 --max-iterations 3000 > logs/train_$(date +%F_%H-%M).out 2>&1 &

# Or as a batch job. Submit from mjwarp/ (or the repo root): train.sh finds its files via
# $SLURM_SUBMIT_DIR. Use your job's partition/account (`echo $SLURM_JOB_PARTITION $SLURM_JOB_ACCOUNT`
# inside a job; e.g. rtx-small and IRI26004 were used before).
sbatch -p rtx-small -N 1 -t 12:00:00 -A <account> -o logs/train_%j.out train.sh --max-iterations 4000

# Continue from a checkpoint
./train.sh --resume logs/nav/<run>/model_<N>.pt --max-iterations 1000

# A single GPU, or a quick sanity run
NGPUS=1 ./train.sh --num-envs 1024 --max-iterations 5
```

- `--num-envs` is **per GPU**. 4096 per GPU uses about 5.4 GB of the 96 GB.
- Each run writes to `logs/nav/<YYYY-mm-dd_HH-MM-SS>/`: TensorBoard events, `model_<it>.pt`
  every 100 iterations plus the final one, and a git diff. `mjwarp/logs/` is gitignored.

### Expected numbers (2x RTX PRO 6000 Blackwell, 4096 envs/GPU)

| | |
|---|---|
| Throughput | about 86k env-steps/s total, about 3.1 s per iteration (32 steps x 8192 envs) |
| 3000 iterations | about 2 h 45 min |
| Learning (noisy env) | `Episode/success` about 0.91 by iteration 20 and 0.97 by iteration 200; `Episode/collision` about 0.03 |
| `Episode/odom_error_m` | about 0.1 (odometry drift at episode end; this is set by the noise model, not learned) |
| `Episode/diverged` | should stay 0 (see Pitfalls) |

## 4. Monitor

```bash
tail -f logs/train_*.out                                   # printed every iteration by rank 0
grep -E "Learning iteration|Episode/success|Episode/collision|ETA" logs/train_*.out | tail -4
tensorboard --logdir logs/nav --port 6006                  # then ssh-tunnel the port from your machine
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv   # both GPUs should be ~95%+
```

## 5. Evaluate, record video, export

```bash
python play.py                       # newest checkpoint under logs/nav, 4 worlds, 60 s video
python play.py --checkpoint logs/nav/<run>/model_<N>.pt --num-envs 9 --seconds 30
python play.py --export              # also writes exported/policy.{onnx,pt,npz} + policy_io.json
python play.py --no-video --export --install   # export, and copy policy.{onnx,npz} + policy_io.json into ../skills/rl_nav_policy/
python verify_deploy.py --checkpoint <the installed model_N.pt> --episodes 8   # robot-code checks + closed-loop skill runs
```

- The video is written to `logs/nav/<run>/videos/model_<N>.mp4`. It is H.264 yuv420p, real
  time at 10 fps, with the worlds tiled in a grid as seen from the overhead `overview` camera.
  Green disks are goals; orange cylinders are obstacles.
- Frames are rendered by mujoco_warp's GPU ray tracer, since OpenGL (EGL/OSMesa) does not work
  on these nodes. They are piped to `ffmpeg`: the one on PATH if present
  (`module load ffmpeg` gives 4.3.2), otherwise the binary bundled with `imageio-ffmpeg` (7.0.2).
- The exported networks include the observation normalizer and return the deterministic
  action. `policy.onnx` (weights inlined, one file) and `policy.npz` hold the same network for
  the robot's onnxruntime and numpy runners.
- `verify_deploy.py` must report max diffs of ~0 for the policies (compared after the clamp to
  [-1, 1]: trained actors output raw actions in the hundreds, where float32 rounding alone
  differs by ~1e-3) and ~1e-7 for the observation; it exits on > 1e-4. The closed-loop
  episodes should mostly print "Reached the goal". Run it after every `--install`.
  With `model_200` of the noisy run: 8/8 reached, true final error 0.17-0.52 m. With
  `model_2999`: 19/20 seeds reached (seed 7 collides).
- **The model_2999 actor is bang-bang**: its raw mean actions have a median magnitude of ~750,
  so almost every command is clamped to full speed and full turn rate. Nothing in training
  penalizes the raw output outside [-1, 1] (the action-rate term sees clamped actions). If that
  is too harsh on the hardware, squash with tanh in the actor or penalize |raw| > 1.
- `play.py` can run on a GPU that is training at the same time; it is small.

To get a video every 200 iterations while a run trains (`EVERY=` and `SECONDS_PER_VIDEO=` override
the interval and length; it exits when training does):

```bash
nohup ./watch_videos.sh > logs/watch_videos.out 2>&1 &   # optional arg: run dir, default newest
```

To export and record automatically when a run finishes, use the torchrun PID from `pgrep -af torchrun`:

```bash
nohup bash -c 'while kill -0 <TORCHRUN_PID> 2>/dev/null; do sleep 60; done; source .venv/bin/activate && python play.py --checkpoint "$(ls logs/nav/<run>/model_*.pt | sort -t_ -k3 -n | tail -1)" --export --install' > logs/post_train_export.out 2>&1 &
```

## Training curves

```bash
python plot_rewards.py              # logs/nav/training_curves.png: reward, success, collision, episode length, one line per run
nohup python plot_rewards.py --watch > logs/plot_rewards.out 2>&1 &   # redraw every minute until training ends
```

## Tuning knobs

- **Task, scene, noise and rewards**: the constants at the top of `nav_env.py`. They are
  grouped as robot interface, world, command path, lidar, odometry, and rewards. Changing
  anything in the robot interface group (`LIDAR_BINS`, the scalings, `ACTION_HISTORY`, the
  speeds) makes old checkpoints incompatible, and needs a re-export and re-install. The skill
  reads everything else it needs from `policy_io.json`.
- **PPO and network**: `TRAIN_CFG` in `train.py`. The numpy runner implements ELU only.
- Checkpoints from the first run (`logs/nav/2026-10-26_18-56-05`, noise-free, 49 observations)
  no longer load in the current env.

## Pitfalls (already handled; don't regress them)

- **Constraint buffers**: mjwarp's default `njmax` of 64 per world overflows when the robot
  touches an obstacle ("nefc overflow" warnings). An overflowing world drops constraints and
  can go NaN. `nav_env.py` passes `NCONMAX=64` and `NJMAX=512`. Worlds that still go
  non-finite are reset as timeouts and counted in `Episode/diverged`.
- **Shared CUDA stream**: `MarsNavEnv.__init__` makes torch and Warp use one non-default
  stream, so the torch views (`wp.to_torch`) of `qpos`, `ctrl`, `sensordata` and `mocap_pos`
  are ordered correctly with Warp's kernels and CUDA graphs. Create the env before any other
  CUDA work in the process, and do not call Warp on a different stream.
- **No host syncs in reset**: `_reset_worlds` samples new poses for every world and keeps
  them only where the mask is set (`torch.where`). Avoid `nonzero` or `.item()` in the
  stepping path.
- **Ray-traced rendering needs `mjw.refit_bvh(m, d, rc)` before every `mjw.render`**, or the
  images show stale poses. `play.py` does this. `MarsWarpEnv.render()` in `warp_env.py` does
  not yet, so its head and wrist camera images are probably stale.
- **The lidar sees parts of the robot itself**: beams about 7-12 deg left of forward return
  0.15-0.19 m. Returns below `LIDAR_MIN_RANGE` (0.2 m) are treated as misses, both in training
  and in `skills/rl_nav_utils.bin_scan`. Keep the two in sync through `deploy_spec()`.
- **Physics runs in 10 ms chunks** (`_chunk_graph`, sensors off), so command latency has that
  resolution. `_forward_graph` computes the sensors once per control step. Don't go back to a
  single graph per control step without moving the latency model.
- **`_sample_layout` must not touch `self.shape_xy`/`shape_yaw`.** It samples for every world,
  and only the masked worlds take the result.
- The goal marker is visual only: `contype=0`, and it sits at z < 4 mm, below the lidar beams
  at 17 cm. Keep any new visual markers out of the beams' plane, or the policy will see them.
- `innate-os/` is a read-only submodule. The robot model comes from it through
  `_driver_pkg.py`; don't edit it.
