# Train and export the RL nav policy for the robot (for an agent on the training machine)

You are an agent on the training machine (TACC Stampede3 GPU node) with a clone of the ICARUS
repo. Your job is to train the MJWarp navigation policy, export it in the format the robot's
controller node (`icarus/rl_nav_node.py`) loads, check it against the trainer, and hand it back
on a git branch. `mjwarp/TRAINING.md` is the full reference; this file is the order to do it in
and the checks that decide whether you're done.

## Why this exists

The robot was given `mjwarp/nav/exported/` from an old run. The node can't use it:

| | What the node needs | What the robot got |
|---|---|---|
| Observations | 53 (40 lidar, 5 goal, 2 velocity, 3 x 2 past actions) | 49 (1 past action) |
| Action scale | 0.4 m/s, 1.0 rad/s | 0.6 m/s, 2.5 rad/s |
| `policy_io.json` | `nav_env.deploy_spec()` format (has `lidar`, `max_forward_speed`, ...) | an older format |
| Files | `policy.onnx` (weights inlined), `policy.npz`, `policy_io.json` | `policy.onnx` + `policy.onnx.data` + `policy.pt`, no `.npz` |

It is most likely the first, noise-free run (`logs/nav/2026-10-26_18-56-05`), which
`TRAINING.md` says no longer loads in the current env. The fix is to train with the current
`nav_env.py` and deliver the output of `play.py --export --install`, not the `exported/` folder.

## Rules

1. **Don't change the robot interface.** Leave these as they are in `mjwarp/nav_env.py`:
   `CONTROL_HZ`, `MAX_FORWARD_SPEED` (0.4), `MAX_TURN_RATE` (1.0), `LIDAR_BINS` (40),
   `ACTION_HISTORY` (3), the lidar ranges, the observation scalings, and `deploy_spec()`. The
   robot's speed limits are set to match them. If you think one of them must change, stop and ask.
2. Don't edit `innate-os/` (read-only submodule), `skills/rl_nav_utils.py`, `skills/rl_nav.py`
   or `icarus/rl_nav_node.py`. If `verify_deploy.py` reports a mismatch, report it and stop. Don't
   "fix" the robot code to match.
3. Deliver only `skills/rl_nav_policy/{policy.onnx,policy.npz,policy_io.json}`. Never copy
   `logs/nav/<run>/exported/` to the robot or commit it; `mjwarp/logs/` stays gitignored.
4. Stop at the first failed check. Report the exact output and wait.
5. Keep a log of commands and key output; the report at the end needs it.

## Phase 0: Repo and node

All commands run from `mjwarp/` unless noted.

- [ ] 0.1 `git fetch && git checkout main && git pull`. Record `git log --oneline -1`.
- [ ] 0.2 `git diff HEAD --stat -- mjwarp/nav_env.py mjwarp/train.py mjwarp/play.py` prints nothing.
      If it does, someone has local changes to the trainer: stop and ask whether to keep them.
- [ ] 0.3 The contract matches what the robot expects:
      ```bash
      grep -nE "^(CONTROL_HZ|MAX_FORWARD_SPEED|MAX_TURN_RATE|LIDAR_BINS|ACTION_HISTORY) =" nav_env.py
      ```
      Expect `MAX_FORWARD_SPEED = 0.4`, `MAX_TURN_RATE = 1.0`, `LIDAR_BINS = 40`, `ACTION_HISTORY = 3`.
- [ ] 0.4 Node checks from `TRAINING.md` section 1: inside a Slurm job (not a login node), GPUs
      visible, enough hours left (~3 h for 3000 iterations on 2 GPUs), and `pgrep -af train.py`
      shows no other run on these GPUs.
- [ ] 0.5 Environment (`TRAINING.md` section 2): `source .venv/bin/activate` (create it first if
      it doesn't exist), then `python test.py` passes.

## Phase 1: Train

- [ ] 1.1 Sanity run: `NGPUS=1 ./train.sh --num-envs 1024 --max-iterations 5`. Must finish without
      a traceback. Note the run dir it created under `logs/nav/` (you can delete it after).
- [ ] 1.2 Full run. If the job has more than ~3.5 h left:
      ```bash
      nohup ./train.sh --num-envs 4096 --max-iterations 3000 > logs/train_$(date +%F_%H-%M).out 2>&1 &
      ```
      Otherwise submit it with `sbatch` as in `TRAINING.md` section 3, with a 12 h limit. Record the
      run dir `logs/nav/<YYYY-mm-dd_HH-MM-SS>/`.
- [ ] 1.3 Monitor (`TRAINING.md` section 4). Checkpoints are written every 100 iterations.
      Expected trends (2x RTX PRO 6000, 4096 envs/GPU):

      | Metric | Expected |
      |---|---|
      | `Episode/success` | ~0.91 by iteration 20, ~0.97 by 200 |
      | `Episode/collision` | ~0.03 |
      | `Episode/diverged` | 0 |
      | Iteration time | ~3.1 s |

      If success is still below 0.8 at iteration 300, or `diverged` stays above 0, stop and report.
      Don't retune rewards or PPO settings unless the operator asks.

## Phase 2: Pass/fail gates on the final checkpoint

Use the last checkpoint: `CKPT=$(ls logs/nav/<run>/model_*.pt | sort -t_ -k3 -n | tail -1)`.

- [ ] 2.1 Final training stats (last 20 iterations in the log): `Episode/success` >= 0.95,
      `Episode/collision` <= 0.05, `Episode/diverged` = 0.
- [ ] 2.2 Export and install into the skill pack:
      ```bash
      python play.py --checkpoint "$CKPT" --no-video --export --install
      ```
      It must print `Installed policy.npz, policy.onnx and policy_io.json into .../skills/rl_nav_policy`.
- [ ] 2.3 Check the installed files against the robot's contract:
      ```bash
      ls ../skills/rl_nav_policy/                    # exactly policy.npz policy.onnx policy_io.json, no policy.onnx.data
      python - <<'EOF'
      import json, onnx, numpy as np
      d = "../skills/rl_nav_policy/"
      s = json.load(open(d + "policy_io.json"))
      size = s["lidar"]["bins"] + 5 + 2 + 2 * s["action_history"]
      print("speeds", s["max_forward_speed"], s["max_turn_rate"], "action_history", s["action_history"], "obs", size)
      m = onnx.load(d + "policy.onnx")
      print("onnx input", [x.dim_value for x in m.graph.input[0].type.tensor_type.shape.dim])
      print("npz obs_mean", np.load(d + "policy.npz")["obs_mean"].shape)
      EOF
      ```
      Pass: `speeds 0.4 1.0 action_history 3 obs 53`, ONNX input `[1, 53]`, npz `(53,)`, and
      `onnx.load` works with no `.data` file next to it.
- [ ] 2.4 Check the robot-side code against the trainer:
      ```bash
      python verify_deploy.py --checkpoint "$CKPT" --episodes 8
      ```
      Pass: every `max diff` line <= 1e-4 (policies ~0 after the clamp, observation ~1e-7), no
      `MISMATCH`, and at least 7 of 8 episodes print "Reached the goal". Record the per-episode lines.
- [ ] 2.5 Optional video for the operator: `python play.py --checkpoint "$CKPT" --num-envs 9 --seconds 30`.
      Record the path of the `.mp4`.
- [ ] 2.6 `md5sum ../skills/rl_nav_policy/*` and record the output. The robot checklist's
      step 0.3 has the old checksums, so the operator needs these.

## Phase 3: Hand it back

- [ ] 3.1 `git status --short` (from the repo root) should show only the three
      `skills/rl_nav_policy/` files as modified. Anything else: list it and ask before continuing.
- [ ] 3.2 Commit only those files on a new branch and push it:
      ```bash
      git checkout -b nav-policy/<run-name>
      git add skills/rl_nav_policy/policy.onnx skills/rl_nav_policy/policy.npz skills/rl_nav_policy/policy_io.json
      git commit -m "RL nav policy: <run> model_<N>"
      git push -u origin nav-policy/<run-name>
      ```
      On the robot, the agent checks this branch out in checklist Phase 0.1 and runs
      `scripts/build.sh`, which installs `skills/rl_nav_policy/` with the node.

## Known behavior (not a failure)

The current trainer produces a bang-bang actor: raw actions are in the hundreds, so nearly every
command is clamped to full speed and full turn rate. Note it in the report; don't change the
actor or rewards to fix it unless the operator asks (`TRAINING.md` describes the options).

## Report (send this back)

```
Code:            main @ <commit>; trainer diffs: <none | list>
Run:             logs/nav/<run>, <iterations>, <GPUs x envs/GPU>, <wall time>
Final stats:     success <..>, collision <..>, diverged <..> (last 20 iterations)
Checkpoint:      model_<N>.pt
Contract:        speeds <..> <..>, action_history <..>, obs <..>, onnx input <..>, npz <..>
verify_deploy:   <each max diff line>; episodes reached <k>/8 (<notes on failures>)
md5sum:          <three lines>
Branch:          nav-policy/<run-name> @ <commit>, pushed <yes/no>
Video:           <path or none>
Failed at:       <phase.step and exact output, or "none">
Observations:    <anything odd>
```
