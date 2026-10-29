"""Runs a trained nav policy, records a top-down video, and exports the policy for deployment.

    python play.py                          # newest checkpoint under logs/nav, 4 worlds, 60 s
    python play.py --checkpoint logs/nav/<run>/model_1000.pt --num-envs 9 --seconds 30
    python play.py --no-video --export      # just export ONNX + TorchScript

Rendering uses mujoco_warp's GPU ray tracer, so it needs no display or OpenGL. Frames are
piped to ffmpeg (H.264, yuv420p, so any player opens it): `ffmpeg` on PATH if there is one
(e.g. `module load ffmpeg`), else the binary bundled with the imageio-ffmpeg package. The
video goes next to the checkpoint (videos/<checkpoint>.mp4), tiling the worlds in a grid, in
real time. Export writes exported/policy.{onnx,pt,npz} and exported/policy_io.json (the
robot-side observation/action contract, nav_env.deploy_spec()) next to the checkpoint; all
include the observation normalizer and output the deterministic (mean) action. --install
also copies policy.npz, policy.onnx and policy_io.json into ../skills/rl_nav_policy/ for the
robot (the RlNavigate skill and icarus/rl_nav_node.py).
"""

import argparse
import copy
import json
import math
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import mujoco_warp as mjw
import numpy as np
import onnx
import torch
import warp as wp
from rsl_rl.runners import OnPolicyRunner

import nav_env
from nav_env import MarsNavEnv
from train import TRAIN_CFG

DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs" / "nav"
# The robot-side runner (skills/rl_nav.py) loads its policy from here.
SKILL_POLICY_DIR = Path(__file__).resolve().parents[1] / "skills" / "rl_nav_policy"


def latest_checkpoint(log_dir: Path) -> Path:
    checkpoints = sorted(log_dir.glob("*/model_*.pt"), key=lambda p: (p.parent.name, int(p.stem.split("_")[1])))
    if not checkpoints:
        raise SystemExit(f"no model_*.pt under {log_dir}")
    return checkpoints[-1]


class OverviewRecorder:
    """Renders the overview camera of every world and tiles them into one frame."""

    def __init__(self, env: MarsNavEnv) -> None:
        self.env = env
        mj_model = env.sim.mj_model
        self.camera_id = mj_model.camera(nav_env.OVERVIEW_CAMERA).id
        width, height = mj_model.cam_resolution[self.camera_id]
        self.context = mjw.create_render_context(
            mj_model,
            env.num_envs,
            render_rgb=[i == self.camera_id for i in range(mj_model.ncam)],
            render_depth=False,
            enabled_geom_groups=[0, 1, 2],
        )
        self.rgb = wp.zeros((env.num_envs, height, width), dtype=wp.vec3)
        self.cols = math.ceil(math.sqrt(env.num_envs))
        self.rows = math.ceil(env.num_envs / self.cols)

    def frame(self) -> np.ndarray:
        model, data = self.env.sim.model, self.env.sim.data
        # render() ray-traces against a BVH built at creation; it must be refit to the current poses.
        mjw.refit_bvh(model, data, self.context)
        mjw.render(model, data, self.context)
        mjw.get_rgb(self.context, self.camera_id, self.rgb)
        images = (self.rgb.numpy().clip(0, 1) * 255).astype(np.uint8)
        n, h, w, _ = images.shape
        grid = np.zeros((self.rows * h, self.cols * w, 3), dtype=np.uint8)
        for i in range(n):
            r, c = divmod(i, self.cols)
            grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = images[i]
        return grid


class Mp4Writer:
    """Streams RGB frames into ffmpeg's stdin, so no frame list is held in memory."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
        self.process = subprocess.Popen(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg exited with code {self.process.returncode}")


def export(runner: OnPolicyRunner, out_dir: Path, install_dir: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_onnx(str(out_dir))
    _inline_onnx_weights(out_dir / "policy.onnx")
    runner.export_policy_to_jit(str(out_dir))
    _export_numpy(runner.alg.get_policy(), out_dir / "policy.npz")
    (out_dir / "policy_io.json").write_text(json.dumps(nav_env.deploy_spec(), indent=2))
    print(f"Exported policy.onnx, policy.pt, policy.npz and policy_io.json to {out_dir}")
    if install_dir is None:
        return
    install_dir.mkdir(parents=True, exist_ok=True)
    for name in ("policy.npz", "policy.onnx", "policy_io.json"):
        shutil.copy2(out_dir / name, install_dir / name)
    print(f"Installed policy.npz, policy.onnx and policy_io.json into {install_dir}")


def _inline_onnx_weights(path: Path) -> None:
    """torch's exporter puts the weights in a policy.onnx.data sidecar; the robot gets one file."""
    model = onnx.load(str(path))
    data = path.with_name(path.name + ".data")
    onnx.save(model, str(path), save_as_external_data=False)
    data.unlink(missing_ok=True)


def _export_numpy(actor: torch.nn.Module, path: Path) -> None:
    """The actor for skills/rl_nav_utils.NavPolicy: normalizer stats and the MLP's linear layers
    (ELU between them, as TRAIN_CFG sets)."""
    if TRAIN_CFG["actor"]["activation"] != "elu":
        raise ValueError("NavPolicy implements ELU only")
    normalizer = actor.obs_normalizer
    linears = [m for m in actor.mlp if isinstance(m, torch.nn.Linear)]
    arrays = {
        "obs_mean": normalizer._mean.squeeze(0),
        "obs_std": normalizer._std.squeeze(0),
        "obs_eps": torch.tensor(normalizer.eps),
        "n_layers": torch.tensor(len(linears)),
    }
    for i, layer in enumerate(linears):
        arrays[f"w{i}"] = layer.weight
        arrays[f"b{i}"] = layer.bias
    np.savez(path, **{k: v.detach().cpu().numpy().astype(np.float32 if v.is_floating_point() else np.int64) for k, v in arrays.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: newest under logs/nav")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument(
        "--install", action="store_true", help=f"with --export, also copy the robot files into {SKILL_POLICY_DIR}"
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or latest_checkpoint(DEFAULT_LOG_DIR)
    print(f"Checkpoint: {checkpoint}")
    torch.manual_seed(args.seed)

    env = MarsNavEnv(args.num_envs, device=args.device)
    runner = OnPolicyRunner(env, copy.deepcopy(TRAIN_CFG), log_dir=None, device=args.device)
    runner.load(str(checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(args.device)

    if args.export:
        export(runner, checkpoint.parent / "exported", SKILL_POLICY_DIR if args.install else None)
    if args.no_video:
        return

    recorder = OverviewRecorder(env)
    video = checkpoint.parent / "videos" / f"{checkpoint.stem}.mp4"
    video.parent.mkdir(exist_ok=True)
    first = recorder.frame()
    writer = Mp4Writer(video, first.shape[1], first.shape[0], nav_env.CONTROL_HZ)
    writer.write(first)
    outcomes = {"success": 0, "collision": 0, "timeout": 0}
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(int(args.seconds * nav_env.CONTROL_HZ)):
            obs, _, _, extras = env.step(policy(obs))
            for key in outcomes:
                outcomes[key] += int(extras["log"][f"Episode/{key}"].sum().item())
            writer.write(recorder.frame())
    writer.close()
    print(f"Episodes finished: {outcomes}")
    print(f"Video: {video}")


if __name__ == "__main__":
    main()
