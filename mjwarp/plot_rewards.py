"""Plots training curves (mean reward, success, collision, episode length) for every run under
logs/nav, one line per run, from the TensorBoard event files. Writes logs/nav/training_curves.png.

    python plot_rewards.py                      # all runs, once
    python plot_rewards.py --runs 2026-10-26_22-48-06
    python plot_rewards.py --watch              # redraw every minute while a train.py is running
"""

import argparse
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

LOG_DIR = Path(__file__).resolve().parent / "logs" / "nav"
PANELS = (
    ("Train/mean_reward", "Mean episode reward", None),
    ("Episode/success", "Success rate", (0.0, 1.0)),
    ("Episode/collision", "Collision rate", (0.0, None)),
    ("Train/mean_episode_length", "Mean episode length (steps)", (0.0, None)),
)
# Categorical palette (dataviz skill default), assigned to runs in a fixed order.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SMOOTHING = 0.9


def load_run(run_dir: Path) -> tuple[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """The run's legend label and its curves. Runs that log odometry error used the noisy env."""
    events = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    events.Reload()
    available = set(events.Tags()["scalars"])
    label = f"{run_dir.name} ({'sim-to-real noise' if 'Episode/odom_error_m' in available else 'noise-free'})"
    curves = {}
    for tag, _, _ in PANELS:
        if tag not in available:
            continue
        points = events.Scalars(tag)
        curves[tag] = (np.array([p.step for p in points]), np.array([p.value for p in points], dtype=float))
    return label, curves


def ema(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values)
    acc = values[0]
    for i, v in enumerate(values):
        acc = SMOOTHING * acc + (1 - SMOOTHING) * v
        out[i] = acc
    return out


def plot(run_dirs: list[Path], out: Path) -> None:
    runs = [load_run(d) for d in run_dirs]
    runs = [(name, curves) for name, curves in runs if curves]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED,
                         "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=SURFACE)
    for ax, (tag, title, ylim) in zip(axes.flat, PANELS, strict=True):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", color=TEXT, fontsize=11)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for i, (name, curves) in enumerate(runs):
            if tag not in curves:
                continue
            steps, values = curves[tag]
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            ax.plot(steps, values, color=color, linewidth=1.0, alpha=0.2)
            smooth = ema(values)
            ax.plot(steps, smooth, color=color, linewidth=2.0, label=name)
            ax.annotate(f"{smooth[-1]:.3g}", (steps[-1], smooth[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT_MUTED)
        if ylim is not None:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.set_xlabel("Training iteration")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, labelcolor=TEXT, title="Run",
               title_fontproperties={"size": 10})
    fig.suptitle(f"Training curves (smoothed, EMA {SMOOTHING}; faint = raw)", x=0.01, ha="left", color=TEXT, fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.8, 0.96))
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out} ({', '.join(name for name, _ in runs)})")


def training_running() -> bool:
    return subprocess.run(["pgrep", "-f", "train.py"], capture_output=True).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", help="run directory names under logs/nav (default: all)")
    parser.add_argument("--out", type=Path, default=LOG_DIR / "training_curves.png")
    parser.add_argument("--watch", action="store_true", help="redraw every 60 s until no train.py is running")
    args = parser.parse_args()
    run_dirs = [LOG_DIR / r for r in args.runs] if args.runs else sorted(d for d in LOG_DIR.iterdir() if d.is_dir())
    plot(run_dirs, args.out)
    while args.watch and training_running():
        time.sleep(60)
        plot(run_dirs if args.runs else sorted(d for d in LOG_DIR.iterdir() if d.is_dir()), args.out)


if __name__ == "__main__":
    main()
