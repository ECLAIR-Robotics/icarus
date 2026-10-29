#!/usr/bin/env bash
# Trains the nav (goal reaching + obstacle avoidance) policy on every GPU of this node, one
# process per GPU. Works in an idev session or as a Slurm batch job:
#
#   mjwarp/train.sh [train.py args...]                 # e.g. --num-envs 4096 --max-iterations 2000
#   sbatch -p rtx-small -N 1 -t 12:00:00 -A <account> mjwarp/train.sh   # submit from the repo root or mjwarp/
#
# Extra arguments go to train.py. NGPUS overrides the GPU count.

set -euo pipefail

MJWARP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# sbatch runs a copy of this script from Slurm's spool dir; find mjwarp/ from the submit dir instead.
if [ ! -f "$MJWARP_DIR/train.py" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    for candidate in "$SLURM_SUBMIT_DIR" "$SLURM_SUBMIT_DIR/mjwarp"; do
        [ -f "$candidate/train.py" ] && MJWARP_DIR="$candidate" && break
    done
fi
cd "$MJWARP_DIR"
[ -f .venv/bin/activate ] || { echo "error: no .venv in $MJWARP_DIR -- see the setup in train.py's docstring" >&2; exit 1; }
source .venv/bin/activate

NGPUS="${NGPUS:-$(nvidia-smi -L | wc -l)}"
echo "==> Training on $NGPUS GPU(s)"
exec torchrun --standalone --nproc_per_node="$NGPUS" train.py "$@"
