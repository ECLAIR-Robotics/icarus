#!/usr/bin/env bash
# While a training run is going, renders a play.py video for every checkpoint whose
# iteration is a multiple of EVERY (default 200), into <run>/videos/model_<N>.mp4.
# Exits once no train.py process is left and every such checkpoint has a video.
#
#   nohup mjwarp/watch_videos.sh [run_dir] > mjwarp/logs/watch_videos.out 2>&1 &
#
# run_dir defaults to the newest run under logs/nav. EVERY and SECONDS_PER_VIDEO override
# the interval and the video length.

set -euo pipefail

MJWARP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MJWARP_DIR"
source .venv/bin/activate

RUN_DIR="${1:-$(ls -d logs/nav/*/ | sort | tail -1)}"
EVERY="${EVERY:-200}"
SECONDS_PER_VIDEO="${SECONDS_PER_VIDEO:-30}"
echo "==> Watching $RUN_DIR: a video every $EVERY iterations"

render_pending() {
    for checkpoint in "$RUN_DIR"/model_*.pt; do
        [ -e "$checkpoint" ] || continue
        iteration="$(basename "$checkpoint" .pt)"
        iteration="${iteration#model_}"
        (( iteration % EVERY == 0 )) || continue
        [ -f "$RUN_DIR/videos/model_${iteration}.mp4" ] && continue
        # Skip checkpoints still being written.
        (( $(date +%s) - $(stat -c %Y "$checkpoint") > 30 )) || continue
        echo "==> Rendering iteration $iteration"
        python play.py --checkpoint "$checkpoint" --seconds "$SECONDS_PER_VIDEO" 2>&1 | grep -E "Episodes|Video|rror" || true
    done
}

while pgrep -f "train.py" > /dev/null; do
    render_pending
    sleep 60
done
render_pending
echo "==> Training finished; watcher done"
