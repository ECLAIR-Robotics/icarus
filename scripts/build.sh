#!/usr/bin/env bash
# Build ICARUS as a colcon overlay on top of innate-os. The same command runs in the
# dev container and natively on the robot, so what compiles here is what ships.
#
#   scripts/build.sh               build the icarus overlay
#   scripts/build.sh --underlay    first build innate-os's ros2_ws (slow; the dev
#                                  container needs this once for innate-os packages)
#   scripts/build.sh --clean       wipe the overlay's build/install/log first
#
# Any other arguments are passed to `colcon build`.
set -eo pipefail

ICARUS_ROOT="${ICARUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INNATE_OS_ROOT="${INNATE_OS_ROOT:-$HOME/innate-os}"
ICARUS_WS="${ICARUS_WS:-$HOME/icarus_ws}"

build_underlay=0
clean=0
colcon_args=()
for arg in "$@"; do
    case "$arg" in
        --underlay) build_underlay=1 ;;
        --clean) clean=1 ;;
        *) colcon_args+=("$arg") ;;
    esac
done

source /opt/ros/humble/setup.bash

if [ "$build_underlay" -eq 1 ]; then
    echo "==> Building innate-os underlay in $INNATE_OS_ROOT/ros2_ws"
    (
        cd "$INNATE_OS_ROOT/ros2_ws"
        colcon build --cmake-args \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    )
fi

underlay="$INNATE_OS_ROOT/ros2_ws/install/setup.bash"
if [ -f "$underlay" ]; then
    source "$underlay"
else
    echo "warning: no innate-os underlay at $underlay; building against plain ROS 2 Humble." >&2
    echo "         Run 'scripts/build.sh --underlay' if icarus needs innate-os packages." >&2
fi

if [ "$clean" -eq 1 ]; then
    rm -rf "$ICARUS_WS/build" "$ICARUS_WS/install" "$ICARUS_WS/log"
fi

mkdir -p "$ICARUS_WS"
cd "$ICARUS_WS"
# --paths, not --base-paths: base-paths would recurse into the innate-os submodule.
echo "==> Building icarus overlay in $ICARUS_WS"
colcon build --symlink-install --paths "$ICARUS_ROOT" "${colcon_args[@]}"

echo
echo "Done. Load it with:  source $ICARUS_WS/install/setup.zsh   (or setup.bash)"
