#!/usr/bin/env bash
# Starts the full MJWarp -> ROS -> Foxglove pipeline: Zenoh router, robot_state_publisher,
# foxglove_bridge, then the MJWarp bridge node, in order. Run from inside the icarus-dev
# container (a shell started via scripts/docker-shell.sh) -- this needs ROS 2 + rclpy,
# which only exist there, not on the bare host.
#
#   mjwarp/run.sh
#
# Ctrl+C stops the bridge node and tears down everything else this script started.

usage() {
    echo "Usage: $0 [-m] [-n ####]"
    echo "  -m          Run a manual control instance"
    echo "  -n ####     Run with N-world instances"
    exit 1
}

RUN_TYPE=""
NUM_OF_INSTANCES=0

while getopts "mn:" flag; do 
    case "${flag}" in 
        m)
            if [ -n "$RUN_TYPE" ]; then usage; fi
            RUN_TYPE="MANUAL"
            NUM_OF_INSTANCES=1
            ;;
        n)
            if [ -n "$RUN_TYPE" ]; then usage; fi
            RUN_TYPE="AUTO"
            NUM_OF_INSTANCES=$OPTARG
            ;;
        *)
            usage
            ;;
    esac
done

if [ -z "$RUN_TYPE" ]; then
    usage
fi

set -euo pipefail

MJWARP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URDF_PATH="/root/icarus/innate-os/ros2_ws/src/mars_bot/mars_sim/urdf/mars.urdf"

note() { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[ -f "$URDF_PATH" ] || die "mars.urdf not found at $URDF_PATH"

if [ "$RUN_TYPE" = "AUTO" ]; then
    # Pure headless physics: no ROS, no Foxglove, no router -- none of it is needed just to
    # step N worlds and let something else consume the data later.
    note "Starting $NUM_OF_INSTANCES worlds headlessly (no visualization, no control)..."
    cd "$MJWARP_DIR"
    exec python3 collect.py "$NUM_OF_INSTANCES"
fi

# RUN_TYPE = MANUAL from here on: this path needs ROS 2 + the innate-os underlay.
[ -f /opt/ros/humble/setup.bash ] || die "ROS 2 not found -- run this inside the icarus-dev container (scripts/docker-shell.sh), not the bare host."
echo "${AMENT_PREFIX_PATH:-}" | grep -q "innate-os/ros2_ws/install" ||
    die "innate-os underlay not sourced -- run 'scripts/build.sh --underlay' once, then open a fresh shell."

# Clean slate: duplicate instances of any of these (stray background jobs left running in
# other terminals) have repeatedly caused silent port conflicts and ROS duplicate-node bugs.
note "Clearing any stray instances from previous runs..."
pkill -f ros_bridge.py 2>/dev/null || true
pkill -f robot_state_publisher 2>/dev/null || true
pkill -f foxglove_bridge 2>/dev/null || true
pkill -f rmw_zenohd 2>/dev/null || true
pkill -f teleop_twist_keyboard 2>/dev/null || true
sleep 1

pids=()
cleanup() {
    note "Shutting down..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

note "Starting Zenoh router..."
ros2 run rmw_zenoh_cpp rmw_zenohd &
pids+=("$!")
sleep 2

note "Starting robot_state_publisher..."
ros2 launch "$MJWARP_DIR/rsp_launch.py" &
pids+=("$!")
sleep 2

note "Starting foxglove_bridge (ws://localhost:8765)..."
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765 &
pids+=("$!")
sleep 2

note "Starting MJWarp bridge node (background)..."
cd "$MJWARP_DIR"
python3 ros_bridge.py &
pids+=("$!")
sleep 2

note "Confirming exactly one of each node is up..."
ros2 node list

note "Starting teleop_twist_keyboard -- this terminal now reads your keystrokes directly."
note "WASD (per its own on-screen legend) drives the base; arm/head hold their actuated zero pose."
note "Ctrl+C here stops teleop and tears down everything else this script started."
ros2 run teleop_twist_keyboard teleop_twist_keyboard
