#!/usr/bin/env bash
# Setup script for ICARUS.
#
# Laptop / desktop (Linux, WSL2, macOS):
#   scripts/setup.sh              everything: submodules, Docker + uv + sim runtime
#                                 (via innate-os's installer), host lint tools, dev image
#   scripts/setup.sh --skip-sim   skip innate-os's simulator installer
#   scripts/setup.sh --skip-image skip building the icarus-dev Docker image
#
# On the MARS robot (native, no Docker; uses the robot's own ~/innate-os):
#   scripts/setup.sh --robot
#
# Not yet cloned? Either clone first:
#   git clone --recursive git@github.com:ECLAIR-Robotics/icarus.git && icarus/scripts/setup.sh
# or run this file from anywhere and it clones into ./icarus.
#
# Safe to re-run: every step checks before it acts.
set -euo pipefail

REPO_URL="${ICARUS_REPO_URL:-git@github.com:ECLAIR-Robotics/icarus.git}"

mode=laptop
skip_sim=0
skip_image=0

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; }

for arg in "$@"; do
    case "$arg" in
        --robot) mode=robot ;;
        --skip-sim) skip_sim=1 ;;
        --skip-image) skip_image=1 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

sudo_cmd() {
    if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi
}

apt_install() {
    have apt-get || die "apt-get not found; install manually: $*"
    sudo_cmd apt-get update -qq
    sudo_cmd apt-get install -y --no-install-recommends "$@"
}

locate_repo() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$script_dir/../package.xml" ] && grep -q '<name>icarus</name>' "$script_dir/../package.xml"; then
        ICARUS_ROOT="$(cd "$script_dir/.." && pwd)"
        return
    fi

    have git || apt_install git
    ICARUS_ROOT="$(pwd)/icarus"
    if [ ! -d "$ICARUS_ROOT/.git" ]; then
        step "Cloning ICARUS into $ICARUS_ROOT"
        # The robot uses its own ~/innate-os, so it skips the ~1 GB submodule.
        if [ "$mode" = robot ]; then
            git clone "$REPO_URL" "$ICARUS_ROOT"
        else
            git clone --recursive "$REPO_URL" "$ICARUS_ROOT"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Laptop
# ---------------------------------------------------------------------------

check_platform() {
    case "$(uname -s)" in
        Linux)
            if grep -qi microsoft /proc/version 2>/dev/null && [[ "$ICARUS_ROOT" == /mnt/* ]]; then
                warn "The repo is on the Windows drive ($ICARUS_ROOT). Docker bind mounts there are very slow;"
                warn "clone into your WSL home (~) instead."
            fi
            ;;
        Darwin)
            have docker || die "Install Docker Desktop (https://docs.docker.com/desktop/install/mac-install/), open it once, then re-run."
            ;;
        *) die "Unsupported OS $(uname -s). Use Linux, WSL2, or macOS." ;;
    esac
}

init_submodules() {
    step "Fetching the innate-os submodule"
    git -C "$ICARUS_ROOT" submodule update --init --recursive
}

# innate-os's installer owns Docker, uv, git, the GL rendering libraries, the LLM
# key prompt and the multi-GB sim runtime download; duplicating it would drift.
install_sim() {
    step "Installing Docker, uv and the Innate simulator runtime (innate-os/scripts/install-sim.sh)"
    echo "When it asks 'Start the simulator now?', answer No to finish ICARUS setup first."
    echo "(If you answer Yes, stop the sim with Ctrl-C and this script continues.)"
    (cd "$ICARUS_ROOT/innate-os" && sh scripts/install-sim.sh) ||
        die "innate-os simulator install failed; see ~/.innate-install.log, fix, and re-run."
}

# ROS/Python deps live in the container; these are only so the host editor can lint.
install_host_tools() {
    step "Installing host lint/type-check tools (ruff, basedpyright, pre-commit) via uv"
    local uv
    uv="$(command -v uv || echo "$HOME/.local/bin/uv")"
    if [ ! -x "$uv" ]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        uv="$HOME/.local/bin/uv"
    fi
    for tool in ruff basedpyright pre-commit; do
        "$uv" tool install --quiet "$tool" || warn "could not install $tool"
    done
}

build_image() {
    step "Building the icarus-dev Docker image"
    "$ICARUS_ROOT/scripts/docker-build.sh" || {
        warn "Image build failed. If Docker was just installed, log out and back in, then: scripts/docker-build.sh"
        return 1
    }
}

print_laptop_next_steps() {
    cat <<EOF

$(printf '\033[1;32m')ICARUS setup complete.$(printf '\033[0m')

  Dev shell (build, test, lint):
    cd $ICARUS_ROOT
    scripts/docker-shell.sh
    scripts/build.sh                      # inside the container

  Simulator (runs on the host; web UI at https://localhost):
    cd $ICARUS_ROOT/innate-os && ./innate-sim up

  Dev shell attached to the running sim (ros2 topic list, ros2 launch icarus ...):
    scripts/docker-shell.sh --sim

  Deploy to the robot:
    ssh into MARS, clone this repo, run: scripts/setup.sh --robot
EOF
}

setup_laptop() {
    check_platform
    init_submodules
    [ "$skip_sim" -eq 1 ] || install_sim
    install_host_tools
    if [ "$skip_image" -eq 0 ]; then
        build_image || exit 1
    fi
    print_laptop_next_steps
}

# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

setup_robot() {
    INNATE_OS_ROOT="${INNATE_OS_ROOT:-$HOME/innate-os}"
    [ -f "$INNATE_OS_ROOT/scripts/innate" ] ||
        die "No innate-os at $INNATE_OS_ROOT. Run this on a provisioned MARS, or set INNATE_OS_ROOT."
    [ -f /opt/ros/humble/setup.bash ] || die "ROS 2 Humble not found at /opt/ros/humble."

    local pinned robot_rev
    pinned="$(git -C "$ICARUS_ROOT" ls-tree HEAD innate-os 2>/dev/null | awk '{print $3}')"
    robot_rev="$(git -C "$INNATE_OS_ROOT" rev-parse HEAD 2>/dev/null || true)"
    if [ -n "$pinned" ] && [ -n "$robot_rev" ] && [ "$pinned" != "$robot_rev" ]; then
        warn "Robot runs innate-os $(git -C "$INNATE_OS_ROOT" describe --tags --always 2>/dev/null),"
        warn "but ICARUS pins ${pinned:0:8}. Code tested in sim may behave differently."
    fi

    step "Installing icarus system dependencies with rosdep"
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    rosdep update --rosdistro humble >/dev/null || warn "rosdep update failed; using cached index"
    rosdep install --from-paths "$ICARUS_ROOT" --ignore-src -y -r ||
        warn "rosdep could not resolve every key; the build will name anything truly missing"

    step "Building the icarus overlay"
    INNATE_OS_ROOT="$INNATE_OS_ROOT" "$ICARUS_ROOT/scripts/build.sh"

    local rc="$HOME/.zshrc" line="[ -f \$HOME/icarus_ws/install/setup.zsh ] && source \$HOME/icarus_ws/install/setup.zsh"
    if [ -f "$rc" ] && ! grep -qF "icarus_ws/install/setup.zsh" "$rc"; then
        printf '\n# ICARUS overlay\n%s\n' "$line" >>"$rc"
    fi

    cat <<EOF

$(printf '\033[1;32m')ICARUS installed on the robot.$(printf '\033[0m')
  Open a new shell (or: source ~/icarus_ws/install/setup.zsh), then:
    ros2 launch icarus icarus.launch.py
  After pulling changes: $ICARUS_ROOT/scripts/build.sh
EOF
}

# ---------------------------------------------------------------------------

locate_repo
if [ "$mode" = robot ]; then
    setup_robot
else
    setup_laptop
fi
