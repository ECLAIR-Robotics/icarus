# syntax=docker/dockerfile:1.7
#
# ICARUS developer image: a coding, build and test environment for the icarus ROS 2
# package, layered on Innate's own simulator dependency image so everything compiles
# against exactly the ROS 2 Humble + Zenoh + apt/pip stack that runs on MARS.
#
# Build and run it through docker/docker-compose.yml (scripts/setup.sh does both).
# Source is bind-mounted, never baked: edits need no image rebuild.
#
#   /root/icarus       this repository (bind mount; includes the innate-os submodule)
#   /root/innate-os    symlink to /root/icarus/innate-os -- innate scripts assume ~/innate-os
#   /root/icarus_ws    colcon overlay workspace for icarus (named volume)

# scripts/setup.sh resolves the published, content-addressed deps image for the pinned
# innate-os submodule (ghcr.io/innate-inc/innate-os-sim-deps:deps-<hash>). The default
# is the tag innate-os's own sim/Dockerfile produces when built locally.
ARG BASE_IMAGE=innate-os-sim-clean-innate:latest
FROM ${BASE_IMAGE} AS icarus-dev

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV ICARUS_ROOT=/root/icarus \
    ICARUS_WS=/root/icarus_ws \
    INNATE_OS_ROOT=/root/innate-os \
    CCACHE_DIR=/root/.cache/ccache \
    INNATE_ZENOH_DISABLE_SHM=1 \
    SKIP_UPDATE_CHECK=1

# Developer tooling on top of Innate's runtime deps. No GUI tools: use Foxglove
# (ws://localhost:8765 while the sim runs) instead of rviz.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        clang-format \
        clangd \
        fd-find \
        gdb \
        git-lfs \
        less \
        nano \
        python3-venv \
        ripgrep \
        rsync \
        vim

# The checks innate-os gates its own PRs on (see innate-os/CLAUDE.md), plus a REPL.
# setuptools is re-pinned last: ament_python on Humble breaks on newer setuptools.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --system \
        basedpyright \
        ipython \
        pre-commit \
        ruff && \
    uv pip install --system 'setuptools==59.6.0'

# The base image made /root/innate-os a directory of empty mount points for
# innate-os's compose file; here the submodule arrives inside the icarus mount.
RUN rm -rf /root/innate-os && \
    ln -s /root/icarus/innate-os /root/innate-os && \
    mkdir -p /root/icarus /root/icarus_ws

# /root/icarus is a host bind mount, so its files keep the host UID while this container
# runs as root; without this, git's ownership check rejects `git status`/submodule
# commands in the dev shell. The host ~/.gitconfig is mounted read-only, so it is set here.
RUN git config --system --add safe.directory /root/icarus && \
    git config --system --add safe.directory /root/icarus/innate-os

# Non-interactive commands (`docker compose run icarus ros2 launch ...`) get the same
# environment an interactive zsh does: ROS, Zenoh transport, underlay, then overlay.
COPY --chmod=755 <<'EOF' /usr/local/bin/icarus-entrypoint
#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
for f in "$INNATE_OS_ROOT/config/dds/setup_dds.zsh" \
         "$INNATE_OS_ROOT/ros2_ws/install/setup.bash" \
         "$ICARUS_WS/install/setup.bash"; do
    [ -f "$f" ] && source "$f"
done
exec "$@"
EOF

# Appended after the base image's Innate block, so the icarus overlay wins.
RUN cat >> /root/.zshrc <<'EOF'

# ----- ICARUS environment -----
[ -f /root/icarus_ws/install/setup.zsh ] && source /root/icarus_ws/install/setup.zsh
export PATH="/root/icarus/scripts:$PATH"
alias icarus-build='/root/icarus/scripts/build.sh'
EOF

WORKDIR /root/icarus
ENTRYPOINT ["/usr/local/bin/icarus-entrypoint"]
CMD ["zsh", "-l"]
