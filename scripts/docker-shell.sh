#!/usr/bin/env bash
# Open a shell in the icarus-dev container, starting it if needed. Every terminal that
# runs this joins the same long-lived container, so builds and running nodes are shared.
#
# The shell opens in the directory you ran this from. Inside this repo that is the same
# spot under /root/icarus; anywhere else, the folder is mounted at its host path (which
# recreates the container if it wasn't mounted yet).
#
#   scripts/docker-shell.sh                  zsh in the container
#   scripts/docker-shell.sh <cmd> [args...]  run one command, e.g. scripts/docker-shell.sh scripts/build.sh
#   scripts/docker-shell.sh --sim            attach to the running simulator's network first
#                                            (start it with: cd innate-os && ./innate-sim up)
#   scripts/docker-shell.sh --restart        recreate the container (closes other open shells)
#   scripts/docker-shell.sh --stop           stop and remove the container (build volumes are kept)
set -euo pipefail

ICARUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ICARUS_ROOT/docker/.env"
COMPOSE=(docker compose -f "$ICARUS_ROOT/docker/docker-compose.yml")

die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
note() { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }

sim=0
restart=0
while [ $# -gt 0 ]; do
    case "$1" in
        --sim) sim=1 ;;
        --restart) restart=1 ;;
        --stop) exec "${COMPOSE[@]}" down ;;
        -h | --help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) break ;;
    esac
    shift
done

[ ! -f /.dockerenv ] || die "You are already inside a container."
docker info >/dev/null 2>&1 || die "Cannot reach the Docker daemon. Is Docker running?"

if [ ! -f "$ENV_FILE" ] || ! docker image inspect icarus-dev:latest >/dev/null 2>&1; then
    note "icarus-dev image not built yet; building it first."
    "$ICARUS_ROOT/scripts/docker-build.sh"
fi

host_pwd="$(pwd -P)"
repo_real="$(cd "$ICARUS_ROOT" && pwd -P)"
case "$host_pwd/" in
    "$repo_real"/*)
        workdir="/root/icarus${host_pwd#"$repo_real"}"
        ;;
    *)
        case "$host_pwd" in
            / | /root | /root/* | /opt | /opt/* | /usr | /usr/* | /etc | /etc/* | /bin | /bin/* | \
                /lib | /lib/* | /proc | /proc/* | /sys | /sys/* | /dev | /dev/*)
                die "Refusing to mount $host_pwd: it would shadow the container's own $host_pwd. cd somewhere else." ;;
        esac
        workdir="$host_pwd"
        export ICARUS_PWD_MOUNT="$host_pwd"
        ;;
esac

container="$("${COMPOSE[@]}" ps -q icarus 2>/dev/null || true)"
current_net=""
if [ -n "$container" ]; then
    current_net="$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$container")"
    if [ -n "${ICARUS_PWD_MOUNT:-}" ] &&
        ! docker inspect -f '{{range .Mounts}}{{.Destination}}{{"\n"}}{{end}}' "$container" | grep -qFx "$workdir"; then
        restart=1
    fi
fi

if [ "$sim" -eq 1 ]; then
    sim_name="$(sed -n 's/^INNATE_SIM_CONTAINER=//p' "$ENV_FILE")"
    sim_id="$(docker inspect -f '{{if .State.Running}}{{.Id}}{{end}}' "$sim_name" 2>/dev/null || true)"
    [ -n "$sim_id" ] ||
        die "Simulator container $sim_name is not running. Start it with: cd innate-os && ./innate-sim up"
    export ICARUS_NETWORK="container:$sim_name"
    # A restarted sim is a new container; a shell still in the old one's network sees nothing.
    [ "$current_net" = "container:$sim_id" ] || [ -z "$container" ] || restart=1
elif [[ "$current_net" == container:* ]]; then
    # Keep an existing sim attachment rather than yanking it out from under other shells.
    export ICARUS_NETWORK="$current_net"
fi

if [ -z "$container" ] || [ "$restart" -eq 1 ]; then
    [ -z "$container" ] || note "Recreating the icarus container (other open shells will close)."
    "${COMPOSE[@]}" up -d --force-recreate icarus >/dev/null ||
        die "Could not start the container. If Docker said 'mounts denied', add $host_pwd to Docker Desktop's File Sharing, or run from a shared folder such as your home directory."
fi

if [ $# -eq 0 ]; then
    set -- zsh -l
fi
tty_flag=()
[ -t 0 ] && [ -t 1 ] || tty_flag=(-T)
exec "${COMPOSE[@]}" exec -w "$workdir" ${tty_flag[@]+"${tty_flag[@]}"} icarus /usr/local/bin/icarus-entrypoint "$@"
