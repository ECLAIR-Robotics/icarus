#!/usr/bin/env bash
# Open a shell in the icarus-dev container, starting it if needed. Every terminal that
# runs this joins the same long-lived container, so builds and running nodes are shared.
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
        -h | --help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

container="$("${COMPOSE[@]}" ps -q icarus 2>/dev/null || true)"
current_net=""
[ -z "$container" ] || current_net="$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$container")"

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
    "${COMPOSE[@]}" up -d --force-recreate icarus >/dev/null
fi

if [ $# -eq 0 ]; then
    set -- zsh -l
fi
tty_flag=()
[ -t 0 ] && [ -t 1 ] || tty_flag=(-T)
exec "${COMPOSE[@]}" exec ${tty_flag[@]+"${tty_flag[@]}"} icarus /usr/local/bin/icarus-entrypoint "$@"
