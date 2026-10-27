#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docker-common.sh"

# docker-common prepares this worktree's directories. Override its shared image
# and Compose project so builds cannot overwrite either source version's image.
export AGENT_HARNESS_IMAGE="agent-harness:${AGENT_HARNESS_USER_SLUG}-rolechange-v4"
export COMPOSE_PROJECT_NAME="agent-harness-${AGENT_HARNESS_USER_SLUG}-rolechange-v4"

case "${1:-}" in
  build)
    docker compose build runtime
    ;;
  unit)
    docker run --rm --network none -e PYTHONDONTWRITEBYTECODE=1 \
      "${AGENT_HARNESS_IMAGE}" python -m unittest discover -s tests -q
    ;;
  lint)
    docker run --rm --network none "${AGENT_HARNESS_IMAGE}" \
      ruff check --no-cache agent_harness benchmarks orchestrations workflows plugins tests
    ;;
  run)
    shift
    docker compose run --rm runtime "$@"
    ;;
  info)
    printf 'image: %s\nproject: %s\nroot: %s\n' \
      "${AGENT_HARNESS_IMAGE}" "${COMPOSE_PROJECT_NAME}" "${AGENT_HARNESS_ROOT_DIR}"
    ;;
  *)
    echo 'Usage: scripts/docker-rolechange-v4.sh {build|unit|lint|info|run <command>...}'
    exit 2
    ;;
esac
