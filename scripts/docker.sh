#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docker-common.sh"

usage() {
  cat <<'EOF'
Usage: scripts/docker.sh <command> [arguments]

Commands:
  build          Build or update the current user's AgentHarness image.
  info           Show the effective image, Compose project, and host paths.
  shell          Open an interactive shell inside the project image.
  run <command>  Run any command inside the project image.
  unit           Run all unit tests inside the project image.
  lint           Run Ruff inside the project image.

Examples:
  ./scripts/docker.sh build
  ./scripts/docker.sh shell
  ./scripts/docker.sh run python -m benchmarks.runner.run_subprocess --help
  ./scripts/docker.sh run python examples/real_problem_orchestration.py
EOF
}

case "${1:-}" in
  build)
    docker compose build runtime
    ;;
  info)
    printf 'user:            %s\n' "${AGENT_HARNESS_USER}"
    printf 'user slug:       %s\n' "${AGENT_HARNESS_USER_SLUG}"
    printf 'image:           %s\n' "${AGENT_HARNESS_IMAGE}"
    printf 'compose project: %s\n' "${COMPOSE_PROJECT_NAME}"
    printf 'results:         %s/\n' "${AGENT_HARNESS_HOST_RESULTS_DIR}"
    printf 'logs:            %s/\n' "${AGENT_HARNESS_HOST_LOGS_DIR}"
    printf 'data:            %s/\n' "${AGENT_HARNESS_HOST_DATA_DIR}"
    ;;
  shell)
    docker compose run --rm runtime bash
    ;;
  run)
    shift
    if [[ "$#" -eq 0 ]]; then
      echo "ERROR: scripts/docker.sh run requires a command." >&2
      exit 2
    fi
    docker compose run --rm runtime "$@"
    ;;
  unit)
    docker compose run --rm runtime \
      python -m unittest discover -s tests -v
    ;;
  lint)
    docker compose run --rm runtime \
      ruff check --no-cache agent_harness benchmarks orchestrations workflows plugins tests
    ;;
  -h | --help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
