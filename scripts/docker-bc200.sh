#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docker-common.sh"

HOST_BC200_RESULTS_DIR="${AGENT_HARNESS_HOST_RESULTS_DIR}/browsecomp_200"
mkdir -p "${HOST_BC200_RESULTS_DIR}"

PROFILE="${PROFILE:-default}"
MODEL_CONFIG="${MODEL_CONFIG:-gpt_5_mini.yaml}"
RUNS="${RUNS:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
SMOKE_LIMIT="${SMOKE_LIMIT:-1}"

compose_run() {
  docker compose run --rm \
    -e "REACT_BASE_MODEL_CONFIG=${MODEL_CONFIG}" \
    runtime \
    python -m benchmarks.runner.run_subprocess "$@"
}

usage() {
  cat <<'EOF'
Usage: scripts/docker-bc200.sh <command> [--model-config FILE] [extra arguments]

Commands:
  build   Build the AgentHarness image.
  info    Show the effective per-user Docker and output names.
  check   Validate the container, dataset registration, and required variables.
  smoke   Run a small evaluation (default: one question, concurrency one).
  full    Run all 200 BrowseComp questions.
  status  Show progress and aggregate accuracy for the full-run output.
  unit    Run the repository unit tests inside the image.
  lint    Run Ruff inside the image.

Environment overrides:
  PROFILE=default|keep5
  MODEL_CONFIG=gpt_5_mini.yaml|deepseek_v4_flash.yaml
  RUNS=1
  CONCURRENCY=1
  SMOKE_LIMIT=1
  OUT_DIR=/app/results/browsecomp_200/react_base-default-gpt_5_mini
  SMOKE_OUT_DIR=/app/results/browsecomp_200/smoke-default-gpt_5_mini
EOF
}

COMMAND="${1:-}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model-config)
      if [[ "$#" -lt 2 ]]; then
        echo "ERROR: --model-config requires a YAML file." >&2
        exit 2
      fi
      MODEL_CONFIG="$2"
      shift 2
      ;;
    --model-config=*)
      MODEL_CONFIG="${1#*=}"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

MODEL_CONFIG_NAME="${MODEL_CONFIG##*/}"
MODEL_CONFIG_STEM="${MODEL_CONFIG_NAME%.yaml}"
OUT_DIR="${OUT_DIR:-/app/results/browsecomp_200/react_base-${PROFILE}-${MODEL_CONFIG_STEM}}"
SMOKE_OUT_DIR="${SMOKE_OUT_DIR:-/app/results/browsecomp_200/smoke-${PROFILE}-${MODEL_CONFIG_STEM}}"

case "${COMMAND}" in
  build)
    ./scripts/docker.sh build
    ;;
  info)
    ./scripts/docker.sh info
    printf 'BC200 results:   %s/\n' "${HOST_BC200_RESULTS_DIR}"
    ;;
  check)
    docker compose run --rm \
      -e "REACT_BASE_MODEL_CONFIG=${MODEL_CONFIG}" \
      runtime \
      python -m benchmarks.runner.docker_preflight \
      --benchmark browsecomp_200
    ;;
  smoke)
    compose_run \
      --benchmark browsecomp_200 \
      --pipeline react_base \
      --profile "${PROFILE}" \
      --runs 1 \
      --limit "${SMOKE_LIMIT}" \
      --concurrency 1 \
      --out "${SMOKE_OUT_DIR}" \
      "${EXTRA_ARGS[@]}"
    ;;
  full)
    compose_run \
      --benchmark browsecomp_200 \
      --pipeline react_base \
      --profile "${PROFILE}" \
      --runs "${RUNS}" \
      --concurrency "${CONCURRENCY}" \
      --out "${OUT_DIR}" \
      "${EXTRA_ARGS[@]}"
    ;;
  status)
    docker compose run --rm \
      runtime \
      python -m benchmarks.runner.check_progress \
      "${OUT_DIR}"
    ;;
  unit)
    ./scripts/docker.sh unit
    ;;
  lint)
    ./scripts/docker.sh lint
    ;;
  *)
    usage
    exit 2
    ;;
esac
