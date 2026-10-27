#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="smoke"

usage() {
  cat <<'EOF'
Usage: scripts/run-bc200-docker.sh [--smoke|--full]

Builds the Docker image, runs the container preflight, unit tests, and lint,
then executes a BC200 task test.

  --smoke  Evaluate a small sample after validation (default).
  --full   Run the smoke test, then evaluate all 200 questions.

Environment overrides are forwarded to scripts/docker-bc200.sh:

  PROFILE=default|keep5
  MODEL_CONFIG=gpt_5_mini.yaml|deepseek_v4_flash.yaml
  SMOKE_LIMIT=1
  RUNS=1
  CONCURRENCY=1
  OUT_DIR=/app/results/browsecomp_200/react_base-default
EOF
}

case "${1:-}" in
  "" | --smoke)
    MODE="smoke"
    ;;
  --full)
    MODE="full"
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed or is not on PATH." >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: Docker Compose v2 is unavailable." >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  echo "ERROR: .env is missing. Create it from .env.example first." >&2
  exit 1
fi
if [[ ! -f benchmarks/datasets/BrowseComp-200/standardized_data.jsonl ]]; then
  echo "ERROR: BrowseComp-200 dataset is missing." >&2
  echo "Expected: benchmarks/datasets/BrowseComp-200/standardized_data.jsonl" >&2
  exit 1
fi

run_step() {
  local label="$1"
  shift
  printf '\n==> %s\n' "${label}"
  "$@"
}

run_step "Resolving per-user Docker isolation" \
  ./scripts/docker-bc200.sh info
run_step "Building AgentHarness Docker image" \
  ./scripts/docker-bc200.sh build
run_step "Running container preflight" \
  ./scripts/docker-bc200.sh check
run_step "Running unit tests inside Docker" \
  ./scripts/docker-bc200.sh unit
run_step "Running Ruff inside Docker" \
  ./scripts/docker-bc200.sh lint
run_step "Running BC200 smoke test inside Docker" \
  ./scripts/docker-bc200.sh smoke

if [[ "${MODE}" == "full" ]]; then
  run_step "Running the complete BC200 evaluation inside Docker" \
    ./scripts/docker-bc200.sh full
  printf '\nBC200 evaluation finished. See the per-user results directory above.\n'
else
  printf '\nSmoke test finished. See the per-user results directory above.\n'
  printf 'Run the full evaluation with:\n'
  printf '  RUNS=1 CONCURRENCY=8 %s --full\n' "$0"
fi
