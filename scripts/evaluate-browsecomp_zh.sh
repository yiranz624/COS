#!/usr/bin/env bash
set -euo pipefail

./scripts/docker.sh run python -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp_zh \
  --pipeline react_base \
  --profile keep5 \
  --runs 1 \
  --concurrency 50 \
  --out ./results/$(date +%F)_browsecomp_zh


# ./scripts/docker.sh run python -m benchmarks.runner.check_progress ./results/$(date +%F)_browsecomp_zh
