#!/usr/bin/env bash

AGENT_HARNESS_ROOT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
)"
cd "${AGENT_HARNESS_ROOT_DIR}"

export DOCKER_UID="$(id -u)"
export DOCKER_GID="$(id -g)"

AGENT_HARNESS_USER="$(id -un)"
AGENT_HARNESS_USER_SLUG="$(
  printf '%s' "${AGENT_HARNESS_USER}" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
)"
if [[ -z "${AGENT_HARNESS_USER_SLUG}" ]]; then
  AGENT_HARNESS_USER_SLUG="uid-${DOCKER_UID}"
fi

export AGENT_HARNESS_USER_SLUG
export AGENT_HARNESS_IMAGE="agent-harness:${AGENT_HARNESS_USER_SLUG}-local"
export COMPOSE_PROJECT_NAME="agent-harness-${AGENT_HARNESS_USER_SLUG}"

AGENT_HARNESS_HOST_RESULTS_DIR="results/${AGENT_HARNESS_USER_SLUG}"
AGENT_HARNESS_HOST_LOGS_DIR="logs/${AGENT_HARNESS_USER_SLUG}"
AGENT_HARNESS_HOST_DATA_DIR="data/${AGENT_HARNESS_USER_SLUG}"

mkdir -p \
  benchmarks/datasets \
  "${AGENT_HARNESS_HOST_RESULTS_DIR}" \
  "${AGENT_HARNESS_HOST_LOGS_DIR}" \
  "${AGENT_HARNESS_HOST_DATA_DIR}"
