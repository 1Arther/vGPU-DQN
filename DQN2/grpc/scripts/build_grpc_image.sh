#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-vgpu-dqn-grpc:v8}
CONTEXT=${CONTEXT:-$(git rev-parse --show-toplevel)}

cd "$CONTEXT"
if command -v docker >/dev/null 2>&1; then
  docker build -f DQN2/grpc/Dockerfile -t "$IMAGE" .
elif command -v nerdctl >/dev/null 2>&1; then
  nerdctl build -f DQN2/grpc/Dockerfile -t "$IMAGE" .
else
  echo "docker or nerdctl is required to build $IMAGE" >&2
  exit 1
fi
