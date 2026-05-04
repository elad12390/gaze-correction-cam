#!/usr/bin/env bash
set -euo pipefail

ECR_REGISTRY="${ECR_REGISTRY:-448049811979.dkr.ecr.us-east-1.amazonaws.com}"
MODEL_REPO="${MODEL_REPO:-models/gaze-correction-cam}"
WEIGHTS_TAG="${WEIGHTS_TAG:-weights-v1}"
AWS_REGION="${AWS_REGION:-us-east-1}"

IMAGE_REPO="${IMAGE_REPO:-448049811979.dkr.ecr.us-east-1.amazonaws.com/workers/gaze-correction}"
IMAGE_TAG="${IMAGE_TAG:-local}"
PLATFORM="${PLATFORM:-linux/arm64}"
PUSH="${PUSH:-0}"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

WEIGHTS_DIR="$(mktemp -d)"
trap 'rm -rf "$WEIGHTS_DIR"' EXIT

echo "[1/3] oras pull weights → ${WEIGHTS_DIR}"
aws ecr get-login-password --region "$AWS_REGION" | \
  oras login --username AWS --password-stdin "$ECR_REGISTRY" >/dev/null
oras pull "${ECR_REGISTRY}/${MODEL_REPO}:${WEIGHTS_TAG}" --output "$WEIGHTS_DIR"

echo "[2/3] docker buildx build → ${IMAGE_REPO}:${IMAGE_TAG}"
EXTRA_ARGS=()
if [[ "$PUSH" == "1" ]]; then
  EXTRA_ARGS+=(--push)
  aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_REGISTRY"
else
  EXTRA_ARGS+=(--load)
fi

docker buildx build \
  --platform "$PLATFORM" \
  --build-context "weights=${WEIGHTS_DIR}" \
  --tag "${IMAGE_REPO}:${IMAGE_TAG}" \
  "${EXTRA_ARGS[@]}" \
  .

echo "[3/3] done"
echo "  Image: ${IMAGE_REPO}:${IMAGE_TAG}"
echo "  Platform: ${PLATFORM}"
echo "  Pushed: $([[ "$PUSH" == "1" ]] && echo yes || echo no)"
