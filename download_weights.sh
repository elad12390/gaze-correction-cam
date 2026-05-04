#!/usr/bin/env bash
set -euo pipefail

ECR_REGISTRY="${ECR_REGISTRY:-448049811979.dkr.ecr.us-east-1.amazonaws.com}"
ECR_REPO="${ECR_REPO:-models/gaze-correction-cam}"
WEIGHTS_TAG="${WEIGHTS_TAG:-weights-v1}"
AWS_REGION="${AWS_REGION:-us-east-1}"

GH_REPO="${GH_REPO:-elad12390/gaze-correction-cam}"
GH_RELEASE="${GH_RELEASE:-weights-v1}"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

LM_OK=0
WEIGHTS_OK=0
if [[ -f lm_feat/shape_predictor_68_face_landmarks.dat ]]; then LM_OK=1; fi
if [[ -f weights/warping_model/flx/12/L/L.index \
   && -f weights/warping_model/flx/12/R/R.index ]]; then WEIGHTS_OK=1; fi

if [[ "$LM_OK" == "1" && "$WEIGHTS_OK" == "1" ]]; then
  echo "[weights] already present — skipping download"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fetch_via_oras() {
  command -v oras >/dev/null 2>&1 || return 1
  command -v aws >/dev/null 2>&1 || return 1
  aws sts get-caller-identity --region "$AWS_REGION" >/dev/null 2>&1 || return 1

  echo "[weights] fetching via ORAS from ${ECR_REGISTRY}/${ECR_REPO}:${WEIGHTS_TAG}"
  aws ecr get-login-password --region "$AWS_REGION" | \
    oras login --username AWS --password-stdin "$ECR_REGISTRY" >/dev/null 2>&1
  oras pull "${ECR_REGISTRY}/${ECR_REPO}:${WEIGHTS_TAG}" --output "$TMP" >/dev/null
  return 0
}

fetch_via_github() {
  local base="https://github.com/${GH_REPO}/releases/download/${GH_RELEASE}"
  echo "[weights] fetching via GitHub Releases from ${base}"
  curl -fsSL --retry 3 --retry-delay 2 -o "$TMP/lm_feat.zip" "${base}/lm_feat.zip"
  curl -fsSL --retry 3 --retry-delay 2 -o "$TMP/weights.zip" "${base}/weights.zip"
}

if ! fetch_via_oras; then
  if [[ "${WEIGHTS_REQUIRE_ECR:-0}" == "1" ]]; then
    echo "[weights] ECR fetch required (WEIGHTS_REQUIRE_ECR=1) but failed — aborting" >&2
    exit 1
  fi
  echo "[weights] ORAS/AWS unavailable — falling back to GitHub Releases (slower)"
  fetch_via_github
fi

unzip -oq "$TMP/lm_feat.zip" -d .
unzip -oq "$TMP/weights.zip" -d .

echo "[weights] OK:"
echo "  $(ls -la lm_feat/shape_predictor_68_face_landmarks.dat 2>/dev/null || echo MISSING)"
echo "  $(ls weights/warping_model/flx/12/L/ 2>/dev/null | wc -l) files in L/"
echo "  $(ls weights/warping_model/flx/12/R/ 2>/dev/null | wc -l) files in R/"
