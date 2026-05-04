#!/usr/bin/env bash
set -euo pipefail

WEIGHTS_RELEASE="${WEIGHTS_RELEASE:-weights-v1}"
WEIGHTS_REPO="${WEIGHTS_REPO:-elad12390/gaze-correction-cam}"
BASE_URL="https://github.com/${WEIGHTS_REPO}/releases/download/${WEIGHTS_RELEASE}"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

LM_OK=0
WEIGHTS_OK=0

if [[ -f lm_feat/shape_predictor_68_face_landmarks.dat ]]; then
  LM_OK=1
fi

if [[ -f weights/warping_model/flx/12/L/L.index \
   && -f weights/warping_model/flx/12/R/R.index ]]; then
  WEIGHTS_OK=1
fi

if [[ "$LM_OK" == "1" && "$WEIGHTS_OK" == "1" ]]; then
  echo "[weights] already present — skipping download"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [[ "$LM_OK" == "0" ]]; then
  echo "[weights] downloading lm_feat.zip from ${BASE_URL}/lm_feat.zip"
  curl -fsSL --retry 3 --retry-delay 2 \
    -o "${TMP}/lm_feat.zip" \
    "${BASE_URL}/lm_feat.zip"
  unzip -oq "${TMP}/lm_feat.zip" -d .
fi

if [[ "$WEIGHTS_OK" == "0" ]]; then
  echo "[weights] downloading weights.zip from ${BASE_URL}/weights.zip"
  curl -fsSL --retry 3 --retry-delay 2 \
    -o "${TMP}/weights.zip" \
    "${BASE_URL}/weights.zip"
  unzip -oq "${TMP}/weights.zip" -d .
fi

echo "[weights] OK:"
echo "  $(ls -la lm_feat/shape_predictor_68_face_landmarks.dat 2>/dev/null || echo MISSING)"
echo "  $(ls weights/warping_model/flx/12/L/ 2>/dev/null | wc -l) files in L/"
echo "  $(ls weights/warping_model/flx/12/R/ 2>/dev/null | wc -l) files in R/"
