#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TP_DIR="${ROOT_DIR}/third_party"
MODE="${1:-full}"

ROBOSUITE_DIR="${TP_DIR}/robosuite"
ROBOMIMIC_DIR="${TP_DIR}/robomimic"
MIMICGEN_DIR="${TP_DIR}/mimicgen"
ROBOSUITE_ORIGIN_URL="https://github.com/jren03/robosuite.git"
ROBOMIMIC_ORIGIN_URL="https://github.com/jren03/robomimic.git"
MIMICGEN_ORIGIN_URL="https://github.com/jren03/mimicgen.git"

# Fork compatibility branches track the original MimicGen-compatible upstream pins:
# - robosuite upstream base:  b9d8d3de5e3dfd1724f4a0e6555246c460407daa
# - robomimic upstream base:  d0b37cf214bd24fb590d182edb6384333f67b661
# - mimicgen upstream base:   v1.0.0 / ea0988523f468ccf7570475f1906023f854962e9
#
# Override at runtime if you prefer a different branch or ref, e.g.
#   ROBOSUITE_REF=main ROBOMIMIC_REF=main MIMICGEN_REF=main ./install_deps.sh
ROBOSUITE_REF_DEFAULT="compat/mimicgen-stack"
ROBOMIMIC_REF_DEFAULT="compat/mimicgen-stack"
MIMICGEN_REF_DEFAULT="compat/mimicgen-stack"
ROBOSUITE_REF="${ROBOSUITE_REF:-$ROBOSUITE_REF_DEFAULT}"
ROBOMIMIC_REF="${ROBOMIMIC_REF:-$ROBOMIMIC_REF_DEFAULT}"
MIMICGEN_REF="${MIMICGEN_REF:-$MIMICGEN_REF_DEFAULT}"

if [[ -t 1 ]]; then
  _CLR_HDR=$'\033[1;36m'  # bold cyan
  _CLR_RST=$'\033[0m'
else
  _CLR_HDR=""
  _CLR_RST=""
fi

hdr() { echo "${_CLR_HDR}==> $*${_CLR_RST}"; }

hdr "sim stack setup (mimicgen-compatible)"
echo "Repo root: ${ROOT_DIR}"
echo "third_party: ${TP_DIR}"
echo ""

command -v git >/dev/null 2>&1 || { echo "git not found"; exit 1; }
command -v timeout >/dev/null 2>&1 || { echo "timeout not found (coreutils)"; exit 1; }
command -v python >/dev/null 2>&1 || {
  echo "python not found. Run this script inside the configured environment."
  exit 1
}

mkdir -p "${TP_DIR}"

clone_or_update_repo() {
  local name="$1"
  local origin_url="$2"
  local dir="$3"
  local ref="$4"

  if [[ ! -d "${dir}/.git" ]]; then
    echo ""
    hdr "Cloning ${name}"
    GIT_TERMINAL_PROMPT=0 timeout 600 git clone --filter=blob:none --no-checkout \
      "${origin_url}" "${dir}"
  fi

  echo ""
  hdr "${name}: checkout ${ref}"
  (
    cd "${dir}"
    git remote set-url origin "${origin_url}"

    GIT_TERMINAL_PROMPT=0 timeout 600 git fetch --tags origin
    GIT_TERMINAL_PROMPT=0 timeout 600 git fetch --depth 1 origin "${ref}"
    git reset --hard FETCH_HEAD
    echo "${name} resolved to:"
    git show -s --format='  %H%n  %ad%n  %s' --date=short HEAD
    echo "  describe: $(git describe --tags --always --dirty 2>/dev/null || echo 'n/a')"
    echo "  origin: $(git remote get-url origin)"
  )
}

prepare_repos() {
  clone_or_update_repo \
    "robosuite" \
    "${ROBOSUITE_ORIGIN_URL}" \
    "${ROBOSUITE_DIR}" \
    "${ROBOSUITE_REF}"

  clone_or_update_repo \
    "robomimic" \
    "${ROBOMIMIC_ORIGIN_URL}" \
    "${ROBOMIMIC_DIR}" \
    "${ROBOMIMIC_REF}"

  clone_or_update_repo \
    "mimicgen" \
    "${MIMICGEN_ORIGIN_URL}" \
    "${MIMICGEN_DIR}" \
    "${MIMICGEN_REF}"
}

verify_imports() {
  echo ""
  hdr "Smoke test imports"
  (cd "${ROOT_DIR}" && python - <<'PY'
import mimicgen  # noqa: F401
import mujoco
import numpy as np
import robomimic  # noqa: F401
import robosuite  # noqa: F401
import torch

print("numpy", np.__version__)
print("mujoco", mujoco.__version__)
print("torch", torch.__version__)
print("robosuite import OK")
print("robomimic import OK")
print("mimicgen import OK")
PY
  )
}

case "${MODE}" in
  prepare)
    prepare_repos
    ;;
  verify)
    verify_imports
    ;;
  full)
    prepare_repos
    verify_imports
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: $0 [prepare|verify|full]"
    exit 1
    ;;
esac

echo ""
hdr "Done"
echo "Tip: run a robosuite demo with:"
echo "  uv run python -m robosuite.demos.demo_random_action"
