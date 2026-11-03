#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cd "${ROOT_DIR}"

if [[ -t 1 ]]; then
    _CLR_HDR=$'\033[1;36m'  # bold cyan
    _CLR_RST=$'\033[0m'
else
    _CLR_HDR=""
    _CLR_RST=""
fi

hdr() { echo "${_CLR_HDR}==> $*${_CLR_RST}"; }

if [ ! -f ".pre-commit-config.yaml" ]; then
    echo "No .pre-commit-config.yaml found in ${ROOT_DIR}; skipping hook install."
    exit 0
fi

if [ -f ".git/hooks/pre-commit" ]; then
    echo "pre-commit hook already installed."
    exit 0
fi

hdr "Installing pre-commit hooks"
echo "Repo: ${ROOT_DIR}"
pre-commit install
echo ""
echo "Done!"
