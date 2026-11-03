#!/bin/bash
set -euo pipefail

# Prevent "source install_deps.sh" from killing the user's shell session.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Do not source this script. Run it as: ./install_deps.sh"
    return 1
fi

ENV_NAME="data"
PYTHON_VERSION="3.10"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -t 1 ]]; then
    _CLR_HDR=$'\033[1;36m'   # bold cyan
    _CLR_OK=$'\033[1;32m'    # bold-ish green
    _CLR_RST=$'\033[0m'
else
    _CLR_HDR=""
    _CLR_OK=""
    _CLR_RST=""
fi

hdr() { echo "${_CLR_HDR}==> $*${_CLR_RST}"; }
ok() { echo "${_CLR_OK}$*${_CLR_RST}"; }

ensure_cmake_for_source_builds() {
    echo ""
    hdr "Ensuring compatible cmake is available for source builds"

    "${CONDA_CMD}" run -n "${ENV_NAME}" python -m pip install "cmake<4"

    if ! "${CONDA_CMD}" run -n "${ENV_NAME}" bash -lc "command -v cmake >/dev/null 2>&1"; then
        echo "cmake is unavailable in '${ENV_NAME}' after installing 'cmake<4'."
        exit 1
    fi

    "${CONDA_CMD}" run -n "${ENV_NAME}" cmake --version
}

hdr "offline_learning_data_study setup"
echo ""

if command -v mamba >/dev/null 2>&1; then
    CONDA_CMD="mamba"
elif command -v conda >/dev/null 2>&1; then
    CONDA_CMD="conda"
else
    echo "Neither mamba nor conda was found."
    echo "Install Miniforge/Mambaforge first:"
    echo "  https://github.com/conda-forge/miniforge"
    exit 1
fi

ok "Using ${CONDA_CMD} for environment management."

if "${CONDA_CMD}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' already exists, skipping creation."
else
    echo ""
    hdr "Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})"
    "${CONDA_CMD}" create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

echo ""
hdr "Ensuring uv is installed in '${ENV_NAME}'"
"${CONDA_CMD}" run -n "${ENV_NAME}" python -m pip install uv

echo ""
PYTHON_PATH=$("${CONDA_CMD}" run -n "${ENV_NAME}" python -c "import sys; print(sys.executable)")
echo "Using Python at: ${PYTHON_PATH}"

ensure_cmake_for_source_builds

echo ""
hdr "Preparing forked sim repositories"
(
    cd "${ROOT_DIR}"
    "${CONDA_CMD}" run -n "${ENV_NAME}" bash -lc "cd '${ROOT_DIR}' && bash bash_scripts/setup_sim_stack.sh prepare"
)

echo ""
hdr "Syncing locked project dependencies"
(
    cd "${ROOT_DIR}"
    tmp_pylock="${ROOT_DIR}/pylock.sync.toml"
    trap 'rm -f "${tmp_pylock}"' EXIT
    "${CONDA_CMD}" run -n "${ENV_NAME}" uv export --format pylock.toml --frozen --all-groups -o "${tmp_pylock}"
    "${CONDA_CMD}" run -n "${ENV_NAME}" uv pip sync --python "${PYTHON_PATH}" "${tmp_pylock}"
)

echo ""
hdr "Verifying sim stack imports"
(
    cd "${ROOT_DIR}"
    "${CONDA_CMD}" run -n "${ENV_NAME}" bash -lc "cd '${ROOT_DIR}' && bash bash_scripts/setup_sim_stack.sh verify"
)

echo ""
hdr "Installing pre-commit hooks (if configured)"
(
    cd "${ROOT_DIR}"
    "${CONDA_CMD}" run -n "${ENV_NAME}" bash -lc "cd '${ROOT_DIR}' && bash bash_scripts/install_precommit_hooks.sh"
)

echo ""
ok "Setup complete"
echo "Next steps:"
echo "  1. Activate the environment: ${CONDA_CMD} activate ${ENV_NAME}"
echo "  2. Run checks when needed: uv run --active pre-commit run --all-files"
