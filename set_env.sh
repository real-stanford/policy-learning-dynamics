#!/bin/bash
# source set_env.sh   (from the repo root, inside the "data" conda env)

export REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Datasets, run directories and figures live under the repo's own outputs/.
export OUTPUTS_ROOT="$REPO_ROOT/outputs"
export TRAINING_ROOT="$OUTPUTS_ROOT/training"
export DATA_ROOT="$OUTPUTS_ROOT/data"
export MIMICGEN_DATA_ROOT="$DATA_ROOT/mimicgen"
export VIS_ROOT="$OUTPUTS_ROOT/vis"

for dir in "$OUTPUTS_ROOT" "$TRAINING_ROOT" "$DATA_ROOT" "$MIMICGEN_DATA_ROOT" "$VIS_ROOT"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

# robosuite_task_zoo (Hammer Cleanup / Kitchen assets) is added via PYTHONPATH rather
# than its editable install, which does not survive multiprocessing 'spawn' env workers.
unset PYTHONPATH
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third_party/robosuite-task-zoo"

# Headless GPU rendering for image-obs rollouts (robosuite/mujoco offscreen). The GPU
# per rank is chosen by MUJOCO_EGL_DEVICE_ID, set in envs/image_parallel/factory.py.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Temp dir on node-local disk: torch.distributed creates one tempdir per process at
# import time; keep them off NFS.
export TMPDIR="${TMPDIR:-/local/real/$USER/tmp}"
mkdir -p "$TMPDIR" 2>/dev/null || { export TMPDIR="/tmp/$USER"; mkdir -p "$TMPDIR"; }
