# OpenVLA-OFT Cluster SIF

This builds an environment-only Apptainer image. The repo is not copied into
the `.sif`; bind the repo at runtime so hook edits do not require rebuilding.
The definition file is self-contained, so it can be launched from any directory.

## Build

```bash
apptainer build --fakeroot /path/to/openvla-oft-env.sif /path/to/openvla-oft-hooks/cluster/openvla-oft.def
```

If fakeroot is unavailable but sudo is available:

```bash
sudo apptainer build /path/to/openvla-oft-env.sif /path/to/openvla-oft-hooks/cluster/openvla-oft.def
```

## Test

```bash
apptainer exec --nv \
  -B "$PWD":/workspace/openvla-oft-hooks \
  openvla-oft-env.sif \
  bash -lc 'cd /workspace/openvla-oft-hooks && python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
```

## Run LIBERO Eval With Hooks

Adjust the host paths for your cluster:

```bash
apptainer exec --nv \
  -B /path/to/openvla-oft-hooks:/workspace/openvla-oft-hooks \
  -B /path/to/hf_cache:/hf_cache \
  -B /path/to/logs:/logs \
  openvla-oft-env.sif \
  bash -lc '
    export HF_HOME=/hf_cache
    export TRANSFORMERS_CACHE=/hf_cache/transformers
    cd /workspace/openvla-oft-hooks
    python experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
      --task_suite_name libero_spatial \
      --hook_config experiments/robot/libero/hooks.yaml \
      --hook_output_dir /logs/hooks
  '
```

## Why This Image Does Not Copy The Repo

The slow build step you saw was caused by copying `.` into the image. That can
be painful on a cluster if the build context includes caches, checkpoints,
datasets, logs, or a `.venv`. This definition only copies the conda environment
file and installs dependencies. Source code is mounted at runtime with `-B`.

## Notes

- Use `--nv` for NVIDIA GPU passthrough.
- Bind a persistent Hugging Face cache, or every job may redownload checkpoints.
- `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` are set for headless cluster runs.
- Flash Attention is not installed here; the hook path needs returned attention
  tensors, and optimized attention kernels often do not expose them.
