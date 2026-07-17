"""Persistence helpers for OpenVLA hook records."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _flatten_dict(data: dict[str, Any], parent_key: str = "", sep: str = "/") -> dict[str, Any]:
    flattened = {}
    for key, value in data.items():
        next_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, next_key, sep=sep))
        else:
            flattened[next_key] = value
    return flattened


class HookRecordWriter:
    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        config_path: str | None,
        hook_cfg: dict[str, Any],
        *,
        policy_tag: str | None = None,
        checkpoint: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0

        provenance_dir = self.output_dir / "output"
        provenance_dir.mkdir(parents=True, exist_ok=True)

        saved_hook_config_path = None
        if config_path is not None:
            saved_hook_config_path = provenance_dir / "hooks.yaml"
            shutil.copy2(config_path, saved_hook_config_path)

        self.manifest_path = provenance_dir / "hook_manifest.json"
        manifest = {
            "created_at": datetime.now().isoformat(),
            "policy_tag": policy_tag,
            "checkpoint": checkpoint,
            "record_dir": str(self.output_dir),
            "hook_config_source": str(config_path) if config_path is not None else None,
            "hook_config_saved": str(saved_hook_config_path) if saved_hook_config_path is not None else None,
            "enabled_hooks": hook_cfg.get("hooks", {}).get("enabled", []),
            "hook_config": hook_cfg,
            "record_format": "numpy.save",
            "record_pattern": "step_0.npy",
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _to_saveable(self, value):
        if isinstance(value, dict):
            return {k: self._to_saveable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_saveable(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._to_saveable(v) for v in value)

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            value = value.numpy()

        try:
            value = np.asarray(value)
        except Exception:
            return value

        if hasattr(value, "dtype") and str(value.dtype) == "bfloat16":
            value = value.astype(np.float32)

        return value

    def save_query(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        hook_records: list[dict[str, Any]] | None = None,
        records: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        if hook_records is None:
            hook_records = records or []

        outputs = dict(outputs or {})
        if metadata is not None:
            outputs["metadata"] = metadata

        data = {
            "inputs": inputs or {},
            "outputs": outputs,
            "hook_records": hook_records,
        }
        data = self._to_saveable(data)
        data = _flatten_dict(data)

        path = self.output_dir / f"step_{self.counter}.npy"
        np.save(path, np.asarray(data, dtype=object), allow_pickle=True)
        self.counter += 1
        return path

    def update_query_metadata(self, path: str | os.PathLike[str], updates: dict[str, Any]) -> None:
        record = np.load(path, allow_pickle=True).item()

        saveable_updates = self._to_saveable(updates)
        for key, value in saveable_updates.items():
            record[f"outputs/metadata/{key}"] = value

        hook_records = record.get("hook_records", [])
        for hook_record in hook_records:
            if isinstance(hook_record, dict):
                hook_record.setdefault("metadata", {}).update(updates)
        record["hook_records"] = self._to_saveable(hook_records)

        np.save(path, np.asarray(record, dtype=object), allow_pickle=True)


def load_hook_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {"hooks": {"enabled": []}}

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required when `hook_config` is set. Install `pyyaml` or unset hook_config."
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "hooks" not in cfg:
        cfg["hooks"] = {}
    cfg["hooks"].setdefault("enabled", [])
    return cfg
