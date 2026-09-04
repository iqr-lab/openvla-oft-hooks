"""Persistence helpers for OpenVLA hook records."""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.robot.openvla_hooks import record_io

logger = logging.getLogger(__name__)


def _flatten_dict(data: dict[str, Any], parent_key: str = "", sep: str = "/") -> dict[str, Any]:
    flattened = {}
    for key, value in data.items():
        next_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, next_key, sep=sep))
        else:
            flattened[next_key] = value
    return flattened


class _AsyncRecordWriter:
    """Background writer for prepared record payloads.

    Encoding (dtype narrowing, byte shuffle, compression) and the disk write both
    happen on worker threads, so neither blocks the rollout. Use more than one
    worker when the destination filesystem is slow enough that a single writer
    cannot keep up (network filesystems, typically).

    Set `log_every` to periodically report where recording time actually goes.
    """

    def __init__(self, *, max_pending_writes: int, encode, num_workers: int = 1, log_every: int = 0):
        self._encode = encode
        self._log_every = log_every
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending_writes)
        self._error: BaseException | None = None
        self._closed = False

        self._stats_lock = threading.Lock()
        self._n = 0
        self._encode_s = 0.0
        self._write_s = 0.0
        self._bytes = 0
        self._blocked_s = 0.0

        self._threads = [
            threading.Thread(target=self._worker, name=f"openvla-hook-writer-{i}", daemon=True)
            for i in range(max(1, num_workers))
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, path: Path, payload: dict) -> None:
        self.raise_if_failed()
        if self._closed:
            raise RuntimeError("Cannot submit write after the writer is closed.")
        start = time.monotonic()
        self._queue.put((path, payload))
        with self._stats_lock:
            self._blocked_s += time.monotonic() - start

    def flush(self) -> None:
        """Block until every queued record has been written."""
        self._queue.join()
        self.raise_if_failed()

    def close(self) -> None:
        if self._closed:
            self.raise_if_failed()
            return
        self._closed = True
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Background hook record write failed.") from self._error

    def _record_stats(self, encode_s: float, write_s: float, nbytes: int) -> None:
        should_log = False
        with self._stats_lock:
            self._n += 1
            self._encode_s += encode_s
            self._write_s += write_s
            self._bytes += nbytes
            should_log = bool(self._log_every) and self._n % self._log_every == 0
            if should_log:
                n, enc, wrt, nb, blk = self._n, self._encode_s, self._write_s, self._bytes, self._blocked_s
        if should_log:
            logger.info(
                "Hook recorder over %d records: encode %.1f ms/step, write %.1f ms/step, "
                "%.2f MB/step, write throughput %.0f MB/s, rollout blocked on queue %.1f ms/step",
                n,
                1000 * enc / n,
                1000 * wrt / n,
                nb / n / 1e6,
                (nb / 1e6) / wrt if wrt > 0 else float("nan"),
                1000 * blk / n,
            )

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, payload = item
                encode_s, write_s, nbytes = self._encode(path, payload)
                self._record_stats(encode_s, write_s, nbytes)
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
            finally:
                self._queue.task_done()


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

        record_cfg = hook_cfg.get("record", {}) or {}
        self._compress = bool(record_cfg.get("compress", True))
        self._encode_kwargs = {
            "float_dtype": str(record_cfg.get("float_dtype", "auto")),
            "codec": str(record_cfg.get("codec", "zstd")),
            "level": int(record_cfg.get("level", 1)),
            "shuffle": bool(record_cfg.get("shuffle", True)),
        }
        self.suffix = record_io.FILE_SUFFIX if self._compress else ".npy"
        if self._compress:
            logger.info("Hook records are compressed: %s", self._encode_kwargs)

        self._writer = (
            _AsyncRecordWriter(
                max_pending_writes=max(1, int(record_cfg.get("max_pending_writes", 4))),
                encode=self._encode_to_disk,
                num_workers=max(1, int(record_cfg.get("writer_threads", 1))),
                log_every=max(0, int(record_cfg.get("log_every", 0))),
            )
            if bool(record_cfg.get("async_write", True))
            else None
        )
        if self._writer is not None:
            atexit.register(self.close)

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
            "record_format": "pirec" if self._compress else "numpy.save",
            "record_pattern": f"step_0{self.suffix}",
            "record_options": self._encode_kwargs if self._compress else None,
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

    def _encode_to_disk(self, output_path: Path, payload: dict) -> tuple[float, float, int]:
        """Serialize one record. Runs on a writer thread when async.

        Returns (encode_seconds, write_seconds, bytes_written) so the writer can
        report where recording time is actually going.
        """
        start = time.monotonic()
        if self._compress:
            blob = record_io.encode_record(payload, **self._encode_kwargs)
            encoded = time.monotonic()
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            tmp_path.write_bytes(blob)
            tmp_path.replace(output_path)
            return encoded - start, time.monotonic() - encoded, len(blob)

        np.save(output_path, np.asarray(payload, dtype=object), allow_pickle=True)
        return 0.0, time.monotonic() - start, output_path.stat().st_size

    def flush(self) -> None:
        """Block until every queued record has reached disk."""
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

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

        path = self.output_dir / f"step_{self.counter}{self.suffix}"
        self.counter += 1

        if self._writer is not None:
            self._writer.submit(path, data)
        else:
            self._encode_to_disk(path, data)
        return path

    def update_query_metadata(self, path: str | os.PathLike[str], updates: dict[str, Any]) -> None:
        # The record may still be sitting in the writer queue, so make sure it
        # has reached disk before reading it back.
        self.flush()

        path = Path(path)
        record = record_io.load_record(path)

        saveable_updates = self._to_saveable(updates)
        for key, value in saveable_updates.items():
            record[f"outputs/metadata/{key}"] = value

        hook_records = record.get("hook_records", [])
        for hook_record in hook_records:
            if isinstance(hook_record, dict):
                hook_record.setdefault("metadata", {}).update(updates)
        record["hook_records"] = self._to_saveable(hook_records)

        self._encode_to_disk(path, record)


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
