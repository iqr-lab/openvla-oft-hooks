"""Format model-side OpenVLA hook payloads into records."""

from __future__ import annotations

from typing import Any

import torch

from experiments.robot.openvla_hooks.hook_runner import emit_all, is_hook_enabled


def _detach(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _detach(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_detach(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_detach(v) for v in value)
    return value


def collect_hook_records(*, payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Build enabled hook records from a model capture payload."""
    if not payload:
        return []

    data = {
        "observation_input": payload["observation_input"],
        "token_spans": payload["token_spans"],
        "prefix_embeddings": payload["prefix_embeddings"],
        "prefix_final_hidden_state": payload["prefix_final_hidden_state"],
        "action_chunks": payload["action_chunks"],
        "prefix_gradients": payload.get("prefix_gradients"),
        "raw_attention_weights": payload.get("raw_attention_weights"),
        "value_vectors": payload.get("value_vectors"),
    }

    records = emit_all(data)
    if metadata:
        for record in records:
            record["metadata"] = dict(metadata)
    return _detach(records)


def hook_context_enabled(hook_context: dict[str, Any] | None) -> bool:
    return bool(hook_context and hook_context.get("enabled"))


def wants_heavy_hook(name: str, hook_context: dict[str, Any] | None) -> bool:
    return hook_context_enabled(hook_context) and is_hook_enabled(name)

