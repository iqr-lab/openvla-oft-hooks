# OpenVLA-OFT Hook Technical Notes

This document describes the OpenVLA-OFT hook implementation in this repo and checks it against the model architecture described in `2502.19645v2.pdf`, "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success."

Scope: these hooks are implemented for the OpenVLA-OFT LIBERO evaluation path using parallel decoding, action chunking, continuous actions, and the L1 regression action head. The paper also studies diffusion and OpenVLA-OFT+ with FiLM, but the default LIBERO OpenVLA-OFT path here is the continuous L1 path without FiLM.

## Model Path During `run_libero_eval`

At a conceptual level, one policy query goes through this path:

1. LIBERO provides an environment observation.
2. `prepare_observation` builds the policy observation:
   - `full_image`: resized third-person image.
   - `wrist_image`: resized wrist camera image.
   - `state`: robot proprioceptive state, assembled from end-effector position, axis-angle orientation, and gripper position.
3. `get_action` calls `get_vla_action`.
4. `get_vla_action`:
   - builds the language prompt, such as `In: What action should the robot take to ...?\nOut:`;
   - preprocesses the image inputs;
   - normalizes proprio when enabled;
   - stores raw observation-level information in `hook_context["observation_input"]`;
   - calls `vla.predict_action`.
5. `predict_action`:
   - appends placeholder action tokens for `ACTION_DIM * NUM_ACTIONS_CHUNK`;
   - appends a stop token;
   - embeds text tokens with the language model embedding table;
   - encodes image views with the fused SigLIP/DINOv2 vision backbone;
   - projects image patches into the LLM embedding space;
   - projects proprio to one extra LLM-space token when enabled;
   - concatenates multimodal embeddings as:

```text
BOS, image patch tokens, optional proprio token, prompt tokens, action placeholder tokens, stop token
```

6. The decoder runs once over the full multimodal sequence.
7. Final decoder hidden states at the action-prediction positions are grouped by action timestep and passed into the L1 action head.
8. The L1 action head maps those hidden states to normalized continuous action values.
9. The model unnormalizes those actions using the dataset statistics and returns one action chunk.
10. `run_libero_eval` executes that full chunk open-loop before querying the model again.

This matches the paper's OpenVLA-OFT recipe: multiple input images, proprio projected into language embedding space, parallel decoding with action placeholders, action chunking, and a continuous L1 action head.

## Important Token Indexing Detail

There are two related but distinct spans:

- `action_tokens`: the literal placeholder action token positions inserted into the input sequence.
- `action_prediction_positions`: the decoder positions whose final hidden states are fed to the action head.

Because the code follows the causal-LM prediction shift, the first action value is predicted from the final prompt position, not from the first literal action placeholder position. In symbols:

```text
action_prediction_start = NUM_PATCHES + NUM_PROMPT_TOKENS
action_token_start      = action_prediction_start + 1
prefix_end              = action_prediction_start + 1
```

So `prefix` ends at the first literal action token and includes the final prompt position. This is why attention hooks use action-prediction positions as queries and prefix positions as keys.

## Saved Record Schema

Records are saved as OpenPI-style `.npy` files:

```python
data = np.load("step_0.npy", allow_pickle=True).item()
```

The top-level object is a flattened dictionary with `/` separators:

```text
inputs/observation/image
inputs/observation/wrist_image
inputs/observation/state
inputs/prompt
outputs/state
outputs/actions
outputs/policy_timing/infer_ms
outputs/metadata/task_id
outputs/metadata/episode_idx
outputs/metadata/query_idx
outputs/metadata/timestep
outputs/metadata/task_description
outputs/metadata/success
hook_records
```

`hook_records` is a list of records:

```python
{
    "hook_name": "...",
    "data": ...,
    "metadata": {
        "task_id": ...,
        "episode_idx": ...,
        "query_idx": ...,
        "timestep": ...,
        "task_description": ...,
        "success": ...,
    },
}
```

The outer schema intentionally matches the OpenPI recorder pattern: `inputs`, `outputs`, and `hook_records`, flattened before saving.

## Hook Configuration

Default config:

```yaml
hooks:
  enabled:
    - observation_input
    - token_spans
    - prefix_embeddings
    - prefix_final_hidden_state
    - prefix_gradients
    - action_chunks
    - raw_attention_weights
    - value_vectors

  action_chunks:
    num_chunks: 1

  raw_attention_weights:
    layers: all

  value_vectors:
    layers: all
```

`raw_attention_weights.layers` and `value_vectors.layers` may be `all` or a list of layer indices.

OpenVLA-OFT with the deterministic L1 action head produces one action chunk per query. The `action_chunks.num_chunks` field is kept for schema similarity with OpenPI, but OpenVLA-OFT does not sample multiple candidate chunks from the L1 head.

## Hook Details

### `observation_input`

Purpose: record what the policy saw at the observation/input level.

Conceptual meaning: this is the bridge between the LIBERO environment and the model sequence. It captures the visual inputs, proprio input, and text prompt before they become the decoder sequence.

Data includes:

- `images/full`: resized third-person image.
- `images/wrist`: resized wrist image, when present.
- `state/raw`: raw proprio before normalization.
- `state/normalized`: normalized proprio passed to the proprio projector.
- `prompt`: full VLA prompt string.
- `task_label`: task description.
- `center_crop`: whether center crop preprocessing was applied.
- `num_images_in_input`: number of image views configured.
- `input_ids`: token IDs after action placeholder and stop token insertion.
- `attention_mask`: corresponding attention mask after sequence extension.
- `pixel_values_shape`: image tensor shape passed to the vision backbone.
- `proprio`: proprio tensor passed into `predict_action`, if enabled.

Correctness check:

- The paper says OpenVLA-OFT can process multiple images and robot state by projecting all inputs into the language embedding space. This hook captures exactly those input sources and the processed token/tensor structure used by the forward pass.

### `token_spans`

Purpose: define where semantic regions live in the decoder sequence.

Conceptual meaning: OpenVLA-OFT turns a multimodal observation into one long decoder sequence. This hook provides the map from conceptual pieces of the robot policy input/output to token indices.

Data fields:

- `bos`: beginning-of-sequence token span.
- `image`: per-image patch spans, for example `full` and `wrist`.
- `proprio`: the proprio token span, when proprio is enabled.
- `prompt`: text prompt span.
- `prefix`: the prefix region used by the main analysis hooks.
- `action_prediction_positions`: positions whose final hidden states feed the action head.
- `action_tokens`: literal placeholder action-token positions.
- `stop`: stop-token span.
- `num_patches`: number of image/proprio/diffusion prefix tokens counted by the model.
- `num_prompt_tokens`: prompt token count excluding BOS.
- `causal_prediction_shift`: currently `1`.

Correctness check:

- Image spans start immediately after BOS.
- Each image contributes `vision_backbone.get_num_patches()` tokens, which is 256 for the OpenVLA setup described in the paper.
- Proprio contributes one projected token after image patches.
- `action_prediction_positions` are shifted left by one relative to literal action placeholders, matching the hidden states actually fed to the action head.
- `prefix` excludes action placeholders and stop token, so prefix-based hooks analyze observation and prompt information rather than action-token slots.

### `prefix_embeddings`

Purpose: record the input embeddings for the prefix before decoder contextualization.

Conceptual meaning: this is the model's pre-decoder representation of observation and instruction. It contains BOS, projected image patches, optional projected proprio, and prompt embeddings, all in the LLM hidden dimension.

Shape:

```text
[batch, prefix_tokens, hidden_dim]
```

What it is:

- after image projection into language embedding space;
- after proprio projection into language embedding space;
- after text token embedding lookup;
- before decoder self-attention layers.

What it is not:

- not raw pixels;
- not raw proprio;
- not final decoder states;
- not action head output.

Correctness check:

- The paper says all input embeddings are concatenated along the sequence dimension before the decoder. This hook records the prefix portion of that exact concatenated embedding sequence.

### `prefix_final_hidden_state`

Purpose: record decoder-final hidden states over the same prefix positions.

Conceptual meaning: this is the contextualized prefix after the Llama-style decoder has mixed information through self-attention.

Shape:

```text
[batch, prefix_tokens, hidden_dim]
```

Difference from `prefix_embeddings`:

- `prefix_embeddings` is before the decoder.
- `prefix_final_hidden_state` is after the final decoder layer.

Correctness check:

- The continuous L1 action head consumes final decoder hidden states. Recording final prefix hidden states gives the comparable post-decoder representation for the observation/prompt side of the model.

### `prefix_gradients`

Purpose: measure sensitivity of the predicted action chunk to the prefix embeddings.

Conceptual meaning: this asks, "If we nudged the observation/prompt embeddings, how would the scalar summary of the predicted action chunk change?"

Shape:

```text
[batch, prefix_tokens, hidden_dim]
```

Implementation:

- clones `multimodal_embeddings[:, :prefix_end, :]` as a differentiable prefix leaf;
- keeps the suffix detached;
- reruns the decoder with the differentiable prefix;
- feeds action-prediction hidden states through the L1 action head;
- computes the gradient of the summed normalized action chunk with respect to the prefix embeddings.

Correctness check:

- The gradient is taken with respect to the prefix input embeddings, not raw pixels or raw proprio.
- The scalar objective is the full predicted normalized action chunk, matching the OpenPI-style "total action output wrt prefix" interpretation.
- This hook requires the L1 action head. It is not implemented for the diffusion reverse-denoising path.

### `action_chunks`

Purpose: record the action chunk produced by one policy query.

Conceptual meaning: OpenVLA-OFT predicts `K` future actions in one forward pass, then the evaluator executes those actions open-loop before re-querying.

Data:

```python
{
    "chunks": array,
    "noises": None,
}
```

Shape:

```text
chunks: [num_chunks, batch, horizon, action_dim]
```

For LIBERO OpenVLA-OFT:

```text
num_chunks = 1
batch = 1
horizon = NUM_ACTIONS_CHUNK = 8
action_dim = ACTION_DIM = 7
```

The stored actions are unnormalized actions, after conversion from normalized action-head outputs back to the dataset action scale.

Correctness check:

- The paper's LIBERO setup uses chunk size `K = 8` and executes full chunks before replanning.
- Deterministic L1 OpenVLA-OFT produces one chunk per query. Multiple stochastic chunks are not natural for this model unless extra sampling logic is introduced outside the L1 head.

### `raw_attention_weights`

Purpose: record decoder attention from action-prediction queries to prefix keys.

Conceptual meaning: this answers, "When forming the hidden states that the action head reads, which prefix tokens did those positions attend to?"

Data fields:

- `weights`: selected attention probabilities.
- `layers`: selected decoder layer indices.
- `key_end`: prefix key boundary.
- `query_start`: first action-prediction query position.
- `query_end`: one past the final action-prediction query position.
- `num_heads`: attention head count.
- `num_layers`: number of recorded layers.

Shape:

```text
weights: [batch, layers, heads, action_prediction_positions, prefix_tokens]
```

What is sliced:

```text
queries = action_prediction_positions
keys    = prefix
```

What it is not:

- not the full raw attention matrix over every query and key;
- not attention from literal action placeholder tokens unless they are also action-prediction positions under the shift;
- not a rollout or causal attribution score.

Correctness check:

- The L1 action head reads `action_prediction_positions`, so those are the right queries for "what did action prediction attend to?"
- Prefix keys isolate observation and instruction information.
- The hook asks the HF decoder to return attentions only when this hook is enabled.

Operational note:

- Some optimized attention implementations do not return attention probabilities. The current OpenVLA eval path does not force flash attention for this reason.

### `value_vectors`

Purpose: record decoder value vectors for prefix tokens.

Conceptual meaning: attention combines values from key positions. This hook records the value-side representation available from prefix tokens before those values are weighted and summed by attention.

Data fields:

- `vectors`: value vectors.
- `layers`: selected decoder layer indices.
- `key_end`: prefix key boundary.
- `num_kv_heads`: number of key/value heads.
- `head_dim`: value head dimension.

Shape:

```text
vectors: [batch, layers, prefix_tokens, num_kv_heads, head_dim]
```

Implementation:

- registers temporary forward hooks on each selected decoder layer's `self_attn.v_proj`;
- captures the value projection output during the normal decoder forward pass;
- slices to prefix token positions;
- reshapes hidden dimension into key/value heads and head dimension.

Correctness check:

- This records prefix value vectors, which line up with the prefix key range used by `raw_attention_weights`.
- It is not the final attention output and not the residual-stream hidden state.

## Correctness Audit Summary

| Hook | Status | Reason |
| --- | --- | --- |
| `observation_input` | Correct for LIBERO OpenVLA-OFT | Captures full image, wrist image, proprio, prompt, token IDs, masks, and tensor shapes used by the actual policy query. |
| `token_spans` | Correct | Accounts for BOS, per-image patch tokens, optional proprio, prompt, causal prediction shift, action placeholders, and stop token. |
| `prefix_embeddings` | Correct | Records pre-decoder prefix embeddings in LLM space, matching the paper's unified latent sequence. |
| `prefix_final_hidden_state` | Correct | Records post-decoder prefix hidden states from the same forward pass that produces action hidden states. |
| `prefix_gradients` | Correct for L1 path | Computes gradient of summed normalized action chunk wrt prefix embeddings. Not a raw-pixel gradient and not implemented for diffusion. |
| `action_chunks` | Correct for L1 OpenVLA-OFT | Records the one deterministic chunk returned by a query, shaped like OpenPI's chunk axis. |
| `raw_attention_weights` | Correct slice | Records action-prediction-query attention onto prefix keys, which is the relevant matrix slice for action-readout analysis. |
| `value_vectors` | Correct slice | Records value projections for prefix tokens at selected decoder layers. |

## Known Scope Limits

- The hooks target the L1 regression / discrete parallel-decoding path. Diffusion-specific hook capture is not implemented.
- `action_chunks.num_chunks > 1` is not used for L1 OpenVLA-OFT because the model deterministically returns one chunk per query.
- `prefix_gradients` is a gradient with respect to LLM-space prefix embeddings, not raw pixels, raw state values, or token IDs.
- `raw_attention_weights` is intentionally sliced to the action-query-to-prefix-key submatrix to avoid huge files and to answer the action-analysis question directly.
- Full runtime verification requires a Torch/OpenVLA environment with the model loaded. The local shell used for this audit does not currently have `torch` installed.

## File Map

- Hook YAML: `experiments/robot/libero/hooks.yaml`
- Hook registry: `experiments/robot/openvla_hooks/hook_runner.py`
- Hook record emitters: `experiments/robot/openvla_hooks/hooks/`
- Hook record formatting: `experiments/robot/openvla_hooks/runtime.py`
- `.npy` writer: `experiments/robot/openvla_hooks/io.py`
- Eval integration: `experiments/robot/libero/run_libero_eval.py`
- Model capture points: `prismatic/extern/hf/modeling_prismatic.py`
- Hook tests: `tests/test_openvla_hooks.py`
