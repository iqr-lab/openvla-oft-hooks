from experiments.robot.openvla_hooks.hook_runner import register_hook


@register_hook("raw_attention_weights")
def emit(data):
    attention = data.get("raw_attention_weights")
    if attention is None:
        return None
    return {
        "hook_name": "raw_attention_weights",
        "data": attention,
    }

