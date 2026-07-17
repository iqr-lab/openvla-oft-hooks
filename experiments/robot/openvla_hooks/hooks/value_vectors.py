from experiments.robot.openvla_hooks.hook_runner import register_hook


@register_hook("value_vectors")
def emit(data):
    value_vectors = data.get("value_vectors")
    if value_vectors is None:
        return None
    return {
        "hook_name": "value_vectors",
        "data": value_vectors,
    }

