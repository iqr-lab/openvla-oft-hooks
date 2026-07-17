from experiments.robot.openvla_hooks.hook_runner import register_hook


@register_hook("prefix_gradients")
def emit(data):
    gradients = data.get("prefix_gradients")
    if gradients is None:
        return None
    return {
        "hook_name": "prefix_gradients",
        "data": gradients,
    }

