from experiments.robot.openvla_hooks.hook_runner import register_hook


@register_hook("observation_input")
def emit(data):
    return {
        "hook_name": "observation_input",
        "data": data["observation_input"],
    }

