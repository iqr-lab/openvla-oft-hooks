"""Utils for evaluating policies in LIBERO simulation environments."""

import logging
import math
import os
import re
from pathlib import Path

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import bddl_utils as _bddl_utils
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


_VARIATION_SUFFIX_RE = re.compile(
    r"(_view_[\d_-]+_initstate_\d+|_(table|tb)_\d+|_initstate_\d+|_level\d+_sample\d+|_add_\d+|_light_\d+|_noise_\d+|_language_\d+)$"
)
_PREAMBLE_LEAK_RE = re.compile(r"^\s*here\s+are\s+\d+\s+variations\b", re.IGNORECASE)
_PROMPT_SUFFIX_RE = re.compile(
    r"\s+(view\s+[\d\s-]+\s+initstate\s+\d+|(table|tb)\s+\d+|initstate\s+\d+|"
    r"level\d+\s+sample\d+|add\s+\d+|light\s+\d+|noise\s+\d+)$",
    re.IGNORECASE,
)


def is_language_variation_name(name: str, category: str | None = None) -> bool:
    """Returns True for LIBERO-Plus language-instruction variation tasks."""
    if "_language_" in name:
        return True
    return category is not None and category.strip().lower() == "language instructions"


def base_task_name(name: str) -> str:
    """Strips LIBERO-Plus variation suffixes to recover the underlying task name."""
    while True:
        new_name = _VARIATION_SUFFIX_RE.sub("", name)
        if new_name == name:
            return name
        name = new_name


def clean_task_prompt(task) -> str:
    """Cleans task.language before it is sent as the model prompt."""
    language = task.language
    if _PREAMBLE_LEAK_RE.match(language):
        base_name = base_task_name(task.name)
        base_bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / f"{base_name}.bddl"
        if base_bddl.exists():
            return _bddl_utils.get_problem_info(str(base_bddl))["language_instruction"]
        logging.warning("No fallback bddl found for leaked prompt on task %r; using corrupted text as-is.", task.name)
        return language

    if "_language_" not in task.name:
        while True:
            cleaned = _PROMPT_SUFFIX_RE.sub("", language)
            if cleaned == language:
                return language
            language = cleaned

    return language


def get_libero_env(task, model_family, resolution=256, seed=0):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = clean_task_prompt(task)
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, output_dir=None, filename=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = output_dir or f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    if filename is None:
        filename = f"{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={processed_task_description}.mp4"
    mp4_path = os.path.join(str(rollout_dir), filename)
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
