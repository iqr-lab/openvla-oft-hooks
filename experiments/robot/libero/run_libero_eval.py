"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    hook_config: Optional[str] = None                # Optional YAML config enabling OpenVLA hook capture
    hook_output_dir: str = "./experiments/logs/hooks" # Root directory for hook .npy records
    save_hook_records: bool = True                   # Whether to persist enabled hook records

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def setup_hooks(cfg: GenerateConfig, run_id: str, log_file=None):
    """Configure optional OpenVLA hook capture."""
    if cfg.hook_config is None:
        try:
            from experiments.robot.openvla_hooks.hook_runner import set_enabled_hooks, set_hook_config

            set_enabled_hooks([])
            set_hook_config({})
        except ImportError:
            pass
        return None, [], {}

    from experiments.robot.openvla_hooks.hook_runner import set_enabled_hooks, set_hook_config
    from experiments.robot.openvla_hooks.io import HookRecordWriter, load_hook_config

    hook_cfg = load_hook_config(cfg.hook_config)
    hooks_cfg = hook_cfg.get("hooks", {})
    enabled_hooks = hooks_cfg.get("enabled", [])
    set_enabled_hooks(enabled_hooks)
    set_hook_config(hooks_cfg)

    writer = None
    if cfg.save_hook_records:
        output_dir = cfg.hook_output_dir
        if cfg.hook_output_dir == "./experiments/logs/hooks":
            output_dir = os.path.join(cfg.hook_output_dir, run_id)
        writer = HookRecordWriter(
            output_dir,
            cfg.hook_config,
            hook_cfg,
            policy_tag=cfg.model_family,
            checkpoint=str(cfg.pretrained_checkpoint),
        )
        log_message(f"Saving OpenVLA hook records to: {output_dir}", log_file)

    log_message(f"Enabled OpenVLA hooks: {enabled_hooks}", log_file)
    return writer, enabled_hooks, hooks_cfg


def write_json(path: Path, data) -> None:
    """Atomically write JSON metadata in the openpi-inference-recorder layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def safe_filename(text: str, max_len: int = 120) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:max_len] or "task"


def setup_recording_index(cfg: GenerateConfig, hook_writer, run_id: str, resize_size):
    """Create openpi-inference-recorder compatible output/index files."""
    if hook_writer is None:
        return None

    root_dir = Path(hook_writer.output_dir)
    output_dir = root_dir / "output"
    videos_dir = root_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    recording = {
        "root_dir": root_dir,
        "output_dir": output_dir,
        "videos_dir": videos_dir,
        "episodes_path": output_dir / "episodes.json",
        "run_summary_path": output_dir / "run_summary.json",
        "task_summary_path": output_dir / "task_summary.json",
        "metadata_path": output_dir / "metadata.json",
        "episode_index": [],
        "task_summaries": [],
        "global_episode_num": 0,
    }

    metadata = {
        "created_at": datetime.now().isoformat(),
        "task_suite_name": cfg.task_suite_name,
        "num_trials_per_task": cfg.num_trials_per_task,
        "seed": cfg.seed,
        "resize_size": resize_size,
        "replan_steps": cfg.num_open_loop_steps,
        "record_dir": str(root_dir),
        "output_dir": str(output_dir),
        "videos_dir": str(videos_dir),
        "run_id": run_id,
        "model_family": cfg.model_family,
        "checkpoint": str(cfg.pretrained_checkpoint),
    }
    write_json(recording["metadata_path"], metadata)
    write_json(recording["episodes_path"], recording["episode_index"])
    write_json(recording["task_summary_path"], recording["task_summaries"])
    return recording


def make_run_summary(cfg: GenerateConfig, recording, num_tasks: int, total_episodes: int, total_successes: int, hook_writer) -> dict:
    return {
        "task_suite_name": cfg.task_suite_name,
        "num_tasks": int(num_tasks),
        "num_episodes": int(total_episodes),
        "num_successes": int(total_successes),
        "overall_success_rate": float(total_successes / total_episodes) if total_episodes else 0.0,
        "num_policy_calls": int(hook_writer.counter if hook_writer is not None else 0),
        "record_dir": str(recording["root_dir"]),
        "output_dir": str(recording["output_dir"]),
        "videos_dir": str(recording["videos_dir"]),
    }


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    task_id=None,
    episode_idx=None,
    hook_writer=None,
    enabled_hooks=None,
    hook_cfg=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    query_idx = 0
    hook_record_paths = []
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                record_inputs = {
                    "observation/image": observation["full_image"].copy(),
                    "observation/state": observation["state"].copy(),
                    "prompt": task_description,
                }
                if "wrist_image" in observation:
                    record_inputs["observation/wrist_image"] = observation["wrist_image"].copy()
                hook_context = None
                if enabled_hooks:
                    hook_context = {
                        "enabled": True,
                        "enabled_hooks": enabled_hooks,
                        "hook_config": hook_cfg or {},
                        "metadata": {
                            "task_id": task_id,
                            "episode_idx": episode_idx,
                            "query_idx": query_idx,
                            "timestep": t,
                            "task_description": task_description,
                            "success": None,
                        },
                    }

                # Query model to get action
                query_start_time = time.monotonic()
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    hook_context=hook_context,
                )
                infer_ms = (time.monotonic() - query_start_time) * 1000
                if hook_context is not None and hook_writer is not None and hook_context.get("records"):
                    record_outputs = {
                        "state": record_inputs["observation/state"],
                        "actions": np.asarray(actions),
                        "policy_timing": {
                            "infer_ms": infer_ms,
                        },
                    }
                    record_path = hook_writer.save_query(
                        inputs=record_inputs,
                        outputs=record_outputs,
                        hook_records=hook_context["records"],
                        metadata=hook_context["metadata"],
                    )
                    hook_record_paths.append(record_path)
                    log_message(f"Saved OpenVLA hook record: {record_path}", log_file)
                query_idx += 1
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    if hook_writer is not None:
        for record_path in hook_record_paths:
            hook_writer.update_query_metadata(record_path, {"success": success})

    return success, replay_images, t


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    hook_writer=None,
    enabled_hooks=None,
    hook_cfg=None,
    recording=None,
    num_tasks=0,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    task_policy_calls = []
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)
        episode_start_idx = hook_writer.counter if hook_writer is not None else 0

        # Run episode
        success, replay_images, num_env_steps = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            task_id,
            episode_idx,
            hook_writer,
            enabled_hooks,
            hook_cfg,
            log_file,
        )
        episode_end_idx = (hook_writer.counter - 1) if hook_writer is not None else episode_start_idx - 1
        num_policy_calls = max(0, episode_end_idx - episode_start_idx + 1)
        task_policy_calls.append(num_policy_calls)

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        if recording is not None:
            suffix = "success" if success else "failure"
            video_filename = (
                f"rollout_task{task_id}_episode{episode_idx}_{safe_filename(str(task_description))}_{suffix}.mp4"
            )
            save_rollout_video(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
                output_dir=recording["videos_dir"],
                filename=video_filename,
            )

            recording["episode_index"].append(
                {
                    "global_episode_num": int(recording["global_episode_num"]),
                    "episode_num": int(episode_idx),
                    "task_id": int(task_id),
                    "task": str(task_description),
                    "start_idx": int(episode_start_idx),
                    "end_idx": int(episode_end_idx),
                    "success": bool(success),
                    "num_policy_calls": int(num_policy_calls),
                    "num_env_steps": int(num_env_steps),
                }
            )
            recording["global_episode_num"] += 1
            write_json(recording["episodes_path"], recording["episode_index"])
            write_json(
                recording["run_summary_path"],
                make_run_summary(cfg, recording, num_tasks, total_episodes, total_successes, hook_writer),
            )
        else:
            save_rollout_video(
                replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
            )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    if recording is not None:
        task_summary = {
            "task_id": int(task_id),
            "task": str(task_description),
            "episodes": int(task_episodes),
            "successes": int(task_successes),
            "success_rate": float(task_successes / task_episodes) if task_episodes else 0.0,
            "mean_policy_calls": float(np.mean(task_policy_calls)) if task_policy_calls else 0.0,
            "min_policy_calls": int(np.min(task_policy_calls)) if task_policy_calls else 0,
            "max_policy_calls": int(np.max(task_policy_calls)) if task_policy_calls else 0,
        }
        recording["task_summaries"].append(task_summary)
        write_json(recording["task_summary_path"], recording["task_summaries"])

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Configure optional OpenVLA hook capture
    hook_writer, enabled_hooks, hook_cfg = setup_hooks(cfg, run_id, log_file)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    recording = setup_recording_index(cfg, hook_writer, run_id, resize_size)

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            hook_writer,
            enabled_hooks,
            hook_cfg,
            recording,
            num_tasks,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    if recording is not None:
        write_json(recording["episodes_path"], recording["episode_index"])
        write_json(recording["task_summary_path"], recording["task_summaries"])
        write_json(
            recording["run_summary_path"],
            make_run_summary(cfg, recording, num_tasks, total_episodes, total_successes, hook_writer),
        )

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
