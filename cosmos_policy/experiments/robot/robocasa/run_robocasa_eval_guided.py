# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
run_robocasa_eval_guided.py

Online guided evaluation on RoboCasa: at each step, the current observation (image)
and task instruction (text) are fed to the teacher (DreamZero); the adapter
produces guidance and the student (Cosmos) outputs actions. Teacher uses
**image + text only** (no state/action inputs).

Usage:
    # Full: student from ckpt_path + adapter & student_trainable from distill ckpt
    uv run -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval_guided \\
        --ckpt_path /workspace/data1/Cosmos-Policy-RoboCasa-Predict2-2B/Cosmos-Policy-RoboCasa-Predict2-2B.pt \\
        --distill_ckpt_path /workspace/outputs/vla_distill_robocasa/checkpoint_00045000.pt \\
        ...

    # Test flow: original student (ckpt_path) + only adapter from distill ckpt (no student_trainable overlay)
    uv run -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval_guided \\
        --ckpt_path /workspace/data1/Cosmos-Policy-RoboCasa-Predict2-2B/Cosmos-Policy-RoboCasa-Predict2-2B.pt \\
        --distill_ckpt_path /workspace/outputs/vla_distill_robocasa/checkpoint_00000101.pt \\
        --adapter_only_from_distill \\
        ...
"""

import ast
import os
import pickle
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import draccus
import h5py
import numpy as np
import torch
import wandb

from cosmos_policy.experiments.robot.cosmos_utils import (
    build_teacher_inputs_image_text_only,
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robocasa.robocasa_utils import save_rollout_video
from cosmos_policy.experiments.robot.robot_utils import DATE_TIME, log_message, setup_logging
from cosmos_policy.utils.utils import jpeg_encode_image, set_seed_everywhere


@dataclass
class GuidedEvalConfig:
    """Config for online guided RoboCasa eval (teacher + adapter + student)."""

    suite: str = "robocasa"
    model_family: str = "cosmos"
    config: str = ""
    ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"

    use_third_person_image: bool = True
    num_third_person_images: int = 2
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = True
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    num_denoising_steps_action: int = 5
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    trained_with_image_aug: bool = True
    chunk_size: int = 32
    num_open_loop_steps: int = 16

    deterministic: bool = True
    deterministic_reset: bool = False
    deterministic_reset_seed: Optional[int] = None

    task_name: str = "PnPCounterToCab"
    num_trials_per_task: int = 50
    env_img_res: int = 224
    robots: str = "PandaMobile"
    controllers: str = "OSC_POSE"
    obj_instance_split: str = "B"
    layout_and_style_ids: str = "((1,1),(2,2),(4,4),(6,9),(7,10))"
    randomize_cameras: bool = False
    # MuJoCo rendering backend. Recommended for container stability:
    # - "osmesa": CPU offscreen rendering (no EGL device selection issues; slower but robust)
    # - "egl": GPU offscreen rendering (requires EGL exposing the target device)
    mujoco_gl: str = "osmesa"
    # Physical GPU id for MuJoCo offscreen EGL rendering (only used when mujoco_gl="egl").
    render_gpu_device_id: int = 7
    # Physical GPU id for PyTorch (teacher+student) inference.
    torch_device_id: int = 7

    local_log_dir: str = "./experiments/logs"
    run_id_note: Optional[str] = None
    use_wandb: bool = False
    wandb_entity: str = "YOUR_ENTITY"
    wandb_project: str = "YOUR_PROJECT"
    seed: int = 195
    randomize_seed: bool = False

    data_collection: bool = False
    jpeg_compress: bool = True

    # Guided eval: teacher + adapter + student
    teacher_path: str = ""
    dreamzero_path: str = ""
    teacher_tokenizer_path: str = ""
    teacher_layer_index: int = 14
    distill_ckpt_path: str = ""
    student_experiment: str = "cosmos_predict2_2b_480p_robocasa_50_demos_per_task"
    # When true: only load adapter_bank from distill_ckpt_path; student stays from ckpt_path (original weights).
    # CLI: --adapter_only_from_distill true (draccus requires explicit value; use "true" or "false").
    adapter_only_from_distill: str = "false"


def validate_config(cfg: GuidedEvalConfig) -> None:
    # Import task registry lazily (robosuite / robocasa import must happen after env vars are set)
    from robocasa.utils.dataset_registry import MULTI_STAGE_TASK_DATASETS, SINGLE_STAGE_TASK_DATASETS

    all_tasks = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}
    if cfg.task_name not in all_tasks:
        raise ValueError(
            f"Task name '{cfg.task_name}' not found. Available: {list(all_tasks.keys())}"
        )
    assert cfg.num_third_person_images == 2
    if (cfg.unnormalize_actions or cfg.normalize_proprio) and not cfg.dataset_stats_path:
        raise ValueError("Must provide dataset_stats_path when unnormalize_actions or normalize_proprio.")
    if not cfg.teacher_path:
        raise ValueError("teacher_path (DreamZero checkpoint) is required for guided eval.")
    if not cfg.distill_ckpt_path or not os.path.isfile(cfg.distill_ckpt_path):
        raise ValueError(f"distill_ckpt_path must be an existing file: {cfg.distill_ckpt_path}")


def _load_tokenizer_fn(teacher_tokenizer_path: str, teacher_path: str, max_length: int = 512) -> Callable[[str], Tuple[torch.Tensor, torch.Tensor]]:
    from cosmos_policy.datasets.robocasa_distill_dataset import _dreamzero_text_clean

    candidates = [
        teacher_tokenizer_path,
        os.path.join(os.path.dirname(teacher_path or ""), "umt5-xxl"),
        os.environ.get("VLA_DISTILL_TOKENIZER_PATH"),
        "/workspace/data1/umt5-xxl",
        "/data1/lingsheng/umt5-xxl",
    ]
    for path in candidates:
        if not path or not os.path.isdir(path):
            continue
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
            def fn(s: str):
                s = _dreamzero_text_clean(s)
                out = tok(
                    s,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    add_special_tokens=True,
                )
                return out["input_ids"].squeeze(0), out["attention_mask"].squeeze(0)
            return fn
        except Exception:
            continue
    raise FileNotFoundError(
        "No teacher tokenizer found. Set --teacher_tokenizer_path to e.g. /workspace/data1/umt5-xxl"
    )


def run_episode_guided(
    cfg: GuidedEvalConfig,
    env,
    task_description: str,
    pipeline,
    dataset_stats,
    tokenizer_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
    episode_idx: int,
    log_file=None,
):
    """Run one episode with online teacher+adapter+student (image+text only for teacher)."""
    # Lazy import (robosuite / mujoco must see correct env vars)
    from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import TASK_MAX_STEPS, prepare_observation

    NUM_STEPS_WAIT = 10
    for _ in range(NUM_STEPS_WAIT):
        dummy_action = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(dummy_action)

    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)
    success = False
    episode_length = 0
    action_queue = deque()
    replay_primary_images = []
    replay_secondary_images = []
    replay_wrist_images = []
    future_image_predictions_list = []

    # Component timing aggregation (teacher/adapter/student inference only).
    # wall clock time and ratio are written at the task level.
    episode_component_time_teacher = 0.0
    episode_component_time_adapter = 0.0
    episode_component_time_student = 0.0
    episode_wall_time_total = 0.0
    episode_query_wall_time_total = 0.0
    num_component_infer_queries = 0
    num_query_calls = 0

    if cfg.data_collection:
        primary_images_list = []
        secondary_images_list = []
        wrist_images_list = []
        proprio_list = []
        actions_list = []

    device = next(pipeline.parameters()).device

    for t in range(max_steps):
        observation = prepare_observation(obs, cfg.flip_images)
        replay_primary_images.append(observation["primary_image"])
        replay_secondary_images.append(observation["secondary_image"])
        replay_wrist_images.append(observation["wrist_image"])
        if cfg.data_collection:
            primary_images_list.append(observation["primary_image"])
            secondary_images_list.append(observation["secondary_image"])
            wrist_images_list.append(observation["wrist_image"])
            proprio_list.append(observation["proprio"])

        if len(action_queue) == 0:
            teacher_inputs = build_teacher_inputs_image_text_only(
                observation,
                task_description,
                tokenizer_fn,
                device=device,
                batch_size=1,
            )
            start_time = time.time()
            action_return_dict = get_action(
                cfg,
                pipeline,
                dataset_stats,
                observation,
                task_description,
                seed=cfg.seed,
                randomize_seed=cfg.randomize_seed,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
                batch_size=1,
                teacher_inputs=teacher_inputs,
            )
            query_time = time.time() - start_time
            log_message(f"Guided query time = {query_time:.3f} sec", log_file)

            episode_query_wall_time_total += float(query_time)
            num_query_calls += 1

            # Pipeline may expose component breakdown timing for generate_samples().
            comp_timing = getattr(pipeline, "_last_component_timing", None)
            if isinstance(comp_timing, dict):
                episode_component_time_teacher += float(comp_timing.get("teacher_time_sec", 0.0))
                episode_component_time_adapter += float(comp_timing.get("adapter_time_sec", 0.0))
                episode_component_time_student += float(comp_timing.get("student_time_sec", 0.0))
                episode_wall_time_total += float(
                    comp_timing.get(
                        "total_wall_time_sec",
                        comp_timing.get("component_time_sec", 0.0),
                    )
                )
                num_component_infer_queries += 1
            else:
                # Fallback: component breakdown not available; keep totals at 0
                pass

            actions = action_return_dict["actions"][: cfg.num_open_loop_steps]
            action_queue.extend(actions)
            if action_return_dict.get("future_image_predictions"):
                future_image_predictions_list.append(action_return_dict["future_image_predictions"])

        action = action_queue.popleft()
        if action.shape[-1] == 7 and env.action_dim == 12:
            mobile_base_action = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
            action = np.concatenate([action, mobile_base_action])
        obs, reward, done, info = env.step(action)
        episode_length += 1
        if cfg.data_collection:
            actions_list.append(action)
        if env._check_success():
            success = True
            log_message(f"  Success at timestep {t}!", log_file)
            break

    log_message(
        f"  Episode {episode_idx}: {'SUCCESS' if success else 'FAILURE'} (length: {episode_length})",
        log_file,
    )

    collected_data = None
    if cfg.data_collection:
        collected_data = dict(
            primary_images=np.stack(primary_images_list, axis=0),
            secondary_images=np.stack(secondary_images_list, axis=0),
            wrist_images=np.stack(wrist_images_list, axis=0),
            proprio=np.stack(proprio_list, axis=0),
            actions=np.stack(actions_list, axis=0),
            success=success,
        )

    episode_component_time_total = (
        episode_component_time_teacher
        + episode_component_time_adapter
        + episode_component_time_student
    )
    episode_component_ratio = None
    if episode_wall_time_total > 0:
        episode_component_ratio = {
            "teacher_ratio": episode_component_time_teacher / episode_wall_time_total,
            "adapter_ratio": episode_component_time_adapter / episode_wall_time_total,
            "student_ratio": episode_component_time_student / episode_wall_time_total,
        }

    return (
        success,
        episode_length,
        replay_primary_images,
        replay_secondary_images,
        replay_wrist_images,
        future_image_predictions_list,
        collected_data,
        dict(
            num_component_infer_queries=num_component_infer_queries,
            num_query_calls=num_query_calls,
            teacher_time_sec=episode_component_time_teacher,
            adapter_time_sec=episode_component_time_adapter,
            student_time_sec=episode_component_time_student,
            component_time_total_sec=episode_component_time_total,
            wall_time_total_sec=episode_wall_time_total,
            query_wall_time_total_sec=episode_query_wall_time_total,
            component_ratio=episode_component_ratio,
        ),
    )


def run_task_guided(
    cfg: GuidedEvalConfig,
    task_name: str,
    pipeline,
    dataset_stats,
    tokenizer_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
    log_file=None,
):
    successes = []
    episode_lengths = []
    total_episodes = 0
    total_successes = 0

    total_component_time_teacher = 0.0
    total_component_time_adapter = 0.0
    total_component_time_student = 0.0
    total_component_time_total = 0.0
    total_wall_time_total = 0.0
    total_num_component_infer_queries = 0
    total_query_wall_time_total = 0.0
    total_num_query_calls = 0

    for episode_idx in range(cfg.num_trials_per_task):
        log_message(f"Starting episode {episode_idx + 1}...", log_file)
        seed = (cfg.seed * episode_idx * 256) if (cfg.deterministic or cfg.deterministic_reset) else None
        # Lazy import: requires robosuite / robocasa to be importable after env vars are set
        from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import create_robocasa_env

        env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
        if cfg.deterministic_reset:
            reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
            set_seed_everywhere(reset_seed)
        env.reset()
        task_description = env.get_ep_meta()["lang"]
        log_message(f"\nTask: {task_description}", log_file)

        success, length, replay_primary, replay_secondary, replay_wrist, future_preds, collected_data, episode_component_timing = (
            run_episode_guided(
                cfg,
                env,
                task_description,
                pipeline,
                dataset_stats,
                tokenizer_fn,
                episode_idx,
                log_file,
            )
        )
        successes.append(success)
        episode_lengths.append(length)
        total_episodes += 1
        if success:
            total_successes += 1

        total_num_component_infer_queries += int(episode_component_timing.get("num_component_infer_queries", 0))
        total_component_time_teacher += float(episode_component_timing.get("teacher_time_sec", 0.0))
        total_component_time_adapter += float(episode_component_timing.get("adapter_time_sec", 0.0))
        total_component_time_student += float(episode_component_timing.get("student_time_sec", 0.0))
        total_component_time_total += float(episode_component_timing.get("component_time_total_sec", 0.0))
        total_wall_time_total += float(episode_component_timing.get("wall_time_total_sec", 0.0))
        total_query_wall_time_total += float(episode_component_timing.get("query_wall_time_total_sec", 0.0))
        total_num_query_calls += int(episode_component_timing.get("num_query_calls", 0))

        rollout_data_dir = os.path.join(cfg.local_log_dir, "rollout_data", f"{task_name}--{DATE_TIME}")
        os.makedirs(rollout_data_dir, exist_ok=True)
        save_rollout_video(
            replay_primary,
            replay_secondary,
            replay_wrist,
            episode_idx,
            success=success,
            task_description=task_description,
            rollout_data_dir=rollout_data_dir,
            log_file=log_file,
        )

        if cfg.data_collection and collected_data is not None and len(collected_data["actions"]) >= 5:
            processed = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:35]
            ep_filename = f"{DATE_TIME}--episode_data--task={processed}--ep={episode_idx}--success={success}.hdf5"
            ep_filepath = os.path.join(rollout_data_dir, ep_filename)
            with h5py.File(ep_filepath, "w") as f:
                for k, v in collected_data.items():
                    if isinstance(v, np.ndarray):
                        is_image = v.ndim == 4 and v.shape[-1] == 3 and v.dtype == np.uint8
                        if is_image and cfg.jpeg_compress:
                            jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in v]
                            if len(jpeg_list) > 1:
                                dt = h5py.vlen_dtype(np.dtype("uint8"))
                                f.create_dataset(k + "_jpeg", data=jpeg_list, dtype=dt)
                        else:
                            f.create_dataset(k, data=v)
                    else:
                        f.attrs[k] = v
                f.attrs["task_description"] = task_description
            log_message(f"Saved episode data to: {ep_filepath}", log_file)

        env.close()
        log_message(f"Success: {success} | completed: {total_episodes} | successes: {total_successes} ({100 * total_successes / total_episodes:.1f}%)", log_file)

    success_rate = float(np.mean(successes))
    avg_length = float(np.mean(episode_lengths))
    log_message(f"Task {task_name} | Success rate: {success_rate:.4f} | Avg length: {avg_length:.1f} | {sum(successes)}/{len(successes)}", log_file)

    # Save component inference timing breakdown (teacher/adapter/student).
    # These timings are measured inside `MoEVLAPipeline.generate_samples()` and exclude
    # preprocessing/query overhead outside the pipeline.
    component_ratio = None
    if total_wall_time_total > 0:
        component_ratio = {
            "teacher_ratio": total_component_time_teacher / total_wall_time_total,
            "adapter_ratio": total_component_time_adapter / total_wall_time_total,
            "student_ratio": total_component_time_student / total_wall_time_total,
        }

    timing_out = {
        "task_name": task_name,
        "num_episodes": cfg.num_trials_per_task,
        "num_component_infer_queries": total_num_component_infer_queries,
        "total_teacher_time_sec": total_component_time_teacher,
        "total_adapter_time_sec": total_component_time_adapter,
        "total_student_time_sec": total_component_time_student,
        "total_component_time_total_sec": total_component_time_total,
        "wall_time_total_sec": total_wall_time_total,
        "total_query_wall_time_total_sec": total_query_wall_time_total,
        "num_query_calls": total_num_query_calls,
        "component_ratio": component_ratio,
    }

    try:
        os.makedirs(cfg.local_log_dir, exist_ok=True)
        timing_out_path = os.path.join(cfg.local_log_dir, f"component_inference_timing--{task_name}--{DATE_TIME}.json")
        import json as _json

        with open(timing_out_path, "w", encoding="utf-8") as f:
            _json.dump(timing_out, f, ensure_ascii=False, indent=2)
        log_message(f"Saved component inference timing to: {timing_out_path}", log_file)
    except Exception as e:
        log_message(f"Failed to save component timing JSON: {e}", log_file)

    if cfg.use_wandb:
        wandb.log({
            f"success_rate/{task_name}": success_rate,
            f"avg_episode_length/{task_name}": avg_length,
            f"num_successes/{task_name}": sum(successes),
            f"num_episodes/{task_name}": len(successes),
        })
    return success_rate, avg_length, successes


@draccus.wrap()
def eval_robocasa_guided(cfg: GuidedEvalConfig) -> float:
    # --- Device / rendering backend selection MUST happen before any robosuite/robocasa import ---
    # Some robosuite modules set MUJOCO_GL at import time (based on macros), so we must set it first.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    _mujoco_gl = (cfg.mujoco_gl or "osmesa").lower().strip()
    if _mujoco_gl not in ("egl", "osmesa", "glx"):
        raise ValueError(f"Invalid mujoco_gl={cfg.mujoco_gl!r}. Use one of: osmesa|egl|glx")
    if _mujoco_gl == "egl":
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(cfg.render_gpu_device_id)
        os.environ["MUJOCO_GL"] = "egl"
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    else:
        # CPU / GLX rendering: do not set MUJOCO_EGL_DEVICE_ID
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        os.environ["MUJOCO_GL"] = _mujoco_gl
        # PYOPENGL_PLATFORM may be set by environment; leave it unchanged for non-EGL backends.

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    set_seed_everywhere(cfg.seed)
    validate_config(cfg)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    # --- PyTorch device (physical id) ---

    device = torch.device(f"cuda:{cfg.torch_device_id}" if torch.cuda.is_available() else "cpu")

    # Ensure project root (where train_vla_distill.py lives) is on path
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    _dreamzero_root = os.path.abspath(cfg.dreamzero_path or os.environ.get("DREAMZERO_PATH", "/workspace/dreamzero"))
    if _dreamzero_root not in sys.path:
        sys.path.insert(0, _dreamzero_root)

    from train_vla_distill import load_teacher_model
    teacher = load_teacher_model(
        cfg.teacher_path,
        device,
        _dreamzero_root,
        "",
        teacher_fp32=False,
    )

    # Lazy import after env var setup (robosuite / mujoco should see MUJOCO_EGL_DEVICE_ID)
    from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import TASK_MAX_STEPS, prepare_observation

    # Load student (Cosmos) from ckpt_path so we get pretrained RoboCasa weights, then overlay distill adapter/trainable
    student, cosmos_config = get_model(cfg)
    assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
        f"Chunk size mismatch: cfg {cfg.chunk_size} vs config {cosmos_config.dataloader_train.dataset.chunk_size}"
    )

    # Build pipeline and load distill checkpoint
    from cosmos_policy.pipeline.moe_vla_pipeline import MoEVLAPipeline, MoEVLAPipelineConfig
    pipeline_config = MoEVLAPipelineConfig(
        teacher_hidden_dim=5120,
        student_hidden_dim=2048,
        adapter_bottleneck_dim=1024,
        adapter_dropout=0.1,
        num_adapter_output_tokens=16,
        num_specialized_experts=4,
        top_k=2,
        gating_hidden_dim=512,
        use_action_fp32=False,
        teacher_layer_index=cfg.teacher_layer_index,
    )
    pipeline = MoEVLAPipeline(teacher_vla=teacher, student_model=student, config=pipeline_config)
    pipeline.freeze_teacher()
    pipeline.to(device)
    pipeline.eval()

    _adapter_only = str(getattr(cfg, "adapter_only_from_distill", "false")).lower() in ("true", "1", "yes")
    ckpt = torch.load(cfg.distill_ckpt_path, map_location="cpu")
    pipeline.adapter_bank.load_state_dict(ckpt["adapter_bank"])
    if not _adapter_only and "student_trainable" in ckpt and len(ckpt["student_trainable"]) > 0:
        pipeline.student.load_state_dict(ckpt["student_trainable"], strict=False)
        log_message("Loaded from distill ckpt: adapter_bank + student_trainable", None)
    else:
        log_message(
            "Loaded from distill ckpt: adapter_bank only (student left from ckpt_path)"
            if _adapter_only
            else "Loaded from distill ckpt: adapter_bank (no student_trainable in ckpt or empty)",
            None,
        )

    tokenizer_fn = _load_tokenizer_fn(
        cfg.teacher_tokenizer_path,
        cfg.teacher_path,
        max_length=512,
    )

    log_file, local_log_filepath, run_id = setup_logging(
        cfg=cfg,
        task_identifier=cfg.task_name,
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note or "guided",
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    log_message(f"Guided eval config: {cfg}", log_file)
    log_message(f"Task: {cfg.task_name} | Trials: {cfg.num_trials_per_task}", log_file)

    success_rate, avg_length, successes = run_task_guided(
        cfg, cfg.task_name, pipeline, dataset_stats, tokenizer_fn, log_file
    )

    log_message("\n" + "=" * 80, log_file)
    log_message("FINAL RESULTS (online guided: teacher image+text -> adapter -> student)", log_file)
    log_message("=" * 80, log_file)
    log_message(f"Task: {cfg.task_name} | Success rate: {success_rate:.4f} | Avg length: {avg_length:.1f}", log_file)
    log_message(f"Total: {sum(successes)}/{len(successes)}", log_file)

    if cfg.use_wandb:
        wandb.log({
            "final_success_rate": success_rate,
            "final_avg_episode_length": avg_length,
            "total_episodes": len(successes),
            "total_successes": sum(successes),
        })
        wandb.save(local_log_filepath)
        wandb.finish()

    if hasattr(pipeline, "cleanup"):
        pipeline.cleanup()
    log_message(f"Results saved to: {local_log_filepath}", log_file)
    return success_rate


if __name__ == "__main__":
    eval_robocasa_guided()
