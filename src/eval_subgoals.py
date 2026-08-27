"""LIBERO-X eval with per-step subgoal instrumentation.

Extends LIBERO-X's eval_template.py: same client-server rollout loop against
an openpi policy server, but additionally queries the environment's BDDL goal
predicates at every simulation step so we can tell, for each multi-step task,
which subgoal(s) the policy completed and the exact step at which it stalled.

Run from the LIBERO-X repo root, e.g.:
  MUJOCO_GL=egl python eval_subgoals.py \
    --scene-group LEVEL1 --load-mode init \
    --task-list-file task_lists/LEVEL1.txt \
    --num-trials-per-task 5 \
    --video-out-path results/LEVEL1

See the top-level README for full setup and scaling notes. Two things worth
knowing before you run this at scale:
  1. Image preprocessing must rotate the camera frame 180 degrees (flip both
     axes) to match openpi's train-time preprocessing -- see the comment at
     the img/wrist_img lines below. Getting this wrong silently collapses
     success rate to ~0% while still looking like the policy is "trying"
     (see README "Known pitfalls").
  2. Runs resume automatically: if --results-out-path already has rows for a
     (task, episode) pair, that pair is skipped on the next run. Safe to
     Ctrl-C and restart, or to let it survive a crash mid-sweep. Pass
     --no-resume to force a clean re-run instead.
"""
import collections
import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import re

import imageio
import numpy as np
import torch
import tqdm
import tyro
from typing import Optional
from natsort import natsorted
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.parse_bddl import parse_bddl_file
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _make_task_segment(scene_name, task_id, task_description, max_len=180):
    base = f"{scene_name}_{str(task_id).zfill(3)}_{task_description.replace(' ', '_')}"
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", base)
    if len(safe) > max_len:
        hash_suffix = hashlib.md5(safe.encode("utf-8")).hexdigest()[:8]
        keep_len = max(1, max_len - len(hash_suffix) - 1)
        safe = f"{safe[:keep_len].rstrip('_')}_{hash_suffix}"
    return safe


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _iter_leaf_predicates(expr):
    """Flatten a (possibly And/Or-nested, or flat-conjunction) goal_state into leaf predicate tuples.

    A leaf predicate is [predicate_name, obj_name, ...] where predicate_name is a
    string and every remaining element is a plain string (not a nested list). A
    goal_state that is a *list of* such leaves (the common flat-conjunction case,
    e.g. [["on","a","b"], ["turnon","c"]]) must NOT be mistaken for a single leaf
    just because it happens to have length 2 or 3.
    """
    if not isinstance(expr, (list, tuple)) or not expr:
        return
    head = expr[0]
    if isinstance(head, str) and head.lower() in ("and", "or"):
        for child in expr[1:]:
            yield from _iter_leaf_predicates(child)
        return
    if isinstance(head, str) and all(not isinstance(x, (list, tuple)) for x in expr[1:]):
        yield tuple(expr)
        return
    # Not a leaf: this level is itself a list of sub-expressions/leaves.
    for child in expr:
        yield from _iter_leaf_predicates(child)


def _predicate_str(pred):
    return " ".join(str(x) for x in pred)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 25

    scene_group: str = "LEVEL1"
    load_mode: str = "init"
    bddl_root: str = "libero/libero_x/bddl"
    init_root: str = "libero/libero_x/init"
    level5_prompt_root: str = "libero/libero_x/LEVEL5"
    task_list_file: str = ""  # newline-delimited bddl filenames; empty = all tasks in level
    num_trials_per_task: int = 5
    num_steps_wait: int = 10
    max_steps: int = 500

    video_out_path: str = "data/eval_videos"
    results_out_path: str = ""
    save_all_videos: bool = False
    save_failure_videos: bool = True
    fps: int = 10

    seed: int = 7
    resume: bool = True  # skip (task, episode) pairs already present in results_out_path


def eval_subgoals(args: Args) -> None:
    np.random.seed(args.seed)

    scene_group = args.scene_group.upper()
    if scene_group not in {"LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4", "LEVEL5"}:
        raise ValueError(f"scene_group must be LEVEL1-LEVEL5, got: {args.scene_group}")
    base_level = "LEVEL4" if scene_group == "LEVEL5" else scene_group

    bddl_dir = pathlib.Path(args.bddl_root) / base_level
    if args.task_list_file:
        wanted = [
            line.strip()
            for line in pathlib.Path(args.task_list_file).read_text().splitlines()
            if line.strip()
        ]
        bddl_files = natsorted(wanted)
        missing = [f for f in bddl_files if not (bddl_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing bddl files in {bddl_dir}: {missing}")
    else:
        bddl_files = natsorted([f for f in os.listdir(bddl_dir) if f.endswith(".bddl")])

    init_dir_root = pathlib.Path(args.init_root) / base_level if args.load_mode.lower() == "init" else None

    output_group = scene_group
    video_out_path = pathlib.Path(args.video_out_path) / output_group
    video_out_path.mkdir(parents=True, exist_ok=True)
    results_out_path = pathlib.Path(args.results_out_path) if args.results_out_path else (
        video_out_path / f"results_{output_group}.jsonl"
    )
    results_out_path.parent.mkdir(parents=True, exist_ok=True)

    completed_episodes = set()  # (task_file, episode_index)
    completed_tasks = set()  # task_file with a task_setup failure already logged
    if args.resume and results_out_path.exists():
        n_prior = 0
        with results_out_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_prior += 1
                if row.get("failure_stage") == "task_setup":
                    completed_tasks.add(row["task_file"])
                else:
                    completed_episodes.add((row["task_file"], row["episode_index"]))
        if n_prior:
            logging.info(
                "Resuming: found %d prior rows in %s (%d completed tasks, %d completed episodes) -- skipping those.",
                n_prior, results_out_path, len(completed_tasks), len(completed_episodes),
            )

    prompt_map = None
    prompt_order = None
    if scene_group == "LEVEL5":
        if not args.level5_prompt_root:
            raise ValueError("LEVEL5 requires level5_prompt_root to be set.")
        prompt_files = sorted(pathlib.Path(args.level5_prompt_root).glob("L5-*.jsonl"))
        prompt_map = {}
        for path in prompt_files:
            tag = path.stem
            per_tag = {}
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                task_id = str(rec.get("task_id", "")).zfill(3)
                variant = rec.get("variant") or ""
                task_desc = rec.get("task_desc", "")
                if not task_id or not task_desc:
                    continue
                per_tag[(task_id, variant)] = task_desc
            prompt_map[tag] = per_tag
        prompt_order = sorted(prompt_map.keys())

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes = 0
    total_successes = 0

    for task_id in tqdm.tqdm(range(len(bddl_files))):
        bddl_file_path = bddl_dir / bddl_files[task_id]
        scene_match = re.search(r"SCENE\d+", bddl_files[task_id])
        scene_name = scene_match.group() if scene_match else base_level

        if str(bddl_file_path) in completed_tasks:
            continue

        try:
            task_description = parse_bddl_file(bddl_file_path)["language"]

            prompt_candidates = None
            if prompt_map is not None:
                m = re.search(r"__T(\d+)(?:__A(\d+))?", bddl_file_path.stem)
                if not m:
                    raise ValueError(f"Could not parse task id from BDDL name: {bddl_file_path.name}")
                key = (m.group(1).zfill(3), f"A{m.group(2)}" if m.group(2) else "")
                prompt_candidates = []
                for tag in prompt_order:
                    per_tag = prompt_map.get(tag, {})
                    if key not in per_tag:
                        raise KeyError(f"Missing prompt for tag {tag}, key={key}, file={bddl_file_path.name}")
                    prompt_candidates.append((tag, per_tag[key]))

            if init_dir_root is not None:
                init_path = init_dir_root / f"{bddl_file_path.stem}.init"
                if not init_path.exists():
                    raise FileNotFoundError(f"Init file not found: {init_path}")
                states_tensor = torch.load(init_path)
                n_avail = len(states_tensor)
                episode_indices = range(min(args.num_trials_per_task, n_avail))
            else:
                states_tensor = None
                episode_indices = range(args.num_trials_per_task)
        except Exception as e:
            # A single malformed/missing task file must not abort a multi-hour unattended sweep.
            logging.error(f"Skipping task {task_id} ({bddl_file_path.name}): {e}")
            _write_result(
                results_out_path, args, scene_name, task_id, bddl_file_path,
                "", 0, False, "", failure_stage="task_setup", error=e,
                subgoal_log=None, steps_taken=0,
            )
            total_episodes += 1
            continue

        task_episodes = 0
        task_successes = 0

        for ep_id in episode_indices:
            if (str(bddl_file_path), ep_id) in completed_episodes:
                continue

            prompt_tag = ""
            if prompt_candidates is not None:
                prompt_idx = (ep_id // 2) % len(prompt_candidates)
                prompt_tag, task_description = prompt_candidates[prompt_idx]

            replay_images = []
            done = False
            t = 0
            subgoal_log = None  # list of dicts, one per leaf predicate

            try:
                env_args = {
                    "bddl_file_name": bddl_file_path,
                    "camera_heights": LIBERO_ENV_RESOLUTION,
                    "camera_widths": LIBERO_ENV_RESOLUTION,
                    "horizon": args.max_steps + args.num_steps_wait + 1,
                }
                env = OffScreenRenderEnv(**env_args)
                env.seed(ep_id)
                env.reset()

                # Underlying robosuite env exposes _eval_predicate / parsed_problem.
                inner = env.env if hasattr(env, "env") else env
                goal_state = inner.parsed_problem["goal_state"]
                leaf_predicates = list(_iter_leaf_predicates(goal_state))
                subgoal_log = [
                    {"predicate": _predicate_str(p), "first_true_step": None, "final_true": False}
                    for p in leaf_predicates
                ]

                if states_tensor is not None:
                    state_vec = states_tensor[ep_id]
                    obs = env.regenerate_obs_from_state(
                        state_vec.numpy() if hasattr(state_vec, "numpy") else state_vec
                    )
                    t = args.num_steps_wait
                else:
                    obs = None
                    t = 0
            except Exception as e:
                logging.error(f"Failed to initialize env for {bddl_file_path} ep {ep_id}: {e}")
                _write_result(
                    results_out_path, args, scene_name, task_id, bddl_file_path,
                    task_description, ep_id, False, "", prompt_tag=prompt_tag,
                    failure_stage="scene_init", error=e, subgoal_log=None, steps_taken=0,
                )
                total_episodes += 1
                task_episodes += 1
                continue

            def _log_subgoals(step_idx):
                if subgoal_log is None:
                    return
                for entry, pred in zip(subgoal_log, leaf_predicates):
                    try:
                        sat = bool(inner._eval_predicate(pred))
                    except Exception:
                        sat = False
                    entry["final_true"] = sat
                    if sat and entry["first_true_step"] is None:
                        entry["first_true_step"] = step_idx

            action_plan = collections.deque()
            steps_taken = 0
            while t < args.max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Match openpi's reference LIBERO client exactly: rotate 180 degrees
                    # (flip both axes) to match pi0/pi0.5's train-time preprocessing.
                    # LIBERO-X's own eval_template.py gates this behind a flag that
                    # defaults to a vertical-only flip, which does not match training.
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate((
                                obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"],
                            )),
                            "prompt": str(task_description),
                        }
                        resp = client.infer(element)
                        action_chunk = resp["actions"]
                        if len(action_chunk) == 0:
                            raise RuntimeError("Policy returned zero actions.")
                        chunk_len = min(len(action_chunk), args.replan_steps)
                        action_plan.extend(action_chunk[:chunk_len])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    steps_taken += 1
                    _log_subgoals(steps_taken)
                    if done:
                        break
                    t += 1
                except Exception as e:
                    logging.error(f"Caught exception during rollout: {e}")
                    break

            total_episodes += 1
            task_episodes += 1
            if done:
                total_successes += 1
                task_successes += 1

            suffix = "success" if done else "failure"
            task_segment = _make_task_segment(scene_name, task_id, task_description)
            ep_tag = f"ep{str(ep_id + 1).zfill(3)}"
            video_path = video_out_path / f"rollout_{task_segment}_{ep_tag}_{suffix}.mp4"

            should_save_video = bool(replay_images) and (
                args.save_all_videos or (args.save_failure_videos and not done)
            )
            if should_save_video:
                try:
                    imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=args.fps)
                except OSError as e:
                    logging.error(f"Failed to write video {video_path}: {e}")
            else:
                video_path = ""

            _write_result(
                results_out_path, args, scene_name, task_id, bddl_file_path,
                task_description, ep_id, done, str(video_path), prompt_tag=prompt_tag,
                subgoal_log=subgoal_log, steps_taken=steps_taken,
            )

        logging.info(
            "Task %s: %d/%d (%.1f%%)", task_id, task_successes, task_episodes,
            (task_successes / task_episodes * 100.0) if task_episodes else 0.0,
        )

    # Recompute totals from the full results file (not just this invocation's
    # counters) so a resumed run reports accurate totals, including episodes
    # that were already logged before this process started.
    all_episodes, all_successes = 0, 0
    if results_out_path.exists():
        with results_out_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                all_episodes += 1
                all_successes += int(bool(row.get("success")))
    success_rate = (all_successes / all_episodes) if all_episodes else 0.0
    logging.info(
        "Total success rate: %.4f (%d/%d) [%d newly run this invocation]",
        success_rate, all_successes, all_episodes, total_episodes,
    )

    summary = {
        "scene_group": output_group,
        "total_episodes": all_episodes,
        "total_successes": all_successes,
        "success_rate": success_rate,
        "newly_run_this_invocation": total_episodes,
    }
    with open(video_out_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _write_result(results_out_path, args, scene_name, task_id, bddl_file_path, task_description,
                   ep_id, done, video_path, prompt_tag="", failure_stage=None, error=None,
                   subgoal_log=None, steps_taken=0):
    result = {
        "scene_group": str(args.scene_group.upper()),
        "scene_name": str(scene_name),
        "task_id": int(task_id),
        "task_file": str(bddl_file_path),
        "task_desc": str(task_description),
        "prompt_tag": str(prompt_tag),
        "episode_index": int(ep_id),
        "success": bool(done),
        "steps_taken": int(steps_taken),
        "video_path": str(video_path),
    }
    if subgoal_log is not None:
        result["num_subgoals"] = len(subgoal_log)
        result["subgoals"] = subgoal_log
        unmet = [i for i, s in enumerate(subgoal_log) if not s["final_true"]]
        result["first_unmet_subgoal_index"] = unmet[0] if unmet else None
        result["num_subgoals_completed"] = sum(1 for s in subgoal_log if s["final_true"])
    if failure_stage:
        result["failure_stage"] = failure_stage
    if error is not None:
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
    with open(results_out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    eval_subgoals(args)
