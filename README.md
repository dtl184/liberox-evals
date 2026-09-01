# LIBERO-X Evaluations

Tools for evaluating robot policies on [LIBERO-X](https://github.com/meituan/LIBERO-X) and analyzing where failed episodes break down.

For each episode, the evaluator records:

* whether the overall task succeeded
* which task subgoals were completed
* when each subgoal was first completed
* which subgoal the policy failed to complete
* the goal predicate associated with that failure
* a video of failed episodes

The repository also includes a failure-labeling pipeline that uses a video-capable vision-language model (VLM) to identify the visible failure mode, localize when the failure begins, explain why the episode failed, and propose a recovery action.

The code was developed and tested with pi 0.5, but it can be used with other policies served through an `openpi-client` compatible policy server.

## LIBERO-X

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) is a widely used benchmark for robot manipulation. It contains a fixed set of tasks across several tabletop and kitchen environments, including picking up objects, placing them in containers, opening drawers, and completing short multi-step instructions.

[LIBERO-X](https://github.com/meituan/LIBERO-X) builds on the same simulation framework but is designed specifically to test robustness and generalization. LIBERO-X changes the scene, objects, layouts, and instructions in increasingly difficult ways.

The benchmark is divided into five levels:

* **Level 1:** Changes object and scene layouts while keeping tasks relatively close to standard LIBERO.
* **Level 2:** Introduces larger changes to object identity, appearance, and placement.
* **Level 3:** Uses more substantially altered scene compositions and combinations of objects.
* **Level 4:** Adds new object attributes such as different colors and shapes. The policy may need to distinguish, for example, a blue bowl from several other bowls in the scene.
* **Level 5:** Uses the same environments and tasks as Level 4, but paraphrases the natural-language instructions.

LIBERO-X is also much larger than the original LIBERO benchmark. LIBERO contains roughly 130 tasks across its standard task suites, while LIBERO-X contains hundreds of tasks at each difficulty level and around 100 new scenes.

LIBERO-X also introduces additional and stricter goal predicates. Standard LIBERO tasks primarily use predicates such as `In`, `On`, `Open`, `Close`, `TurnOn`, `TurnOff`, and `Stack`. LIBERO-X adds `ExactIn`, `UprightOn`, and `SideOn`.

For example, LIBERO-X can distinguish between:

* placing a bottle on a surface
* placing the bottle upright on the surface
* placing the bottle on its side

This makes it possible to identify more precise manipulation failures than with the original LIBERO predicates alone.

## Repository structure

```text
src/
  eval_subgoals.py        Run evaluations and record task progress
  sample_tasks.py         Select a smaller subset of LIBERO-X tasks
  analyze_results.py      Summarize evaluation results

scripts/
  setup_env.sh            Set up LIBERO-X and openpi
  download_checkpoint.sh  Download a policy checkpoint
  serve_policy.sh         Start the policy server
  run_sweep.sh            Evaluate across LIBERO-X levels

analysis/
  label_failures.py       Identify failed subgoals from simulator results
  sample_for_vlm.py       Select a subset of failures for VLM analysis
  label_with_vlm.py       Diagnose failures, localize failure onset, and propose recovery
  build_gallery.py        Build a browsable gallery of labeled failure videos
```

## Requirements

You will need:

* Linux
* an NVIDIA GPU
* `conda`
* `uv`
* `git`
* `gsutil`

This setup has been tested on RTX 4090 GPUs with 24 GB of VRAM.

You should also have roughly 50 GB of free disk space for LIBERO-X, openpi, and the pi 0.5 checkpoint.

## Setup

Choose a directory where LIBERO-X and openpi should be installed:

```bash
./scripts/setup_env.sh /path/to/workspace
```

This will:

1. clone LIBERO-X
2. clone openpi
3. create a `liberox` conda environment
4. install the required dependencies

Then download the default pi 0.5 LIBERO checkpoint:

```bash
./scripts/download_checkpoint.sh /path/to/workspace
```

The checkpoint is about 12 GB.

## Running an evaluation

### 1. Start the policy server

In one terminal:

```bash
./scripts/serve_policy.sh /path/to/workspace
```

Leave this running while evaluations are being performed.

### 2. Choose tasks

LIBERO-X contains hundreds of tasks per level, so it is often useful to evaluate on a smaller sample first.

For example, to select 20 tasks from each level:

```bash
python src/sample_tasks.py \
  --libero-x-root /path/to/workspace/LIBERO-X \
  --out-dir task_lists \
  --n-per-level 20
```

### 3. Run the evaluation

In another terminal:

```bash
TASK_LIST_DIR=task_lists TRIALS_PER_TASK=5 \
  ./scripts/run_sweep.sh /path/to/workspace /path/to/results
```

By default, this evaluates Levels 1-5 sequentially.

Runs can be stopped and restarted with the same command. Episodes already present in the results file are skipped automatically.

To evaluate only particular levels:

```bash
LEVELS="LEVEL1 LEVEL2" \
TASK_LIST_DIR=task_lists \
TRIALS_PER_TASK=5 \
./scripts/run_sweep.sh /path/to/workspace /path/to/results
```

## Analyze the evaluation results

After the evaluation finishes:

```bash
python src/analyze_results.py \
  --results-dir /path/to/results \
  --out-json summary.json
```

The evaluator stores one JSON record for each episode.

An example looks like:

```json
{
  "task_desc": "place the black bowl to the right of the bowl drainer",
  "success": false,
  "steps_taken": 500,
  "num_subgoals": 1,
  "subgoals": [
    {
      "predicate": "in akita_black_bowl_1 bowl_drainer_1_right_region",
      "first_true_step": null,
      "final_true": false
    }
  ],
  "num_subgoals_completed": 0
}
```

`first_true_step` records when a subgoal was first completed.

This lets the evaluator distinguish between:

* a subgoal that was never completed
* a subgoal that was completed but later undone

For example, the robot may correctly place a bowl and then accidentally knock it out of position later in the episode.

# Failure labeling

After receiving failed LIBERO-X rollouts, the failure-labeling pipeline first uses simulator predicates to determine which subgoal was not achieved. A video-capable VLM then reviews the rollout to determine the visible failure mode, identify when the behavior responsible for failure begins, explain the failure, and propose a recovery action.

The full pipeline is:

```text
LIBERO-X evaluation
        ↓
results_LEVEL*.jsonl
+ failure videos
        ↓
label_failures.py
        ↓
failure_labels.jsonl
        ↓
sample_for_vlm.py       optional
        ↓
vlm_sample.jsonl
        ↓
label_with_vlm.py
        ↓
vlm_labeled.jsonl
        ↓
build_gallery.py        optional
        ↓
gallery.html
```

## 1. Generate simulator-based failure labels

First run:

```bash
python analysis/label_failures.py \
  --results-dir /path/to/results \
  --out-manifest /path/to/results/failure_labels.jsonl
```

This step does not use a VLM. It reads the goal-predicate information recorded during evaluation and determines where each failed episode stopped making progress.

For example, it may produce:

```json
{
  "failure_category": "multi_step_failed_after_first_subgoal",
  "failing_predicate_type": "uprighton",
  "failing_predicate": "uprighton bottle_1 table_region",
  "failing_subgoal_index": 1,
  "detail": "never satisfied"
}
```

For a multi-step task, the script can distinguish between cases such as:

* failing on the first subgoal
* successfully completing earlier subgoals but failing later
* briefly satisfying a goal and then undoing it

## 2. Optionally sample failures for VLM analysis

If the evaluation contains many failed episodes, it may be unnecessary or expensive to send every video to a VLM.

A representative subset can be generated with:

```bash
python analysis/sample_for_vlm.py \
  --manifest /path/to/results/failure_labels.jsonl \
  --out /path/to/results/vlm_sample.jsonl
```

The sampler selects failures across different failure categories and predicate types.

If you want the VLM to analyze every failure, this step can be skipped and `failure_labels.jsonl` can be passed directly to `label_with_vlm.py`.

## 3. Diagnose and temporally localize failures with a VLM

`label_with_vlm.py` currently uses Gemini because it supports native video input. Set a Gemini API key and install the video-labeling dependencies:

```bash
export GEMINI_API_KEY=...
pip install google-genai decord imageio imageio-ffmpeg
```

Then run:

```bash
python analysis/label_with_vlm.py \
  --sample /path/to/results/vlm_sample.jsonl \
  --out /path/to/results/vlm_labeled.jsonl \
  --backend gemini \
  --model gemini-3.1-pro-preview
```

The VLM receives:

* the complete rollout video
* the original task instruction
* the exact simulator predicate that was not satisfied
* simulator-derived information about the failure

It is asked to determine:

1. the primary visible failure mode
2. when the failure behavior begins
3. why the episode failed
4. what the robot should do to recover

Failure timestep labeling is performed in two stages:

1. **Coarse pass.** Gemini receives the complete rollout video and estimates the failure-onset time in seconds.
2. **Frame-level refinement.** A short window around the coarse estimate is extracted from the original rollout. Each original frame is written as one second of a temporary 1-FPS video, allowing the VLM to select the earliest frame where the identified failure becomes visible.

By default, the refinement window covers 3 seconds before and after the coarse estimate. This can be changed with:

```bash
--refine-window-seconds 3.0
```

The second pass can also be disabled:

```bash
--no-refine
```

For `timeout` failures, no refinement pass is needed; the final frame is used directly.

### Failure taxonomy

The current failure categories are:

* **`grasp_failure`** — the robot attempts to grasp the relevant object but does not successfully acquire it, including repeated unsuccessful grasp attempts.
* **`stuck_or_no_progress`** — the robot becomes stuck, freezes, repeatedly executes ineffective behavior, or otherwise stops making meaningful progress toward the task.
* **`placement_or_insertion_failure`** — the robot reaches the relevant target but fails the required placement, insertion, position, or object orientation.
* **`unstable_or_dangerous_behavior`** — the robot exhibits unstable, erratic, unexpected, or potentially dangerous motion that prevents successful task completion.
* **`object_displacement`** — the robot unintentionally knocks over, pushes away, drops, or otherwise displaces an object in a way that contributes to task failure.
* **`wrong_object_or_target`** — the robot manipulates the wrong object or moves the correct object toward the wrong destination.
* **`timeout_or_insufficient_progress`** — no discrete mistake is clearly identifiable and the robot simply does not complete the task before the episode ends.
* **`other`** — the observed failure does not fit one of the categories above.

### VLM output

The VLM annotations are appended to the simulator-derived failure record. An example looks like:

```json
{
  "vlm_failure_mode": "placement_or_insertion_failure",
  "vlm_failure_onset_type": "obvious_mistake",
  "vlm_failure_onset_seconds": 12.43,
  "vlm_failure_onset_timestamp": "00:12.4",
  "vlm_failure_onset_frame": 248,
  "vlm_failure_onset_step": 248,
  "vlm_coarse_failure_onset_seconds": 12.5,
  "vlm_temporal_refined": true,
  "vlm_confidence": "high",
  "vlm_temporal_confidence": "high",
  "vlm_failure_reason": "The robot moved the bottle into the target region but released it on its side, so the upright placement condition was not satisfied.",
  "vlm_recovery_action": "Re-grasp the bottle, rotate it upright, and place it back inside the target region before releasing it.",
  "vlm_justification": "The rollout shows the bottle being released horizontally in the target area.",
  "vlm_temporal_justification": "This is the first frame where the bottle begins to fall onto its side."
}
```

`vlm_failure_onset_frame` is a 0-based index into the rollout video. Because `eval_subgoals.py` records one rollout image for each action-step observation, the same value is also stored as `vlm_failure_onset_step` for alignment with other timestep-level analyses.

The output also retains the coarse response, refinement response, video metadata, and token usage for later analysis.

## 4. Build a failure video gallery

After labeling, the results can be viewed in a local HTML gallery:

```bash
python analysis/build_gallery.py \
  --manifest /path/to/results/vlm_labeled.jsonl \
  --out /path/to/results/gallery.html \
  --root /path/to/results
```

The gallery pairs each failed rollout video with information such as:

* the task instruction
* the simulator-derived failed predicate
* the failure category
* the VLM's failure diagnosis
* the proposed recovery action
* the VLM's justification

This makes it easier to browse failures and compare common failure modes across tasks or benchmark levels.

## References

* [LIBERO-X — Robustness Litmus for Vision-Language-Action Models](https://arxiv.org/abs/2602.06556)
* [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
* [openpi](https://github.com/Physical-Intelligence/openpi)
* [VLA-SAFE](https://vla-safe.github.io/)
