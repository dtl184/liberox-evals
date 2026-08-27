# LIBERO-X Evaluations

Tools for evaluating robot policies on [LIBERO-X](https://github.com/meituan/LIBERO-X).

For each episode, the evaluator records:

* whether the overall task succeeded
* which parts of the task were completed
* when each part was first completed
* which part the policy failed to complete
* a video of failed episodes

The code was developed and tested with pi 0.5, but it can be used with other policies served through an `openpi-client` compatible policy server.

## LIBERO-X

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) is a widely used benchmark for robot manipulation. It contains a fixed set of tasks across several tabletop and kitchen environments, with tasks such as picking up objects, placing them in containers, opening drawers, and completing short multi-step instructions.

[LIBERO-X](https://github.com/meituan/LIBERO-X) builds on the same underlying simulation framework, but is designed specifically to test robustness and generalization. LIBERO-X changes the scene, objects, and instructions in increasingly difficult ways.

The benchmark is divided into five levels:

* **Level 1:** Changes object and scene layouts while keeping the task relatively close to standard LIBERO.
* **Level 2:** Introduces larger changes to object identity, appearance, and placement.
* **Level 3:** Uses more substantially altered scene compositions and combinations of objects.
* **Level 4:** Adds new object attributes such as different colors and shapes. The policy must use these attributes to identify the correct object, for example distinguishing a blue bowl from other bowls in the scene.
* **Level 5:** Uses the same environments and tasks as Level 4, but paraphrases the natural-language instructions.

LIBERO-X is also much larger than the original LIBERO benchmark. LIBERO contains roughly 130 tasks across its standard task suites, while LIBERO-X contains hundreds of tasks at each difficulty level and around 100 new scenes.

LIBERO-X also introduces additional and stricter goal predicates. Standard LIBERO tasks primarily use predicates such as In, On, Open, Close, TurnOn, TurnOff, and Stack. LIBERO-X adds ExactIn, UprightOn, and SideOn. For example, it can distinguish between simply placing a bottle on a surface, placing it upright, and placing it on its side.


## Repository structure

```text
src/
  eval_subgoals.py       Run evaluations and record task progress
  sample_tasks.py        Select a smaller subset of LIBERO-X tasks
  analyze_results.py     Summarize evaluation results

scripts/
  setup_env.sh           Set up LIBERO-X and openpi
  download_checkpoint.sh Download a policy checkpoint
  serve_policy.sh        Start the policy server
  run_sweep.sh           Evaluate across LIBERO-X levels

analysis/
  label_failures.py
  claude_label_failures.py
  build_gallery.py
```

The `analysis/` directory contains optional tools for looking more closely at failure videos.

## Requirements

You will need:

* Linux
* an NVIDIA GPU
* `conda`
* `uv`
* `git`
* `gsutil`

Tested this setup on our GPUs, RTX 4090 with 24 GB of VRAM.

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

LIBERO-X contains hundreds of tasks per level, so it's useful to take a sample of tasks at each level when evaluating a policy. 

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

By default, this evaluates Levels 1–5 sequentially.

Runs can be stopped and restarted with the same command. Episodes that are already present in the results file are skipped automatically.

To evaluate only particular levels:

```bash
LEVELS="LEVEL1 LEVEL2" \
TASK_LIST_DIR=task_lists \
TRIALS_PER_TASK=5 \
./scripts/run_sweep.sh /path/to/workspace /path/to/results
```

## Analyze the results

After the evaluation finishes:

```bash
python src/analyze_results.py \
  --results-dir /path/to/results \
  --out-json summary.json
```

The evaluator stores one JSON record for each episode.

A simplified example looks like:

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

`first_true_step` records when that part of the task was first completed.

This lets the evaluator distinguish between:

* a subgoal that was never completed
* a subgoal that was completed but later undone

For example, the robot might correctly place a bowl and then accidentally knock it out of position later in the episode.

## Failure videos

The default sweep saves videos for failed episodes.

The tools in `analysis/` can be used to organize these videos and inspect common failure modes.

For example:

```bash
python analysis/build_gallery.py ...
```

There is also an optional Claude-based video labeling script for cases where the task result alone does not explain the failure, such as distinguishing between:

* failing to approach the object
* picking up the wrong object
* failing to grasp
* placing an object incorrectly

See `analysis/claude_label_failures.py` for options.

## Important: image rotation

When evaluating π0 or π0.5, the camera image must be rotated **180 degrees** before being sent to the policy.

The reference openpi LIBERO code uses this preprocessing during evaluation. Without it, the policy may still appear to move toward objects but its success rate can fall close to zero.

`eval_subgoals.py` applies the correct rotation automatically.

If you use this repo with another policy, check the preprocessing expected by that policy before evaluating it.

## References

* [LIBERO-X — Robustness Litmus for Vision-Language-Action Models](https://arxiv.org/abs/2602.06556)
* [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
* [openpi](https://github.com/Physical-Intelligence/openpi)
