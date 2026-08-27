# LIBERO-X Evaluations

A reusable pipeline for running VLA policies against [LIBERO-X](https://github.com/meituan/LIBERO-X) and getting more than a pass/fail number back: every goal predicate in every task is evaluated at every simulation step, so for any multi-step task you know exactly which subgoal the policy completed and which one it stalled on.

Built and test with pi 0.5, but only assumes an [openpi-client](https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client)-compatible websocket policy server, so any policy served that way works.

## What is LIBERO-X, and how does it differ from LIBERO?

**[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)** (NeurIPS 2023) is a benchmark for lifelong robot learning: four fixed task suites (Spatial, Object, Goal, "10"/Long), about 130 tasks total, across a handful of tabletop/kitchen scenes with a fixed, small object set. It's designed to test structured knowledge transfer across a controlled task sequence, not robustness to novelty.

**LIBERO-X** (Meituan, RSS 2026) is built on the same underlying engine (it vendors LIBERO's robosuite/BDDL codebase directly) but is explicitly a *robustness* benchmark — it asks whether a policy generalizes or has memorized specific scenes. Concretely:

- **5 progressively harder difficulty levels (L1→L5)**, not 4 flat suites. L1–L3 escalate perturbation across spatial layout, object identity/texture, and scene composition. L4 adds new object *attributes* (color/shape variants, e.g. "the teal bowl" instead of "the black bowl"). **L5 reuses L4's exact scenes** but swaps in paraphrased natural-language instructions, isolating linguistic robustness from visual difficulty — which is why this pipeline treats L5 as "L4's task files, different prompt" rather than a fifth independent task set (see `run_sweep.sh`).
- **~100 novel scenes and 600–826 tasks per level** (vs. LIBERO's ~130 tasks, period), using new object asset libraries not present in original LIBERO.
- **3 new goal predicates** beyond LIBERO's `On`/`In`/`Open`/`Close`/`TurnOn`/`TurnOff`/`Stack`: `ExactIn` (stricter containment), `UprightOn`, and `SideOn` — more precise placement requirements than plain `On`/`In`.
- **A separate 2,520-demo training set** (600 tasks, 100 scenes) for fine-tuning models *on LIBERO-X itself*, which is the benchmark's intended evaluation setup. A policy evaluated zero-shot (fine-tuned only on vanilla LIBERO, as in this pipeline's default checkpoint) is being asked to generalize much further than the benchmark's own reported numbers assume — expect a large gap from LIBERO-X's own paper results unless you fine-tune on their training set first.

## Repo layout

```
src/
  eval_subgoals.py     # core eval harness: rollout + per-step subgoal telemetry
  sample_tasks.py       # stratified (or full-coverage) task-list sampler
  analyze_results.py    # aggregate SR / subgoal-failure / predicate-type reports
scripts/
  setup_env.sh           # clone LIBERO-X + openpi, create the client conda env
  download_checkpoint.sh # pull an openpi checkpoint (default: pi05_libero)
  serve_policy.sh        # start the policy server
  run_sweep.sh            # run eval_subgoals.py across LEVEL1-5
analysis/                 # optional: VLM-based qualitative failure labeling
  label_failures.py       # exact, predicate-grounded failure-mode labels (no VLM)
  claude_label_failures.py # Claude-as-judge visual failure-mode labels (see below)
  build_gallery.py        # local HTML gallery pairing videos with labels
```

## Prerequisites

- NVIDIA GPU with ≥24GB VRAM (validated on an RTX 4090), recent driver
- ~50GB free disk for LIBERO-X + openpi + a checkpoint; more if you keep failure videos from a large sweep (~200KB/episode)
- `conda`, `uv`, `git`, `gsutil` (or manual download access to `gs://openpi-assets`)

## Setup

```bash
./scripts/setup_env.sh /path/to/workspace
./scripts/download_checkpoint.sh /path/to/workspace          # ~12GB, pi05_libero by default
```

This clones `LIBERO-X/` and `openpi/` into the workspace and creates a `liberox` conda env (Python 3.9) for the simulation/client side. openpi manages its own `uv`-based venv for the policy server.

**A note on the client env:** `conda activate` inside a non-interactive script does not reliably persist across shell invocations — it's easy to end up silently installing packages into the wrong (system) Python instead of the `liberox` env, which then produces confusing version-conflict errors deep in robosuite/mujoco. `setup_env.sh` and `run_sweep.sh` both invoke the env's Python by absolute path (`$(conda info --base)/envs/liberox/bin/python`) for exactly this reason — do the same in any script you add, rather than `conda activate` + bare `python`.

## Running

**1. Start the policy server** (leave running in its own terminal/background process):
```bash
./scripts/serve_policy.sh /path/to/workspace
```

**2. (Optional) Sample a task subset.** LIBERO-X has 600–826 tasks *per level* — running everything at high trial counts is a multi-day job (see Scaling below). For a pilot, sample a stratified subset biased toward multi-step tasks:
```bash
python src/sample_tasks.py --libero-x-root /path/to/workspace/LIBERO-X \
  --out-dir task_lists --n-per-level 20
```
Omit `--n-per-level` to sample every task (full coverage, no downsampling).

**3. Run the sweep:**
```bash
TASK_LIST_DIR=task_lists TRIALS_PER_TASK=5 \
  ./scripts/run_sweep.sh /path/to/workspace /path/to/results
```
Or run a single level directly for finer control (`src/eval_subgoals.py --help` for all options):
```bash
MUJOCO_GL=egl $(conda info --base)/envs/liberox/bin/python src/eval_subgoals.py \
  --scene-group LEVEL1 --load-mode init \
  --task-list-file task_lists/LEVEL1.txt --num-trials-per-task 5 \
  --video-out-path /path/to/results/LEVEL1 \
  --results-out-path /path/to/results/LEVEL1/results_LEVEL1.jsonl
```

**4. Analyze:**
```bash
python src/analyze_results.py --results-dir /path/to/results --out-json summary.json
```

## Scaling to 500–1000+ episodes

Measured throughput on a single RTX 4090: **~14 seconds/episode**, essentially constant regardless of task difficulty or success rate (a failing episode still runs its full step budget — nothing here is bottlenecked on GPU compute; it's dominated by MuJoCo physics + offscreen rendering). At that rate:

| Episodes | Wall clock |
|---|---|
| 500 | ~2 hours |
| 1,000 | ~4 hours |
| Full LEVEL1 (600 tasks × 5 trials = 3,000) | ~12 hours |

This is comfortably a single-GPU, single-process job at the 500–1000 episode scale — no need to reach for parallelism there. A few things that matter more at that scale than at pilot scale:

- **Resume is automatic.** `eval_subgoals.py` skips any `(task, episode)` pair already present in `--results-out-path` on startup. A multi-hour sweep can be safely Ctrl-C'd or crash and picked back up with the identical command — nothing gets duplicated, and `summary.json`'s totals are recomputed from the full results file, not just what ran in that invocation. Pass `--no-resume` to force a clean re-run.
- **A single bad task file can't kill an unattended run.** Both the per-task setup (BDDL parsing, init-state loading) and the per-episode rollout are wrapped so one failure is logged as a `failure_stage` row and the sweep continues — this matters once you're running thousands of episodes unattended overnight.
- **`run_sweep.sh` uses `set -uo pipefail`, not `-e`** — deliberately, so one `eval_subgoals.py` process exiting nonzero (e.g. an unhandled crash) doesn't stop the remaining levels from running.
- **Video storage is bounded by default.** `--save-failure-videos --no-save-all-videos` (the `run_sweep.sh` default) only keeps videos for failed episodes; at a typical LIBERO-X zero-shot success rate that's still most episodes. Each is ~150–300KB, so 1,000 episodes is well under 300MB — not a real disk concern, but worth knowing if you flip on `--save-all-videos`.
- **If you do need to go faster:** the bottleneck is per-process simulation, not the GPU (which sits mostly idle during a sweep — inference is a small fraction of each episode's wall time). Sharding a task list across multiple `eval_subgoals.py` processes hitting the same policy server (which handles concurrent websocket connections) should parallelize close to linearly, but this pipeline hasn't been load-tested that way — validate on a small shard before trusting a large unattended parallel run.

## Output format

One JSON object per line in `results_LEVELn.jsonl`:

```jsonc
{
  "scene_group": "LEVEL1", "scene_name": "SCENE10", "task_id": 2,
  "task_file": "libero/libero_x/bddl/LEVEL1/....bddl",
  "task_desc": "place the black bowl to the right of the bowl drainer and place the green bowl to the left of the bowl drainer",
  "episode_index": 0, "success": false, "steps_taken": 500,
  "video_path": ".../rollout_..._failure.mp4",
  "num_subgoals": 2,
  "subgoals": [
    {"predicate": "in akita_black_bowl_1 bowl_drainer_1_right_region", "first_true_step": null, "final_true": false},
    {"predicate": "in akita_green_bowl_1 bowl_drainer_1_left_region", "first_true_step": null, "final_true": false}
  ],
  "first_unmet_subgoal_index": 0,       // which subgoal (0-indexed) it never completed
  "num_subgoals_completed": 0
}
```

`first_true_step` vs. `final_true` distinguishes "never attempted" (`first_true_step: null`) from "achieved it, then lost it" (`first_true_step` set, `final_true: false`) — e.g. a bowl placed correctly and then knocked out of position by a later action.

## Known pitfalls

**The single most important thing to get right: image preprocessing must rotate the camera frame 180° (flip both axes), not just vertically.** π0/π0.5 were trained with this rotation; openpi's own reference LIBERO client applies it unconditionally with the comment *"rotate 180 degrees to match train preprocessing."* LIBERO-X's own `eval_template.py` (which this pipeline is adapted from) gates this behind a `--flip-images` flag that **defaults to off** — get this wrong and success rate silently collapses to near-zero while rollout videos still look "active" (the policy reaches roughly toward the right area, just never completes precise manipulation), which reads exactly like a real zero-shot generalization failure rather than a bug. `eval_subgoals.py` always applies the full flip; if you're adapting this pipeline for a different policy, verify what rotation *that* policy's own reference client uses before trusting a suspiciously-low success rate.

**Verify a new checkpoint/setup against vanilla LIBERO first**, not LIBERO-X directly. LIBERO-X's own scenes are novel enough that a genuinely broken harness and genuine zero-shot difficulty look identical (both give ~0%). Point `--bddl-root` at LIBERO-X's vendored copy of original LIBERO (`libero/libero/bddl_files/<suite>`, e.g. `libero_goal`) with `--load-mode bddl` instead of `init`, and confirm you land near the policy's published vanilla-LIBERO number before trusting any LIBERO-X result. (π0.5/`pi05_libero` should land close to 96–98% on `libero_goal`.)

**`ExactIn`-type tasks are extremely hard for a zero-shot checkpoint.** In our runs, first-subgoal completion for `exactin` predicates was under 1%, versus 7–13% for `on`/`in`. If your predicate-type breakdown shows `exactin` dominating the failure count, that's consistent with prior results, not necessarily a new problem.

## The `analysis/` extras (optional)

Beyond pass/fail and subgoal telemetry, `analysis/label_failures.py` turns the exact-predicate data into a browsable, categorically-labeled video manifest (no VLM, fully deterministic — it reads what `eval_subgoals.py` already logged). `analysis/build_gallery.py` renders any labeled manifest as a local HTML gallery you can serve with `python -m http.server` and view over a forwarded port.

For a *qualitative* read on failure modes the predicate data can't distinguish (e.g. "never approached the object" vs. "grasped the wrong object" vs. "placed it imprecisely"), `analysis/claude_label_failures.py` uses Claude (`claude-opus-4-8` by default) as a visual judge over sampled still frames with a describe-then-classify prompt — validated against a local open-weight VLM (Qwen2.5-VL) which we found, after spot-checking its output against raw frames, to be unreliable for this fine-grained a task (it collapsed to guessing from the task description text rather than the video). Roughly $0.02/video at current pricing. Requires `ANTHROPIC_API_KEY` and `pip install -r analysis/requirements.txt`.

## References

- LIBERO-X — Wang et al., *Robustness Litmus for Vision-Language-Action Models*, RSS 2026. [arXiv:2602.06556](https://arxiv.org/abs/2602.06556)
- LIBERO — Liu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, NeurIPS 2023.
- π0.5 / openpi — Physical Intelligence, [github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
