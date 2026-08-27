"""Stratified task sampler for LIBERO-X evaluation.

Picks N tasks per level, biased toward multi-predicate (multi-step) goals,
spread across distinct scenes, and writes one task-list file per level
(newline-delimited bddl filenames) that eval_subgoals.py consumes via
--task-list-file. Run with no --n-per-level to sample every task in each
level instead (full coverage, no sampling).

Usage:
    python sample_tasks.py --libero-x-root /path/to/LIBERO-X --out-dir task_lists \
        [--n-per-level 20] [--multi-step-fraction 0.7] [--levels LEVEL1 LEVEL2 LEVEL3 LEVEL4] [--seed 7]
"""
import argparse
import json
import pathlib
import random
import re

GOAL_BLOCK_RE = re.compile(r":goal\s*\((.*)\)\s*\)\s*$", re.DOTALL)
PRED_RE = re.compile(r"\(\s*([A-Za-z][A-Za-z0-9_]*)\b")


def count_predicates(bddl_path: pathlib.Path) -> int:
    text = bddl_path.read_text(errors="ignore")
    goal_idx = text.find(":goal")
    if goal_idx == -1:
        return 1
    goal_text = text[goal_idx:]
    preds = PRED_RE.findall(goal_text)
    preds = [p for p in preds if p.lower() not in ("and", "or", "goal")]
    return max(1, len(preds))


def scene_of(fname: str) -> str:
    m = re.search(r"SCENE\d+", fname)
    return m.group() if m else fname


def pick_diverse(pool, k, seen_scenes):
    chosen = []
    for f in pool:
        if len(chosen) >= k:
            break
        sc = scene_of(f)
        if sc not in seen_scenes:
            chosen.append(f)
            seen_scenes.add(sc)
    for f in pool:
        if len(chosen) >= k:
            break
        if f not in chosen:
            chosen.append(f)
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libero-x-root", required=True, help="path to a LIBERO-X checkout (dir containing libero/libero_x/bddl)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-level", type=int, default=None, help="tasks to sample per level; omit for every task (full coverage)")
    ap.add_argument("--multi-step-fraction", type=float, default=0.7, help="fraction of the sample biased toward multi-predicate (multi-step) tasks")
    ap.add_argument("--levels", nargs="+", default=["LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4"],
                     help="levels to sample from (LEVEL5 reuses LEVEL4's task files at eval time, so it is not sampled separately)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    bddl_root = pathlib.Path(args.libero_x_root) / "libero" / "libero_x" / "bddl"
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for level in args.levels:
        level_dir = bddl_root / level
        if not level_dir.is_dir():
            raise FileNotFoundError(f"No such level directory: {level_dir}")
        files = sorted(f.name for f in level_dir.glob("*.bddl"))
        pred_counts = {f: count_predicates(level_dir / f) for f in files}

        if args.n_per_level is None:
            picked = files
        else:
            multi = [f for f, n in pred_counts.items() if n >= 2]
            single = [f for f, n in pred_counts.items() if n == 1]
            random.shuffle(multi)
            random.shuffle(single)

            n_multi = round(args.n_per_level * args.multi_step_fraction)
            n_single = args.n_per_level - n_multi

            seen_scenes = set()
            picked = pick_diverse(multi, n_multi, seen_scenes) + pick_diverse(single, n_single, seen_scenes)
            picked = picked[: args.n_per_level]

        out_path = out_dir / f"{level}.txt"
        out_path.write_text("\n".join(sorted(picked)) + "\n")

        summary[level] = {
            "n_tasks_total": len(files),
            "n_sampled": len(picked),
            "n_multi_step_sampled": sum(1 for f in picked if pred_counts[f] >= 2),
            "n_single_step_sampled": sum(1 for f in picked if pred_counts[f] == 1),
        }
        print(f"{level} -> {out_path} ({len(picked)} of {len(files)} tasks)")

    (out_dir / "sample_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
