"""Reusable failure-labeling pipeline for LIBERO-X (or any eval_subgoals.py) rollouts.

Turns the per-step subgoal telemetry that eval_subgoals.py already logs into a
labeled video manifest for semantic-alignment / RLHF-style review: for every
failed episode, decide WHAT kind of failure it was (never grasped/placed vs.
lost the plan after step 1 vs. a specific predicate type like exactin/close/
turnon) using exact ground truth from the simulator, not a VLM guess.

Labels are derived, not annotated by hand, so this scales to any future
eval_subgoals.py output directory with zero manual work.

Usage:
    python label_failures.py --results-dir /path/to/results_full \
        --out-manifest /path/to/failure_labels.jsonl \
        [--organize-dir /path/to/labeled_videos] [--summary]

Re-run any time a new eval_subgoals.py sweep finishes; point --results-dir at
the new output and it produces a fresh manifest (and, optionally, a symlink
tree organized by label for quick browsing) without touching prior runs.
"""
import argparse
import json
import pathlib
from collections import Counter


def predicate_type(pred_str: str) -> str:
    return pred_str.split()[0]


def classify(record: dict) -> dict:
    """Return a dict of labels for one episode record from a results_LEVEL*.jsonl file."""
    if record.get("failure_stage") == "task_setup":
        return {"failure_category": "task_setup_error", "failing_predicate_type": None,
                "failing_predicate": None, "detail": record.get("error_message", "")}
    if record.get("failure_stage") == "scene_init":
        return {"failure_category": "scene_init_error", "failing_predicate_type": None,
                "failing_predicate": None, "detail": record.get("error_message", "")}

    if record["success"]:
        return {"failure_category": "success", "failing_predicate_type": None,
                "failing_predicate": None, "detail": ""}

    subgoals = record.get("subgoals") or []
    num_subgoals = record.get("num_subgoals", len(subgoals) or 1)
    unmet_idx = record.get("first_unmet_subgoal_index")

    if not subgoals:
        # No subgoal telemetry (e.g. older run) -- fall back to a generic label.
        return {"failure_category": "unknown_failure", "failing_predicate_type": None,
                "failing_predicate": None, "detail": ""}

    # Which subgoal to attribute the failure to: the first one never satisfied.
    if unmet_idx is None:
        # Shouldn't happen for a genuine failure, but guard anyway.
        unmet_idx = next((i for i, sg in enumerate(subgoals) if not sg["final_true"]), len(subgoals) - 1)
    unmet_idx = min(unmet_idx, len(subgoals) - 1)
    failing_sg = subgoals[unmet_idx]
    ftype = predicate_type(failing_sg["predicate"])

    was_transiently_true = failing_sg["first_true_step"] is not None and not failing_sg["final_true"]

    if num_subgoals == 1:
        category = "single_step_lost" if was_transiently_true else "single_step_incomplete"
    else:
        if unmet_idx == 0:
            category = "multi_step_stalled_at_first_subgoal"
        else:
            category = "multi_step_failed_after_first_subgoal"

    return {
        "failure_category": category,
        "failing_predicate_type": ftype,
        "failing_predicate": failing_sg["predicate"],
        "failing_subgoal_index": unmet_idx,
        "detail": "lost after being briefly satisfied" if was_transiently_true else "never satisfied",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="Directory containing LEVEL*/results_LEVEL*.jsonl")
    ap.add_argument("--out-manifest", default=None, help="Output JSONL manifest path (default: <results-dir>/failure_labels.jsonl)")
    ap.add_argument("--organize-dir", default=None, help="If set, build a symlink tree <organize-dir>/<label>/<video>.mp4 for browsing")
    ap.add_argument("--include-success", action="store_true", help="Also label successful episodes (default: failures only)")
    ap.add_argument("--summary", action="store_true", help="Print a label-distribution summary")
    args = ap.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    out_manifest = pathlib.Path(args.out_manifest) if args.out_manifest else results_dir / "failure_labels.jsonl"
    result_files = sorted(results_dir.glob("LEVEL*/results_LEVEL*.jsonl"))
    if not result_files:
        raise SystemExit(f"No results_LEVEL*.jsonl files found under {results_dir}")

    manifest_rows = []
    category_counts = Counter()
    predicate_counts = Counter()
    combo_counts = Counter()

    for rf in result_files:
        level = rf.parent.name
        for line in rf.open():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["success"] and not args.include_success:
                continue
            labels = classify(record)
            if labels["failure_category"] == "success" and not args.include_success:
                continue

            video_path = record.get("video_path") or ""
            row = {
                "level": level,
                "task_id": record.get("task_id"),
                "task_desc": record.get("task_desc"),
                "scene_name": record.get("scene_name"),
                "task_file": record.get("task_file"),
                "episode_index": record.get("episode_index"),
                "num_subgoals": record.get("num_subgoals", 1),
                "steps_taken": record.get("steps_taken"),
                "video_path": video_path,
                "video_exists": bool(video_path) and pathlib.Path(video_path).exists(),
                **labels,
            }
            manifest_rows.append(row)
            category_counts[labels["failure_category"]] += 1
            if labels["failing_predicate_type"]:
                predicate_counts[labels["failing_predicate_type"]] += 1
                combo_counts[(labels["failure_category"], labels["failing_predicate_type"])] += 1

    with out_manifest.open("w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_with_video = sum(1 for r in manifest_rows if r["video_exists"])
    print(f"Wrote {len(manifest_rows)} labeled rows to {out_manifest}")
    print(f"  {n_with_video} have an on-disk video, {len(manifest_rows) - n_with_video} do not (e.g. task_setup/scene_init errors, or successes when videos aren't saved for them)")

    if args.organize_dir:
        organize_root = pathlib.Path(args.organize_dir)
        n_linked = 0
        for row in manifest_rows:
            if not row["video_exists"]:
                continue
            label = row["failure_category"]
            if row["failing_predicate_type"]:
                label = f"{label}__{row['failing_predicate_type']}"
            label_dir = organize_root / label
            label_dir.mkdir(parents=True, exist_ok=True)
            src = pathlib.Path(row["video_path"]).resolve()
            # Video basenames repeat across levels (same scene/task template shows
            # up at multiple difficulty levels), so prefix with level to avoid
            # silently dropping videos on a filename collision.
            dst = label_dir / f"{row['level']}__{src.name}"
            if not dst.exists():
                dst.symlink_to(src)
                n_linked += 1
            elif not dst.is_symlink() or dst.resolve() != src:
                print(f"  WARNING: unresolved collision at {dst}, skipping {src}")
        print(f"Organized {n_linked} video symlinks under {organize_root}/<label>/")

    if args.summary:
        print("\n=== failure_category distribution ===")
        for cat, n in category_counts.most_common():
            print(f"  {cat:38s} {n:5d}")
        print("\n=== failing_predicate_type distribution ===")
        for pt, n in predicate_counts.most_common():
            print(f"  {pt:12s} {n:5d}")
        print("\n=== category x predicate_type (top 15) ===")
        for (cat, pt), n in combo_counts.most_common(15):
            print(f"  {cat:38s} x {pt:10s} {n:5d}")


if __name__ == "__main__":
    main()
