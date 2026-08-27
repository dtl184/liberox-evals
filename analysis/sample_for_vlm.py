"""Stratified sampler that picks failure videos for VLM review from a
failure_labels.jsonl manifest produced by label_failures.py.

Usage:
    python sample_for_vlm.py --manifest failure_labels.jsonl --out vlm_sample.jsonl
"""
import argparse
import json
import random
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--per-combo-single-step", type=int, default=2)
ap.add_argument("--per-combo-multistep-stalled", type=int, default=2)
ap.add_argument("--all-multistep-past-first", action="store_true", default=True,
                 help="include every multi_step_failed_after_first_subgoal episode (default on, it's the rarest/most interesting category)")
args = ap.parse_args()

random.seed(args.seed)

rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
by_combo = defaultdict(list)
for r in rows:
    by_combo[(r["failure_category"], r["failing_predicate_type"])].append(r)

sample = []
for (cat, ptype), group in by_combo.items():
    if cat == "multi_step_failed_after_first_subgoal" and args.all_multistep_past_first:
        sample.extend(group)
        continue
    k = args.per_combo_single_step if cat == "single_step_incomplete" else args.per_combo_multistep_stalled
    sample.extend(random.sample(group, min(k, len(group))))

print(f"Sampled {len(sample)} of {len(rows)} total labeled failures")
by_cat = defaultdict(int)
for r in sample:
    by_cat[r["failure_category"]] += 1
for cat, n in by_cat.items():
    print(f"  {cat}: {n}")

with open(args.out, "w") as f:
    for r in sample:
        f.write(json.dumps(r) + "\n")
