"""Aggregate analysis for eval_subgoals.py output.

Reads every LEVEL*/results_LEVEL*.jsonl under --results-dir and reports:
  - success rate per level and overall, split by single- vs multi-step tasks
  - where multi-step failures stall (first_unmet_subgoal_index distribution)
  - completion rate of the first subgoal, broken down by predicate type
    (on/in/exactin/close/open/turnon/turnoff/uprighton/sideon/...)
  - for tasks that get past subgoal 0, success rate by (step1_type -> step2_type)

Usage:
    python analyze_results.py --results-dir results/ [--out-json summary.json]
"""
import argparse
import json
import pathlib
from collections import Counter


def predicate_type(pred_str: str) -> str:
    return pred_str.split()[0]


def load_all(results_dir: pathlib.Path):
    by_level = {}
    for path in sorted(results_dir.glob("LEVEL*/results_LEVEL*.jsonl")):
        level = path.parent.name
        rows = [json.loads(l) for l in path.open() if l.strip()]
        by_level.setdefault(level, []).extend(rows)
    return by_level


def per_level_summary(by_level):
    summary = {}
    for level, rows in by_level.items():
        n = len(rows)
        succ = sum(r["success"] for r in rows)
        single = [r for r in rows if r.get("num_subgoals", 1) == 1]
        multi = [r for r in rows if r.get("num_subgoals", 1) >= 2]
        summary[level] = {
            "episodes": n,
            "successes": succ,
            "success_rate": succ / n if n else 0.0,
            "single_step_episodes": len(single),
            "single_step_successes": sum(r["success"] for r in single),
            "multi_step_episodes": len(multi),
            "multi_step_successes": sum(r["success"] for r in multi),
        }
    return summary


def subgoal_failure_breakdown(rows):
    multi = [r for r in rows if r.get("num_subgoals", 1) >= 2]
    fails = [r for r in multi if not r["success"]]
    idx_dist = Counter(r["first_unmet_subgoal_index"] for r in fails if r.get("first_unmet_subgoal_index") is not None)
    return {
        "multi_step_episodes": len(multi),
        "multi_step_failures": len(fails),
        "first_unmet_subgoal_index_distribution": dict(sorted(idx_dist.items())),
    }


def predicate_type_breakdown(rows):
    multi = [r for r in rows if r.get("num_subgoals", 1) >= 2 and r.get("subgoals")]
    sg0_total, sg0_done = Counter(), Counter()
    combo_total, combo_succ = Counter(), Counter()
    for r in multi:
        sgs = r["subgoals"]
        t0 = predicate_type(sgs[0]["predicate"])
        sg0_total[t0] += 1
        if sgs[0]["final_true"]:
            sg0_done[t0] += 1
        if len(sgs) >= 2:
            t1 = predicate_type(sgs[1]["predicate"])
            combo_total[(t0, t1)] += 1
            if r["success"]:
                combo_succ[(t0, t1)] += 1
    step1_rates = {t: {"n": n, "completed": sg0_done[t], "rate": sg0_done[t] / n} for t, n in sg0_total.items()}
    combo_rates = {
        f"{t0}->{t1}": {"n": n, "full_success": combo_succ[(t0, t1)], "rate": combo_succ[(t0, t1)] / n}
        for (t0, t1), n in combo_total.items()
    }
    return {"step1_completion_by_type": step1_rates, "step1_to_step2_combo_success": combo_rates}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    by_level = load_all(results_dir)
    if not by_level:
        raise SystemExit(f"No LEVEL*/results_LEVEL*.jsonl files found under {results_dir}")

    all_rows = [r for rows in by_level.values() for r in rows]
    report = {
        "per_level": per_level_summary(by_level),
        "overall": per_level_summary({"ALL": all_rows})["ALL"],
        "subgoal_failure_breakdown": subgoal_failure_breakdown(all_rows),
        "predicate_type_breakdown": predicate_type_breakdown(all_rows),
    }

    print(f"{'LEVEL':10}{'SR':>8}  {'n':>6}   single-step        multi-step")
    for level, s in report["per_level"].items():
        ss = f"{s['single_step_successes']}/{s['single_step_episodes']}"
        ms = f"{s['multi_step_successes']}/{s['multi_step_episodes']}"
        print(f"{level:10}{100*s['success_rate']:6.1f}%  {s['episodes']:6d}   {ss:14}   {ms:14}")
    o = report["overall"]
    print(f"\nOVERALL: {o['successes']}/{o['episodes']} = {100*o['success_rate']:.1f}%")

    sfb = report["subgoal_failure_breakdown"]
    print(f"\nMulti-step failures ({sfb['multi_step_failures']}/{sfb['multi_step_episodes']}), by first-unmet-subgoal-index:")
    for idx, n in sfb["first_unmet_subgoal_index_distribution"].items():
        print(f"  subgoal index {idx}: {n} ({100*n/sfb['multi_step_failures']:.1f}%)" if sfb["multi_step_failures"] else f"  subgoal index {idx}: {n}")

    ptb = report["predicate_type_breakdown"]
    print("\nSubgoal-0 (first step) completion rate by predicate type:")
    for t, s in sorted(ptb["step1_completion_by_type"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {t:12s} n={s['n']:4d}  completed={s['completed']:3d}  ({100*s['rate']:.1f}%)")

    print("\nFull-task success rate by (step1_type -> step2_type):")
    for combo, s in sorted(ptb["step1_to_step2_combo_success"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {combo:24s} n={s['n']:4d}  success={s['full_success']:3d}  ({100*s['rate']:.1f}%)")

    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"\nWrote full report to {args.out_json}")


if __name__ == "__main__":
    main()
