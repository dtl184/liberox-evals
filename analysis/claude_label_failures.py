"""Claude-as-judge failure-mode labeling for LIBERO-X rollout videos.

Same taxonomy, same still-frame + chain-of-thought approach as the local
Qwen2.5-VL v2 pipeline (vlm_label_failures.py), but calling Claude Opus 4.8
via the Anthropic API instead of a locally-hosted VLM. Built to let the two
be compared directly on the same 69-video sample.

Reusable: point --sample at any JSONL with at least {task_desc,
failing_predicate, video_path} rows (e.g. the output of sample_for_vlm.py)
and it appends claude_* fields, writing a fresh manifest.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python claude_label_failures.py --sample vlm_sample.jsonl --out claude_labeled.jsonl \
        [--num-frames 8] [--limit 5]
"""
import argparse
import base64
import io
import json
import pathlib
import re
import sys
import time

import anthropic
from decord import VideoReader, cpu
from PIL import Image

TAXONOMY = """\
- never_approached: the arm never meaningfully moved toward the relevant object/mechanism
- failed_grasp: the arm reached and attempted to grasp the object but never successfully picked it up
- dropped_object: the object was demonstrably picked up, then visibly fell or was released away from any target while still in transit
- imprecise_placement: the object was carried to approximately the right area and released there, but placement missed the precise target region (close but not correct)
- wrong_target: the object was carried to and released in a clearly different location than the target, or the wrong object was manipulated
- actuation_failure: for open/close/turn-on/turn-off tasks, the arm reached the mechanism (drawer, stove knob, microwave) but failed to actuate it correctly
- idle_or_erratic: the arm barely moved, froze, or moved with no clear purpose for most of the episode
- other: none of the above fit; explain in justification"""

PROMPT_TEMPLATE = """You are reviewing frames from a failed robot manipulation rollout in a simulated tabletop/kitchen environment. The images above are frames shown in time order, evenly spaced from the start to the end of the episode, each preceded by a text label naming its position.

Task instruction given to the robot: "{task_desc}"
The specific goal that was NOT achieved by the end of the episode: {failing_predicate}
({detail})

First, in 2-4 sentences, describe concretely what you observe happening across the frames: where does the arm go, does it contact/grasp an object, does the object move, and where does everything end up relative to where it needed to be. Be specific about what you can actually see in each frame -- do not assume or invent events (e.g. do not say an object was "dropped" unless you can see it separated from the gripper and away from the target in a later frame).

Then, based ONLY on what you just described, classify HOW the robot failed, choosing exactly one category:
{taxonomy}

End your response with a line starting with "ANSWER:" followed by ONLY a JSON object:
ANSWER: {{"failure_mode": "<one category from the list above>", "confidence": "high|medium|low", "justification": "<one concise sentence citing what you observed>"}}"""


def extract_frames(video_path: str, num_frames: int):
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    idx = [round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]
    frames = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(f) for f in frames]


def frame_to_b64_jpeg(frame: Image.Image) -> str:
    buf = io.BytesIO()
    frame.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def parse_json_response(text: str) -> dict:
    marker_idx = text.rfind("ANSWER:")
    search_text = text[marker_idx:] if marker_idx != -1 else text
    matches = re.findall(r"\{.*?\}", search_text, re.DOTALL)
    if not matches:
        return {"failure_mode": "parse_error", "confidence": "low", "justification": text[:300]}
    try:
        parsed = json.loads(matches[-1])
        parsed["_reasoning"] = text[:marker_idx].strip() if marker_idx != -1 else ""
        return parsed
    except json.JSONDecodeError:
        return {"failure_mode": "parse_error", "confidence": "low", "justification": text[:300]}


def label_one(client, row, num_frames, max_retries=4):
    frames = extract_frames(row["video_path"], num_frames)
    prompt = PROMPT_TEMPLATE.format(
        task_desc=row.get("task_desc", ""),
        failing_predicate=row.get("failing_predicate", ""),
        detail=row.get("detail", ""),
        taxonomy=TAXONOMY,
    )

    content = []
    for fi, frame in enumerate(frames):
        label = "start" if fi == 0 else ("end" if fi == len(frames) - 1 else f"t={fi}/{len(frames)-1}")
        content.append({"type": "text", "text": f"Frame {fi+1} ({label}):"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_to_b64_jpeg(frame)},
        })
    content.append({"type": "text", "text": prompt})

    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=1024,
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": "high"},
                messages=[{"role": "user", "content": content}],
            )
            break
        except anthropic.RateLimitError as e:
            last_exc = e
            time.sleep(min(60, 2 ** attempt * 2))
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last_exc = e
                time.sleep(min(60, 2 ** attempt * 2))
            else:
                raise
        except anthropic.APIConnectionError as e:
            last_exc = e
            time.sleep(min(60, 2 ** attempt * 2))
    else:
        raise last_exc

    text_blocks = [b.text for b in response.content if b.type == "text"]
    thinking_blocks = [b.thinking for b in response.content if b.type == "thinking"]
    response_text = "\n".join(text_blocks).strip()

    parsed = parse_json_response(response_text)
    return {
        "claude_failure_mode": parsed.get("failure_mode"),
        "claude_confidence": parsed.get("confidence"),
        "claude_justification": parsed.get("justification"),
        "claude_reasoning": parsed.get("_reasoning", ""),
        "claude_thinking_summary": "\n".join(thinking_blocks).strip(),
        "claude_raw_response": response_text,
        "claude_usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    client = anthropic.Anthropic()

    rows = [json.loads(l) for l in open(args.sample) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    out_path = pathlib.Path(args.out)
    n_ok = 0
    total_in, total_out = 0, 0
    with out_path.open("w") as out_f:
        for i, row in enumerate(rows):
            video_path = row.get("video_path")
            if not video_path or not pathlib.Path(video_path).exists():
                row["claude_failure_mode"] = "video_missing"
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                continue

            t0 = time.time()
            try:
                result = label_one(client, row, args.num_frames)
            except Exception as e:
                print(f"[{i+1}/{len(rows)}] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
                row["claude_failure_mode"] = "api_error"
                row["claude_justification"] = f"{type(e).__name__}: {e}"
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                continue
            elapsed = time.time() - t0

            row.update(result)
            total_in += result["claude_usage"]["input_tokens"]
            total_out += result["claude_usage"]["output_tokens"]

            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            n_ok += 1
            print(f"[{i+1}/{len(rows)}] {elapsed:.1f}s  {row['failure_category']:38s} -> {row['claude_failure_mode']}", file=sys.stderr)

    est_cost = total_in / 1e6 * 5.0 + total_out / 1e6 * 25.0
    print(f"\nDone. Labeled {n_ok}/{len(rows)} videos -> {out_path}", file=sys.stderr)
    print(f"Tokens: {total_in} in / {total_out} out. Estimated cost: ${est_cost:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
