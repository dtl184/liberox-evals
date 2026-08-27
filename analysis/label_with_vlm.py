"""VLM-based failure diagnosis and recovery labeling for LIBERO-X rollouts.

For each failed rollout, the VLM receives:
    - the task instruction
    - the exact simulator predicate that was not achieved
    - simulator-derived failure information
    - evenly sampled frames from the rollout video

The VLM then:
    1. describes what happened,
    2. classifies the visible failure mode,
    3. explains why the episode failed, and
    4. proposes what the robot should do next to recover.

The currently implemented backend is Anthropic with Opus 4.8

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...

    python analysis/label_with_vlm.py \
        --sample vlm_sample.jsonl \
        --out vlm_labeled.jsonl

Optional:
    --num-frames 8
    --limit 5
    --backend anthropic
    --model claude-opus-4-8
"""

import argparse
import base64
import io
import json
import pathlib
import sys
import time

import anthropic
from decord import VideoReader, cpu
from PIL import Image


# Failure modes
TAXONOMY = """\
- grasp_failure: the robot attempts to grasp the relevant object but does not successfully acquire it, including repeated unsuccessful grasp attempts
- stuck_or_no_progress: the robot becomes stuck, freezes, repeatedly executes ineffective behavior, or otherwise stops making meaningful progress toward the task
- placement_or_insertion_failure: the robot reaches the relevant target but fails the required placement, insertion, position, or object orientation
- unstable_or_dangerous_behavior: the robot exhibits unstable, erratic, unexpected, or potentially dangerous motion that prevents successful task completion
- object_displacement: the robot unintentionally knocks over, pushes away, drops, or otherwise displaces an object in a way that contributes to task failure
- wrong_object_or_target: the robot manipulates the wrong object or moves the correct object toward the wrong destination
- other: the observed failure does not fit one of the categories above; explain the failure clearly"""


PROMPT_TEMPLATE = """You are reviewing frames from a failed robot manipulation rollout in a simulated tabletop or kitchen environment.

The images above are frames shown in chronological order and sampled evenly from the beginning to the end of the episode.

Task instruction:
"{task_desc}"

The simulator reports that this goal was NOT achieved by the end of the episode:
{failing_predicate}

Additional simulator information:
{detail}

Your job is to determine what happened, why the episode failed, and what the robot should do next to recover.

First, describe what you can concretely observe across the frames in 2-4 sentences. Focus on:
- where the robot arm moves,
- which object or mechanism it interacts with,
- whether a grasp succeeds,
- whether the relevant object moves,
- whether an object is dropped, knocked over, or displaced,
- whether the robot becomes stuck or behaves erratically,
- where the relevant object ends up,
- and how the final state differs from the requested goal.

Only claim events that are supported by the provided frames. Do not invent actions that are not visible. For example, do not say that an object was dropped unless later frames provide evidence that it left the gripper away from the intended target.

Then classify the PRIMARY visible failure using exactly one of these categories:

{taxonomy}

If more than one failure occurs, select the category that best explains why the task ultimately failed. You may mention secondary failures in the failure reason.

Next, explain the immediate reason the episode failed. This should be specific to the observed rollout and more informative than the broad failure category.

For example:

"The robot reached the correct bowl but repeatedly closed the gripper beside it, so the bowl was never lifted."

"The robot transported the bottle to the correct region but released it horizontally, while the goal required the bottle to remain upright."

"The robot missed the target region during placement and then began moving erratically instead of attempting to correct the placement."

Finally, propose a recovery action.

The recovery action should describe what the robot should do NEXT FROM THE OBSERVED FINAL STATE. Do not simply repeat the original task instruction.

A useful recovery action should identify:
1. what object or mechanism the robot should interact with,
2. what corrective action it should perform, and
3. what condition it should achieve before continuing with the rest of the task.

Good recovery examples include:

- "Move the gripper back above the bowl, center the gripper around it, and retry the grasp before continuing toward the target."
- "Re-grasp the bottle, rotate it upright, and place it back inside the target region before releasing it."
- "Release the incorrect object, move to the requested object, and establish a successful grasp before resuming the task."
- "Move back to the drawer handle, establish contact with the handle, and pull outward until the drawer is visibly open."
- "Stop the erratic motion, return the arm to a stable pose above the workspace, and retry the failed placement."

If the frames do not provide enough information to confidently determine the exact recovery action, propose the safest reasonable retry of the failed step rather than inventing unseen state.

End your response with a line beginning with "ANSWER:" followed by ONLY a JSON object with this exact structure:

ANSWER: {{
    "failure_mode": "<one category from the taxonomy>",
    "confidence": "high|medium|low",
    "failure_reason": "<concise explanation of why this rollout failed>",
    "recovery_action": "<specific next action the robot should take from the observed final state>",
    "justification": "<one concise sentence citing the visual evidence supporting the diagnosis>"
}}"""


def extract_frames(video_path: str, num_frames: int):
    """Extract evenly spaced frames from a rollout video."""
    if num_frames < 2:
        raise ValueError("--num-frames must be at least 2")

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)

    if total == 0:
        raise ValueError(f"Video contains no frames: {video_path}")

    idx = [
        round(i * (total - 1) / (num_frames - 1))
        for i in range(num_frames)
    ]

    frames = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(frame) for frame in frames]


def frame_to_b64_jpeg(frame: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded JPEG."""
    buf = io.BytesIO()
    frame.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def parse_json_response(text: str) -> dict:
    """Extract the final ANSWER JSON object from the VLM response."""
    marker_idx = text.rfind("ANSWER:")

    if marker_idx != -1:
        search_text = text[marker_idx + len("ANSWER:"):].strip()
        reasoning = text[:marker_idx].strip()
    else:
        search_text = text.strip()
        reasoning = ""

    # First try parsing everything following ANSWER: directly.
    try:
        parsed = json.loads(search_text)
        parsed["_reasoning"] = reasoning
        return parsed
    except json.JSONDecodeError:
        pass

    # Fall back to finding the first JSON object in the remaining text.
    start = search_text.find("{")

    if start == -1:
        return {
            "failure_mode": "parse_error",
            "confidence": "low",
            "failure_reason": "",
            "recovery_action": "",
            "justification": text[:300],
            "_reasoning": reasoning,
        }

    try:
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(search_text[start:])
        parsed["_reasoning"] = reasoning
        return parsed
    except json.JSONDecodeError:
        return {
            "failure_mode": "parse_error",
            "confidence": "low",
            "failure_reason": "",
            "recovery_action": "",
            "justification": text[:300],
            "_reasoning": reasoning,
        }


def create_vlm_client(backend: str):
    """Create the client for the selected VLM backend."""
    if backend == "anthropic":
        return anthropic.Anthropic()

    # Add other backends here as desired

    raise ValueError(f"Unsupported VLM backend: {backend}")


def call_vlm(
    backend,
    client,
    frames,
    prompt,
    model,
    max_retries=4,
):
    """Send rollout frames and the diagnosis prompt to the selected VLM."""

    content = []

    for fi, frame in enumerate(frames):
        if fi == 0:
            label = "start"
        elif fi == len(frames) - 1:
            label = "end"
        else:
            label = f"t={fi}/{len(frames) - 1}"

        content.append({
            "type": "text",
            "text": f"Frame {fi + 1} ({label}):",
        })

        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame_to_b64_jpeg(frame),
            },
        })

    content.append({
        "type": "text",
        "text": prompt,
    })

    last_exc = None

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1400,
                thinking={
                    "type": "adaptive",
                    "display": "summarized",
                },
                output_config={
                    "effort": "high",
                },
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
            )

            return response

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

    raise last_exc


def extract_response_text(backend, response):
    """Extract plain text from a VLM backend response."""
    if backend == "anthropic":
        text_blocks = [
            block.text
            for block in response.content
            if block.type == "text"
        ]
        return "\n".join(text_blocks).strip()

    raise ValueError(f"Unsupported VLM backend: {backend}")


def extract_usage(backend, response):
    """Return normalized token-usage information when available."""
    if backend == "anthropic":
        return {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    return {
        "input_tokens": 0,
        "output_tokens": 0,
    }


def label_one(
    backend,
    client,
    row,
    num_frames,
    model,
):
    """Diagnose one failed rollout and propose a recovery action."""

    frames = extract_frames(
        row["video_path"],
        num_frames,
    )

    prompt = PROMPT_TEMPLATE.format(
        task_desc=row.get("task_desc", ""),
        failing_predicate=row.get(
            "failing_predicate",
            "",
        ),
        detail=row.get("detail", ""),
        taxonomy=TAXONOMY,
    )

    response = call_vlm(
        backend=backend,
        client=client,
        frames=frames,
        prompt=prompt,
        model=model,
    )

    response_text = extract_response_text(
        backend=backend,
        response=response,
    )

    parsed = parse_json_response(response_text)

    usage = extract_usage(
        backend=backend,
        response=response,
    )

    return {
        "vlm_failure_mode": parsed.get(
            "failure_mode"
        ),
        "vlm_confidence": parsed.get(
            "confidence"
        ),
        "vlm_failure_reason": parsed.get(
            "failure_reason"
        ),
        "vlm_recovery_action": parsed.get(
            "recovery_action"
        ),
        "vlm_justification": parsed.get(
            "justification"
        ),
        "vlm_reasoning": parsed.get(
            "_reasoning",
            "",
        ),
        "vlm_raw_response": response_text,
        "vlm_usage": usage,
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Use a VLM to diagnose LIBERO-X failure videos "
            "and propose recovery actions."
        )
    )

    ap.add_argument(
        "--sample",
        required=True,
        help=(
            "Input JSONL manifest containing task_desc, "
            "failing_predicate, and video_path."
        ),
    )

    ap.add_argument(
        "--out",
        required=True,
        help="Output labeled JSONL manifest.",
    )

    ap.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help=(
            "Number of evenly spaced video frames "
            "to send to the VLM."
        ),
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N episodes.",
    )

    ap.add_argument(
        "--backend",
        choices=["anthropic"],
        default="anthropic",
    )

    ap.add_argument(
        "--model",
        default="claude-opus-4-8",
    )

    args = ap.parse_args()

    client = create_vlm_client(args.backend)

    with open(args.sample) as f:
        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if args.limit:
        rows = rows[:args.limit]

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_ok = 0
    total_in = 0
    total_out = 0

    with out_path.open("w") as out_f:
        for i, row in enumerate(rows):
            video_path = row.get("video_path")

            if (
                not video_path
                or not pathlib.Path(video_path).exists()
            ):
                row.update({
                    "vlm_backend": args.backend,
                    "vlm_model": args.model,
                    "vlm_failure_mode": "video_missing",
                    "vlm_confidence": "low",
                    "vlm_failure_reason": "",
                    "vlm_recovery_action": "",
                    "vlm_justification": (
                        "Rollout video is missing."
                    ),
                })

                out_f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                out_f.flush()
                continue

            t0 = time.time()

            try:
                result = label_one(
                    backend=args.backend,
                    client=client,
                    row=row,
                    num_frames=args.num_frames,
                    model=args.model,
                )

            except Exception as e:
                print(
                    f"[{i + 1}/{len(rows)}] "
                    f"ERROR: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )

                row.update({
                    "vlm_backend": args.backend,
                    "vlm_model": args.model,
                    "vlm_failure_mode": "api_error",
                    "vlm_confidence": "low",
                    "vlm_failure_reason": "",
                    "vlm_recovery_action": "",
                    "vlm_justification": (
                        f"{type(e).__name__}: {e}"
                    ),
                })

                out_f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                out_f.flush()
                continue

            elapsed = time.time() - t0

            row.update(result)

            row["vlm_backend"] = args.backend
            row["vlm_model"] = args.model

            total_in += result[
                "vlm_usage"
            ].get("input_tokens", 0)

            total_out += result[
                "vlm_usage"
            ].get("output_tokens", 0)

            out_f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            out_f.flush()

            n_ok += 1

            failure_category = row.get(
                "failure_category",
                "unknown",
            )

            failure_mode = row.get(
                "vlm_failure_mode",
                "unknown",
            )

            print(
                f"[{i + 1}/{len(rows)}] "
                f"{elapsed:.1f}s  "
                f"{failure_category:38s} "
                f"-> {failure_mode}",
                file=sys.stderr,
            )

    print(
        f"\nDone. Labeled "
        f"{n_ok}/{len(rows)} videos "
        f"-> {out_path}",
        file=sys.stderr,
    )

    print(
        f"Tokens: "
        f"{total_in} in / "
        f"{total_out} out.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()