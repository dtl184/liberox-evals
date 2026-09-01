"""
VLM-based failure diagnosis, temporal localization, and recovery labeling
for LIBERO-X rollouts.

For each failed rollout, the VLM receives:
    - the entire rollout video
    - the task instruction
    - the exact simulator predicate that was not achieved
    - simulator-derived failure information

The VLM then:
    1. identifies the primary visible failure mode,
    2. identifies when failure behavior begins,
    3. explains why the episode failed, and
    4. proposes a recovery action.

Failure onset follows a SAFE-style operational definition:

    - obvious_mistake:
        The earliest obvious task-relevant mistake that contributes to
        the eventual failure and is not subsequently recovered from.

    - operator_intervention:
        The earliest point at which a reasonable human operator would
        decide that the policy needs help to complete the task.

    - timeout:
        No earlier failure event can be identified. The policy continues
        making plausible progress or simply fails to finish before the
        episode horizon. The final timestep is used.

Transient mistakes that the robot later successfully recovers from
should NOT be labeled as failure onset.

Temporal localization is done in two stages:

    1. The complete rollout video is sent to the VLM to obtain a coarse
       failure-onset timestamp.

    2. A short window around that timestamp is extracted. Each original
       rollout frame in that window is written as one second of a new
       1-FPS video. The VLM then identifies the first frame at which the
       failure becomes visible.

This produces a frame-level VLM annotation while still allowing the
model to see the complete rollout before making the decision.

The current backend is Gemini because it supports native video input.
Other video-capable VLMs can be added by implementing VLMBackend.

Requirements:
    pip install google-genai decord imageio imageio-ffmpeg

Environment:
    export GEMINI_API_KEY=...

Usage:
    python analysis/label_with_vlm.py \
        --sample failure_labels.jsonl \
        --out vlm_labeled.jsonl

Optional:
    --backend gemini
    --model gemini-3.1-pro-preview
    --limit 20
    --refine-window-seconds 3.0
    --no-refine
"""

import argparse
import json
import pathlib
import sys
import tempfile
import time
from abc import ABC, abstractmethod

import imageio.v2 as imageio
from decord import VideoReader, cpu


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

TAXONOMY = """\
- grasp_failure: the robot attempts to grasp the relevant object but does not successfully acquire it, including repeated unsuccessful grasp attempts
- stuck_or_no_progress: the robot becomes stuck, freezes, repeatedly executes ineffective behavior, or otherwise stops making meaningful progress toward the task
- placement_or_insertion_failure: the robot reaches the relevant target but fails the required placement, insertion, position, or object orientation
- unstable_or_dangerous_behavior: the robot exhibits unstable, erratic, unexpected, or potentially dangerous motion that prevents successful task completion
- object_displacement: the robot unintentionally knocks over, pushes away, drops, or otherwise displaces an object in a way that contributes to task failure
- wrong_object_or_target: the robot manipulates the wrong object or moves the correct object toward the wrong destination
- timeout_or_insufficient_progress: no discrete mistake is clearly identifiable and the robot simply does not complete the task before the episode ends
- other: the observed failure does not fit one of the categories above; explain the failure clearly
"""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

COARSE_PROMPT_TEMPLATE = """You are reviewing the COMPLETE video of a failed robot manipulation rollout in LIBERO-X.

The video begins at rollout time 0.0 seconds and has a duration of approximately {duration_seconds:.2f} seconds.

Task instruction:
"{task_desc}"

The simulator reports that this goal predicate was NOT satisfied at the end of the episode:
{failing_predicate}

Additional simulator information:
{detail}

The simulator predicate tells you WHAT goal condition was unsatisfied at the end. It does NOT tell you when the behavior that caused the failure began.

Your job is to determine:

1. the primary visible failure mode,
2. when the failure behavior begins,
3. why the episode failed,
4. and what corrective action should be taken.

FAILURE ONSET DEFINITION

Use exactly one of the following onset types:

obvious_mistake:
    Use this when there is an identifiable task-relevant mistake such as
    dropping an object, knocking an object away, manipulating the wrong
    object, making a failed placement, or otherwise performing an action
    that contributes to the eventual failure.

    The onset should be the EARLIEST point where the mistake responsible
    for the eventual failure becomes apparent.

operator_intervention:
    Use this when there is not one discrete mistake, but the robot reaches
    a point where a reasonable human operator would decide that the policy
    needs assistance.

    Examples include repeatedly attempting an ineffective action, becoming
    stuck, moving erratically, or clearly ceasing to make useful progress.

timeout:
    Use this when there is no defensible earlier failure point. If the robot
    continues behaving plausibly but simply does not finish before the time
    limit, the failure onset is the END of the rollout.

IMPORTANT RULES

- Do NOT treat the goal predicate being false early in the rollout as failure.
  Most goal predicates are naturally false until the robot completes the task.

- Do NOT label a temporary mistake as failure onset if the robot later
  successfully recovers from that mistake.

- Use the earliest mistake or intervention point that actually explains the
  eventual failed outcome.

- Do not use hindsight to label normal task execution as failure merely because
  you know the episode eventually fails.

- If there is no identifiable earlier point, use onset_type="timeout" and set
  failure_onset_seconds to approximately {duration_seconds:.2f}.

FAILURE TAXONOMY

Choose exactly one:

{taxonomy}

RECOVERY ACTION

The recovery action should describe what the robot should do at or immediately
after the identified failure onset in order to recover.

It should identify:
1. what object or mechanism to interact with,
2. what corrective action to perform,
3. what condition should be achieved before continuing.

Return ONLY a JSON object with this structure:

{{
    "failure_mode": "<one category from the taxonomy>",
    "onset_type": "obvious_mistake|operator_intervention|timeout",
    "failure_onset_seconds": <number>,
    "failure_window_start_seconds": <number>,
    "failure_window_end_seconds": <number>,
    "confidence": "high|medium|low",
    "failure_reason": "<specific explanation of why this rollout failed>",
    "recovery_action": "<specific corrective action from the failure state>",
    "justification": "<concise visual evidence for the failure type and onset>"
}}
"""


REFINE_PROMPT_TEMPLATE = """You are reviewing a TEMPORALLY MAGNIFIED clip from a failed robot rollout.

The complete rollout was already reviewed by another pass of the same VLM.

The coarse analysis identified:

Failure mode:
{failure_mode}

Failure onset type:
{onset_type}

Failure reason:
{failure_reason}

Coarse onset estimate in the original rollout:
{coarse_seconds:.2f} seconds

Task instruction:
"{task_desc}"

Failed simulator predicate:
{failing_predicate}

TEMPORAL MAPPING

This clip has been deliberately slowed down.

Each SECOND of this clip corresponds to exactly ONE FRAME from the original
rollout.

Therefore:

    refined clip second 0 = original rollout frame {start_frame}
    refined clip second 1 = original rollout frame {start_frame_plus_one}
    refined clip second 2 = original rollout frame {start_frame_plus_two}
    ...

The clip contains {num_frames} original rollout frames.

Your job is ONLY to refine the temporal location of the failure.

Find the EARLIEST frame in this clip where the failure identified above becomes
visibly apparent or where operator intervention becomes justified.

Remember:

- Do not select a temporary mistake if the robot recovers from it.
- Do not select a frame merely because the task is not complete yet.
- Select the first frame associated with the behavior responsible for the
  eventual failure.
- If the coarse event is not actually visible in this clip, return null.

Return ONLY:

{{
    "refined_second": <integer from 0 to {max_second}, or null>,
    "confidence": "high|medium|low",
    "justification": "<brief description of what changes at this frame>"
}}
"""


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict:
    """Extract a JSON object from a VLM response."""

    if not text:
        raise ValueError("Empty VLM response")

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")

    if start == -1:
        raise ValueError(
            f"No JSON object found in VLM response: {text[:300]}"
        )

    decoder = json.JSONDecoder()

    try:
        parsed, _ = decoder.raw_decode(cleaned[start:])
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse VLM JSON response: {text[:500]}"
        ) from e


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def get_video_metadata(video_path):
    """Return frame count, FPS, and final-frame timestamp."""

    vr = VideoReader(
        str(video_path),
        ctx=cpu(0),
    )

    num_frames = len(vr)

    if num_frames == 0:
        raise ValueError(
            f"Video contains no frames: {video_path}"
        )

    fps = float(vr.get_avg_fps())

    if fps <= 0:
        raise ValueError(
            f"Invalid FPS for video: {video_path}"
        )

    # Timestamp of the final visible frame.
    duration_seconds = (
        (num_frames - 1) / fps
        if num_frames > 1
        else 0.0
    )

    return {
        "num_frames": num_frames,
        "fps": fps,
        "duration_seconds": duration_seconds,
    }


def make_refinement_clip(
    video_path,
    output_path,
    center_frame,
    radius_frames,
):
    """
    Extract a window around center_frame.

    The output video is written at 1 FPS. Therefore each second of the
    refinement video corresponds exactly to one original video frame.
    """

    vr = VideoReader(
        str(video_path),
        ctx=cpu(0),
    )

    total_frames = len(vr)

    start_frame = max(
        0,
        center_frame - radius_frames,
    )

    end_frame = min(
        total_frames - 1,
        center_frame + radius_frames,
    )

    indices = list(
        range(
            start_frame,
            end_frame + 1,
        )
    )

    frames = vr.get_batch(
        indices
    ).asnumpy()

    imageio.mimwrite(
        str(output_path),
        frames,
        fps=1,
    )

    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "num_frames": len(indices),
    }


def clamp(value, low, high):
    return max(
        low,
        min(high, value),
    )


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_timestamp(seconds):
    """Format seconds as MM:SS.s."""

    if seconds is None:
        return ""

    seconds = max(
        0.0,
        float(seconds),
    )

    minutes = int(
        seconds // 60
    )

    remaining = (
        seconds
        - 60 * minutes
    )

    return (
        f"{minutes:02d}:"
        f"{remaining:04.1f}"
    )


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class VLMBackend(ABC):
    """
    Interface for video-capable VLM backends.

    To add another backend, implement generate_video() and register
    the class in BACKENDS below.
    """

    def __init__(
        self,
        model,
        max_retries=4,
    ):
        self.model = model
        self.max_retries = max_retries

    @abstractmethod
    def generate_video(
        self,
        video_path,
        prompt,
    ):
        """
        Return:

            {
                "text": "...",
                "usage": {
                    "input_tokens": int,
                    "output_tokens": int
                }
            }
        """
        raise NotImplementedError


class GeminiBackend(VLMBackend):
    """Gemini native-video backend."""

    def __init__(
        self,
        model="gemini-3.1-pro-preview",
        max_retries=4,
    ):
        super().__init__(
            model=model,
            max_retries=max_retries,
        )

        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "Gemini backend requires google-genai. "
                "Install with: pip install google-genai"
            ) from e

        self.genai = genai
        self.client = genai.Client()

    def _wait_for_file(
        self,
        uploaded,
    ):
        """Wait until Gemini has processed the uploaded video."""

        while True:
            state = getattr(
                uploaded,
                "state",
                None,
            )

            state_name = getattr(
                state,
                "name",
                str(state) if state else "",
            )

            if state_name == "ACTIVE":
                return uploaded

            if state_name == "FAILED":
                raise RuntimeError(
                    "Gemini video processing failed"
                )

            time.sleep(2)

            uploaded = self.client.files.get(
                name=uploaded.name
            )

    @staticmethod
    def _extract_usage(response):
        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        if usage is None:
            return {
                "input_tokens": 0,
                "output_tokens": 0,
            }

        input_tokens = getattr(
            usage,
            "prompt_token_count",
            0,
        ) or 0

        output_tokens = getattr(
            usage,
            "candidates_token_count",
            0,
        ) or 0

        return {
            "input_tokens": int(
                input_tokens
            ),
            "output_tokens": int(
                output_tokens
            ),
        }

    def _generate_once(
        self,
        video_path,
        prompt,
    ):
        uploaded = None

        try:
            uploaded = self.client.files.upload(
                file=str(video_path)
            )

            uploaded = self._wait_for_file(
                uploaded
            )

            # Video is intentionally placed before the text prompt.
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    uploaded,
                    prompt,
                ],
            )

            text = getattr(
                response,
                "text",
                None,
            )

            if not text:
                raise RuntimeError(
                    "Gemini returned no text"
                )

            return {
                "text": text.strip(),
                "usage": self._extract_usage(
                    response
                ),
            }

        finally:
            if uploaded is not None:
                try:
                    self.client.files.delete(
                        name=uploaded.name
                    )
                except Exception:
                    # Uploaded files are temporary anyway. Failure to
                    # delete should not invalidate an episode label.
                    pass

    def generate_video(
        self,
        video_path,
        prompt,
    ):
        last_exc = None

        for attempt in range(
            self.max_retries
        ):
            try:
                return self._generate_once(
                    video_path=video_path,
                    prompt=prompt,
                )

            except Exception as e:
                last_exc = e

                if (
                    attempt
                    == self.max_retries - 1
                ):
                    break

                wait_seconds = min(
                    60,
                    2 ** attempt * 2,
                )

                time.sleep(
                    wait_seconds
                )

        raise last_exc


BACKENDS = {
    "gemini": GeminiBackend,
}


def create_backend(
    backend_name,
    model,
):
    if backend_name not in BACKENDS:
        available = ", ".join(
            sorted(BACKENDS)
        )

        raise ValueError(
            f"Unsupported VLM backend: "
            f"{backend_name}. "
            f"Available: {available}"
        )

    return BACKENDS[
        backend_name
    ](
        model=model
    )


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def coarse_label(
    backend,
    row,
    video_path,
    metadata,
):
    """Analyze the complete rollout."""

    prompt = COARSE_PROMPT_TEMPLATE.format(
        duration_seconds=metadata[
            "duration_seconds"
        ],
        task_desc=row.get(
            "task_desc",
            "",
        ),
        failing_predicate=row.get(
            "failing_predicate",
            "",
        ),
        detail=row.get(
            "detail",
            "",
        ),
        taxonomy=TAXONOMY,
    )

    result = backend.generate_video(
        video_path=video_path,
        prompt=prompt,
    )

    parsed = parse_json_response(
        result["text"]
    )

    return {
        "parsed": parsed,
        "text": result["text"],
        "usage": result["usage"],
    }


def refine_failure_onset(
    backend,
    row,
    video_path,
    metadata,
    coarse,
    refine_window_seconds,
):
    """
    Convert the VLM's coarse timestamp into a frame-level annotation.
    """

    parsed = coarse["parsed"]

    onset_type = str(
        parsed.get(
            "onset_type",
            "",
        )
    ).strip()

    duration = metadata[
        "duration_seconds"
    ]

    fps = metadata["fps"]

    num_frames = metadata[
        "num_frames"
    ]

    coarse_seconds = safe_float(
        parsed.get(
            "failure_onset_seconds"
        ),
        duration,
    )

    coarse_seconds = clamp(
        coarse_seconds,
        0.0,
        duration,
    )

    # Timeout means there is intentionally no earlier event.
    if onset_type == "timeout":
        final_frame = (
            num_frames - 1
        )

        return {
            "frame": final_frame,
            "seconds": duration,
            "refined": False,
            "refinement_text": "",
            "refinement_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "refinement_confidence":
                parsed.get(
                    "confidence",
                    "low",
                ),
            "refinement_justification":
                "No earlier failure event identified; "
                "using the final frame.",
        }

    coarse_frame = int(
        round(
            coarse_seconds * fps
        )
    )

    coarse_frame = int(
        clamp(
            coarse_frame,
            0,
            num_frames - 1,
        )
    )

    radius_frames = max(
        1,
        int(
            round(
                refine_window_seconds
                * fps
            )
        ),
    )

    with tempfile.TemporaryDirectory(
        prefix="liberox_failure_refine_"
    ) as temp_dir:

        refinement_path = (
            pathlib.Path(temp_dir)
            / "refinement.mp4"
        )

        clip = make_refinement_clip(
            video_path=video_path,
            output_path=refinement_path,
            center_frame=coarse_frame,
            radius_frames=radius_frames,
        )

        prompt = REFINE_PROMPT_TEMPLATE.format(
            failure_mode=parsed.get(
                "failure_mode",
                "unknown",
            ),
            onset_type=onset_type,
            failure_reason=parsed.get(
                "failure_reason",
                "",
            ),
            coarse_seconds=coarse_seconds,
            task_desc=row.get(
                "task_desc",
                "",
            ),
            failing_predicate=row.get(
                "failing_predicate",
                "",
            ),
            start_frame=clip[
                "start_frame"
            ],
            start_frame_plus_one=(
                clip["start_frame"] + 1
            ),
            start_frame_plus_two=(
                clip["start_frame"] + 2
            ),
            num_frames=clip[
                "num_frames"
            ],
            max_second=(
                clip["num_frames"] - 1
            ),
        )

        refinement = backend.generate_video(
            video_path=refinement_path,
            prompt=prompt,
        )

        refinement_parsed = (
            parse_json_response(
                refinement["text"]
            )
        )

    refined_second = (
        refinement_parsed.get(
            "refined_second"
        )
    )

    if refined_second is None:
        # Refinement could not confidently identify the event.
        # Preserve the coarse estimate.
        final_frame = coarse_frame

        return {
            "frame": final_frame,
            "seconds": (
                final_frame / fps
            ),
            "refined": False,
            "refinement_text":
                refinement["text"],
            "refinement_usage":
                refinement["usage"],
            "refinement_confidence":
                refinement_parsed.get(
                    "confidence",
                    "low",
                ),
            "refinement_justification":
                refinement_parsed.get(
                    "justification",
                    "",
                ),
        }

    try:
        refined_second = int(
            round(
                float(
                    refined_second
                )
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        refined_second = None

    if refined_second is None:
        final_frame = coarse_frame

    else:
        refined_second = int(
            clamp(
                refined_second,
                0,
                clip["num_frames"] - 1,
            )
        )

        final_frame = (
            clip["start_frame"]
            + refined_second
        )

    final_frame = int(
        clamp(
            final_frame,
            0,
            num_frames - 1,
        )
    )

    final_seconds = (
        final_frame / fps
    )

    return {
        "frame": final_frame,
        "seconds": final_seconds,
        "refined": True,
        "refinement_text":
            refinement["text"],
        "refinement_usage":
            refinement["usage"],
        "refinement_confidence":
            refinement_parsed.get(
                "confidence",
                "low",
            ),
        "refinement_justification":
            refinement_parsed.get(
                "justification",
                "",
            ),
    }


def label_one(
    backend,
    row,
    refine=True,
    refine_window_seconds=3.0,
):
    """Label one failed rollout."""

    video_path = pathlib.Path(
        row["video_path"]
    )

    metadata = get_video_metadata(
        video_path
    )

    coarse = coarse_label(
        backend=backend,
        row=row,
        video_path=video_path,
        metadata=metadata,
    )

    parsed = coarse["parsed"]

    if refine:
        temporal = refine_failure_onset(
            backend=backend,
            row=row,
            video_path=video_path,
            metadata=metadata,
            coarse=coarse,
            refine_window_seconds=(
                refine_window_seconds
            ),
        )

    else:
        onset_type = parsed.get(
            "onset_type",
            "",
        )

        coarse_seconds = safe_float(
            parsed.get(
                "failure_onset_seconds"
            ),
            metadata[
                "duration_seconds"
            ],
        )

        coarse_seconds = clamp(
            coarse_seconds,
            0.0,
            metadata[
                "duration_seconds"
            ],
        )

        if onset_type == "timeout":
            frame = (
                metadata[
                    "num_frames"
                ]
                - 1
            )

        else:
            frame = int(
                round(
                    coarse_seconds
                    * metadata["fps"]
                )
            )

            frame = int(
                clamp(
                    frame,
                    0,
                    metadata[
                        "num_frames"
                    ]
                    - 1,
                )
            )

        temporal = {
            "frame": frame,
            "seconds": (
                frame
                / metadata["fps"]
            ),
            "refined": False,
            "refinement_text": "",
            "refinement_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "refinement_confidence": "",
            "refinement_justification": "",
        }

    coarse_usage = coarse[
        "usage"
    ]

    refine_usage = temporal[
        "refinement_usage"
    ]

    total_usage = {
        "input_tokens": (
            coarse_usage.get(
                "input_tokens",
                0,
            )
            + refine_usage.get(
                "input_tokens",
                0,
            )
        ),
        "output_tokens": (
            coarse_usage.get(
                "output_tokens",
                0,
            )
            + refine_usage.get(
                "output_tokens",
                0,
            )
        ),
    }

    coarse_onset_seconds = (
        safe_float(
            parsed.get(
                "failure_onset_seconds"
            )
        )
    )

    failure_window_start = (
        safe_float(
            parsed.get(
                "failure_window_start_seconds"
            )
        )
    )

    failure_window_end = (
        safe_float(
            parsed.get(
                "failure_window_end_seconds"
            )
        )
    )

    onset_frame = temporal[
        "frame"
    ]

    onset_seconds = temporal[
        "seconds"
    ]

    return {
        "vlm_failure_mode":
            parsed.get(
                "failure_mode"
            ),

        "vlm_failure_onset_type":
            parsed.get(
                "onset_type"
            ),

        "vlm_failure_onset_seconds":
            onset_seconds,

        "vlm_failure_onset_timestamp":
            format_timestamp(
                onset_seconds
            ),

        # 0-based video frame index.
        #
        # eval_subgoals.py records one rollout image for each
        # action-step observation, so this also provides the
        # temporal index for probe alignment.
        "vlm_failure_onset_frame":
            onset_frame,

        "vlm_failure_onset_step":
            onset_frame,

        "vlm_coarse_failure_onset_seconds":
            coarse_onset_seconds,

        "vlm_failure_window_start_seconds":
            failure_window_start,

        "vlm_failure_window_end_seconds":
            failure_window_end,

        "vlm_temporal_refined":
            temporal["refined"],

        "vlm_confidence":
            parsed.get(
                "confidence"
            ),

        "vlm_temporal_confidence":
            temporal[
                "refinement_confidence"
            ],

        "vlm_failure_reason":
            parsed.get(
                "failure_reason"
            ),

        "vlm_recovery_action":
            parsed.get(
                "recovery_action"
            ),

        "vlm_justification":
            parsed.get(
                "justification"
            ),

        "vlm_temporal_justification":
            temporal[
                "refinement_justification"
            ],

        "vlm_video_fps":
            metadata["fps"],

        "vlm_video_num_frames":
            metadata[
                "num_frames"
            ],

        "vlm_video_duration_seconds":
            metadata[
                "duration_seconds"
            ],

        "vlm_raw_response":
            coarse["text"],

        "vlm_refinement_raw_response":
            temporal[
                "refinement_text"
            ],

        "vlm_usage":
            total_usage,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser(
        description=(
            "Use a video-capable VLM to diagnose "
            "LIBERO-X failure rollouts, identify "
            "failure onset, and propose recovery."
        )
    )

    ap.add_argument(
        "--sample",
        required=True,
        help=(
            "Input JSONL manifest containing "
            "task_desc, failing_predicate, "
            "and video_path."
        ),
    )

    ap.add_argument(
        "--out",
        required=True,
        help=(
            "Output labeled JSONL manifest."
        ),
    )

    ap.add_argument(
        "--backend",
        choices=sorted(
            BACKENDS.keys()
        ),
        default="gemini",
        help=(
            "Video-capable VLM backend."
        ),
    )

    ap.add_argument(
        "--model",
        default=(
            "gemini-3.1-pro-preview"
        ),
        help=(
            "Model name for the selected backend."
        ),
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only process the first N episodes."
        ),
    )

    ap.add_argument(
        "--refine-window-seconds",
        type=float,
        default=3.0,
        help=(
            "Seconds before and after the coarse "
            "failure estimate to inspect during "
            "frame-level refinement."
        ),
    )

    ap.add_argument(
        "--no-refine",
        action="store_true",
        help=(
            "Disable the second frame-level "
            "temporal refinement pass."
        ),
    )

    args = ap.parse_args()

    backend = create_backend(
        backend_name=args.backend,
        model=args.model,
    )

    with open(
        args.sample,
        encoding="utf-8",
    ) as f:
        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if args.limit is not None:
        rows = rows[
            :args.limit
        ]

    out_path = pathlib.Path(
        args.out
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_ok = 0

    total_input_tokens = 0
    total_output_tokens = 0

    with out_path.open(
        "w",
        encoding="utf-8",
    ) as out_f:

        for i, row in enumerate(
            rows
        ):

            video_path = row.get(
                "video_path"
            )

            if (
                not video_path
                or not pathlib.Path(
                    video_path
                ).exists()
            ):

                row.update({
                    "vlm_backend":
                        args.backend,

                    "vlm_model":
                        args.model,

                    "vlm_failure_mode":
                        "video_missing",

                    "vlm_failure_onset_type":
                        "",

                    "vlm_failure_onset_seconds":
                        None,

                    "vlm_failure_onset_frame":
                        None,

                    "vlm_failure_onset_step":
                        None,

                    "vlm_confidence":
                        "low",

                    "vlm_failure_reason":
                        "",

                    "vlm_recovery_action":
                        "",

                    "vlm_justification":
                        "Rollout video is missing.",
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
                    backend=backend,
                    row=row,
                    refine=(
                        not args.no_refine
                    ),
                    refine_window_seconds=(
                        args.refine_window_seconds
                    ),
                )

            except Exception as e:

                print(
                    f"[{i + 1}/{len(rows)}] "
                    f"ERROR: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )

                row.update({
                    "vlm_backend":
                        args.backend,

                    "vlm_model":
                        args.model,

                    "vlm_failure_mode":
                        "api_error",

                    "vlm_failure_onset_type":
                        "",

                    "vlm_failure_onset_seconds":
                        None,

                    "vlm_failure_onset_frame":
                        None,

                    "vlm_failure_onset_step":
                        None,

                    "vlm_confidence":
                        "low",

                    "vlm_failure_reason":
                        "",

                    "vlm_recovery_action":
                        "",

                    "vlm_justification":
                        (
                            f"{type(e).__name__}: "
                            f"{e}"
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

            elapsed = (
                time.time() - t0
            )

            row.update(
                result
            )

            row["vlm_backend"] = (
                args.backend
            )

            row["vlm_model"] = (
                args.model
            )

            usage = result.get(
                "vlm_usage",
                {},
            )

            total_input_tokens += (
                usage.get(
                    "input_tokens",
                    0,
                )
            )

            total_output_tokens += (
                usage.get(
                    "output_tokens",
                    0,
                )
            )

            out_f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            out_f.flush()

            n_ok += 1

            failure_mode = row.get(
                "vlm_failure_mode",
                "unknown",
            )

            onset_type = row.get(
                "vlm_failure_onset_type",
                "unknown",
            )

            onset_seconds = row.get(
                "vlm_failure_onset_seconds"
            )

            if onset_seconds is None:
                onset_text = "?"
            else:
                onset_text = (
                    f"{onset_seconds:.2f}s"
                )

            refined = row.get(
                "vlm_temporal_refined",
                False,
            )

            print(
                f"[{i + 1}/{len(rows)}] "
                f"{elapsed:.1f}s  "
                f"{failure_mode}  "
                f"onset={onset_text}  "
                f"type={onset_type}  "
                f"refined={refined}",
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
        f"{total_input_tokens} in / "
        f"{total_output_tokens} out.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()