"""Build an HTML gallery for inspecting LIBERO-X VLM failure labels.

Shows:
- full rollout video
- predicted failure onset
- visual failure marker on a timeline
- current playback position
- button to play from 3 seconds before failure
- optional automatic pause at the predicted failure
- failure mode, reason, and recovery action

Usage:
    python analysis/build_gallery.py \
        --manifest vlm_labeled.jsonl \
        --out /home/train/libero_x_eval/gallery.html \
        --root /home/train/libero_x_eval
"""

import argparse
import html
import json
import pathlib
from string import Template


def esc(value):
    if value is None:
        return ""
    return html.escape(str(value))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


ap = argparse.ArgumentParser()

ap.add_argument(
    "--manifest",
    required=True,
)

ap.add_argument(
    "--out",
    required=True,
)

ap.add_argument(
    "--root",
    required=True,
    help="Root directory from which videos will be served.",
)

args = ap.parse_args()


with open(args.manifest, encoding="utf-8") as f:
    rows = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


root = pathlib.Path(args.root).resolve()

cards = []


for r in rows:

    video_path = r.get("video_path")

    if not video_path:
        continue

    video_abs = pathlib.Path(video_path).resolve()

    try:
        video_src = video_abs.relative_to(root)
    except ValueError:
        video_src = video_abs

    onset = as_float(
        r.get("vlm_failure_onset_seconds")
    )

    window_start = as_float(
        r.get("vlm_failure_window_start_seconds")
    )

    window_end = as_float(
        r.get("vlm_failure_window_end_seconds")
    )

    onset_attr = (
        str(onset)
        if onset is not None
        else ""
    )

    window_start_attr = (
        str(window_start)
        if window_start is not None
        else ""
    )

    window_end_attr = (
        str(window_end)
        if window_end is not None
        else ""
    )

    onset_text = (
        f"{onset:.2f}s"
        if onset is not None
        else "not available"
    )

    failure_mode = r.get(
        "vlm_failure_mode",
        ""
    )

    onset_type = r.get(
        "vlm_failure_onset_type",
        ""
    )

    failure_reason = r.get(
        "vlm_failure_reason",
        ""
    )

    recovery_action = r.get(
        "vlm_recovery_action",
        ""
    )

    justification = r.get(
        "vlm_justification",
        ""
    )

    temporal_justification = r.get(
        "vlm_temporal_justification",
        ""
    )

    failing_predicate = r.get(
        "failing_predicate",
        ""
    )

    tags = []

    if r.get("failure_category"):
        tags.append(
            "<span class='tag simulator'>"
            + esc(r["failure_category"])
            + "</span>"
        )

    if r.get("failing_predicate_type"):
        tags.append(
            "<span class='tag predicate'>"
            + esc(r["failing_predicate_type"])
            + "</span>"
        )

    if failure_mode:
        tags.append(
            "<span class='tag vlm'>"
            + esc(failure_mode)
            + "</span>"
        )

    if onset_type:
        tags.append(
            "<span class='tag onset'>"
            + esc(onset_type)
            + "</span>"
        )

    timeline = ""

    if onset is not None:
        timeline = """
        <div class="controls">

            <button
                type="button"
                class="before-button"
            >
                ▶ Play 3s before failure
            </button>

            <button
                type="button"
                class="jump-button"
            >
                Jump to failure
            </button>

            <label class="pause-control">
                <input
                    type="checkbox"
                    class="pause-at-failure"
                    checked
                >
                pause at failure
            </label>

        </div>

        <div class="timeline-wrap">

            <div class="timeline">

                <div class="failure-window"></div>

                <div class="failure-marker">
                    <span>FAILURE</span>
                </div>

                <div class="playback-marker"></div>

            </div>

            <div class="timeline-labels">
                <span>0:00</span>
                <span class="current-time">0:00</span>
                <span class="duration">--:--</span>
            </div>

        </div>
        """

    failure_reason_html = ""

    if failure_reason:
        failure_reason_html = f"""
        <div class="section">
            <div class="section-title">
                Failure reason
            </div>
            <div>
                {esc(failure_reason)}
            </div>
        </div>
        """

    recovery_html = ""

    if recovery_action:
        recovery_html = f"""
        <div class="section recovery">
            <div class="section-title">
                Recovery action
            </div>
            <div>
                {esc(recovery_action)}
            </div>
        </div>
        """

    justification_html = ""

    if justification:
        justification_html = f"""
        <div class="section">
            <div class="section-title">
                VLM justification
            </div>
            <div>
                {esc(justification)}
            </div>
        </div>
        """

    temporal_html = ""

    if temporal_justification:
        temporal_html = f"""
        <div class="section temporal">
            <div class="section-title">
                Temporal justification
            </div>
            <div>
                {esc(temporal_justification)}
            </div>
        </div>
        """

    onset_frame = r.get(
        "vlm_failure_onset_frame"
    )

    frame_html = ""

    if onset_frame is not None:
        frame_html = (
            "<div><strong>Video frame:</strong> "
            + esc(onset_frame)
            + "</div>"
        )

    cards.append(
        f"""
        <div
            class="card"
            data-onset="{esc(onset_attr)}"
            data-window-start="{esc(window_start_attr)}"
            data-window-end="{esc(window_end_attr)}"
        >

            <div class="video-wrap">

                <video
                    controls
                    preload="metadata"
                    muted
                >
                    <source
                        src="{esc(video_src)}"
                        type="video/mp4"
                    >
                </video>

                <div class="failure-overlay">
                    VLM FAILURE ONSET
                </div>

            </div>

            {timeline}

            <div class="meta">

                <div class="task">
                    {esc(r.get("task_desc", ""))}
                </div>

                <div class="tags">
                    {''.join(tags)}
                </div>

                <div class="failure-info">

                    <div>
                        <strong>
                            Predicted failure onset:
                        </strong>
                        {esc(onset_text)}
                    </div>

                    {frame_html}

                    <div>
                        <strong>
                            Onset criterion:
                        </strong>
                        {esc(onset_type)}
                    </div>

                    <div>
                        <strong>
                            Confidence:
                        </strong>
                        {esc(r.get("vlm_confidence", ""))}
                    </div>

                </div>

                <div class="section">

                    <div class="section-title">
                        Failed predicate
                    </div>

                    <code>
                        {esc(failing_predicate)}
                    </code>

                </div>

                {failure_reason_html}

                {recovery_html}

                {justification_html}

                {temporal_html}

                <div class="footer">
                    {esc(r.get("scene_group", ""))}
                    /
                    {esc(r.get("scene_name", ""))}
                    ·
                    {esc(r.get("vlm_model", ""))}
                </div>

            </div>

        </div>
        """
    )


PAGE = Template(
r"""<!doctype html>

<html>

<head>

<meta charset="utf-8">

<title>LIBERO-X Failure Gallery</title>

<style>

body {
    font-family: -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;

    background: #111;
    color: #eee;

    margin: 0;
    padding: 20px;
}

h1 {
    font-size: 20px;
}

#filter {
    margin-bottom: 20px;
}

#filter input {
    width: min(500px, 100%);
    box-sizing: border-box;

    padding: 9px;

    background: #222;
    color: #eee;

    border: 1px solid #444;
    border-radius: 5px;
}

.grid {
    display: grid;

    grid-template-columns:
        repeat(auto-fill, minmax(400px, 1fr));

    gap: 18px;
}

.card {
    background: #1b1b1b;

    border: 1px solid #333;
    border-radius: 8px;

    overflow: hidden;
}

.video-wrap {
    position: relative;
    background: black;
}

video {
    width: 100%;
    display: block;
}

.failure-overlay {
    display: none;

    position: absolute;

    top: 12px;
    left: 50%;

    transform: translateX(-50%);

    background: rgba(190, 20, 20, 0.95);

    color: white;

    padding: 7px 12px;

    border-radius: 4px;

    font-weight: bold;
    font-size: 12px;

    pointer-events: none;
}

.failure-overlay.active {
    display: block;
}

.controls {
    display: flex;

    align-items: center;
    flex-wrap: wrap;

    gap: 8px;

    padding: 10px 12px 4px;
}

.controls button {
    padding: 6px 9px;

    background: #333;
    color: #eee;

    border: 1px solid #555;
    border-radius: 4px;

    cursor: pointer;
}

.controls button:hover {
    background: #444;
}

.pause-control {
    margin-left: auto;

    color: #aaa;

    font-size: 12px;
}

.timeline-wrap {
    padding: 16px 14px 10px;
}

.timeline {
    position: relative;

    height: 12px;

    border-radius: 6px;

    background: #444;
}

.failure-window {
    display: none;

    position: absolute;

    top: 0;
    bottom: 0;

    background: rgba(255, 160, 0, 0.35);
}

.failure-marker {
    position: absolute;

    top: -6px;
    bottom: -6px;

    width: 3px;

    background: #ff3b30;

    z-index: 3;
}

.failure-marker span {
    position: absolute;

    bottom: 22px;
    left: 50%;

    transform: translateX(-50%);

    background: #b91919;
    color: white;

    padding: 2px 4px;

    border-radius: 3px;

    font-size: 9px;
    font-weight: bold;

    white-space: nowrap;
}

.playback-marker {
    position: absolute;

    top: -3px;

    width: 3px;
    height: 18px;

    background: #5badff;

    z-index: 2;
}

.timeline-labels {
    display: flex;

    justify-content: space-between;

    margin-top: 5px;

    color: #888;

    font-size: 10px;
}

.current-time {
    color: #5badff;
}

.meta {
    padding: 12px 14px;

    font-size: 13px;
    line-height: 1.4;
}

.task {
    font-size: 14px;
    font-weight: 600;

    margin-bottom: 8px;
}

.tags {
    margin-bottom: 8px;
}

.tag {
    display: inline-block;

    margin-right: 5px;
    margin-bottom: 4px;

    padding: 2px 7px;

    border-radius: 10px;

    font-size: 11px;
}

.tag.simulator {
    background: #2a4d69;
}

.tag.predicate {
    background: #4d692a;
}

.tag.vlm {
    background: #69402a;
}

.tag.onset {
    background: #6a3030;
}

.failure-info {
    margin: 8px 0 12px;

    padding: 7px 9px;

    background: #251919;

    border-left: 3px solid #ff3b30;
}

.section {
    margin-top: 10px;
}

.section-title {
    margin-bottom: 3px;

    color: #999;

    font-size: 11px;
    font-weight: bold;

    text-transform: uppercase;
}

.section code {
    color: #bbb;

    font-size: 11px;
}

.recovery {
    padding-left: 8px;

    border-left: 3px solid #428a56;
}

.temporal {
    padding-left: 8px;

    border-left: 3px solid #a36c24;
}

.footer {
    margin-top: 12px;

    color: #777;

    font-size: 11px;
}

</style>

</head>

<body>

<h1>$count labeled failure videos</h1>

<div id="filter">

    <input
        id="q"
        placeholder="filter by task, predicate, failure type..."
    >

</div>

<div class="grid">

$cards

</div>


<script>

function formatTime(seconds) {

    if (!Number.isFinite(seconds)) {
        return "--:--";
    }

    const minutes = Math.floor(
        seconds / 60
    );

    const remaining =
        seconds - minutes * 60;

    return (
        String(minutes)
        + ":"
        + remaining
            .toFixed(1)
            .padStart(4, "0")
    );
}


function setupCard(card) {

    const video =
        card.querySelector("video");

    if (!video) {
        return;
    }

    const onset =
        parseFloat(card.dataset.onset);

    const windowStart =
        parseFloat(
            card.dataset.windowStart
        );

    const windowEnd =
        parseFloat(
            card.dataset.windowEnd
        );

    const failureMarker =
        card.querySelector(
            ".failure-marker"
        );

    const playbackMarker =
        card.querySelector(
            ".playback-marker"
        );

    const failureWindow =
        card.querySelector(
            ".failure-window"
        );

    const currentTime =
        card.querySelector(
            ".current-time"
        );

    const durationText =
        card.querySelector(
            ".duration"
        );

    const overlay =
        card.querySelector(
            ".failure-overlay"
        );

    const jumpButton =
        card.querySelector(
            ".jump-button"
        );

    const beforeButton =
        card.querySelector(
            ".before-button"
        );

    const pauseCheckbox =
        card.querySelector(
            ".pause-at-failure"
        );

    let alreadyPaused = false;


    function updateStaticTimeline() {

        const duration =
            video.duration;

        if (
            !Number.isFinite(duration)
            || duration <= 0
        ) {
            return;
        }

        if (durationText) {
            durationText.textContent =
                formatTime(duration);
        }

        if (
            Number.isFinite(onset)
            && failureMarker
        ) {

            let percent =
                onset / duration * 100;

            percent = Math.max(
                0,
                Math.min(100, percent)
            );

            failureMarker.style.left =
                percent + "%";
        }


        if (
            Number.isFinite(windowStart)
            && Number.isFinite(windowEnd)
            && failureWindow
        ) {

            let start =
                windowStart
                / duration
                * 100;

            let end =
                windowEnd
                / duration
                * 100;

            start = Math.max(
                0,
                Math.min(100, start)
            );

            end = Math.max(
                start,
                Math.min(100, end)
            );

            failureWindow.style.left =
                start + "%";

            failureWindow.style.width =
                (end - start) + "%";

            failureWindow.style.display =
                "block";
        }
    }


    video.addEventListener(
        "loadedmetadata",
        updateStaticTimeline
    );


    video.addEventListener(
        "timeupdate",
        () => {

            const duration =
                video.duration;

            const current =
                video.currentTime;


            if (
                Number.isFinite(duration)
                && duration > 0
                && playbackMarker
            ) {

                let percent =
                    current
                    / duration
                    * 100;

                percent = Math.max(
                    0,
                    Math.min(
                        100,
                        percent
                    )
                );

                playbackMarker.style.left =
                    percent + "%";
            }


            if (currentTime) {
                currentTime.textContent =
                    formatTime(current);
            }


            if (
                Number.isFinite(onset)
            ) {

                const nearFailure =
                    Math.abs(
                        current - onset
                    ) < 0.4;

                if (overlay) {

                    overlay.classList.toggle(
                        "active",
                        nearFailure
                    );
                }


                if (
                    current < onset - 0.5
                ) {
                    alreadyPaused = false;
                }


                if (
                    pauseCheckbox
                    && pauseCheckbox.checked
                    && !alreadyPaused
                    && current >= onset
                ) {

                    alreadyPaused = true;

                    video.pause();

                    video.currentTime =
                        onset;

                    if (overlay) {
                        overlay.classList.add(
                            "active"
                        );
                    }
                }
            }
        }
    );


    if (jumpButton) {

        jumpButton.addEventListener(
            "click",
            () => {

                alreadyPaused = true;

                video.pause();

                video.currentTime =
                    onset;

                if (overlay) {
                    overlay.classList.add(
                        "active"
                    );
                }
            }
        );
    }


    if (beforeButton) {

        beforeButton.addEventListener(
            "click",
            () => {

                alreadyPaused = false;

                video.currentTime =
                    Math.max(
                        0,
                        onset - 3
                    );

                video.play();
            }
        );
    }
}


document
    .querySelectorAll(".card")
    .forEach(setupCard);


const search =
    document.getElementById("q");


search.addEventListener(
    "input",
    () => {

        const query =
            search.value.toLowerCase();

        document
            .querySelectorAll(".card")
            .forEach(
                card => {

                    const match =
                        card.innerText
                            .toLowerCase()
                            .includes(query);

                    card.style.display =
                        match
                        ? ""
                        : "none";
                }
            );
    }
);

</script>

</body>

</html>hrilabrules

"""
)


html_out = PAGE.substitute(
    count=len(cards),
    cards="\n".join(cards),
)


pathlib.Path(
    args.out
).write_text(
    html_out,
    encoding="utf-8",
)


print(
    f"Wrote gallery with "
    f"{len(cards)} videos to "
    f"{args.out}"
)