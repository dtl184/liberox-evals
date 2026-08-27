"""Build a browsable HTML gallery pairing failure videos with their labels.

Reusable: point --manifest at any JSONL with a video_path field plus label
fields (works with label_failures.py's failure_labels.jsonl, or
vlm_label_failures.py's vlm_labeled_*.jsonl -- it picks up whichever label
fields are present: failure_category/failing_predicate_type from the exact
pipeline, vlm_failure_mode/vlm_justification/vlm_reasoning from the VLM one).

Usage:
    python build_gallery.py --manifest vlm_labeled_sample_v2.jsonl --out gallery.html --root /home/train/libero_x_eval
"""
import argparse
import html
import json
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--root", required=True, help="server root the gallery will be served from (video src paths are made relative to this)")
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
root = pathlib.Path(args.root).resolve()

def esc(s):
    return html.escape(str(s)) if s is not None else ""

cards = []
for r in rows:
    vp = r.get("video_path")
    if not vp:
        continue
    vp_abs = pathlib.Path(vp).resolve()
    try:
        rel = vp_abs.relative_to(root)
    except ValueError:
        rel = vp_abs  # fall back to absolute if outside root
    label_bits = []
    if r.get("failure_category"):
        label_bits.append(f"<span class='tag cat'>{esc(r['failure_category'])}</span>")
    if r.get("failing_predicate_type"):
        label_bits.append(f"<span class='tag pred'>{esc(r['failing_predicate_type'])}</span>")
    if r.get("vlm_failure_mode"):
        label_bits.append(f"<span class='tag vlm'>qwen7b: {esc(r['vlm_failure_mode'])}</span>")
    if r.get("claude_failure_mode"):
        label_bits.append(f"<span class='tag claude'>claude: {esc(r['claude_failure_mode'])} ({esc(r.get('claude_confidence',''))})</span>")
    justification = r.get("claude_justification") or r.get("vlm_justification") or r.get("detail") or ""
    reasoning = r.get("claude_reasoning") or r.get("vlm_reasoning") or ""
    other_justification = r.get("vlm_justification") if r.get("claude_failure_mode") and r.get("vlm_justification") else ""
    cards.append(f"""
    <div class="card">
      <video controls preload="none" muted>
        <source src="{esc(rel)}" type="video/mp4">
      </video>
      <div class="meta">
        <div class="task">{esc(r.get('task_desc',''))}</div>
        <div class="tags">{''.join(label_bits)}</div>
        <div class="pred-str"><code>{esc(r.get('failing_predicate',''))}</code></div>
        {f'<div class="justification">{esc(justification)}</div>' if justification else ''}
        {f'<div class="other-justification">qwen7b said: {esc(other_justification)}</div>' if other_justification else ''}
        {f'<details><summary>Reasoning</summary>{esc(reasoning)}</details>' if reasoning else ''}
        <div class="level">{esc(r.get('level',''))} / {esc(r.get('scene_name',''))}</div>
      </div>
    </div>""")

html_out = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Failure video gallery</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background:#111; color:#eee; margin:0; padding:20px; }}
  h1 {{ font-size:18px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(300px,1fr)); gap:16px; }}
  .card {{ background:#1b1b1b; border:1px solid #333; border-radius:8px; overflow:hidden; }}
  video {{ width:100%; display:block; background:#000; }}
  .meta {{ padding:10px 12px; font-size:13px; }}
  .task {{ font-weight:600; margin-bottom:6px; }}
  .tags {{ margin-bottom:6px; }}
  .tag {{ display:inline-block; font-size:11px; padding:2px 7px; border-radius:10px; margin-right:5px; margin-bottom:4px; }}
  .tag.cat {{ background:#2a4d69; }}
  .tag.pred {{ background:#4d692a; }}
  .tag.vlm {{ background:#69402a; }}
  .tag.claude {{ background:#3d2a69; }}
  .other-justification {{ margin-top:4px; color:#888; font-size:12px; }}
  .pred-str code {{ font-size:11px; color:#aaa; }}
  .justification {{ margin-top:6px; color:#ccc; font-style:italic; }}
  details {{ margin-top:6px; color:#999; font-size:12px; }}
  .level {{ margin-top:6px; color:#777; font-size:11px; }}
  #filter {{ margin-bottom:16px; }}
  #filter input {{ padding:6px; width:300px; background:#222; border:1px solid #444; color:#eee; border-radius:4px; }}
</style></head>
<body>
<h1>{len(cards)} labeled failure videos</h1>
<div id="filter"><input id="q" placeholder="filter by task text or label..." oninput="filterCards()"></div>
<div class="grid" id="grid">
{''.join(cards)}
</div>
<script>
function filterCards() {{
  const q = document.getElementById('q').value.toLowerCase();
  document.querySelectorAll('.card').forEach(c => {{
    c.style.display = c.innerText.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
</script>
</body></html>"""

pathlib.Path(args.out).write_text(html_out)
print(f"Wrote gallery with {len(cards)} videos to {args.out}")
