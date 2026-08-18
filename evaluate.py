"""
Automated quality metrics for a trained Setter checkpoint / generation pipeline:
generates routes across a grid of grades/angles, reports validity rate and basic
composition stats, saves a visual HTML report of the actual climbs, and appends a
compact record to eval_history.jsonl so quality can be tracked across changes over
time (not just eyeballed once and forgotten).

Run standalone after any change to the model or generation code:
    python evaluate.py [checkpoint] --label "description of what changed"

Or import run_evaluation() / evaluate_checkpoint() and call from other code (used by
training.py at the end of every training run).
"""
import argparse
import base64
import collections
import datetime
import io
import json
import os
import statistics
import sys

import torch

from setter import (
    Setter,
    VOCAB_SIZE,
    load_checkpoint_state_dict,
    hold_id_to_xy,
    v_grade_to_normalized_difficulty,
    normalized_difficulty_to_v_grade,
)
from generate import generate_route, decode_holds, drawClimb, _route_is_valid
from critic import load_critic

DEFAULT_GRADES = [1, 4, 7, 10, 13, 16]
DEFAULT_ANGLES = [0, 10, 20, 30, 40, 50, 60, 70]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(SCRIPT_DIR, "eval_history.jsonl")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "eval_reports")


def evaluate_checkpoint(model, device="cpu", grades=None, angles=None, seed=123, with_images=False, critic=None):
    """
    Returns a dict summary: overall + per-grade validity rate, avg holds, avg y-span.
    If with_images, each row also carries a PIL image of the rendered climb.
    """
    grades = grades or DEFAULT_GRADES
    angles = angles or DEFAULT_ANGLES
    model.eval()
    torch.manual_seed(seed)

    rows = []
    for grade in grades:
        for angle in angles:
            tokens = generate_route(model, grade=grade, angle=angle, device=device, critic=critic)
            climb = decode_holds(tokens)
            holds = [h for h in climb if h not in ("[START]", "[END]")]
            coords = [hold_id_to_xy(int(h[1:h.index("r")])) for h in holds]
            coords = [c for c in coords if c is not None]
            y_span = (max(c[1] for c in coords) - min(c[1] for c in coords)) if len(coords) > 1 else 0
            row = {
                "grade": grade, "angle": angle,
                "valid": _route_is_valid(climb),
                "n_holds": len(holds), "y_span": y_span,
            }
            if critic is not None:
                route_ids = torch.tensor([tokens], device=device)
                angle_tensor = torch.tensor([angle / 70.0], device=device)
                with torch.no_grad():
                    predicted_norm = critic(route_ids, angle_tensor).item()
                row["critic_v_grade"] = normalized_difficulty_to_v_grade(predicted_norm)
                row["critic_diff_units"] = abs(
                    predicted_norm - v_grade_to_normalized_difficulty(grade)
                ) * 21.0  # back to raw difficulty units, for a finer-grained average than V-grade rounding
            if with_images:
                row["image"] = drawClimb(climb, save=False)
            rows.append(row)

    total = len(rows)
    valid = sum(1 for r in rows if r["valid"])
    by_grade = collections.defaultdict(list)
    for r in rows:
        by_grade[r["grade"]].append(r)

    per_grade = {}
    for grade, grade_rows in by_grade.items():
        n = len(grade_rows)
        v = sum(1 for r in grade_rows if r["valid"])
        stats = {
            "valid_rate": v / n,
            "avg_holds": statistics.mean(r["n_holds"] for r in grade_rows),
            "avg_y_span": statistics.mean(r["y_span"] for r in grade_rows),
        }
        if critic is not None:
            stats["avg_critic_diff"] = statistics.mean(r["critic_diff_units"] for r in grade_rows)
        per_grade[grade] = stats

    summary = {
        "total": total,
        "valid_rate": valid / total,
        "avg_holds": statistics.mean(r["n_holds"] for r in rows),
        "avg_y_span": statistics.mean(r["y_span"] for r in rows),
        "per_grade": per_grade,
        "rows": rows,
    }
    if critic is not None:
        summary["avg_critic_diff"] = statistics.mean(r["critic_diff_units"] for r in rows)
    return summary


def print_summary(summary):
    has_critic = "avg_critic_diff" in summary
    print(f"Evaluated {summary['total']} routes "
          f"({len(summary['per_grade'])} grades x {summary['total'] // len(summary['per_grade'])} angles)")
    critic_bit = f"  critic grade-match error: {summary['avg_critic_diff']:.2f} difficulty units" if has_critic else ""
    print(f"  Overall valid rate: {summary['valid_rate']*100:.0f}%  "
          f"avg holds: {summary['avg_holds']:.1f}  avg y-span: {summary['avg_y_span']:.0f}px{critic_bit}")
    for grade in sorted(summary["per_grade"]):
        s = summary["per_grade"][grade]
        critic_bit = f"  critic_err={s['avg_critic_diff']:.2f}" if has_critic else ""
        print(f"    V{grade}: valid={s['valid_rate']*100:.0f}%  "
              f"avg_holds={s['avg_holds']:.1f}  avg_y_span={s['avg_y_span']:.0f}px{critic_bit}")


def append_history(summary, label, checkpoint_path=None, val_loss=None):
    """Appends a compact (no images) record to eval_history.jsonl for trend tracking."""
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "checkpoint": checkpoint_path,
        "val_loss": val_loss,
        "total": summary["total"],
        "valid_rate": round(summary["valid_rate"], 4),
        "avg_holds": round(summary["avg_holds"], 2),
        "avg_y_span": round(summary["avg_y_span"], 1),
        "per_grade": {
            str(g): {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}
            for g, s in summary["per_grade"].items()
        },
    }
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def print_history(last_n=15):
    if not os.path.exists(HISTORY_PATH):
        print("No evaluation history yet - run an evaluation first.")
        return
    with open(HISTORY_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"{'timestamp':<20} {'label':<32} {'valid%':>7} {'avg_holds':>10} {'avg_y_span':>11}")
    for r in records[-last_n:]:
        print(f"{r['timestamp']:<20} {r['label'][:32]:<32} "
              f"{r['valid_rate']*100:>6.0f}% {r['avg_holds']:>10.1f} {r['avg_y_span']:>11.0f}")


def _image_to_b64_jpeg(img, scale=0.4, quality=80):
    thumb = img.resize((int(img.width * scale), int(img.height * scale)))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_report_html(summary, label, checkpoint_path=None, val_loss=None):
    """Builds a self-contained HTML report (same visual language as the review-batch
    artifact) with every generated route's image and stats, grouped by grade."""
    font_path = "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"
    try:
        with open(font_path, "rb") as f:
            font_b64 = base64.b64encode(f.read()).decode("ascii")
        font_face = f'''@font-face {{
      font-family: "DIN Cond";
      src: url(data:font/ttf;base64,{font_b64}) format("truetype");
      font-weight: 700;
    }}'''
    except FileNotFoundError:
        font_face = ""

    by_grade = collections.defaultdict(list)
    for r in summary["rows"]:
        by_grade[r["grade"]].append(r)

    def card(r):
        pill = "pill-good" if r["valid"] else "pill-bad"
        text = "VALID" if r["valid"] else "REJECTED"
        img_b64 = _image_to_b64_jpeg(r["image"])
        critic_html = ""
        if "critic_v_grade" in r:
            diff = abs(r["critic_v_grade"] - r["grade"])
            critic_class = "critic-good" if diff == 0 else ("critic-ok" if diff <= 1 else "critic-bad")
            critic_html = f'<span class="critic-tag {critic_class}">critic thinks V{r["critic_v_grade"]}</span>'
        return f'''
      <figure class="card">
        <div class="thumb-wrap">
          <img src="data:image/jpeg;base64,{img_b64}" alt="V{r['grade']} at {r['angle']} degrees" />
          <span class="angle-tag">{r['angle']}&deg;</span>
        </div>
        <figcaption>
          <span class="pill {pill}">{text}</span>
          {critic_html}
          <span class="stat-line">{r['n_holds']} holds &middot; span {r['y_span']}px</span>
        </figcaption>
      </figure>'''

    rows_html = []
    for grade in sorted(by_grade):
        grade_rows = sorted(by_grade[grade], key=lambda r: r["angle"])
        s = summary["per_grade"][grade]
        cards = "".join(card(r) for r in grade_rows)
        critic_meta = f" &middot; critic err {s['avg_critic_diff']:.2f}" if "avg_critic_diff" in s else ""
        rows_html.append(f'''
    <section class="grade-row">
      <div class="grade-label">
        <span class="grade-num">V{grade}</span>
        <span class="grade-meta">{s['valid_rate']*100:.0f}% valid &middot; avg {s['avg_holds']:.1f} holds{critic_meta}</span>
      </div>
      <div class="card-strip">{cards}</div>
    </section>''')

    meta_bits = [f"label: <strong>{label}</strong>"]
    if checkpoint_path:
        meta_bits.append(f"checkpoint: <span class=\"mono\">{os.path.basename(checkpoint_path)}</span>")
    if val_loss is not None:
        meta_bits.append(f"val loss: <span class=\"mono\">{val_loss:.4f}</span>")
    meta_line = " &middot; ".join(meta_bits)

    critic_stat_html = ""
    if "avg_critic_diff" in summary:
        critic_stat_html = f'''
    <div class="stat"><span class="num mono">{summary['avg_critic_diff']:.2f}</span><span class="label">Critic grade error</span></div>'''

    return f'''<!doctype html>
<title>Route Quality Eval</title>
<style>
{font_face}
:root {{
  --ink: #1D1B18; --ink-raised: #29261F; --ink-border: #3C382E;
  --chalk: #F4F1E7; --chalk-raised: #FFFFFF; --chalk-border: #DEDACB;
  --accent: #2F9E96; --accent-strong: #1F7A73;
  --good: #4F9D5C; --good-bg: rgba(79,157,92,0.14);
  --bad: #C25B3F; --bad-bg: rgba(194,91,63,0.14);
  --warn: #C99A3A;
  --bg: var(--chalk); --surface: var(--chalk-raised); --border: var(--chalk-border);
  --text: #211F1A; --text-muted: #6B6558; --text-faint: #8C8676;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --bg: var(--ink); --surface: var(--ink-raised); --border: var(--ink-border); --text: #F1EEE4; --text-muted: #B6AF9E; --text-faint: #8A8474; }}
}}
:root[data-theme="dark"] {{ --bg: var(--ink); --surface: var(--ink-raised); --border: var(--ink-border); --text: #F1EEE4; --text-muted: #B6AF9E; --text-faint: #8A8474; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.5; padding: 0 0 3rem; }}
.din {{ font-family: "DIN Cond", "Arial Narrow", sans-serif; font-weight: 700; text-transform: uppercase; letter-spacing: 0.02em; }}
.mono {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }}
header.masthead {{ max-width: 72rem; margin: 0 auto; padding: 2.5rem 2rem 1.5rem; border-bottom: 1px solid var(--border); }}
h1.din {{ font-size: clamp(2rem, 5vw, 3.2rem); line-height: 0.92; margin: 0 0 0.6rem; text-wrap: balance; }}
h1.din span {{ color: var(--accent-strong); }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) h1.din span {{ color: var(--accent); }} }}
:root[data-theme="dark"] h1.din span {{ color: var(--accent); }}
.subhead {{ max-width: 46rem; color: var(--text-muted); font-size: 0.95rem; margin: 0 0 1.25rem; }}
.stat-strip {{ display: flex; flex-wrap: wrap; gap: 2.25rem; padding-top: 1.25rem; border-top: 1px solid var(--border); }}
.stat {{ display: flex; flex-direction: column; gap: 0.15rem; }}
.stat .num {{ font-size: 1.7rem; font-weight: 700; }}
.stat .label {{ font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-faint); }}
main {{ max-width: 72rem; margin: 0 auto; padding: 0 2rem; }}
.grade-row {{ padding: 2rem 0; border-bottom: 1px solid var(--border); }}
.grade-row:last-child {{ border-bottom: none; }}
.grade-label {{ display: flex; align-items: baseline; gap: 0.9rem; margin-bottom: 1rem; }}
.grade-num {{ font-family: "DIN Cond", "Arial Narrow", sans-serif; font-weight: 700; font-size: 2.1rem; color: var(--accent-strong); }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) .grade-num {{ color: var(--accent); }} }}
:root[data-theme="dark"] .grade-num {{ color: var(--accent); }}
.grade-meta {{ font-size: 0.85rem; color: var(--text-faint); }}
.card-strip {{ display: grid; grid-template-columns: repeat(8, minmax(7rem, 1fr)); gap: 0.75rem; overflow-x: auto; }}
@media (max-width: 900px) {{ .card-strip {{ grid-template-columns: repeat(4, minmax(7rem, 1fr)); }} }}
.card {{ margin: 0; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }}
.thumb-wrap {{ position: relative; aspect-ratio: 1284 / 1413; background: var(--ink); }}
.thumb-wrap img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
.angle-tag {{ position: absolute; top: 0.35rem; left: 0.35rem; font-family: "DIN Cond", "Arial Narrow", sans-serif; font-weight: 700; font-size: 0.85rem; background: rgba(0,0,0,0.55); color: #F4F1E7; padding: 0.08rem 0.35rem; border-radius: 3px; }}
figcaption {{ padding: 0.45rem 0.55rem 0.6rem; display: flex; flex-direction: column; gap: 0.3rem; }}
.pill {{ align-self: flex-start; font-size: 0.6rem; font-weight: 700; letter-spacing: 0.06em; padding: 0.12rem 0.4rem; border-radius: 999px; }}
.pill-good {{ background: var(--good-bg); color: var(--good); }}
.pill-bad {{ background: var(--bad-bg); color: var(--bad); }}
.critic-tag {{ align-self: flex-start; font-size: 0.62rem; font-weight: 600; }}
.critic-good {{ color: var(--good); }}
.critic-ok {{ color: var(--warn); }}
.critic-bad {{ color: var(--bad); }}
.stat-line {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 0.62rem; color: var(--text-faint); }}
</style>
<header class="masthead">
  <h1 class="din">Route Quality <span>Evaluation</span></h1>
  <p class="subhead">{meta_line}</p>
  <div class="stat-strip">
    <div class="stat"><span class="num mono">{summary['total']}</span><span class="label">Routes</span></div>
    <div class="stat"><span class="num mono">{summary['valid_rate']*100:.0f}%</span><span class="label">Valid</span></div>
    <div class="stat"><span class="num mono">{summary['avg_holds']:.1f}</span><span class="label">Avg holds</span></div>
    <div class="stat"><span class="num mono">{summary['avg_y_span']:.0f}px</span><span class="label">Avg y-span</span></div>{critic_stat_html}
  </div>
</header>
<main>
{"".join(rows_html)}
</main>
'''


def run_evaluation(model, device="cpu", label=None, checkpoint_path=None, val_loss=None,
                    grades=None, angles=None, seed=123, save_report=True, save_history=True, critic=None):
    """
    Full evaluation: generate + score routes, print a summary, optionally save a visual
    HTML report and append to eval_history.jsonl. This is the one function to call
    after any change to the model or generation pipeline.
    """
    label = label or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary = evaluate_checkpoint(model, device=device, grades=grades, angles=angles,
                                   seed=seed, with_images=save_report, critic=critic)
    print_summary(summary)

    if save_history:
        append_history(summary, label, checkpoint_path=checkpoint_path, val_loss=val_loss)
        print(f"  -> appended to {HISTORY_PATH}")

    if save_report:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:60]
        report_path = os.path.join(
            REPORTS_DIR, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_label}.html"
        )
        html = build_report_html(summary, label, checkpoint_path=checkpoint_path, val_loss=val_loss)
        with open(report_path, "w") as f:
            f.write(html)
        print(f"  -> visual report: {report_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="kilter_setter_best.pt")
    parser.add_argument("--label", default=None, help="Short description of what changed since the last eval")
    parser.add_argument("--history", action="store_true", help="Print recent evaluation history and exit")
    parser.add_argument("--no-report", action="store_true", help="Skip the visual HTML report")
    parser.add_argument("--no-critic", action="store_true", help="Ignore critic_best.pt even if present")
    args = parser.parse_args()

    if args.history:
        print_history()
        sys.exit(0)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = Setter(vocab_size=VOCAB_SIZE).to(device)
    try:
        model.load_state_dict(load_checkpoint_state_dict(args.checkpoint, map_location=device))
    except FileNotFoundError:
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    critic = None if args.no_critic else load_critic(os.path.join(SCRIPT_DIR, "critic_best.pt"), device=device)
    print("Using critic-based reranking" if critic else "No critic checkpoint found, using hold-count proxy")

    run_evaluation(model, device=device, label=args.label, checkpoint_path=args.checkpoint,
                    save_report=not args.no_report, critic=critic)
