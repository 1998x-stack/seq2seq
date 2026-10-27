"""Self-contained HTML ablation dashboard backed by actual experiment artifacts.

Rendering is intentionally read-only. It must never launch training or invent
scores for missing runs. The offline HTML bundles plotly.js locally/inline.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from seq2seq.experiments import summarize


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    report = summarize(root)  # refuse invalid comparisons across data fingerprints
    plan = _json(root / "plan.json") if (root / "plan.json").is_file() else []
    runs: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/seed-*/result.json")):
        result = _json(path)
        rows = []
        metric_log = path.parent / "metrics.jsonl"
        if metric_log.is_file():
            for line in metric_log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        metadata = _json(path.parent / "metadata.json") if (
            path.parent / "metadata.json"
        ).is_file() else {}
        runs.append({
            "name": result["name"], "seed": result["seed"],
            "metrics": result["metrics"], "history": rows,
            "parameters": metadata.get("parameters"),
            "device": metadata.get("device"),
            "run_dir": str(path.parent.relative_to(root)),
        })
    expected = {(r["name"], int(r["seed"])) for r in plan}
    completed = {(r["name"], int(r["seed"])) for r in runs}
    if not expected:
        # Historical runs may lack a plan file; do not silently claim completeness.
        expected = completed
    return {
        "root": str(root), "report": report, "runs": runs,
        "planned": len(expected), "completed": len(completed),
        "missing": sorted(expected - completed),
        "has_plan": bool(plan),
    }


def render_dashboard(
    root: str | Path, output: str | Path | None = None,
) -> Path:
    """Produce an offline HTML with train curves, BLEU distribution and buckets."""
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("Install plotting extras: pip install -e '.[viz]'") from exc

    data = collect(root)
    runs = data["runs"]
    if not runs:
        raise ValueError("No completed runs available to visualize")

    def figure_html(fig: Any, *, bundle: bool = False) -> str:
        fig.update_layout(
            template="plotly_white", font={"family": "Arial, sans-serif"},
            margin={"l": 55, "r": 25, "t": 48, "b": 58},
            height=420, hovermode="closest",
        )
        return plot(fig, output_type="div", include_plotlyjs=bundle,
                    config={"responsive": True, "displaylogo": False})

    # Each run is one series, with validation solid and train dotted.
    curves = make_subplots(rows=1, cols=2, subplot_titles=("Token NLL", "Token accuracy"))
    for run in runs:
        name = f"{run['name']} · seed {run['seed']}"
        history = sorted(run["history"], key=lambda x: x["epoch"])
        if not history:
            continue
        epochs = [row["epoch"] for row in history]
        for col, metric in ((1, "nll"), (2, "token_accuracy")):
            for split, dash in (("validation", "solid"), ("train", "dot")):
                vals = [row.get(split, {}).get(metric) for row in history]
                curves.add_trace(go.Scatter(
                    x=epochs, y=vals, mode="lines+markers", name=f"{name} / {split}",
                    line={"dash": dash},
                    showlegend=(col == 1),
                    legendgroup=f"{name}:{split}",
                    hovertemplate=(
                        f"Epoch %{{x}}<br>Value %{{y:.4f}}"
                        f"<extra>{name} / {split}</extra>"
                    ),
                ), row=1, col=col)
    curves.update_xaxes(title_text="Epoch")
    curves.update_yaxes(title_text="NLL (nats/token)", row=1, col=1)
    curves.update_yaxes(title_text="Accuracy (fraction)", range=[0, 1], row=1, col=2)
    curves_html = figure_html(curves, bundle=True)

    scores = go.Figure()
    for name in sorted({r["name"] for r in runs}):
        subset = [r for r in runs if r["name"] == name]
        points = [
            (r["seed"], r["metrics"].get("bleu"))
            for r in subset if r["metrics"].get("bleu") is not None
        ]
        if points:
            scores.add_trace(go.Box(
                y=[float(v) for _, v in points],
                name=name, boxpoints="all", jitter=0.2, pointpos=0,
                customdata=[[seed] for seed, _ in points],
                hovertemplate=(
                    f"BLEU %{{y:.2f}}<br>Seed %{{customdata[0]}}"
                    f"<extra>{name}</extra>"
                ),
            ))
    scores.update_layout(title="Test corpus BLEU across seeds", yaxis_title="BLEU (0–100)")
    score_html = figure_html(scores)

    # Stratification: each bar is the *real* group BLEU for each completed run.
    buckets = ("01-10", "11-20", "21-40", "41+")
    profiles = go.Figure()
    for run in runs:
        b = run["metrics"].get("length_buckets", {})
        profiles.add_trace(go.Bar(
            x=list(buckets),
            y=[(b.get(k) or {}).get("bleu") for k in buckets],
            name=f"{run['name']} · {run['seed']}",
            customdata=[[(b.get(k) or {}).get("count", 0)] for k in buckets],
            hovertemplate="Length %{x}<br>BLEU %{y:.2f}<br>N %{customdata[0]}<extra></extra>",
        ))
    profiles.update_layout(
        title="BLEU by source length · per-run (not pooled)",
        xaxis_title="Source tokens", yaxis_title="Corpus BLEU", barmode="group",
    )
    profile_html = figure_html(profiles)

    # Provenance is visible: users should never mistake incomplete experiments
    # or a synthetic copy task for full scientific translation benchmarks.
    labels = [f"{r['name']} / seed {r['seed']}" for r in runs]
    lines = []
    for run, label in zip(runs, labels):
        met = run["metrics"]
        def fmt(value: Any) -> str:
            return "—" if value is None else f"{float(value):.4f}"
        lines.append(
            "<tr><td>" + html.escape(label) + "</td>"
            + "<td>" + fmt(met.get("bleu")) + "</td>"
            + "<td>" + fmt(met.get("nll")) + "</td>"
            + "<td>" + fmt(met.get("exact_match")) + "</td>"
            + "<td>" + html.escape(str(run["parameters"] or "—")) + "</td>"
            + "<td>" + html.escape(run["run_dir"]) + "</td></tr>"
        )
    missing = "".join(
        "<li>" + html.escape(f"{name} / seed {seed}") + "</li>"
        for name, seed in data["missing"]
    )
    warn = (
        "INCOMPLETE — this report contains partial runs; do not draw final conclusions."
        if data["missing"] or not data["has_plan"]
        else "All recorded plan entries have finished. Verify the plan scope before publication."
    )
    fingerprint = html.escape(str(data["report"]["data_fingerprint"]))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seq2Seq Ablation Dashboard</title>
<style>
body{{font:15px/1.5 system-ui,-apple-system,sans-serif;max-width:1500px;margin:0 auto;padding:28px;background:#f7f8fb;color:#152337}}
header,section{{background:white;padding:22px 26px;border:1px solid #e4e8ef;border-radius:12px;margin-bottom:18px;overflow:hidden}}
h1{{font-size:25px;margin:0 0 8px}}h2{{font-size:18px;margin:0 0 12px}}
small,.muted{{color:#61718a}}.stats{{display:flex;gap:24px;flex-wrap:wrap;margin:14px 0}}.stats strong{{font-size:23px;display:block}}
.flag{{border-left:4px solid #bc7c18;padding:8px 12px;background:#fff8eb}}
table{{width:100%;border-collapse:collapse;font-size:13px;min-width:680px}}th,td{{padding:10px 9px;text-align:left;border-bottom:1px solid #e6ebf2}}th{{background:#f1f4f9}}
.scroll{{overflow:auto}}code{{overflow-wrap:anywhere}}.plotly-graph-div{{max-width:100%}}
</style></head><body>
<header><h1>Seq2Seq · Ablation Dashboard</h1>
<p class="muted">Read-only offline analysis of actual experiment artifacts. No synthetic metrics.</p>
<div class="stats"><div><strong>{data["completed"]}</strong>Completed runs</div>
<div><strong>{data["planned"]}</strong>Planned runs</div>
<div><strong>{len(data["report"]["models"])}</strong>Architecture variants observed</div></div>
<p class="flag">{html.escape(warn)}</p>
<small>Dataset fingerprint: <code>{fingerprint}</code></small></header>
<section><h2>Training and validation history</h2>{curves_html}</section>
<section><h2>Seed-level BLEU distributions</h2>{score_html}</section>
<section><h2>Length stratification</h2>{profile_html}</section>
<section><h2>Measured run table</h2><div class="scroll"><table>
<thead><tr><th>Variant / seed</th><th>Test BLEU</th><th>Test token NLL</th>
<th>Exact match</th><th>Parameters</th><th>Run artifact path</th></tr></thead>
<tbody>{''.join(lines)}</tbody></table></div></section>
<section><h2>Unfinished runs</h2>
{('<ul>' + missing + '</ul>') if missing else '<p>None in recorded plan.</p>'}</section>
<footer class="muted">Fixed split fingerprint does not establish book-disjoint evaluation.
 BLEU uses greedy decoded sequences; small length buckets may be unstable.
 See docs/stage02.md for methodology.</footer>
</body></html>"""
    dest = Path(output) if output else Path(root) / "dashboard.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".html.tmp")
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(dest)
    return dest
