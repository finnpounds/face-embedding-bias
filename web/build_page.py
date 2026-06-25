"""
Build a self-contained ``web/index.html`` project page for a personal website.

Reads the generated figures (``figures/*.png``) and metrics
(``results/metrics.json``), inlines the images as base64 so the output is a
single portable file, and writes ``web/index.html``.

Run after ``src/bias_pipeline.py``:
    python web/build_page.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RESULTS = ROOT / "results"
OUT = ROOT / "web" / "index.html"

RACE_ORDER = ["Black", "East Asian", "Indian", "Latino Hispanic",
              "Middle Eastern", "Southeast Asian", "White"]


def img(name: str) -> str:
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def main():
    m = json.loads((RESULTS / "metrics.json").read_text())
    dl = m["directional_leakage"]
    auc = m["holdout_ovsr_auc"]
    n = m["n_faces"]

    # per-race AUC rows
    auc_rows = "\n".join(
        f"<tr><td>{r}</td><td>{auc['arc'][r]:.2f}</td><td>{auc['vit'][r]:.2f}</td></tr>"
        for r in RACE_ORDER
    )
    macro_arc = auc["arc"]["macro_avg"]
    macro_vit = auc["vit"]["macro_avg"]

    html = TEMPLATE.format(
        n=n,
        macro_arc=f"{macro_arc:.2f}", macro_vit=f"{macro_vit:.2f}",
        arc_gap=dl["arc"]["cosine_gap"], arc_knn=f"{dl['arc']['knn5']:.2f}",
        arc_auc=f"{dl['arc']['probe_auc']:.2f}",
        vit_gap=dl["vit"]["cosine_gap"], vit_knn=f"{dl['vit']['knn5']:.2f}",
        vit_auc=f"{dl['vit']['probe_auc']:.2f}",
        auc_rows=auc_rows,
        per_race_auc=img("per_race_auc.png"),
        umap_arc=img("umap_arc.png"),
        umap_vit=img("umap_vit.png"),
        samples=img("sample_faces_by_race.png"),
    )
    OUT.write_text(html)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Demographic Bias in Open-Source Face-Recognition Embeddings</title>
<style>
  :root {{
    --bg: #0f1117; --panel: #171a22; --ink: #e8eaed; --muted: #9aa0aa;
    --accent: #6ea8fe; --accent2: #f0883e; --line: #262b36; --good: #4cc38a;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  .wrap {{ max-width: 860px; margin: 0 auto; padding: 0 22px 96px; }}
  header {{ padding: 72px 0 28px; border-bottom: 1px solid var(--line); }}
  .eyebrow {{ color: var(--accent); font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; font-size: 13px; }}
  h1 {{ font-size: 2.15rem; line-height: 1.18; margin: 12px 0 14px; }}
  .lede {{ font-size: 1.15rem; color: var(--muted); margin: 0; }}
  .meta {{ margin-top: 22px; font-size: 14px; color: var(--muted); }}
  .meta a {{ color: var(--accent); text-decoration: none; }}
  .meta a:hover {{ text-decoration: underline; }}
  h2 {{ font-size: 1.4rem; margin: 52px 0 14px; }}
  h3 {{ font-size: 1.05rem; margin: 30px 0 8px; color: var(--ink); }}
  p {{ color: #cfd3da; }}
  .kpis {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 30px 0 8px; }}
  .kpi {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
    padding: 18px; text-align: center; }}
  .kpi .num {{ font-size: 1.9rem; font-weight: 700; color: var(--accent); }}
  .kpi .lab {{ font-size: 12.5px; color: var(--muted); margin-top: 4px; }}
  figure {{ margin: 24px 0; }}
  figure img {{ width: 100%; border-radius: 12px; background: #fff; padding: 10px;
    border: 1px solid var(--line); }}
  figcaption {{ color: var(--muted); font-size: 13.5px; margin-top: 8px; text-align: center; }}
  .twocol {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 15px; }}
  th, td {{ padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--muted); font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: .03em; }}
  td:not(:first-child), th:not(:first-child) {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:last-child td {{ font-weight: 700; }}
  .callout {{ background: var(--panel); border-left: 3px solid var(--accent2);
    border-radius: 8px; padding: 14px 18px; margin: 22px 0; color: #d6dae1; }}
  .pill {{ display: inline-block; background: var(--panel); border: 1px solid var(--line);
    border-radius: 999px; padding: 4px 12px; font-size: 13px; color: var(--muted); margin: 2px 4px 2px 0; }}
  ul {{ color: #cfd3da; }}
  footer {{ margin-top: 60px; padding-top: 22px; border-top: 1px solid var(--line);
    color: var(--muted); font-size: 14px; }}
  footer a {{ color: var(--accent); text-decoration: none; }}
  @media (max-width: 640px) {{
    .kpis {{ grid-template-columns: 1fr; }} .twocol {{ grid-template-columns: 1fr; }}
    h1 {{ font-size: 1.7rem; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">Machine-Learning Fairness · Computer Vision</div>
    <h1>Demographic Bias in Open-Source<br>Face-Recognition Embeddings</h1>
    <p class="lede">Two popular, freely available vision models &mdash; ArcFace and ViT &mdash;
      were never trained to encode race. A simple linear probe recovers it from their
      embeddings anyway. Here's how much, and why it matters for anyone shipping them.</p>
    <div class="meta">
      <span class="pill">FairFace · {n} aligned faces</span>
      <span class="pill">ArcFace (CNN) vs. ViT (transformer)</span>
      <span class="pill">GDPR · CCPA · NIST AI-RMF</span>
      <br><br>
      Finn Pounds &nbsp;·&nbsp; EAI6400 &nbsp;·&nbsp;
      <a href="https://github.com/finnpounds/face-embedding-bias">View the code on GitHub →</a>
    </div>
  </header>

  <section>
    <div class="kpis">
      <div class="kpi"><div class="num">{macro_vit}</div><div class="lab">ViT held-out AUC for race (0.5 = chance)</div></div>
      <div class="kpi"><div class="num">{macro_arc}</div><div class="lab">ArcFace held-out AUC for race</div></div>
      <div class="kpi"><div class="num">&gt;96%</div><div class="lab">Detector keep-rate across every group</div></div>
    </div>

    <h2>The question</h2>
    <p>Open-source face models are a few <code>pip install</code>s from production
      &mdash; identity verification in finance, patient check-in in healthcare,
      loyalty kiosks in retail. If the <em>embeddings</em> these systems pass around
      secretly encode protected attributes like race, an organization can be
      processing special-category data &mdash; and risking disparate impact &mdash;
      <strong>without ever explicitly collecting it.</strong></p>

    <h2>The headline result</h2>
    <p>Training a logistic-regression probe on 70% of the embeddings and scoring it
      on a held-out 30%, race is recovered far above chance for every group &mdash;
      and, surprisingly, the general-purpose <strong>ViT leaks more than the
      face-specialized ArcFace</strong>.</p>
    <figure>
      <img src="{per_race_auc}" alt="Per-race held-out AUC bar chart">
      <figcaption>One-vs-rest held-out AUC per group. The dashed line is random chance (0.5).</figcaption>
    </figure>

    <table>
      <thead><tr><th>Group</th><th>ArcFace</th><th>ViT</th></tr></thead>
      <tbody>
        {auc_rows}
        <tr><td>Macro average</td><td>{macro_arc}</td><td>{macro_vit}</td></tr>
      </tbody>
    </table>

    <div class="callout">
      <strong>Why it matters:</strong> "We never store race" is not a sufficient
      privacy claim when the embeddings you <em>do</em> store make race trivially
      recoverable. Under GDPR Art. 9, CCPA, and the NIST AI Risk Management
      Framework, that inferability can pull a system into scope for special-category
      data, consent, and fairness obligations.
    </div>

    <h2>The embeddings cluster by race</h2>
    <p>UMAP projections of each embedding space, colored by group. Demographic
      structure is visible in both &mdash; not a designed feature of either model.</p>
    <div class="twocol">
      <figure><img src="{umap_arc}" alt="UMAP of ArcFace embeddings"><figcaption>ArcFace (CNN, 512-d)</figcaption></figure>
      <figure><img src="{umap_vit}" alt="UMAP of ViT embeddings"><figcaption>ViT (transformer, 768-d)</figcaption></figure>
    </div>

    <h2>How it was measured</h2>
    <p>A race-balanced subset of FairFace (1,000 images each across 7 groups) is
      aligned with MTCNN, then embedded twice &mdash; once with ArcFace
      (<code>buffalo_l</code>, InsightFace) and once with ViT-Base
      (<code>vit-base-patch16-224-in21k</code>, HuggingFace). Leakage is then probed
      several ways:</p>
    <table>
      <thead><tr><th>Signal</th><th>ArcFace</th><th>ViT</th></tr></thead>
      <tbody>
        <tr><td>Cosine gap (within − between race)</td><td>{arc_gap}</td><td>{vit_gap}</td></tr>
        <tr><td>kNN-5 accuracy</td><td>{arc_knn}</td><td>{vit_knn}</td></tr>
        <tr><td>Linear-probe AUC</td><td>{arc_auc}</td><td>{vit_auc}</td></tr>
      </tbody>
    </table>
    <figure>
      <img src="{samples}" alt="MTCNN-aligned FairFace samples by group">
      <figcaption>MTCNN-aligned sample faces, one row per group.</figcaption>
    </figure>

    <h2>Honest caveats</h2>
    <ul>
      <li>"Race" here is FairFace's coarse 7-way social annotation &mdash; the point
        is leakage of a <em>labeled sensitive attribute</em>, not an endorsement of
        the categories.</li>
      <li>These probes are <em>linear</em>; non-linear attacks would likely extract
        <strong>more</strong>, so the numbers are a lower bound.</li>
      <li>Findings describe these two checkpoints; other models will differ.</li>
    </ul>
  </section>

  <footer>
    Finn Pounds · EAI6400 (AI Ethics &amp; Governance), 2025 ·
    <a href="https://github.com/finnpounds/face-embedding-bias">Code &amp; full report on GitHub</a>
  </footer>
</div>
</body>
</html>
"""

if __name__ == "__main__":
    main()
