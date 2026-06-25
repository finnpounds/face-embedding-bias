"""
Demographic-bias analysis pipeline for open-source face-recognition embeddings.

Compares how strongly race leaks into ArcFace (CNN) vs. ViT (transformer)
embeddings on a race-balanced subset of FairFace.

Stages
------
1. Load a race-balanced image subset (filename pattern: ``<Race>_<idx>.jpg``).
2. Align faces with MTCNN (112x112 crops).
3. Extract ArcFace (512-d) and ViT (768-d) embeddings, streaming so that the
   full set of face crops is never held in memory at once.
4. Quantify directional leakage (cosine gap, kNN, linear-probe AUC), a held-out
   one-vs-rest probe, within-race similarity, and detector keep-rates.
5. Write metrics to ``results/`` and labeled figures to ``figures/``.

Embeddings are cached to ``embed_cache.npz`` so figures/metrics can be
re-derived without recomputing.

Usage
-----
    python src/bias_pipeline.py                # full run (uses cache if present)
    python src/bias_pipeline.py --force        # ignore cache, recompute
    python src/bias_pipeline.py --figures-only # only redraw from the cache
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "fairface_subset"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
CACHE_FILE = ROOT / "embed_cache.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Consistent, color-blind-friendly palette + stable race ordering for plots.
RACE_ORDER = [
    "Black", "East Asian", "Indian", "Latino Hispanic",
    "Middle Eastern", "Southeast Asian", "White",
]
PALETTE = dict(zip(RACE_ORDER, sns.color_palette("colorblind", len(RACE_ORDER))))

sns.set_theme(style="white", context="talk")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def race_from_fname(stem: str) -> str:
    """``Latino_Hispanic_0042`` -> ``Latino Hispanic``."""
    m = re.match(r"(.+)_\d+$", stem)
    token = m.group(1) if m else stem
    return token.replace("_", " ")


def list_images() -> pd.DataFrame:
    paths = sorted(p for p in DATA_ROOT.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise FileNotFoundError(
            f"No images in {DATA_ROOT}. Download the FairFace subset first "
            "(see the notebook's download cell or the README)."
        )
    df = pd.DataFrame(
        {"path": [str(p) for p in paths],
         "race": [race_from_fname(p.stem) for p in paths]}
    )
    return df


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def load_models():
    from facenet_pytorch import MTCNN
    from insightface.model_zoo import get_model
    from transformers import AutoImageProcessor, AutoModel

    # post_process=False keeps crops in [0, 255] RGB. The default (True)
    # standardizes to ~[-1, 1], which — if later rescaled by *255 and clamped —
    # destroys ~half the pixels and the color balance before they reach the models.
    mtcnn = MTCNN(image_size=112, margin=0, thresholds=[0.6, 0.7, 0.7],
                  keep_all=False, post_process=False, device=DEVICE)

    arc = get_model("buffalo_l", root="~/.insightface",
                    providers=["CPUExecutionProvider"])

    vit_proc = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k", use_fast=True)
    vit = AutoModel.from_pretrained(
        "google/vit-base-patch16-224-in21k").to(DEVICE).eval()
    return mtcnn, arc, vit_proc, vit


def _resample_bicubic():
    try:
        return Image.Resampling.BICUBIC          # Pillow >= 10
    except AttributeError:
        return Image.BICUBIC                      # Pillow < 10


def align_face(mtcnn, pil_img, target=256):
    """Upscale small images, then MTCNN-align -> HxWx3 uint8 RGB, or None."""
    w, h = pil_img.size
    if min(w, h) < target:
        scale = target / min(w, h)
        pil_img = pil_img.resize((round(w * scale), round(h * scale)),
                                 _resample_bicubic())
    crop = mtcnn(pil_img)
    if crop is None:
        return None
    # crop is already C x H x W in [0, 255] thanks to post_process=False.
    crop = crop.permute(1, 2, 0).clamp(0, 255)
    return crop.cpu().numpy().astype("uint8")


@torch.inference_mode()
def emb_arc(arc, np_img):
    bgr = np_img[:, :, ::-1].astype("uint8")     # RGB -> BGR
    return arc.get_feat(bgr).squeeze().astype("float32")   # already unit-norm


@torch.inference_mode()
def emb_vit(vit_proc, vit, np_img):
    pil = Image.fromarray(np_img, mode="RGB")
    out = vit(**vit_proc(images=pil, return_tensors="pt").to(DEVICE))
    return out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype("float32")


# --------------------------------------------------------------------------- #
# Embedding extraction (streaming -> low memory)
# --------------------------------------------------------------------------- #
def build_embeddings(samples_per_race_grid: int = 5):
    df = list_images()
    attempt_counts = df["race"].value_counts()
    mtcnn, arc, vit_proc, vit = load_models()

    arc_rows, vit_rows, labels = [], [], []
    # Keep a few crops per race for the sample-grid figure (everything else freed).
    grid: dict[str, list[np.ndarray]] = {r: [] for r in RACE_ORDER}

    for row in tqdm(df.itertuples(), total=len(df), desc="align+embed"):
        crop = align_face(mtcnn, Image.open(row.path).convert("RGB"))
        if crop is None:
            continue
        arc_rows.append(emb_arc(arc, crop))
        vit_rows.append(emb_vit(vit_proc, vit, crop))
        labels.append(row.race)
        if len(grid.get(row.race, [])) < samples_per_race_grid:
            grid[row.race].append(crop)

    arc_emb = np.stack(arc_rows)
    vit_emb = np.stack(vit_rows)
    # L2-normalize ViT (ArcFace is already unit-norm); makes cosine geometry comparable.
    vit_emb = vit_emb / np.linalg.norm(vit_emb, axis=1, keepdims=True)
    labels = np.array(labels)

    keep_counts = pd.Series(labels).value_counts()
    keep_rate = (keep_counts / attempt_counts).fillna(0)

    # Force a 1-D object array of per-group stacks. Building it element-by-element
    # avoids numpy collapsing uniform-shape stacks into one 5-D array.
    grid_arr = np.empty(len(RACE_ORDER), dtype=object)
    for i, r in enumerate(RACE_ORDER):
        grid_arr[i] = (np.stack(grid[r]) if grid[r]
                       else np.zeros((0, 112, 112, 3), "uint8"))

    np.savez_compressed(
        CACHE_FILE,
        arc=arc_emb, vit=vit_emb, labels=labels,
        attempt=attempt_counts.reindex(RACE_ORDER).values,
        keep=keep_counts.reindex(RACE_ORDER).fillna(0).values,
        grid=grid_arr,
    )
    print(f"Cached embeddings -> {CACHE_FILE.name}  "
          f"(arc {arc_emb.shape}, vit {vit_emb.shape})")
    return load_cache()


def load_cache():
    d = np.load(CACHE_FILE, allow_pickle=True)
    keep_rate = pd.Series(
        np.where(d["attempt"] > 0, d["keep"] / np.maximum(d["attempt"], 1), 0.0),
        index=RACE_ORDER, name="keep_rate")
    # Coerce each group's crops back to a clean (k, 112, 112, 3) uint8 array,
    # robust to how the object array was stored.
    grid = {r: np.asarray(g, dtype="uint8") for r, g in zip(RACE_ORDER, d["grid"])}
    return d["arc"], d["vit"], d["labels"], keep_rate, grid


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def cosine_gap(X, y):
    sims = cosine_similarity(X)
    y = np.asarray(y)
    same = y[:, None] == y
    return float(sims[same].mean() - sims[~same].mean())


def knn_acc(X, y, k=5):
    return float(KNeighborsClassifier(k, metric="cosine").fit(X, y).score(X, y))


def probe_auc(X, y):
    codes = pd.Categorical(y).codes
    aucs = []
    for r in np.unique(codes):
        yb = (codes == r).astype(int)
        clf = LogisticRegression(max_iter=500, n_jobs=-1).fit(X, yb)
        aucs.append(roc_auc_score(yb, clf.predict_proba(X)[:, 1]))
    return float(np.mean(aucs))


def ovsr_holdout_auc(X, y, test_size=0.30):
    """Per-race AUC from a single stratified 70/30 split (CV-tuned L2 probe)."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED)
    scores = {}
    for r in RACE_ORDER:
        clf = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegressionCV(Cs=10, cv=5, max_iter=5000, n_jobs=-1),
        ).fit(Xtr, (ytr == r).astype(int))
        scores[r] = float(roc_auc_score((yte == r).astype(int),
                                        clf.predict_proba(Xte)[:, 1]))
    scores["macro_avg"] = float(np.mean([scores[r] for r in RACE_ORDER]))
    return scores


def within_race_similarity(X, y):
    sims = cosine_similarity(X)
    out = {}
    for r in RACE_ORDER:
        idx = np.where(y == r)[0]
        if len(idx) > 1:
            block = sims[np.ix_(idx, idx)]
            out[r] = float(block[np.triu_indices(len(idx), 1)].mean())
    return out


def compute_metrics(arc, vit, labels, keep_rate):
    metrics = {"directional_leakage": {}, "holdout_ovsr_auc": {},
               "within_race_similarity": {}, "detector_keep_rate": {}}
    for name, X in [("arc", arc), ("vit", vit)]:
        metrics["directional_leakage"][name] = {
            "cosine_gap": round(cosine_gap(X, labels), 4),
            "knn5": round(knn_acc(X, labels), 4),
            "probe_auc": round(probe_auc(X, labels), 4),
        }
        metrics["holdout_ovsr_auc"][name] = {
            k: round(v, 4) for k, v in ovsr_holdout_auc(X, labels).items()}
        metrics["within_race_similarity"][name] = {
            k: round(v, 4) for k, v in within_race_similarity(X, labels).items()}
    metrics["detector_keep_rate"] = {k: round(float(v), 4)
                                     for k, v in keep_rate.items()}
    metrics["n_faces"] = int(len(labels))
    return metrics


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_umap(X, labels, title, out_path, n_neighbors=15, min_dist=0.1):
    from umap import UMAP
    xy = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
              metric="cosine", random_state=SEED).fit_transform(X)
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "Race": labels})
    # Shuffle draw order so no single group sits on top of the overplot.
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.scatterplot(data=df, x="x", y="y", hue="Race", hue_order=RACE_ORDER,
                    palette=PALETTE, s=14, alpha=0.75, linewidth=0, ax=ax)
    ax.set(title=title, xlabel="UMAP-1", ylabel="UMAP-2")
    ax.legend(title="Race", bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False, markerscale=1.6)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path.name)


def plot_per_race_auc(metrics, out_path):
    rows = []
    for model in ("arc", "vit"):
        for r in RACE_ORDER:
            rows.append({"Race": r,
                         "Model": "ArcFace" if model == "arc" else "ViT",
                         "AUC": metrics["holdout_ovsr_auc"][model][r]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Race", y="AUC", hue="Model",
                palette={"ArcFace": "#4C72B0", "ViT": "#DD8452"}, ax=ax)
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.text(0.01, 0.505, "chance (0.5)", color="grey", fontsize=11,
            transform=ax.get_yaxis_transform())
    ax.set(ylim=(0.4, 1.0), ylabel="Held-out one-vs-rest AUC",
           title="How easily a linear probe recovers race from each embedding")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="", frameon=False)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path.name)


def plot_sample_grid(grid, out_path, k=5):
    races = [r for r in RACE_ORDER if len(grid[r])]
    fig, axs = plt.subplots(len(races), k, figsize=(k * 1.7, len(races) * 1.7))
    for r_i, race in enumerate(races):
        crops = grid[race]
        for c_i in range(k):
            ax = axs[r_i, c_i]
            ax.axis("off")
            if c_i < len(crops):
                ax.imshow(np.asarray(crops[c_i], dtype="uint8"))
            if c_i == 0:
                ax.text(-0.15, 0.5, race, transform=ax.transAxes,
                        ha="right", va="center", fontsize=12)
    fig.suptitle("MTCNN-aligned FairFace samples by group", y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path.name)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_results(metrics):
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    dl = pd.DataFrame(metrics["directional_leakage"]).T.rename(
        index={"arc": "ArcFace", "vit": "ViT"})
    auc = pd.DataFrame(metrics["holdout_ovsr_auc"]).rename(
        columns={"arc": "ArcFace", "vit": "ViT"})
    within = pd.DataFrame(metrics["within_race_similarity"]).rename(
        columns={"arc": "ArcFace", "vit": "ViT"})
    keep = pd.Series(metrics["detector_keep_rate"], name="keep_rate").to_frame()

    with (RESULTS_DIR / "summary.md").open("w") as f:
        f.write(f"# Results summary ({metrics['n_faces']} aligned faces)\n\n")
        f.write("## Directional leakage\n\n" + dl.to_markdown() + "\n\n")
        f.write("## Held-out one-vs-rest AUC\n\n" + auc.to_markdown() + "\n\n")
        f.write("## Within-race cosine similarity\n\n" + within.to_markdown() + "\n\n")
        f.write("## Detector / alignment keep-rate\n\n" + keep.to_markdown() + "\n")
    print("wrote results/metrics.json and results/summary.md")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="recompute embeddings even if the cache exists")
    ap.add_argument("--figures-only", action="store_true",
                    help="only redraw figures/metrics from the cache")
    args = ap.parse_args()

    FIG_DIR.mkdir(exist_ok=True)

    if CACHE_FILE.exists() and not args.force:
        print(f"Loading cached embeddings from {CACHE_FILE.name}")
        arc, vit, labels, keep_rate, grid = load_cache()
    else:
        if args.figures_only:
            raise SystemExit("--figures-only needs an existing embed_cache.npz")
        arc, vit, labels, keep_rate, grid = build_embeddings()

    print("Computing metrics ...")
    metrics = compute_metrics(arc, vit, labels, keep_rate)
    write_results(metrics)

    print("Drawing figures ...")
    plot_umap(arc, labels, "ArcFace embeddings (CNN) — UMAP", FIG_DIR / "umap_arc.png")
    plot_umap(vit, labels, "ViT embeddings (transformer) — UMAP", FIG_DIR / "umap_vit.png")
    plot_per_race_auc(metrics, FIG_DIR / "per_race_auc.png")
    plot_sample_grid(grid, FIG_DIR / "sample_faces_by_race.png")

    print("\nDone. Macro-avg held-out AUC — "
          f"ArcFace {metrics['holdout_ovsr_auc']['arc']['macro_avg']}, "
          f"ViT {metrics['holdout_ovsr_auc']['vit']['macro_avg']}")


if __name__ == "__main__":
    main()
