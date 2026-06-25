"""
Download a race-balanced subset of FairFace into ``fairface_subset/``.

Streams the HuggingFace ``HuggingFaceM4/FairFace`` dataset and saves the first
``N_PER_RACE`` images for each of the 7 race groups, named ``<Race>_<idx>.jpg``
so the rest of the pipeline can recover labels from filenames.

Usage:
    python src/download_data.py            # 1000 images/group (7000 total)
    python src/download_data.py --n 200    # smaller subset
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "fairface_subset"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000,
                    help="images per race group (default 1000)")
    args = ap.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    counter: dict[str, int] = {}

    ds = load_dataset("HuggingFaceM4/FairFace", "0.25",
                      split="train", streaming=True, cache_dir=str(ROOT / ".hf_cache"))
    to_str = ds.features["race"].int2str

    for ex in tqdm(ds, desc="Streaming FairFace"):
        race = to_str(ex["race"]).replace("_", " ")
        counter.setdefault(race, 0)
        if counter[race] >= args.n:
            if all(v >= args.n for v in counter.values()):
                break
            continue
        ex["image"].convert("RGB").save(
            DATA_ROOT / f"{race.replace(' ', '_')}_{counter[race]:04d}.jpg", quality=95)
        counter[race] += 1

    print(f"Saved {sum(counter.values())} images -> {DATA_ROOT}")
    print(counter)


if __name__ == "__main__":
    main()
