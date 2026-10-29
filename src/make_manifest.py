import argparse, json, random
from collections import Counter
from pathlib import Path
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SPLITS = {"train", "val", "test"}

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="manifest.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.input)
    rows, bad = [], []

    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        rel = p.relative_to(root)
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
                mode = im.mode
        except Exception as e:
            bad.append({"path": str(rel), "error": str(e)})
            continue

        parts = rel.parts
        split = parts[0] if parts[0] in SPLITS else None
        rows.append({
            "path": str(rel),
            "label": parts[-2] if len(parts) > 1 else None,
            "width": w,
            "height": h,
            "mode": mode,
            "split": split,
            "split_provisional": split is None,
        })

    rng = random.Random(args.seed)
    for r in rows:
        if r["split"] is None:
            x = rng.random()
            r["split"] = "train" if x < 0.8 else "val" if x < 0.9 else "test"

    with open(args.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if bad:
        with open(args.output + ".bad.json", "w") as f:
            json.dump(bad, f, indent=2)

    print(f"{len(rows)} imagens válidas, {len(bad)} com erro")
    print("Classes:", dict(Counter(r["label"] for r in rows)))
    print("Splits:", dict(Counter(r["split"] for r in rows)))

if __name__ == "__main__":
    main()