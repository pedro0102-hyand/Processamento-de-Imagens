import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", type=int, default=200, help="imagens por classe p/ cor média")
    ap.add_argument("--grid", type=int, default=5, help="exemplos por classe na grade")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in open(args.manifest)]

    # 1) Classes x splits
    classes = sorted({r["label"] for r in rows})
    splits = ["train", "val", "test"]
    cnt = Counter((r["label"], r["split"]) for r in rows)
    print(f"{'classe':28s}" + "".join(f"{s:>8s}" for s in splits))
    for c in classes:
        print(f"{c:28s}" + "".join(f"{cnt[(c, s)]:8d}" for s in splits))

    x = np.arange(len(classes))
    plt.figure(figsize=(9, 4))
    for i, s in enumerate(splits):
        plt.bar(x + i * 0.27, [cnt[(c, s)] for c in classes], 0.27, label=s)
    plt.xticks(x + 0.27, classes, rotation=45, ha="right")
    plt.ylabel("imagens"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "class_counts.png", dpi=120); plt.close()

    # 2) Dimensões e modo
    print("\nDimensões (w x h):", dict(Counter((r["width"], r["height"]) for r in rows)))
    print("Modos:", dict(Counter(r["mode"] for r in rows)))

    # 3) Cor média por classe (amostra)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r)
    print("\nCor média RGB por classe (amostra):")
    for c in classes:
        items = rng.sample(by_class[c], min(args.sample, len(by_class[c])))
        means = [np.asarray(Image.open(root / r["path"]).convert("RGB")).reshape(-1, 3).mean(0)
                 for r in items]
        m = np.mean(means, axis=0)
        print(f"{c:28s} R={m[0]:6.1f} G={m[1]:6.1f} B={m[2]:6.1f}")

    # 4) Grade de exemplos
    fig, axes = plt.subplots(len(classes), args.grid, figsize=(args.grid * 1.6, len(classes) * 1.6))
    for i, c in enumerate(classes):
        items = rng.sample(by_class[c], min(args.grid, len(by_class[c])))
        for j in range(args.grid):
            ax = axes[i][j]; ax.axis("off")
            if j < len(items):
                ax.imshow(Image.open(root / items[j]["path"]).convert("RGB"))
            if j == 0:
                ax.set_title(c, fontsize=7, loc="left")
    plt.tight_layout()
    plt.savefig(out / "examples.png", dpi=120); plt.close()
    print(f"\nGráficos salvos em {out}/")


if __name__ == "__main__":
    main()