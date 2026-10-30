import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

M = np.array([[0.4124564, 0.3575761, 0.1804375],
              [0.2126729, 0.7151522, 0.0721750],
              [0.0193339, 0.1191920, 0.9503041]])

MINV = np.linalg.inv(M)
WHITE = np.array([0.95047, 1.0, 1.08883])
D = 6 / 29
MODES = {"none": [], "ab": [1, 2], "full": [0, 1, 2]}   # canais L=0, a=1, b=2

def rgb2lab(rgb):
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    t = (lin @ M.T) / WHITE
    f = np.where(t > D ** 3, np.cbrt(t), t / (3 * D ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)

def lab2rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    t = np.where(f > D, f ** 3, 3 * D ** 2 * (f - 4 / 29))
    lin = np.clip((t * WHITE) @ MINV.T, 0, 1)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)

def normalize_image(rgb, ref, mode="ab", gmin=0.5, gmax=2.0):
    """Reinhard por tile: leva média e desvio de cada canal LAB para os da referência.
    O ganho (desvio_ref / desvio_tile) é limitado a [gmin, gmax] para não amplificar ruído."""
    if mode == "none":
        return rgb
    lab = rgb2lab(rgb)
    for k in MODES[mode]:
        m, s = lab[..., k].mean(), lab[..., k].std()
        g = float(np.clip(ref["std"][k] / (s + 1e-6), gmin, gmax))
        lab[..., k] = (lab[..., k] - m) * g + ref["mean"][k]
    return lab2rgb(lab)

def tile_stats(path):
    lab = rgb2lab(np.asarray(Image.open(path).convert("RGB"))).reshape(-1, 3)
    return lab.mean(0), lab.std(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--config", default="configs/color_norm.json")
    ap.add_argument("--bg-label", default="background")
    ap.add_argument("--gmin", type=float, default=0.5)
    ap.add_argument("--gmax", type=float, default=2.0)
    ap.add_argument("--n-grid", type=int, default=8)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(exist_ok=True)

    # Só treino e validação; fundo fica de fora (seria rejeitado pelo passo A).
    rows = [json.loads(l) for l in open(args.manifest)]
    rows = [r for r in rows if r["split"] in ("train", "val") and r["label"] != args.bg_label]
    for r in rows:
        r["mean"], r["std"] = tile_stats(root / r["path"])
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]

    # Referência calculada SÓ no treino.
    ref = {"mean": np.mean([r["mean"] for r in train], 0),
           "std": np.mean([r["std"] for r in train], 0)}
    print(f"Referência (treino, {len(train)} tiles sem fundo):")
    print(f"  média LAB = {np.round(ref['mean'], 1)}  desvio dentro do tile = {np.round(ref['std'], 1)}")

    # Teste de ida e volta RGB <-> LAB
    err = max(int(np.abs(lab2rgb(rgb2lab(a)).astype(int) - a).max())
              for a in (np.asarray(Image.open(root / r["path"]).convert("RGB")) for r in rows[:50]))
    print(f"  erro máximo de ida e volta RGB-LAB-RGB: {err} (esperado <= 1)")

    # Tabela por classe (validação)
    classes = sorted({r["label"] for r in val})
    print("\nValidação, por classe (média LAB dos tiles | desvio entre tiles | % ganho limitado)")
    print(f"{'classe':28s}{'L':>7s}{'a':>7s}{'b':>7s} |{'dL':>6s}{'da':>6s}{'db':>6s} |{'cL':>5s}{'ca':>5s}{'cb':>5s}")
    cmeans, cwithin = [], []
    for c in classes:
        mu = np.array([r["mean"] for r in val if r["label"] == c])
        sd = np.array([r["std"] for r in val if r["label"] == c])
        g = ref["std"] / (sd + 1e-6)
        clip = ((g < args.gmin) | (g > args.gmax)).mean(0) * 100
        cmeans.append(mu.mean(0)); cwithin.append(mu.std(0))
        print(f"{c:28s}" + "".join(f"{v:7.1f}" for v in mu.mean(0)) + " |"
              + "".join(f"{v:6.1f}" for v in mu.std(0)) + " |"
              + "".join(f"{v:5.0f}" for v in clip))
    between, within = np.std(cmeans, 0), np.mean(cwithin, 0)
    print(f"\nEntre classes (desvio das médias): {np.round(between, 1)}")
    print(f"Dentro das classes (desvio médio):  {np.round(within, 1)}")
    print(f"Razão entre/dentro (L, a, b):       {np.round(between / within, 2)}")

    # Grade antes/depois nos tiles mais atípicos em croma (a,b) dentro da própria classe
    allr = train + val
    for r in allr:
        mu = np.array([x["mean"][1:] for x in allr if x["label"] == r["label"]])
        r["dist"] = float(np.linalg.norm(r["mean"][1:] - np.median(mu, 0)))
    top = sorted(allr, key=lambda r: -r["dist"])[:args.n_grid]
    fig, axes = plt.subplots(3, len(top), figsize=(len(top) * 1.8, 6.2), squeeze=False)
    for j, r in enumerate(top):
        a = np.asarray(Image.open(root / r["path"]).convert("RGB"))
        for i, mode in enumerate(["none", "ab", "full"]):
            axes[i][j].imshow(normalize_image(a, ref, mode, args.gmin, args.gmax))
            axes[i][j].axis("off")
        axes[0][j].set_title(f"{r['label'][:14]}\nd={r['dist']:.0f}", fontsize=7)
    for i, name in enumerate(["original", "normaliza a,b", "normaliza L,a,b"]):
        axes[i][0].text(-0.1, 0.5, name, transform=axes[i][0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "color_norm_before_after.png", dpi=120)
    plt.close()
    print(f"\nTiles mais atípicos (distância de croma à mediana da classe):")
    for r in top:
        print(f"  {r['label']:28s} {r['split']:6s} d={r['dist']:.1f}  {r['path']}")

    Path(args.config).parent.mkdir(exist_ok=True)
    json.dump({"method": "reinhard_lab_por_tile",
               "ref": {"mean": ref["mean"].tolist(), "std": ref["std"].tolist()},
               "gain_clip": [args.gmin, args.gmax], "fit_split": "train",
               "fit_excludes": args.bg_label, "n_fit": len(train)},
              open(args.config, "w"), indent=2)
    print(f"\nSalvo: {args.config} e {out}/color_norm_before_after.png")

if __name__ == "__main__":
    main()