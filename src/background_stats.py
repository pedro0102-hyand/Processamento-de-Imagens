import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# caracteristicas a serem extraidas das imagens
MEASURES = ["brilho", "desvio_cinza", "saturacao", "iqr_cinza"]

def measures(path):
    im = Image.open(path).convert("RGB") # abre a imagem e converte no modelo RGB (vermelho, verde, azul)
    gray = np.asarray(im.convert("L"), dtype=np.float32) # converte para escala de cinza
    sat = np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1] # medir saturacao da imagem HSV (matiz, saturacao, valor)
    q25, q75 = np.percentile(gray, [25, 75]) # calculo estatístico de quartis da imagem
    return [gray.mean(), gray.std(), sat.mean(), q75 - q25] 

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in open(args.manifest)]
    rows = [r for r in rows if r["split"] in args.splits]  # sem teste

    data = [(r["path"], r["label"], r["split"], *measures(root / r["path"])) for r in rows]

    with open(out / "background_measures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split", *MEASURES])
        w.writerows(data)

    classes = sorted({d[1] for d in data})
    vals = {c: np.array([d[3:] for d in data if d[1] == c]) for c in classes}

    print(f"{'classe':28s}" + "".join(f"{m:>22s}" for m in MEASURES))
    for c in classes:
        med, p5, p95 = np.median(vals[c], 0), np.percentile(vals[c], 5, 0), np.percentile(vals[c], 95, 0)
        print(f"{c:28s}" + "".join(f"{med[i]:8.1f} [{p5[i]:5.1f}-{p95[i]:5.1f}]" for i in range(len(MEASURES))))

    fig, axes = plt.subplots(1, len(MEASURES), figsize=(20, 4))
    for i, m in enumerate(MEASURES):
        for c in classes:
            axes[i].hist(vals[c][:, i], bins=40, histtype="step",
                         linewidth=2.5 if c == "background" else 1, label=c)
        axes[i].set_title(m)
    axes[0].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out / "background_hist.png", dpi=120)
    print(f"\nSalvo: {out}/background_measures.csv e {out}/background_hist.png")

if __name__ == "__main__":
    main()