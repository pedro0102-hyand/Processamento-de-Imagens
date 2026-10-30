import argparse, csv, json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load(csv_path, split):

    rows = [r for r in csv.DictReader(open(csv_path)) if r["split"] == split]
    paths = [r["path"] for r in rows]
    labels = np.array([r["label"] for r in rows])
    X = np.array([[float(r["brilho"]), float(r["desvio_cinza"])] for r in rows])
    return paths, labels, X

def predict(X, t, lo=None, hi=None):

    pred = X[:, 1] < t                       # pouca textura
    if lo is not None:
        pred = pred | (X[:, 0] < lo) | (X[:, 0] > hi)   # muito escuro/claro
    return pred

def prf(y, pred):

    tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp, fp, fn

def fit_rule1(y, X):

    cands = [(prf(y, predict(X, t))[2], dict(t=float(t))) for t in np.arange(1, 31, 0.5)]
    return max(cands, key=lambda c: c[0])

def fit_rule2(y, X):

    best = (-1, None)
    for t in np.arange(1, 31, 1.0):
        for lo in range(60, 125, 5):
            for hi in range(190, 245, 5):
                f = prf(y, predict(X, t, lo, hi))[2]
                if f > best[0]:
                    best = (f, dict(t=float(t), lo=lo, hi=hi))
    return best

def save_grid(paths, labels, idx, root, fname, title, ncols=6, maxn=12):

    idx = list(idx)[:maxn]
    if not idx:
        return
    nrows = (len(idx) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 2.0), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(Image.open(root / paths[i]).convert("RGB"))
        ax.set_title(labels[i][:14], fontsize=7)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--measures", default="eda_out/background_measures.csv")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--config", default="configs/background_rule.json")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)

    _, ytr_l, Xtr = load(args.measures, "train")
    pv, yv_l, Xv = load(args.measures, "val")
    ytr, yv = ytr_l == "background", yv_l == "background"

    # Parâmetros escolhidos SÓ no treino; validação só avalia; teste não é usado.
    f1, p1 = fit_rule1(ytr, Xtr)
    f2, p2 = fit_rule2(ytr, Xtr)
    rules = {
        "regra1_desvio": (f1, p1, lambda X, p: predict(X, p["t"])),
        "regra2_desvio+brilho": (f2, p2, lambda X, p: predict(X, p["t"], p["lo"], p["hi"])),
    }

    print(f"{'regra':24s}{'F1 treino':>10s}{'prec val':>10s}{'rec val':>9s}{'F1 val':>8s}  params")
    best_name, best_f = None, -1
    for name, (ftr, p, fn) in rules.items():
        prec, rec, f, *_ = prf(yv, fn(Xv, p))
        print(f"{name:24s}{ftr:10.3f}{prec:10.3f}{rec:9.3f}{f:8.3f}  {p}")
        if ftr > best_f:
            best_name, best_f = name, ftr

    _, p, fn = rules[best_name]
    pred = fn(Xv, p)
    *_, tp, fpn, fnn = prf(yv, pred)
    print(f"\nRegra escolhida (por F1 no treino): {best_name}")
    print(f"Validação: TP={tp} FP={fpn} FN={fnn}")
    print("Falsos positivos por classe (tecido rejeitado):", dict(Counter(yv_l[~yv & pred])))

    fn_idx = np.where(yv & ~pred)[0]
    if len(fn_idx):
        print(f"Falsos negativos (fundo aceito): desvio {Xv[fn_idx,1].min():.1f}-{Xv[fn_idx,1].max():.1f}, "
              f"brilho {Xv[fn_idx,0].min():.0f}-{Xv[fn_idx,0].max():.0f}")

    out.mkdir(exist_ok=True)
    save_grid(pv, yv_l, np.where(~yv & pred)[0], root, out / "bg_falsos_positivos.png", "Falsos positivos (tecido rejeitado)")
    save_grid(pv, yv_l, fn_idx, root, out / "bg_falsos_negativos.png", "Falsos negativos (fundo aceito)")

    Path(args.config).parent.mkdir(exist_ok=True)
    json.dump({"rule": best_name, "params": p, "fit_split": "train"},
              open(args.config, "w"), indent=2)
    print(f"\nSalvo: {args.config} e grades em {out}/")

if __name__ == "__main__":
    main()