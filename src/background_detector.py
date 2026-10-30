import argparse, csv, json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

COLS = {"brilho": 0, "desvio_cinza": 1, "iqr_cinza": 2}

# carregamendo dos dados
def load(csv_path, split):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["split"] == split]
    paths = [r["path"] for r in rows]
    labels = np.array([r["label"] for r in rows]) # cria um array contendo a classificacao das imagens
    X = np.array([[float(r[c]) for c in COLS] for r in rows])
    return paths, labels, X

# classificacao básica conforme as caracetriscas da imagem (brilho, desvio_ciza e iqr_cinza)
def predict(X, p):
    pred = X[:, COLS[p["medida"]]] < p["t"]          # pouca textura
    if "lo" in p:
        pred = pred | (X[:, 0] < p["lo"]) | (X[:, 0] > p["hi"])   # muito escuro/claro
    return pred

# avaliacao das métricas de classificacao
def prf(y, pred):
    tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f, tp, fp, fn

# ajuste para encontrar os melhores parametros
def fit(y, X, medida, com_brilho):
    best = (-1, None)
    ts = np.arange(0.5, 31, 0.5)
    los = range(60, 125, 5) if com_brilho else [None]
    his = range(190, 256, 5) if com_brilho else [None]
    for t in ts:
        for lo in los:
            for hi in his:
                p = {"medida": medida, "t": float(t)}
                if com_brilho:
                    p.update(lo=lo, hi=hi)
                f = prf(y, predict(X, p))[2]
                if f > best[0]:
                    best = (f, p)
    return best

def motivo(x, p):

    m = []
    if x[COLS[p["medida"]]] < p["t"]:
        m.append(f"{p['medida']}={x[COLS[p['medida']]]:.1f}<{p['t']}")
    if "lo" in p and x[0] < p["lo"]:
        m.append(f"brilho={x[0]:.0f}<{p['lo']}")
    if "lo" in p and x[0] > p["hi"]:
        m.append(f"brilho={x[0]:.0f}>{p['hi']}")
    return ", ".join(m)

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
    specs = {
        "r1_desvio": ("desvio_cinza", False),
        "r2_desvio+brilho": ("desvio_cinza", True),
        "r3_iqr": ("iqr_cinza", False),
        "r4_iqr+brilho": ("iqr_cinza", True),
    }
    fitted = {n: fit(ytr, Xtr, m, b) for n, (m, b) in specs.items()}

    print(f"{'regra':20s}{'F1 treino':>10s}{'prec val':>10s}{'rec val':>9s}{'F1 val':>8s}  params")
    best_name, best_f = None, -1
    for n, (ftr, p) in fitted.items():
        prec, rec, f, *_ = prf(yv, predict(Xv, p))
        print(f"{n:20s}{ftr:10.3f}{prec:10.3f}{rec:9.3f}{f:8.3f}  {p}")
        if ftr > best_f:
            best_name, best_f = n, ftr

    p = fitted[best_name][1]
    pred = predict(Xv, p)
    *_, tp, fpn, fnn = prf(yv, pred)
    print(f"\nRegra escolhida (por F1 no treino): {best_name}")
    print(f"Validação: TP={tp} FP={fpn} FN={fnn}")

    print("\nFalsos positivos (tecido rejeitado) e motivo:")
    for i in np.where(~yv & pred)[0]:
        print(f"  {yv_l[i]:24s} {motivo(Xv[i], p)}")

    fn_idx = np.where(yv & ~pred)[0]
    if len(fn_idx):
        print(f"\nFalsos negativos (fundo aceito): mediana desvio={np.median(Xv[fn_idx,1]):.1f} "
              f"iqr={np.median(Xv[fn_idx,2]):.1f} brilho={np.median(Xv[fn_idx,0]):.0f}")

    out.mkdir(exist_ok=True)
    save_grid(pv, yv_l, np.where(~yv & pred)[0], root, out / "bg_falsos_positivos.png", "Falsos positivos (tecido rejeitado)")
    save_grid(pv, yv_l, fn_idx, root, out / "bg_falsos_negativos.png", "Falsos negativos (fundo aceito)")

    Path(args.config).parent.mkdir(exist_ok=True)
    json.dump({"rule": best_name, "params": p, "fit_split": "train"},
              open(args.config, "w"), indent=2)
    print(f"\nSalvo: {args.config} e grades em {out}/")


if __name__ == "__main__":
    main()