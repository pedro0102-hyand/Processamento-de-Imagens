import argparse, json, platform
from collections import Counter
from pathlib import Path
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from baseline import rf
from color_norm import rgb2lab
from evaluate import extract, ABL     

# mean b channel (LAB) of each tile: negative = bluish (typical H&E), positive = yellowish/orange
def mean_b(rows, root):
    return np.array([rgb2lab(np.asarray(Image.open(root / r["path"]).convert("RGB"))).reshape(-1, 3)[:, 2].mean()
                     for r in rows])

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--root", default="data")
    ap.add_argument("--norm-config", default="configs/color_norm.json")
    ap.add_argument("--abl-config", default="results/ablacao.json")
    ap.add_argument("--pred", default="results/predicoes_teste.jsonl")
    ap.add_argument("--out", default="results/analise_erros.json")
    ap.add_argument("--fig-dir", default="eda_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=4, help="number of confusion pairs to inspect")
    args = ap.parse_args()
    root = Path(args.root)

    nc = json.load(open(args.norm_config))
    ref, (gmin, gmax) = nc["ref"], nc["gain_clip"]
    hp = json.load(open(args.abl_config))["resultados"][ABL["ab"]]["hiperparametros"]

    rows = [json.loads(l) for l in open(args.manifest)]
    tr_rows = [r for r in rows if r["split"] == "train"]
    va_rows = [r for r in rows if r["split"] == "val"]
    preds = [json.loads(l) for l in open(args.pred)]          
    yte = np.array([p["label"] for p in preds])
    pte = np.array([p["pred"] for p in preds])
    conf = np.array([p["conf"] for p in preds])
    ytr = np.array([r["label"] for r in tr_rows])
    yva = np.array([r["label"] for r in va_rows])
    classes = sorted(set(ytr) | set(yva) | set(yte))
    K = len(classes)

    # validation predictions of the train-only model (same pipeline and hyperparameters, seed fixed)
    Xtr = extract(tr_rows, root, "ab", ref, gmin, gmax)
    Xva = extract(va_rows, root, "ab", ref, gmin, gmax)
    pva = rf(args.seed, hp).fit(Xtr, ytr).predict(Xva)
    print(f"Modelo só treino, validação: F1 macro {f1_score(yva, pva, average='macro'):.3f} "
          f"(deve bater com a ablação)  |  teste (modelo principal): {f1_score(yte, pte, average='macro'):.3f}")
    print("Atenção: validação = modelo treinado só no treino; teste = modelo principal (treino + validação).")

    cm_te = confusion_matrix(yte, pte, labels=classes)
    cm_va = confusion_matrix(yva, pva, labels=classes)
    rate = lambda cm, i, j: cm[i, j] / max(cm[i].sum(), 1)

    # 1) the most frequent confusions: validation rate vs test rate
    off = sorted(((cm_te[i, j], i, j) for i in range(K) for j in range(K) if i != j), reverse=True)[:args.top]
    print(f"\n1) Taxa de confusão (% da classe real): validação vs teste")
    print(f"{'real → previsto':62s}{'val %':>8s}{'teste %':>9s}{'n teste':>9s}")
    pairs = []
    for n, i, j in off:
        print(f"{classes[i] + ' → ' + classes[j]:62s}{100 * rate(cm_va, i, j):8.1f}{100 * rate(cm_te, i, j):9.1f}{n:9d}")
        pairs.append(dict(real=classes[i], pred=classes[j], val_pct=100 * rate(cm_va, i, j),
                          teste_pct=100 * rate(cm_te, i, j), n_teste=int(n)))

    # 2) validation reweighted with the test class proportions (label-shift check)
    n_va, n_te = Counter(yva), Counter(yte)
    w = np.array([(n_te[c] / len(yte)) / (n_va[c] / len(yva)) for c in yva])
    _, _, f_va, _ = precision_recall_fscore_support(yva, pva, labels=classes, zero_division=0)
    _, _, f_vw, _ = precision_recall_fscore_support(yva, pva, labels=classes, sample_weight=w, zero_division=0)
    _, _, f_te, _ = precision_recall_fscore_support(yte, pte, labels=classes, zero_division=0)
    rng = np.random.default_rng(0)
    boots = np.array([f1_score(yte[i], pte[i], labels=classes, average=None, zero_division=0)
                      for i in (rng.integers(0, len(yte), len(yte)) for _ in range(500))])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    print(f"\n2) F1 por classe: validação, validação com as proporções do teste, e teste (IC 95%)")
    print(f"{'classe':38s}{'val':>7s}{'val repond.':>13s}{'teste':>8s}{'IC 95% teste':>20s}")
    for i, c in enumerate(classes):
        print(f"{c:38s}{f_va[i]:7.3f}{f_vw[i]:13.3f}{f_te[i]:8.3f}   [{lo[i]:.3f}, {hi[i]:.3f}]")
    print(f"{'macro':38s}{f_va.mean():7.3f}{f_vw.mean():13.3f}{f_te.mean():8.3f}")

    # 3) color shortcut check: tiles with positive mean b (yellowish/orange) in train and test
    b_tr, b_te = mean_b(tr_rows, root), mean_b([{"path": p["path"]} for p in preds], root)
    print(f"\n3) Tiles com b médio > 0 (amarelados/alaranjados; a referência de cor tem b = {ref['mean'][2]:.1f})")
    print(f"{'classe':38s}{'treino':>14s}{'teste':>14s}")
    color = {}
    for c in classes:
        a, b = (ytr == c), (yte == c)
        na, nb = int((b_tr[a] > 0).sum()), int((b_te[b] > 0).sum())
        color[c] = dict(treino=na, n_treino=int(a.sum()), teste=nb, n_teste=int(b.sum()))
        print(f"{c:38s}{na:6d} /{a.sum():5d}{nb:6d} /{b.sum():5d}")
    sel = b_te > 0
    if sel.any():
        print(f"\n   Teste, tiles com b > 0 (n = {sel.sum()}), acerto {100 * (yte[sel] == pte[sel]).mean():.0f}% "
              f"(geral {100 * (yte == pte).mean():.0f}%)")
        for c in classes:
            m = sel & (yte == c)
            if m.any():
                wrong = Counter(pte[m & (pte != c)]).most_common(1)
                print(f"   {c:38s} n={int(m.sum()):3d}  corretos={int((pte[m] == c).sum()):3d}"
                      + (f"  erro mais comum → {wrong[0][0]} ({wrong[0][1]})" if wrong else ""))

    # 4) random errors of the most frequent confusions (not only the most confident ones)
    Path(args.fig_dir).mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for p in pairs:
        ie = np.where((yte == p["real"]) & (pte == p["pred"]))[0]
        ik = np.where((yte == p["real"]) & (pte == p["real"]))[0]
        if len(ie) == 0:
            continue
        pe = rng.choice(ie, min(12, len(ie)), replace=False)
        pk = rng.choice(ik, min(6, len(ik)), replace=False) if len(ik) else []
        fig, axes = plt.subplots(3, 6, figsize=(11, 6.8), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel()[:12], pe):
            ax.imshow(Image.open(root / preds[i]["path"]).convert("RGB"))
            ax.set_title(f"erro, conf. {conf[i]:.2f}", fontsize=7)
        for ax, i in zip(axes.ravel()[12:], pk):
            ax.imshow(Image.open(root / preds[i]["path"]).convert("RGB"))
            ax.set_title("correto", fontsize=7)
        fig.suptitle(f"{p['real']} → {p['pred']} ({p['n_teste']} casos): 12 erros aleatórios e 6 acertos da classe real", fontsize=9)
        plt.tight_layout()
        plt.savefig(Path(args.fig_dir) / f"erro_{p['real'][:12]}_para_{p['pred'][:12]}.png", dpi=120)
        plt.close()
    print(f"\n4) Figuras por par de confusão salvas em {args.fig_dir}/ (erro_<real>_para_<previsto>.png)")

    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({"observacao": "Análise de erros feita depois do teste. Não altera o pipeline: só gera recomendações.",
               "confusoes": pairs,
               "f1_por_classe": {c: {"val": float(f_va[i]), "val_reponderado": float(f_vw[i]), "teste": float(f_te[i]),
                                     "ic95_teste": [float(lo[i]), float(hi[i])]} for i, c in enumerate(classes)},
               "tiles_b_positivo": color,
               "versoes": {"python": platform.python_version(), "numpy": np.__version__,
                           "scikit-learn": sklearn.__version__}},
              open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"Salvo: {args.out}")

if __name__ == "__main__":
    main()