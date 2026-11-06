import argparse, json, platform, time
from pathlib import Path
import numpy as np
import sklearn
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from color_norm import rgb2lab
import skimage
from skimage.feature import local_binary_pattern
from skimage.filters import sobel

# feature names: 20 color features (L, a, b from LAB and S from HSV) + 32 texture features
COR = [f"{c}_{s}" for c in ["L", "a", "b", "S"] for s in ["mean", "std", "p10", "p50", "p90"]]
TEX = ([f"lbp8_{i}" for i in range(10)] + [f"lbp16_{i}" for i in range(18)]
       + ["grad_mean", "grad_std", "grad_p50", "grad_p90"])

NAMES = COR + TEX
IDX_COR = list(range(len(COR)))
IDX_TEX = list(range(len(COR), len(NAMES)))
IDX_MEDIA = [COR.index(n) for n in ("L_mean", "a_mean", "b_mean")]

# extract features from a tile:
# color: mean, std, p10, p50, p90 for L, a, b and S (saturation)
# texture: uniform LBP histograms (radius 1 and 2) + Sobel gradient magnitude stats
def features(path):

    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img)
    lab = rgb2lab(rgb).reshape(-1, 3)
    sat = np.asarray(img.convert("HSV"), dtype=np.float32)[..., 1].ravel()

    f = []

    for c in (lab[:, 0], lab[:, 1], lab[:, 2], sat):
        p10, p50, p90 = np.percentile(c, [10, 50, 90])
        f += [c.mean(), c.std(), p10, p50, p90]

    gray = np.asarray(img.convert("L"))

    for P, R in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        f += list(np.bincount(lbp.astype(int).ravel(), minlength=P + 2) / lbp.size)

    g = sobel(gray / 255.0).ravel()
    f += [g.mean(), g.std(), *np.percentile(g, [50, 90])]
    return f

# create a random forest classifier with the given seed and parameters
def rf(seed, params):
    return RandomForestClassifier(n_estimators = 200, class_weight = "balanced", n_jobs = -1, random_state = seed, **params)

# compute the macro F1 score of a model trained on (Xtr, ytr) and evaluated on (Xv, yv)
def f1m(model, Xtr, ytr, Xv, yv):
    model.fit(Xtr, ytr)
    return f1_score(yv, model.predict(Xv), average="macro")

# for one feature set: pick hyperparameters on validation, measure seed variation, train the final model
def run_set(cols, seed, seeds, grid, Xtr, ytr, Xv, yv):

    Xt, Xe = Xtr[:, cols], Xv[:, cols]
    scores = [(f1m(rf(seed, p), Xt, ytr, Xe, yv), p) for p in grid]
    _, best_p = max(scores, key=lambda s: s[0])
    f_seeds = [f1m(rf(s, best_p), Xt, ytr, Xe, yv) for s in seeds]

    t0 = time.time()
    model = rf(seed, best_p).fit(Xt, ytr)
    t_fit = time.time() - t0
    pred = model.predict(Xe)
    return dict(params=best_p, f1=f1_score(yv, pred, average="macro"), f_seeds=f_seeds,
                pred=pred, model=model, t_fit=t_fit)

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results/baseline_textura.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    root = Path(args.root)

    # read the manifest and extract features and labels (train and val only, test is never read here)
    rows = [json.loads(l) for l in open(args.manifest)]
    rows = [r for r in rows if r["split"] in ("train", "val")]
    t0 = time.time()
    X = np.array([features(root / r["path"]) for r in rows])
    t_feat = time.time() - t0
    y = np.array([r["label"] for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    Xtr, ytr, Xv, yv = X[tr], y[tr], X[~tr], y[~tr]
    print(f"Features: {len(COR)} de cor + {len(TEX)} de textura, {tr.sum()} treino, "
          f"{(~tr).sum()} validação ({t_feat:.0f}s para extrair)")

    # baseline: dummy classifier (most frequent class)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    f_dummy = f1_score(yv, dummy.predict(Xv), average="macro")

    # same model, same grid and same seed for each feature set
    grid = [dict(max_depth=d, min_samples_leaf=m) for d in (None, 10, 20) for m in (1, 3)]
    sets = {"cor média (L,a,b)": IDX_MEDIA, "cor": IDX_COR, "textura": IDX_TEX,
            "cor + textura": IDX_COR + IDX_TEX}
    res = {n: run_set(c, args.seed, args.seeds, grid, Xtr, ytr, Xv, yv) for n, c in sets.items()}

    print(f"\n{'configuração':22s}{'nº feat':>8s}{'F1 macro (val)':>16s}{'seeds (média ± dp)':>22s}  hiperparâmetros")
    print(f"{'classe mais frequente':22s}{0:8d}{f_dummy:16.3f}")
    for n, r in res.items():
        print(f"{n:22s}{len(sets[n]):8d}{r['f1']:16.3f}"
              f"{np.mean(r['f_seeds']):14.3f} ± {np.std(r['f_seeds']):.3f}  {r['params']}")

    # per-class F1: color vs color + texture
    classes = sorted(set(yv))
    f_cor = f1_score(yv, res["cor"]["pred"], labels=classes, average=None)
    f_all = f1_score(yv, res["cor + textura"]["pred"], labels=classes, average=None)
    print(f"\n{'F1 por classe (val)':38s}{'cor':>8s}{'cor+tex':>9s}{'diferença':>11s}")
    for c, a, b in zip(classes, f_cor, f_all):
        print(f"{c:38s}{a:8.3f}{b:9.3f}{b - a:+11.3f}")

    model = res["cor + textura"]["model"]
    names = [NAMES[i] for i in sets["cor + textura"]]
    top = np.argsort(model.feature_importances_)[::-1][:8]
    print("\nFeatures mais importantes (cor + textura):",
          ", ".join(f"{names[i]} ({model.feature_importances_[i]:.2f})" for i in top))

    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({
        "experimento": "baseline_random_forest_cor_vs_textura", "pre_processamento": "nenhum",
        "seed": args.seed, "seeds_extras": args.seeds, "n_estimators": 200,
        "f1_macro_dummy": float(f_dummy),
        "resultados": {n: {"n_features": len(sets[n]), "hiperparametros": r["params"],
                           "f1_macro_val": float(r["f1"]),
                           "f1_macro_val_seeds": [float(f) for f in r["f_seeds"]],
                           "tempo_treino_s": round(r["t_fit"], 1)} for n, r in res.items()},
        "f1_por_classe": {"cor": dict(zip(classes, map(float, f_cor))),
                          "cor + textura": dict(zip(classes, map(float, f_all)))},
        "tempo_features_s": round(t_feat, 1),
        "versoes": {"python": platform.python_version(), "numpy": np.__version__,
                    "scikit-learn": sklearn.__version__, "scikit-image": skimage.__version__},
    }, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSalvo: {args.out}")

if __name__ == "__main__":
    main()