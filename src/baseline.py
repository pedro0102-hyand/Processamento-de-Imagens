import argparse, json, platform, time
from pathlib import Path
import numpy as np
import sklearn
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from color_norm import rgb2lab

CHANNELS = ["L", "a", "b", "S"] 
STATS = ["mean", "std", "p10", "p50", "p90"]
NAMES = [f"{c}_{s}" for c in CHANNELS for s in STATS]

# extract features from a tile: mean, std, p10, p50, p90 for L, a, b and S (saturation)
def features(path):

    rgb = np.asarray(Image.open(path).convert("RGB")) 
    lab = rgb2lab(rgb).reshape(-1, 3)
    sat = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32)[..., 1].ravel()

    f = []

    for c in (lab[:, 0], lab[:, 1], lab[:, 2], sat):
        p10, p50, p90 = np.percentile(c, [10, 50, 90])
        f += [c.mean(), c.std(), p10, p50, p90]
    return f

# create a random forest classifier with the given seed and parameters
def rf(seed, params):
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed, **params)

# compute the macro F1 score of a model trained on (Xtr, ytr) and evaluated on (Xv, yv)
def f1m(model, Xtr, ytr, Xv, yv):
    model.fit(Xtr, ytr)
    return f1_score(yv, model.predict(Xv), average="macro")

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results/baseline_cor.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    root = Path(args.root)

    # read the manifest and extract features and labels
    rows = [json.loads(l) for l in open(args.manifest)]
    rows = [r for r in rows if r["split"] in ("train", "val")]
    t0 = time.time()
    X = np.array([features(root / r["path"]) for r in rows])
    t_feat = time.time() - t0
    y = np.array([r["label"] for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    Xtr, ytr, Xv, yv = X[tr], y[tr], X[~tr], y[~tr]
    print(f"Features: {X.shape[1]} por imagem, {tr.sum()} treino, {(~tr).sum()} validação "
          f"({t_feat:.0f}s para extrair)")

    # baseline: dummy classifier (most frequent class)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    f_dummy = f1_score(yv, dummy.predict(Xv), average="macro")

    # grid search for the best hyperparameters of a random forest classifier
    grid = [dict(max_depth=d, min_samples_leaf=m) for d in (None, 10, 20) for m in (1, 3)]
    scores = [(f1m(rf(args.seed, p), Xtr, ytr, Xv, yv), p) for p in grid]
    print("\nHiperparâmetros (F1 macro na validação):")
    for s, p in scores:
        print(f"  {p}  ->  {s:.3f}")
    best_f, best_p = max(scores, key=lambda s: s[0])

    # best model trained on the full training set
    idx = [NAMES.index(n) for n in ("L_mean", "a_mean", "b_mean")]
    f_mean = f1m(rf(args.seed, best_p), Xtr[:, idx], ytr, Xv[:, idx], yv)

    # best model trained on the full training set with different seeds
    f_seeds = [f1m(rf(s, best_p), Xtr, ytr, Xv, yv) for s in args.seeds]

    t0 = time.time()
    model = rf(args.seed, best_p).fit(Xtr, ytr)
    t_fit = time.time() - t0
    pred = model.predict(Xv)
    f_final = f1_score(yv, pred, average="macro")

    print(f"\nMelhores hiperparâmetros: {best_p}")
    print(f"\n{'configuração':38s}{'F1 macro (val)':>15s}")
    print(f"{'classe mais frequente':38s}{f_dummy:15.3f}")
    print(f"{'floresta só com cor média (L,a,b)':38s}{f_mean:15.3f}")
    print(f"{'floresta com features de cor':38s}{f_final:15.3f}")
    print(f"\nVariação entre seeds {args.seeds}: {np.mean(f_seeds):.3f} ± {np.std(f_seeds):.3f}")
    print("\nPor classe (validação):")
    print(classification_report(yv, pred, digits=3, zero_division=0))
    imp = np.argsort(model.feature_importances_)[::-1][:5]
    print("Features mais importantes:", ", ".join(f"{NAMES[i]} ({model.feature_importances_[i]:.2f})" for i in imp))

    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({
        "experimento": "baseline_cor_random_forest", "pre_processamento": "nenhum",
        "seed": args.seed, "hiperparametros": best_p, "n_estimators": 200,
        "f1_macro_val": float(f_final),
        "f1_macro_val_seeds": [float(f) for f in f_seeds],
        "f1_macro_dummy": float(f_dummy), "f1_macro_cor_media": float(f_mean),
        "por_classe": classification_report(yv, pred, output_dict=True, zero_division=0),
        "tempo_features_s": round(t_feat, 1), "tempo_treino_s": round(t_fit, 1),
        "versoes": {"python": platform.python_version(), "numpy": np.__version__,
                    "scikit-learn": sklearn.__version__},
    }, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSalvo: {args.out}")

if __name__ == "__main__":
    main()