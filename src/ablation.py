import argparse, io, json, platform, time
from pathlib import Path
import numpy as np
import sklearn
from PIL import Image
from sklearn.metrics import f1_score
from baseline import features, rf                    
from background_detector import predict as bg_predict  
from color_norm import normalize_image                

GRID = [dict(max_depth=d, min_samples_leaf=m) for d in (None, 10, 20) for m in (1, 3)]

# gray-level measures used by the background rule: brightness, std and IQR (same order as background_detector.COLS)
def measure(rgb):

    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32)
    q25, q75 = np.percentile(gray, [25, 75])
    return [gray.mean(), gray.std(), q75 - q25]

# baseline.features() reads a file, so a normalized array goes through an in-memory PNG (lossless)
def features_from_array(rgb):

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)
    return features(buf)

# one configuration: pick hyperparameters on validation, repeat with other seeds, final predictions (seed fixed)
def run(Xall, tr, ytr, yv, mask, post, seed, seeds):

    Xt, yt = Xall[tr], ytr
    if mask is not None:
        Xt, yt = Xt[mask], yt[mask]
    Xe = Xall[~tr]

    def score(model):
        return f1_score(yv, post(model.predict(Xe)), average="macro")

    scores = [(score(rf(seed, p).fit(Xt, yt)), p) for p in GRID]
    _, best_p = max(scores, key=lambda s: s[0])
    f_seeds = [score(rf(s, best_p).fit(Xt, yt)) for s in seeds]
    pred = post(rf(seed, best_p).fit(Xt, yt).predict(Xe))
    return dict(params=best_p, f1=f1_score(yv, pred, average="macro"), f_seeds=f_seeds, pred=pred)

# paired bootstrap on the validation set: 95% interval of (F1 macro of config - F1 macro of reference)
def boot_delta(yv, p_ref, p_cfg, n=500, seed=0):

    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(yv), len(yv))
        d.append(f1_score(yv[i], p_cfg[i], average="macro") - f1_score(yv[i], p_ref[i], average="macro"))
    return np.percentile(d, [2.5, 97.5])

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--root", default="data")
    ap.add_argument("--bg-config", default="configs/background_rule.json")
    ap.add_argument("--norm-config", default="configs/color_norm.json")
    ap.add_argument("--out", default="results/ablacao.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    root = Path(args.root)

    bg = json.load(open(args.bg_config))
    nc = json.load(open(args.norm_config))
    ref, (gmin, gmax) = nc["ref"], nc["gain_clip"]

    # train and val only, test is never read here
    rows = [json.loads(l) for l in open(args.manifest)]
    rows = [r for r in rows if r["split"] in ("train", "val")]
    y = np.array([r["label"] for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    ytr, yv = y[tr], y[~tr]
    imgs = [np.asarray(Image.open(root / r["path"]).convert("RGB")) for r in rows]

    # background rule is applied on the ORIGINAL images (before any normalization)
    flag = bg_predict(np.array([measure(a) for a in imgs]), bg["params"])
    flag_v = flag[~tr]
    print(f"Regra de fundo ({bg['rule']}): {bg['params']}")
    print(f"  validação: rejeita {flag_v.sum()} imagens, {(flag_v & (yv == 'background')).sum()} são background de verdade; "
          f"{((~flag_v) & (yv == 'background')).sum()} fundos escapam")

    # features for the three versions of the images
    t0 = time.time()
    X_none = np.array([features(root / r["path"]) for r in rows])
    X_ab = np.array([features_from_array(normalize_image(a, ref, "ab", gmin, gmax)) for a in imgs])
    X_full = np.array([features_from_array(normalize_image(a, ref, "full", gmin, gmax)) for a in imgs])
    t_feat = time.time() - t0
    print(f"  features extraídas em {t_feat:.0f}s")

    not_bg = ytr != "background"
    override = lambda p: np.where(flag_v, "background", p)
    cfgs = {
        "1 sem pré-proc.":          (X_none, None, lambda p: p),
        "2 fundo: regra + 8 cl.":   (X_none, not_bg, override),   # model never sees the background class
        "3 fundo: regra sobrepõe":  (X_none, None, override),     # 9-class model, rule overrides its output
        "4 normaliz. ab":           (X_ab, None, lambda p: p),
        "5 normaliz. full":         (X_full, None, lambda p: p),
    }
    res = {n: run(X, tr, ytr, yv, m, post, args.seed, args.seeds) for n, (X, m, post) in cfgs.items()}

    names = list(cfgs)
    ref_pred = res[names[0]]["pred"]
    print(f"\n{'configuração':26s}{'F1 macro':>9s}{'seeds (média ± dp)':>22s}{'Δ vs 1':>9s}{'IC 95% do Δ':>20s}")
    for n in names:
        r = res[n]
        if n == names[0]:
            r["delta"], r["ic"] = 0.0, (0.0, 0.0)
        else:
            r["delta"], r["ic"] = r["f1"] - res[names[0]]["f1"], boot_delta(yv, ref_pred, r["pred"])
        print(f"{n:26s}{r['f1']:9.3f}{np.mean(r['f_seeds']):14.3f} ± {np.std(r['f_seeds']):.3f}"
              f"{r['delta']:+9.3f}   [{r['ic'][0]:+.3f}, {r['ic'][1]:+.3f}]")

    classes = sorted(set(yv))
    f_cls = {n: f1_score(yv, res[n]["pred"], labels=classes, average=None) for n in names}
    print(f"\n{'F1 por classe (val)':38s}" + "".join(f"{n.split()[0]:>7s}" for n in names))
    for i, c in enumerate(classes):
        print(f"{c:38s}" + "".join(f"{f_cls[n][i]:7.3f}" for n in names))

    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({
        "experimento": "ablacao_preprocessamento_random_forest_cor_textura",
        "seed": args.seed, "seeds_extras": args.seeds, "n_estimators": 200,
        "regra_fundo": bg, "normalizacao": {"metodo": nc["method"], "gain_clip": nc["gain_clip"]},
        "resultados": {n: {"hiperparametros": res[n]["params"], "f1_macro_val": float(res[n]["f1"]),
                           "f1_macro_val_seeds": [float(f) for f in res[n]["f_seeds"]],
                           "delta_vs_1": float(res[n]["delta"]), "ic95_delta": [float(v) for v in res[n]["ic"]],
                           "f1_por_classe": dict(zip(classes, map(float, f_cls[n])))} for n in names},
        "tempo_features_s": round(t_feat, 1),
        "versoes": {"python": platform.python_version(), "numpy": np.__version__,
                    "scikit-learn": sklearn.__version__},
    }, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSalvo: {args.out}")

if __name__ == "__main__":
    main()