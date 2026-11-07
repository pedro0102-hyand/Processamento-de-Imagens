import argparse, json, platform, time
from pathlib import Path
import numpy as np
import sklearn
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from baseline import features, rf, NAMES
from ablation import features_from_array, boot_delta
from color_norm import normalize_image

# ablation config name for each normalization mode (used to read the hyperparameters chosen on validation)
ABL = {"ab": "4 normaliz. ab", "none": "1 sem pré-proc.", "full": "5 normaliz. full"}

# features of every row, with the given normalization mode applied before the extraction
def extract(rows, root, mode, ref, gmin, gmax):
    out = []
    for r in rows:
        if mode == "none":
            out.append(features(root / r["path"]))
        else:
            a = np.asarray(Image.open(root / r["path"]).convert("RGB"))
            out.append(features_from_array(normalize_image(a, ref, mode, gmin, gmax)))
    return np.array(out)

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--root", default="data")
    ap.add_argument("--norm-config", default="configs/color_norm.json")
    ap.add_argument("--abl-config", default="results/ablacao.json")
    ap.add_argument("--out", default="results/avaliacao_teste.json")
    ap.add_argument("--pred-out", default="results/predicoes_teste.jsonl")
    ap.add_argument("--model-out", default="models/modelo_final.joblib")
    ap.add_argument("--fig-dir", default="eda_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)

    # the test set is used ONCE: refuse to run again unless --force is given
    if Path(args.out).exists() and not args.force:
        print(f"{args.out} já existe: o teste já foi usado. Use --force só se souber o que está fazendo.")
        return

    nc = json.load(open(args.norm_config))
    ref, (gmin, gmax) = nc["ref"], nc["gain_clip"]
    abl = json.load(open(args.abl_config))["resultados"]
    hp = {m: abl[n]["hiperparametros"] for m, n in ABL.items()}

    rows = [json.loads(l) for l in open(args.manifest)]
    y = np.array([r["label"] for r in rows])
    split = np.array([r["split"] for r in rows])
    te, tr_only, tr_val = split == "test", split == "train", split != "test"
    classes = sorted(set(y))
    yte = y[te]

    t0 = time.time()
    X = {m: extract(rows, root, m, ref, gmin, gmax) for m in ABL}
    t_feat = time.time() - t0
    print(f"{te.sum()} imagens de teste, {tr_val.sum()} de treino+validação ({t_feat:.0f}s para extrair features)")

    # main model: normalization ab, trained on train + val. The others are secondary analyses.
    plan = {
        "principal: ab, treino+val":    ("ab", tr_val),
        "ab, só treino":                ("ab", tr_only),
        "sem pré-proc., treino+val":    ("none", tr_val),
        "full, treino+val":             ("full", tr_val),
    }
    
    res = {}
    for name, (mode, mask) in plan.items():
        t0 = time.time()
        model = rf(args.seed, hp[mode]).fit(X[mode][mask], y[mask])
        t_fit = time.time() - t0
        t0 = time.time()
        proba = model.predict_proba(X[mode][te])
        t_pred = time.time() - t0
        pred = model.classes_[proba.argmax(1)]
        res[name] = dict(model=model, proba=proba, pred=pred, t_fit=t_fit, t_pred=t_pred,
                         f1=f1_score(yte, pred, average="macro"), acc=accuracy_score(yte, pred))

    main_name = list(plan)[0]
    main = res[main_name]
    rng = np.random.default_rng(0)
    boot = [f1_score(yte[i], main["pred"][i], average="macro")
            for i in (rng.integers(0, len(yte), len(yte)) for _ in range(500))]
    ic = np.percentile(boot, [2.5, 97.5])

    val = abl[ABL["ab"]]
    print(f"\nReferência (validação, ab, treinado só no treino): F1 macro {val['f1_macro_val']:.3f} "
          f"(média de 5 seeds {np.mean(val['f1_macro_val_seeds']):.3f})")
    print(f"\n{'configuração (no teste)':30s}{'F1 macro':>10s}{'acurácia':>10s}")
    for n, r in res.items():
        print(f"{n:30s}{r['f1']:10.3f}{r['acc']:10.3f}")
    print(f"\nModelo principal: F1 macro {main['f1']:.3f}, IC 95% [{ic[0]:.3f}, {ic[1]:.3f}] (bootstrap no teste)")

    # secondary question: does normalization help on the test patients? (paired bootstrap)
    d_none = boot_delta(yte, res["sem pré-proc., treino+val"]["pred"], main["pred"])
    d_full = boot_delta(yte, res["full, treino+val"]["pred"], main["pred"])
    print(f"Δ F1 (ab − sem pré-proc.): {main['f1'] - res['sem pré-proc., treino+val']['f1']:+.3f}  IC 95% [{d_none[0]:+.3f}, {d_none[1]:+.3f}]")
    print(f"Δ F1 (ab − full):          {main['f1'] - res['full, treino+val']['f1']:+.3f}  IC 95% [{d_full[0]:+.3f}, {d_full[1]:+.3f}]")

    # per-class metrics of the main model
    p, rc, f, sup = precision_recall_fscore_support(yte, main["pred"], labels=classes, zero_division=0)
    print(f"\n{'classe (teste)':38s}{'precisão':>9s}{'recall':>8s}{'F1':>7s}{'n':>6s}")
    for i, c in enumerate(classes):
        print(f"{c:38s}{p[i]:9.3f}{rc[i]:8.3f}{f[i]:7.3f}{sup[i]:6d}")

    # confusion matrix
    cm = confusion_matrix(yte, main["pred"], labels=classes)
    print("\nMatriz de confusão (linhas = real, colunas = previsto):")
    print(" " * 4 + "".join(f"{j:5d}" for j in range(len(classes))))
    for i, c in enumerate(classes):
        print(f"{i:3d} " + "".join(f"{v:5d}" for v in cm[i]) + f"   {c}")
    off = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j]
    print("Confusões mais frequentes:", "; ".join(f"{a} → {b} ({n})" for n, a, b in sorted(off, reverse=True)[:5]))

    Path(args.fig_dir).mkdir(exist_ok=True)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("previsto"); ax.set_ylabel("real"); ax.set_title("Matriz de confusão (teste)")
    plt.tight_layout(); plt.savefig(Path(args.fig_dir) / "matriz_confusao_teste.png", dpi=120); plt.close()

    # most confident errors of the main model
    te_rows = [r for r, t in zip(rows, te) if t]
    err = np.where(main["pred"] != yte)[0]
    err = err[np.argsort(-main["proba"][err].max(1))][:18]
    fig, axes = plt.subplots(3, 6, figsize=(11, 6.5), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), err):
        ax.imshow(Image.open(root / te_rows[i]["path"]).convert("RGB"))
        ax.set_title(f"{yte[i][:11]} → {main['pred'][i][:11]}\nconf. {main['proba'][i].max():.2f}", fontsize=6)
    fig.suptitle("Erros do modelo principal com maior confiança (teste)", fontsize=9)
    plt.tight_layout(); plt.savefig(Path(args.fig_dir) / "erros_teste.png", dpi=120); plt.close()

    # artifacts: predictions, model, results
    Path(args.pred_out).parent.mkdir(exist_ok=True)
    with open(args.pred_out, "w") as fh:
        for r, pr, pb in zip(te_rows, main["pred"], main["proba"]):
            fh.write(json.dumps({"path": r["path"], "label": r["label"], "pred": str(pr),
                                 "conf": round(float(pb.max()), 4)}, ensure_ascii=False) + "\n")
    Path(args.model_out).parent.mkdir(exist_ok=True)
    joblib.dump({"model": main["model"], "normalizacao": "ab", "ref": ref, "gain_clip": [gmin, gmax],
                 "features": NAMES, "classes": classes}, args.model_out)

    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({
        "declaracao": "Pipeline (normalização ab, cor+textura, random forest) e treino com treino+validação "
                      "decididos antes de usar o teste. O teste foi avaliado uma única vez.",
        "seed": args.seed, "hiperparametros": hp, "n_teste": int(te.sum()),
        "principal": {"f1_macro": float(main["f1"]), "acuracia": float(main["acc"]),
                      "ic95_f1_macro": [float(v) for v in ic],
                      "por_classe": {c: {"precisao": float(p[i]), "recall": float(rc[i]), "f1": float(f[i]),
                                         "n": int(sup[i])} for i, c in enumerate(classes)},
                      "matriz_confusao": {"classes": classes, "valores": cm.tolist()},
                      "tempo_treino_s": round(main["t_fit"], 2), "tempo_predicao_s": round(main["t_pred"], 3)},
        "secundarias": {n: {"f1_macro": float(r["f1"]), "acuracia": float(r["acc"])} for n, r in res.items()},
        "delta_ab_vs_sem": {"delta": float(main["f1"] - res["sem pré-proc., treino+val"]["f1"]),
                            "ic95": [float(v) for v in d_none]},
        "delta_ab_vs_full": {"delta": float(main["f1"] - res["full, treino+val"]["f1"]),
                             "ic95": [float(v) for v in d_full]},
        "referencia_validacao": {"f1_macro": val["f1_macro_val"], "f1_macro_seeds": val["f1_macro_val_seeds"]},
        "tempo_features_s": round(t_feat, 1),
        "versoes": {"python": platform.python_version(), "numpy": np.__version__,
                    "scikit-learn": sklearn.__version__},
    }, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSalvo: {args.out}, {args.pred_out}, {args.model_out} e figuras em {args.fig_dir}/")

if __name__ == "__main__":
    main()