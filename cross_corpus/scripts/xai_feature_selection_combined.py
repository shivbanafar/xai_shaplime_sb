"""
XAI-based feature selection (SHAP + LIME combined) for cross-corpus SER.

Protocol (no test leakage):
  1. Per-corpus z-normalize SAVEE (train) and EMO-DB (test).
  2. Fit a linear SVM on SAVEE (full 2048 WavLM dims).
  3. Compute TWO global feature-importance rankings, ON SAVEE ONLY:
       - SHAP : shap.LinearExplainer, mean|SHAP| summed over classes.
       - LIME : LimeTabularExplainer, mean|weight| aggregated over many SAVEE
                instances (global aggregation of local explanations).
  4. Min-max normalize each ranking and AVERAGE -> combined importance.
  5. Keep top-k combined dims; retrain MAML from scratch on SAVEE's top-k
     (seed 42, identical hyper-params) and evaluate on EMO-DB.
     Sweep k in {128,256,512,1024,2048}.
  6. Compare to the full-feature MAML (k=2048).
"""
import os, sys, warnings
import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")

BASE_DIR = "/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
sys.path.insert(0, os.path.join(BASE_DIR, "cross_corpus", "scripts"))
import cross_corpus_train as cc          # seeds np/torch=42, defines MLP/train_maml/...

OUT_DIR = os.path.join(BASE_DIR, "cross_corpus", "results")
SEED = 42
SHARED = cc.SHARED
SVM_C = 0.1
KS = [128, 256, 512, 1024, 2048]
LIME_N_INSTANCES = 150      # SAVEE instances aggregated for global LIME importance
LIME_N_SAMPLES = 1000       # perturbations per instance


def reseed(s=SEED):
    np.random.seed(s); torch.manual_seed(s)


def minmax(v):
    v = np.asarray(v, dtype=float)
    rng = v.max() - v.min()
    return (v - v.min()) / rng if rng > 0 else np.zeros_like(v)


def svm_proba_fn(svm):
    def f(X):
        d = svm.decision_function(X)
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return f


def shap_importance(svm, Zs):
    expl = shap.LinearExplainer(svm, Zs)
    sv = expl.shap_values(Zs)
    if isinstance(sv, list):
        return np.sum([np.mean(np.abs(c), axis=0) for c in sv], axis=0)
    sv = np.abs(sv)
    return sv.mean(axis=0).sum(axis=-1) if sv.ndim == 3 else sv.mean(axis=0)


def lime_importance(svm, Zs, n_feat):
    proba = svm_proba_fn(svm)
    expl = LimeTabularExplainer(
        Zs, mode="classification", discretize_continuous=False,
        feature_names=[str(i) for i in range(n_feat)], random_state=SEED)
    imp = np.zeros(n_feat)
    rng = np.random.RandomState(SEED)
    sub = rng.choice(len(Zs), min(LIME_N_INSTANCES, len(Zs)), replace=False)
    for n, i in enumerate(sub):
        lbl = int(np.argmax(proba(Zs[i:i+1])[0]))
        e = expl.explain_instance(Zs[i], proba, num_features=n_feat,
                                  num_samples=LIME_N_SAMPLES, labels=(lbl,))
        for fidx, w in e.as_map()[lbl]:
            imp[fidx] += abs(w)
        if (n + 1) % 30 == 0:
            print(f"      LIME {n+1}/{len(sub)} instances")
    return imp / len(sub)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    L = []
    def log(m=""):
        print(m); L.append(str(m))

    log("=" * 64)
    log("XAI FEATURE SELECTION (SHAP + LIME combined)  SAVEE -> EMO-DB")
    log("=" * 64)

    Xs, ys = cc.load("savee"); Xt, yt = cc.load("embodb")
    le = LabelEncoder().fit(SHARED)
    ys_e, yt_e = le.transform(ys), le.transform(yt)
    Zs = StandardScaler().fit_transform(Xs)
    Zt = StandardScaler().fit_transform(Xt)
    log(f"Device {device} | SAVEE {Xs.shape} | EMO-DB {Xt.shape}")

    svm = LinearSVC(C=SVM_C, max_iter=5000, random_state=SEED).fit(Zs, ys_e)
    log(f"SVM (full {Zs.shape[1]}-dim) train-acc={accuracy_score(ys_e, svm.predict(Zs)):.3f}")

    log("\n[SHAP] LinearExplainer on SAVEE ...")
    imp_shap = shap_importance(svm, Zs)
    log(f"  SHAP top-5 dims: {np.argsort(imp_shap)[::-1][:5].tolist()}")

    log(f"\n[LIME] aggregating {LIME_N_INSTANCES} SAVEE instances "
        f"(num_samples={LIME_N_SAMPLES}) ...")
    imp_lime = lime_importance(svm, Zs, Zs.shape[1])
    log(f"  LIME top-5 dims: {np.argsort(imp_lime)[::-1][:5].tolist()}")

    combined = minmax(imp_shap) + minmax(imp_lime)
    order = np.argsort(combined)[::-1]
    # rank-agreement diagnostic
    s_top = set(np.argsort(imp_shap)[::-1][:512].tolist())
    l_top = set(np.argsort(imp_lime)[::-1][:512].tolist())
    log(f"  combined top-5 dims: {order[:5].tolist()}")
    log(f"  SHAP/LIME top-512 overlap: {len(s_top & l_top)}/512")

    log("\n[SWEEP] retraining MAML on top-k combined dims")
    results = []
    for k in KS:
        idx = order[:k]
        reseed(SEED)
        m = cc.train_maml(Zs[:, idx], le.inverse_transform(ys_e), len(SHARED), device)
        mf = cc.fine_tune_maml(m, Zs[:, idx], ys_e, device); mf.eval()
        with torch.no_grad():
            pred = torch.argmax(
                mf(torch.tensor(Zt[:, idx], dtype=torch.float32, device=device)), 1
            ).cpu().numpy()
        acc = accuracy_score(yt_e, pred); f1 = f1_score(yt_e, pred, average="macro")
        tag = "  (full baseline)" if k == 2048 else ""
        log(f"  k={k:<5d}  acc={acc*100:5.1f}%   macroF1={f1:.3f}{tag}")
        results.append((k, acc, f1))

    log("\n" + "=" * 64)
    log("SUMMARY — MAML accuracy vs # SHAP+LIME-selected WavLM dims")
    log("=" * 64)
    log(f"{'k (top dims)':<16}{'Accuracy':>12}{'macro-F1':>12}")
    log("-" * 40)
    for k, acc, f1 in results:
        log(f"{k:<16}{acc*100:>11.1f}%{f1:>12.3f}")
    best = max(results, key=lambda r: r[1])
    full = [r for r in results if r[0] == 2048][0]
    log("-" * 40)
    log(f"Best: k={best[0]} at {best[1]*100:.1f}%  |  full(2048)={full[1]*100:.1f}%")
    log("Verdict: " + ("SELECTION HELPS" if best[0] != 2048 and best[1] > full[1]
                       else "full features best — selection does NOT beat baseline"))

    with open(os.path.join(OUT_DIR, "xai_feature_selection_combined.txt"), "w") as f:
        f.write("\n".join(L))
    log(f"\nWritten to {os.path.join(OUT_DIR, 'xai_feature_selection_combined.txt')}")


if __name__ == "__main__":
    main()
