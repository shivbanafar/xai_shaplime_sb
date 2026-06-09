"""
XAI-based feature selection for cross-corpus SER  (Option A: SHAP on SVM).

Protocol (no test leakage):
  1. Per-corpus z-normalize SAVEE (train) and EMO-DB (test).
  2. Fit a linear SVM on SAVEE (full 2048 WavLM dims) and compute SHAP feature
     importances with shap.LinearExplainer  ON SAVEE ONLY (never EMO-DB).
  3. Rank the 2048 dims by mean|SHAP| (summed over classes); keep the top-k.
  4. Retrain MAML from scratch on SAVEE's top-k dims (seed 42, identical
     hyper-params) and evaluate on EMO-DB.  Sweep k in {128,256,512,1024,2048}.
  5. Compare to the full-feature MAML (k=2048 == the 74.9% baseline).

Selection uses SOURCE (SAVEE) importances only, so this is an honest
"XAI importance -> select features -> train MAML" pipeline.
"""
import os, sys, warnings
import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import shap

warnings.filterwarnings("ignore")

BASE_DIR = "/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
sys.path.insert(0, os.path.join(BASE_DIR, "cross_corpus", "scripts"))
# reuse the exact MAML building blocks from the training script
import cross_corpus_train as cc  # sets np/torch seed 42 at import, defines MLP/train_maml/...

OUT_DIR = os.path.join(BASE_DIR, "cross_corpus", "results")
SEED = 42
SHARED = cc.SHARED
SVM_C = 0.1
KS = [128, 256, 512, 1024, 2048]


def reseed(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)


def shap_importance(Zs, ys_e, le):
    """Mean|SHAP| per WavLM dim from a linear SVM, computed on SAVEE only."""
    svm = LinearSVC(C=SVM_C, max_iter=5000, random_state=SEED).fit(Zs, ys_e)
    print(f"  SVM (full {Zs.shape[1]}-dim) fit for SHAP ranking; "
          f"train-acc={accuracy_score(ys_e, svm.predict(Zs)):.3f}")
    expl = shap.LinearExplainer(svm, Zs)
    sv = expl.shap_values(Zs)            # list[n_classes] of (n_samples, n_features) OR array
    if isinstance(sv, list):
        imp = np.sum([np.mean(np.abs(c), axis=0) for c in sv], axis=0)
    else:
        sv = np.abs(sv)
        imp = sv.mean(axis=0).sum(axis=-1) if sv.ndim == 3 else sv.mean(axis=0)
    return imp                            # (2048,)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    log_lines = []
    def log(m=""):
        print(m); log_lines.append(str(m))

    log("=" * 64)
    log("XAI-BASED FEATURE SELECTION (Option A: SHAP on SVM)  SAVEE -> EMO-DB")
    log("=" * 64)

    Xs, ys = cc.load("savee")
    Xt, yt = cc.load("embodb")
    le = LabelEncoder().fit(SHARED)
    ys_e, yt_e = le.transform(ys), le.transform(yt)
    log(f"Device {device} | SAVEE train {Xs.shape} | EMO-DB test {Xt.shape}")

    # per-corpus z-norm (same DA as the 74.9% run)
    Zs = StandardScaler().fit_transform(Xs)
    Zt = StandardScaler().fit_transform(Xt)

    # SHAP importances on SAVEE
    log("\n[SHAP] ranking 2048 WavLM dims via LinearExplainer (SAVEE only) ...")
    imp = shap_importance(Zs, ys_e, le)
    order = np.argsort(imp)[::-1]         # most→least important
    log(f"  done. top-5 dim indices: {order[:5].tolist()}")

    # k-sweep: retrain MAML on top-k SAVEE features, eval on EMO-DB
    log("\n[SWEEP] retraining MAML on top-k SHAP-selected dims")
    results = []
    for k in KS:
        idx = order[:k]
        Zs_k, Zt_k = Zs[:, idx], Zt[:, idx]
        reseed(SEED)
        m = cc.train_maml(Zs_k, le.inverse_transform(ys_e), len(SHARED), device)
        mf = cc.fine_tune_maml(m, Zs_k, ys_e, device); mf.eval()
        with torch.no_grad():
            pred = torch.argmax(
                mf(torch.tensor(Zt_k, dtype=torch.float32, device=device)), 1
            ).cpu().numpy()
        acc = accuracy_score(yt_e, pred); f1 = f1_score(yt_e, pred, average="macro")
        tag = "  (full baseline)" if k == 2048 else ""
        log(f"  k={k:<5d}  acc={acc*100:5.1f}%   macroF1={f1:.3f}{tag}")
        results.append((k, acc, f1))

    log("\n" + "=" * 64)
    log("SUMMARY — MAML accuracy vs # SHAP-selected WavLM dims")
    log("=" * 64)
    log(f"{'k (top dims)':<16}{'Accuracy':>12}{'macro-F1':>12}")
    log("-" * 40)
    for k, acc, f1 in results:
        log(f"{k:<16}{acc*100:>11.1f}%{f1:>12.3f}")
    best = max(results, key=lambda r: r[1])
    full = [r for r in results if r[0] == 2048][0]
    log("-" * 40)
    log(f"Best: k={best[0]} at {best[1]*100:.1f}%  |  full(2048)={full[1]*100:.1f}%")
    verdict = ("SELECTION HELPS" if best[0] != 2048 and best[1] > full[1]
               else "full features best — selection does NOT beat baseline")
    log(f"Verdict: {verdict}")

    with open(os.path.join(OUT_DIR, "xai_feature_selection.txt"), "w") as f:
        f.write("\n".join(log_lines))
    log(f"\nWritten to {os.path.join(OUT_DIR, 'xai_feature_selection.txt')}")


if __name__ == "__main__":
    main()
