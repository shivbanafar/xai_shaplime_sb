"""
Reverse-direction cross-corpus SER:  train EMO-DB (German) -> test SAVEE (English).
Mirror of cross_corpus_train.py (which goes SAVEE -> EMO-DB), per-corpus z-norm.

Goal: check whether the fear / happiness story is symmetric.
  - If FEAR fails because its expression DIVERGES across languages, it should be
    the worst-transferring class in BOTH directions.
  - If HAPPINESS fails because it is intrinsically CONFUSABLE, it should also be
    poor in both directions, regardless of which language is the source.

Reports SVM + MAML accuracy and per-class F1 on SAVEE, plus a geometry-only
nearest-centroid transfer (EMO-DB centroids -> classify SAVEE) and the
(symmetric) residual cross-corpus centroid distances.
"""
import os, sys, warnings
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.spatial.distance import cosine as cos_dist

warnings.filterwarnings("ignore")
BASE_DIR = "/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
sys.path.insert(0, os.path.join(BASE_DIR, "cross_corpus", "scripts"))
import cross_corpus_train as cc

OUT_DIR = os.path.join(BASE_DIR, "cross_corpus", "results")
SHARED = cc.SHARED


def nearest_centroid(Xsrc_n, ysrc, Xtgt_n, ytgt, classes):
    cs = {c: Xsrc_n[ysrc == c].mean(0) for c in classes}
    C = np.stack([cs[c] for c in classes])
    pred = np.array(classes)[np.linalg.norm(Xtgt_n[:, None, :] - C[None], axis=2).argmin(1)]
    rec = {c: float((pred[ytgt == c] == c).mean()) for c in classes}
    return rec, float((pred == ytgt).mean()), cs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    L = []
    def log(m=""): print(m); L.append(str(m))

    Xs, ys = cc.load("savee"); Xe, ye = cc.load("embodb")
    le = LabelEncoder().fit(SHARED)
    ys_e, ye_e = le.transform(ys), le.transform(ye)

    log("=" * 64)
    log("REVERSE CROSS-CORPUS:  train EMO-DB (Ger) -> test SAVEE (Eng)")
    log("=" * 64)
    log(f"EMO-DB train {Xe.shape} | SAVEE test {Xs.shape}")

    # per-corpus z-norm
    Xe_n = StandardScaler().fit_transform(Xe)
    Xs_n = StandardScaler().fit_transform(Xs)

    # SVM + MAML, EMO-DB -> SAVEE  (reuse the project's run())
    summary = cc.run("per-corpus z-norm  (EMO-DB -> SAVEE)",
                     Xe_n, Xs_n, ye_e, ys_e, le, device, log)

    # geometry-only nearest-centroid transfer, reverse direction
    log("\n" + "-" * 64)
    log("NEAREST-CENTROID TRANSFER (EMO-DB centroids -> SAVEE; per-class recall)")
    log("-" * 64)
    rec, overall, _ = nearest_centroid(Xe_n, ye, Xs_n, ys, SHARED)
    # residual cross-corpus centroid distance (symmetric across directions)
    cs_s = {c: Xs_n[ys == c].mean(0) for c in SHARED}
    cs_e = {c: Xe_n[ye == c].mean(0) for c in SHARED}
    log(f"  overall nearest-centroid acc = {overall*100:.1f}%")
    log(f"  {'emotion':<10}{'NC recall':>11}{'cos-dist':>10}")
    for c in SHARED:
        log(f"  {c:<10}{rec[c]*100:>10.1f}%{cos_dist(cs_s[c], cs_e[c]):>10.3f}")

    with open(os.path.join(OUT_DIR, "reverse_results.txt"), "w") as f:
        f.write("\n".join(L))
    log(f"\nWritten to {os.path.join(OUT_DIR, 'reverse_results.txt')}")


if __name__ == "__main__":
    main()
