"""
Why does per-corpus z-norm help, and why do fear/happiness still fail?
SAVEE (English) -> EMO-DB (German), WavLM-only 2048-dim.  No model retraining.

Three measurements, each under (A) naive source-scaler and (B) per-corpus z-norm:

  1. DOMAIN SEPARABILITY
     Train a linear classifier to predict the CORPUS of a sample (SAVEE vs
     EMO-DB), 5-fold CV.  Accuracy near 50% => the two feature spaces are
     indistinguishable (well aligned).  Also report proxy A-distance
     d_A = 2(1 - 2*err).  This is non-trivial: per-dim z-norm only zeroes the
     marginals, so a multivariate classifier can still separate via structure.

  2. CLASS-CONDITIONAL CROSS-CORPUS CENTROID DISTANCE
     For each emotion, distance between the SAVEE centroid and the EMO-DB
     centroid (relative to each corpus's own mean).  Large residual distance
     after z-norm = that emotion is positioned differently across languages,
     which should predict poor transfer.

  3. NEAREST-CENTROID TRANSFER (geometry-only proxy for accuracy)
     Build SAVEE class centroids; assign each EMO-DB sample to the nearest one;
     report per-class recall.  This depends ONLY on feature geometry (no MAML),
     so it isolates whether the representation itself supports transfer.

Finally we correlate per-class centroid distance with per-class transfer recall,
and with the reported MAML per-class F1, to see if "the emotions whose clouds
stay far apart are exactly the ones that fail" holds.

Outputs: cross_corpus/results/znorm_analysis.txt
         cross_corpus/results/tsne_znorm.png
         cross_corpus/results/centroid_distance.png
"""
import os, sys, warnings
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine as cos_dist
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

BASE_DIR = "/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
sys.path.insert(0, os.path.join(BASE_DIR, "cross_corpus", "scripts"))
import cross_corpus_train as cc   # provides load(), SHARED, seeds

OUT_DIR = os.path.join(BASE_DIR, "cross_corpus", "results")
SEED = 42
SHARED = cc.SHARED

# Reported MAML per-class F1 (per-corpus z-norm, 74.9% run; from results.txt)
MAML_F1 = {"anger": 0.82, "disgust": 0.76, "fear": 0.66,
           "happiness": 0.62, "neutral": 0.82, "sadness": 0.74}


def naive(Xs, Xt):
    sc = StandardScaler().fit(Xs)
    return sc.transform(Xs), sc.transform(Xt)


def percorpus(Xs, Xt):
    return StandardScaler().fit_transform(Xs), StandardScaler().fit_transform(Xt)


def domain_separability(Xs_n, Xt_n):
    X = np.vstack([Xs_n, Xt_n])
    y = np.r_[np.zeros(len(Xs_n)), np.ones(len(Xt_n))].astype(int)
    # PCA-reduce before the linear domain classifier: standard proxy-A-distance
    # setup, and avoids the below-chance overfitting of a 2048-dim classifier.
    clf = make_pipeline(PCA(n_components=50, random_state=SEED),
                        LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()
    err = 1.0 - acc
    d_a = 2.0 * (1.0 - 2.0 * err)          # proxy A-distance
    return acc, max(d_a, 0.0)


def centroids(Xn, y, classes):
    return {c: Xn[y == c].mean(axis=0) for c in classes}


def class_centroid_dist(Xs_n, ys, Xt_n, yt, classes):
    cs, ct = centroids(Xs_n, ys, classes), centroids(Xt_n, yt, classes)
    out = {}
    for c in classes:
        eu = float(np.linalg.norm(cs[c] - ct[c]))
        co = float(cos_dist(cs[c], ct[c]))
        out[c] = (eu, co)
    return out, cs


def nearest_centroid_transfer(cs, Xt_n, yt, classes):
    """Assign each EMO-DB sample to nearest SAVEE class centroid; per-class recall."""
    C = np.stack([cs[c] for c in classes])           # (n_cls, dim)
    d = np.linalg.norm(Xt_n[:, None, :] - C[None, :, :], axis=2)  # (N, n_cls)
    pred = np.array(classes)[d.argmin(axis=1)]
    recall = {}
    for c in classes:
        m = yt == c
        recall[c] = float((pred[m] == c).mean()) if m.sum() else float("nan")
    overall = float((pred == yt).mean())
    return recall, overall


def tsne_figure(Xs_a, Xt_a, Xs_b, Xt_b, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (Xs_n, Xt_n, title) in zip(
            axes, [(Xs_a, Xt_a, "Naive source-scaler"),
                   (Xs_b, Xt_b, "Per-corpus z-norm")]):
        Z = TSNE(n_components=2, init="pca", perplexity=30,
                 random_state=SEED).fit_transform(np.vstack([Xs_n, Xt_n]))
        zs, zt = Z[:len(Xs_n)], Z[len(Xs_n):]
        ax.scatter(zs[:, 0], zs[:, 1], s=10, alpha=0.6, label="SAVEE (Eng)", c="#1f77b4")
        ax.scatter(zt[:, 0], zt[:, 1], s=10, alpha=0.6, label="EMO-DB (Ger)", c="#d62728")
        ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Corpus alignment in WavLM space (t-SNE)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    L = []
    def log(m=""): print(m); L.append(str(m))

    Xs, ys = cc.load("savee"); Xt, yt = cc.load("embodb")
    classes = SHARED
    log("=" * 64)
    log("WHY Z-NORM HELPS / WHY FEAR & HAPPINESS FAIL   SAVEE -> EMO-DB")
    log("=" * 64)
    log(f"SAVEE {Xs.shape} | EMO-DB {Xt.shape} | classes {classes}")
    for c in classes:
        log(f"  {c:<10} SAVEE n={int((ys==c).sum()):<4} EMO-DB n={int((yt==c).sum())}")

    Xs_a, Xt_a = naive(Xs, Xt)
    Xs_b, Xt_b = percorpus(Xs, Xt)

    # 1. domain separability
    log("\n" + "-" * 64)
    log("1) DOMAIN SEPARABILITY  (corpus classifier; 50% = aligned)")
    log("-" * 64)
    acc_a, da_a = domain_separability(Xs_a, Xt_a)
    acc_b, da_b = domain_separability(Xs_b, Xt_b)
    log(f"  Naive source-scaler : acc={acc_a*100:5.1f}%   proxy A-dist={da_a:.3f}")
    log(f"  Per-corpus z-norm   : acc={acc_b*100:5.1f}%   proxy A-dist={da_b:.3f}")
    log(f"  => separability change: {acc_a*100:.1f}% -> {acc_b*100:.1f}%")

    # 2. class-conditional centroid distance
    log("\n" + "-" * 64)
    log("2) CLASS-CONDITIONAL CROSS-CORPUS CENTROID DISTANCE")
    log("-" * 64)
    dist_a, _ = class_centroid_dist(Xs_a, ys, Xt_a, yt, classes)
    dist_b, cs_b = class_centroid_dist(Xs_b, ys, Xt_b, yt, classes)
    log(f"  {'emotion':<10}{'naive eu':>10}{'znorm eu':>10}{'znorm cos':>11}")
    for c in classes:
        log(f"  {c:<10}{dist_a[c][0]:>10.2f}{dist_b[c][0]:>10.2f}{dist_b[c][1]:>11.3f}")

    # 3. nearest-centroid transfer (geometry-only)
    log("\n" + "-" * 64)
    log("3) NEAREST-CENTROID TRANSFER (z-norm; geometry-only per-class recall)")
    log("-" * 64)
    recall_b, overall_b = nearest_centroid_transfer(cs_b, Xt_b, yt, classes)
    log(f"  overall nearest-centroid acc = {overall_b*100:.1f}%")
    log(f"  {'emotion':<10}{'NC recall':>11}{'znorm cos-d':>13}{'MAML F1':>9}")
    for c in classes:
        log(f"  {c:<10}{recall_b[c]*100:>10.1f}%{dist_b[c][1]:>13.3f}{MAML_F1[c]:>9.2f}")

    # correlations
    order = classes
    cosd = np.array([dist_b[c][1] for c in order])
    ncrec = np.array([recall_b[c] for c in order])
    mamlf1 = np.array([MAML_F1[c] for c in order])
    r1, p1 = spearmanr(cosd, ncrec)
    r2, p2 = spearmanr(cosd, mamlf1)
    log("\n" + "-" * 64)
    log("CORRELATIONS  (Spearman; negative = farther apart -> worse transfer)")
    log("-" * 64)
    log(f"  centroid cos-dist  vs  nearest-centroid recall : rho={r1:+.2f} (p={p1:.3f})")
    log(f"  centroid cos-dist  vs  MAML per-class F1        : rho={r2:+.2f} (p={p2:.3f})")

    # rank the worst-aligned emotions
    worst = sorted(order, key=lambda c: -dist_b[c][1])
    log(f"\n  Emotions ranked by residual cross-corpus distance (worst first):")
    log("    " + " > ".join(f"{c}({dist_b[c][1]:.2f})" for c in worst))

    # bar figure: residual centroid distance vs MAML F1
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(order))
    ax.bar(x - 0.2, cosd, 0.4, label="cross-corpus cos-dist (z-norm)", color="#d62728")
    ax2 = ax.twinx()
    ax2.plot(x, mamlf1, "o-", color="#1f77b4", label="MAML per-class F1")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("centroid cosine distance"); ax2.set_ylabel("MAML F1")
    ax.set_title("Residual cross-corpus distance vs transfer quality")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.92), fontsize=8)
    fig.tight_layout()
    cd_png = os.path.join(OUT_DIR, "centroid_distance.png")
    fig.savefig(cd_png, dpi=200, bbox_inches="tight"); plt.close(fig)
    log(f"\n  saved {cd_png}")

    # t-SNE figure
    log("  computing t-SNE (this takes a moment) ...")
    tsne_png = os.path.join(OUT_DIR, "tsne_znorm.png")
    tsne_figure(Xs_a, Xt_a, Xs_b, Xt_b, tsne_png)
    log(f"  saved {tsne_png}")

    with open(os.path.join(OUT_DIR, "znorm_analysis.txt"), "w") as f:
        f.write("\n".join(L))
    log(f"\nWritten to {os.path.join(OUT_DIR, 'znorm_analysis.txt')}")


if __name__ == "__main__":
    main()
