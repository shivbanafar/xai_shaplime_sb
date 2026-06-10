"""
Improve reverse transfer (EMO-DB -> SAVEE) WITHOUT test-label leakage.

Diagnosis from reverse_cross_corpus.py: MAML collapsed toward 'anger'
(recall 0.95, precision 0.26) because the EMO-DB *training* set is imbalanced
(anger 127 vs disgust 46).  Principled, leakage-free fixes:

  1. class-balanced fine-tuning  (inverse-frequency CE weights on EMO-DB)
  2. balanced SVM                (class_weight='balanced')
  3. multi-seed MAML ensemble    (average softmax over 3 seeds; variance control)

All choices are set a priori from the diagnosis, NOT selected on SAVEE accuracy.
Per-corpus z-norm kept (same recipe as the forward 74.9% run).
"""
import os, sys, copy, warnings
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")
BASE_DIR = "/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
sys.path.insert(0, os.path.join(BASE_DIR, "cross_corpus", "scripts"))
import cross_corpus_train as cc

OUT_DIR = os.path.join(BASE_DIR, "cross_corpus", "results")
SHARED = cc.SHARED
SEEDS = [42, 123, 7]


def class_weights(y_e, n_cls):
    counts = np.bincount(y_e, minlength=n_cls).astype(float)
    w = counts.sum() / (n_cls * np.clip(counts, 1, None))
    return w


def weighted_finetune(m, X, y_e, dev, w, epochs=300):
    f = copy.deepcopy(m)
    opt = optim.Adam(f.parameters(), lr=1e-4, weight_decay=1e-2)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev),
                             label_smoothing=0.1)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y_e, dtype=torch.long, device=dev)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)
    for _ in range(epochs):
        f.train()
        for xb, yb in dl:
            opt.zero_grad(); ce(f(xb), yb).backward(); opt.step()
        sch.step()
    f.eval()
    return f


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    L = []
    def log(m=""): print(m); L.append(str(m))

    Xs, ys = cc.load("savee"); Xe, ye = cc.load("embodb")
    le = LabelEncoder().fit(SHARED)
    ys_e, ye_e = le.transform(ys), le.transform(ye)
    Xe_n = StandardScaler().fit_transform(Xe)     # train = EMO-DB
    Xs_n = StandardScaler().fit_transform(Xs)      # test  = SAVEE
    n_cls = len(SHARED)

    log("=" * 64)
    log("IMPROVED REVERSE  (EMO-DB -> SAVEE), leakage-free")
    log("=" * 64)
    w = class_weights(ye_e, n_cls)
    log("EMO-DB class counts: " +
        ", ".join(f"{c}={int((ye_e==i).sum())}" for i, c in enumerate(SHARED)))
    log("inverse-freq weights: " + ", ".join(f"{c}={w[i]:.2f}" for i, c in enumerate(SHARED)))

    # --- balanced SVM ---
    pca = PCA(n_components=300, random_state=42).fit(Xe_n)
    svm = LinearSVC(C=0.1, class_weight="balanced", max_iter=5000,
                    random_state=42).fit(pca.transform(Xe_n), ye_e)
    sp = svm.predict(pca.transform(Xs_n))
    log(f"\n[SVM balanced]  acc={accuracy_score(ys_e,sp)*100:.1f}%  "
        f"macroF1={f1_score(ys_e,sp,average='macro'):.3f}")

    # --- class-balanced MAML, 3-seed ensemble ---
    probs = np.zeros((len(Xs_n), n_cls))
    per_seed = []
    for s in SEEDS:
        np.random.seed(s); torch.manual_seed(s)
        m = cc.train_maml(Xe_n, le.inverse_transform(ye_e), n_cls, dev)
        f = weighted_finetune(m, Xe_n, ye_e, dev, w)
        with torch.no_grad():
            p = torch.softmax(f(torch.tensor(Xs_n, dtype=torch.float32, device=dev)), 1).cpu().numpy()
        probs += p
        a = accuracy_score(ys_e, p.argmax(1))
        per_seed.append(a)
        log(f"  seed {s}: MAML acc={a*100:.1f}%")
    pred = (probs / len(SEEDS)).argmax(1)
    acc = accuracy_score(ys_e, pred); mf1 = f1_score(ys_e, pred, average="macro")

    log(f"\n[MAML balanced + {len(SEEDS)}-seed ensemble]  "
        f"acc={acc*100:.1f}%  macroF1={mf1:.3f}  (per-seed mean {np.mean(per_seed)*100:.1f}%)")
    log(classification_report(ys_e, pred, target_names=SHARED, zero_division=0))

    log("Fear/happiness recall (improved, reverse):")
    for c in ["fear", "happiness"]:
        i = list(SHARED).index(c)
        m = ys_e == i
        log(f"  {c:<10} recall={ (pred[m]==i).mean()*100:5.1f}%")

    with open(os.path.join(OUT_DIR, "reverse_improved.txt"), "w") as f:
        f.write("\n".join(L))
    log(f"\nWritten to {os.path.join(OUT_DIR,'reverse_improved.txt')}")


if __name__ == "__main__":
    main()
