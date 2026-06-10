"""
Reverse transfer (EMO-DB -> SAVEE), enhanced recipe — pre-committed, leakage-free.

Recipe fixed a priori (NOT selected on SAVEE accuracy):
  per-corpus z-norm  ->  L2-normalise embeddings (unit length)
  ->  class-balanced fine-tuning (inverse-freq CE weights)
  ->  5-seed softmax ensemble.

L2-normalisation is a standard embedding-transfer step (direction over magnitude)
and, to stay honest, would be applied identically to the forward direction if
adopted in the paper.  We report whatever this single recipe yields.
"""
import os, sys, copy, warnings
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")
BASE_DIR = "/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
sys.path.insert(0, os.path.join(BASE_DIR, "cross_corpus", "scripts"))
import cross_corpus_train as cc

OUT_DIR = os.path.join(BASE_DIR, "cross_corpus", "results")
SHARED = cc.SHARED
SEEDS = [42, 123, 7, 2024, 99]


def class_weights(y_e, n_cls):
    c = np.bincount(y_e, minlength=n_cls).astype(float)
    return c.sum() / (n_cls * np.clip(c, 1, None))


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
    f.eval(); return f


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    L = []
    def log(m=""): print(m); L.append(str(m))

    Xs, ys = cc.load("savee"); Xe, ye = cc.load("embodb")
    le = LabelEncoder().fit(SHARED)
    ys_e, ye_e = le.transform(ys), le.transform(ye)
    n_cls = len(SHARED)

    # per-corpus z-norm  ->  L2-normalise rows
    Xe_n = normalize(StandardScaler().fit_transform(Xe))
    Xs_n = normalize(StandardScaler().fit_transform(Xs))
    w = class_weights(ye_e, n_cls)

    log("=" * 64)
    log("REVERSE ENHANCED  (EMO-DB -> SAVEE): z-norm + L2 + balanced + 5-seed")
    log("=" * 64)

    pca = PCA(n_components=300, random_state=42).fit(Xe_n)
    svm = LinearSVC(C=0.1, class_weight="balanced", max_iter=5000,
                    random_state=42).fit(pca.transform(Xe_n), ye_e)
    sp = svm.predict(pca.transform(Xs_n))
    log(f"[SVM balanced + L2]  acc={accuracy_score(ys_e,sp)*100:.1f}%  "
        f"macroF1={f1_score(ys_e,sp,average='macro'):.3f}")

    probs = np.zeros((len(Xs_n), n_cls)); per = []
    for s in SEEDS:
        np.random.seed(s); torch.manual_seed(s)
        m = cc.train_maml(Xe_n, le.inverse_transform(ye_e), n_cls, dev)
        f = weighted_finetune(m, Xe_n, ye_e, dev, w)
        with torch.no_grad():
            p = torch.softmax(f(torch.tensor(Xs_n, dtype=torch.float32, device=dev)), 1).cpu().numpy()
        probs += p; a = accuracy_score(ys_e, p.argmax(1)); per.append(a)
        log(f"  seed {s}: acc={a*100:.1f}%")
    pred = (probs / len(SEEDS)).argmax(1)
    acc = accuracy_score(ys_e, pred); mf1 = f1_score(ys_e, pred, average="macro")
    log(f"\n[MAML ensemble]  acc={acc*100:.1f}%  macroF1={mf1:.3f}  "
        f"(per-seed mean {np.mean(per)*100:.1f}% +/- {np.std(per)*100:.1f})")
    log(classification_report(ys_e, pred, target_names=SHARED, zero_division=0))

    with open(os.path.join(OUT_DIR, "reverse_improved2.txt"), "w") as f:
        f.write("\n".join(L))
    log(f"\nWritten to {os.path.join(OUT_DIR,'reverse_improved2.txt')}")


if __name__ == "__main__":
    main()
