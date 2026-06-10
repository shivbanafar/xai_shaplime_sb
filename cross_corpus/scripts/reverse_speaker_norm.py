"""
Reverse transfer (EMO-DB -> SAVEE) with PER-SPEAKER z-normalisation.

Motivation (a priori, from the reverse-direction diagnosis): SAVEE is only
4 idiosyncratic male speakers, so the corpus-level statistics are dominated by
speaker identity.  Normalising each SPEAKER by their own mean/std removes
speaker-specific offsets the same way per-corpus z-norm removes corpus offsets.
This uses speaker IDs (filename metadata) only — never emotion labels — so it
remains unsupervised w.r.t. the task.

Recipe (fixed before evaluation):
  per-SPEAKER z-norm (both corpora)  ->  class-balanced fine-tuning
  ->  3-seed softmax ensemble  (the recipe that reached 60.2% with corpus-norm).

Speaker alignment: the cache was built by os.listdir enumeration; we re-run the
same enumeration and VERIFY the reconstructed labels match the cached y
position-by-position before trusting the speaker assignment (hard assert).

EMO-DB speaker = first 2 chars of filename (e.g. 03a01Wa.wav -> '03').
SAVEE  speaker = prefix before '_'      (e.g. DC_a01.wav    -> 'DC').
"""
import os, sys, re, copy, warnings
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")
BASE_DIR = "/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, os.path.join(BASE_DIR, "cross_corpus", "scripts"))
import cross_corpus_train as cc

OUT_DIR = os.path.join(BASE_DIR, "cross_corpus", "results")
SHARED = cc.SHARED
SEEDS = [42, 123, 7]


# ── speaker recovery (mirror run_all_datasets.py enumeration exactly) ────────
def enum_emodb(ddir):
    EMAP = {"W": "anger", "L": "boredom", "E": "disgust", "A": "fear",
            "F": "happiness", "T": "sadness", "N": "neutral"}
    labels, spk = [], []
    for f in os.listdir(ddir):
        if not f.endswith(".wav") or len(f) < 6: continue
        emo = EMAP.get(f[5])
        if emo: labels.append(emo); spk.append(f[:2])
    return np.array(labels), np.array(spk)


def enum_savee(ddir):
    EMAP = {"a": "anger", "d": "disgust", "f": "fear", "h": "happiness",
            "n": "neutral", "sa": "sadness", "su": "surprise"}
    labels, spk = [], []
    for f in os.listdir(ddir):
        if not f.endswith(".wav"): continue
        m = re.match(r"([A-Z]+)_([a-z]+)\d+\.wav", f)
        if not m: continue
        emo = EMAP.get(m.group(2))
        if emo: labels.append(emo); spk.append(m.group(1))
    return np.array(labels), np.array(spk)


def load_with_speakers(key, enum_fn, ddir):
    d = np.load(os.path.join(BASE_DIR, "cache", f"{key}_wavlm_large_combined.npz"),
                allow_pickle=True)
    X = d["X"]; y_cache = np.array([str(v) for v in d["y"]])
    y_enum, spk = enum_fn(ddir)
    assert len(y_enum) == len(y_cache) and (y_enum == y_cache).all(), \
        f"{key}: enumeration does not reproduce cached label order — speaker IDs unsafe"
    m = np.isin(y_cache, SHARED)
    return X[m][:, :cc.WAVLM_DIM], y_cache[m], spk[m]


def per_speaker_znorm(X, spk):
    Xn = np.empty_like(X, dtype=float)
    for s in np.unique(spk):
        m = spk == s
        Xn[m] = StandardScaler().fit_transform(X[m])
    return Xn


# ── balanced fine-tune (same as the 60.2% recipe) ────────────────────────────
def class_weights(y_e, n_cls):
    c = np.bincount(y_e, minlength=n_cls).astype(float)
    return c.sum() / (n_cls * np.clip(c, 1, None))


def weighted_finetune(m, X, y_e, dev, w, epochs=300):
    f = copy.deepcopy(m)
    opt = optim.Adam(f.parameters(), lr=1e-4, weight_decay=1e-2)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev),
                             label_smoothing=0.1)
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32, device=dev),
                                  torch.tensor(y_e, dtype=torch.long, device=dev)),
                    batch_size=64, shuffle=True)
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

    log("=" * 64)
    log("REVERSE (EMO-DB -> SAVEE): PER-SPEAKER z-norm + balanced + 3-seed")
    log("=" * 64)

    Xe, ye, spk_e = load_with_speakers("embodb", enum_emodb, os.path.join(DATA_DIR, "EMBODB"))
    Xs, ys, spk_s = load_with_speakers("savee",  enum_savee, os.path.join(DATA_DIR, "SAVEE"))
    log(f"speaker alignment verified | EMO-DB speakers {sorted(set(spk_e))} | "
        f"SAVEE speakers {sorted(set(spk_s))}")

    le = LabelEncoder().fit(SHARED)
    ye_e, ys_e = le.transform(ye), le.transform(ys)
    n_cls = len(SHARED)

    Xe_n = per_speaker_znorm(Xe, spk_e)   # train
    Xs_n = per_speaker_znorm(Xs, spk_s)   # test
    w = class_weights(ye_e, n_cls)

    pca = PCA(n_components=300, random_state=42).fit(Xe_n)
    svm = LinearSVC(C=0.1, class_weight="balanced", max_iter=5000,
                    random_state=42).fit(pca.transform(Xe_n), ye_e)
    sp = svm.predict(pca.transform(Xs_n))
    log(f"\n[SVM balanced + speaker-norm]  acc={accuracy_score(ys_e,sp)*100:.1f}%  "
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

    with open(os.path.join(OUT_DIR, "reverse_speaker_norm.txt"), "w") as f:
        f.write("\n".join(L))
    log(f"Written to {os.path.join(OUT_DIR,'reverse_speaker_norm.txt')}")


if __name__ == "__main__":
    main()
