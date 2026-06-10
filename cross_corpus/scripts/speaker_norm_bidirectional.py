"""
Bidirectional cross-corpus SER under PER-SPEAKER z-normalisation (leakage-free).

Runs BOTH directions with one consistent recipe so the table is comparable:
  per-speaker z-norm  ->  class-balanced fine-tuning  ->  3-seed softmax ensemble.

Per-speaker normalisation standardises each speaker by their own feature
mean/std (uses speaker IDs from filenames, never emotion labels).  It assumes
test utterances can be grouped by speaker at inference time — a standard, still
unsupervised SER assumption that we state explicitly.

Speaker IDs are recovered by re-running the cache enumeration and asserting the
reconstructed label order matches the cached y (so speaker assignment is safe).
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


def enum_emodb(ddir):
    EMAP = {"W": "anger", "L": "boredom", "E": "disgust", "A": "fear",
            "F": "happiness", "T": "sadness", "N": "neutral"}
    lab, spk = [], []
    for f in os.listdir(ddir):
        if not f.endswith(".wav") or len(f) < 6: continue
        e = EMAP.get(f[5])
        if e: lab.append(e); spk.append(f[:2])
    return np.array(lab), np.array(spk)


def enum_savee(ddir):
    EMAP = {"a": "anger", "d": "disgust", "f": "fear", "h": "happiness",
            "n": "neutral", "sa": "sadness", "su": "surprise"}
    lab, spk = [], []
    for f in os.listdir(ddir):
        if not f.endswith(".wav"): continue
        m = re.match(r"([A-Z]+)_([a-z]+)\d+\.wav", f)
        if not m: continue
        e = EMAP.get(m.group(2))
        if e: lab.append(e); spk.append(m.group(1))
    return np.array(lab), np.array(spk)


ENUM = {"savee": (enum_savee, os.path.join(DATA_DIR, "SAVEE")),
        "embodb": (enum_emodb, os.path.join(DATA_DIR, "EMBODB"))}


def load_spk(key):
    d = np.load(os.path.join(BASE_DIR, "cache", f"{key}_wavlm_large_combined.npz"),
                allow_pickle=True)
    X = d["X"]; yc = np.array([str(v) for v in d["y"]])
    fn, ddir = ENUM[key]; ye, spk = fn(ddir)
    assert len(ye) == len(yc) and (ye == yc).all(), f"{key}: enum/cache mismatch"
    m = np.isin(yc, SHARED)
    return X[m][:, :cc.WAVLM_DIM], yc[m], spk[m]


def per_speaker(X, spk):
    Xn = np.empty_like(X, dtype=float)
    for s in np.unique(spk):
        m = spk == s
        Xn[m] = StandardScaler().fit_transform(X[m])
    return Xn


def cw(y_e, k):
    c = np.bincount(y_e, minlength=k).astype(float)
    return c.sum() / (k * np.clip(c, 1, None))


def finetune(m, X, y_e, dev, w, epochs=300):
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


def run_dir(src, tgt, le, dev, log):
    Xsr, ysr, spk_sr = load_spk(src)
    Xtg, ytg, spk_tg = load_spk(tgt)
    Xsr_n, Xtg_n = per_speaker(Xsr, spk_sr), per_speaker(Xtg, spk_tg)
    ysr_e, ytg_e = le.transform(ysr), le.transform(ytg)
    k = len(SHARED); w = cw(ysr_e, k)

    pca = PCA(n_components=300, random_state=42).fit(Xsr_n)
    svm = LinearSVC(C=0.1, class_weight="balanced", max_iter=5000,
                    random_state=42).fit(pca.transform(Xsr_n), ysr_e)
    svm_acc = accuracy_score(ytg_e, svm.predict(pca.transform(Xtg_n)))

    probs = np.zeros((len(Xtg_n), k)); per = []
    for s in SEEDS:
        np.random.seed(s); torch.manual_seed(s)
        m = cc.train_maml(Xsr_n, le.inverse_transform(ysr_e), k, dev)
        f = finetune(m, Xsr_n, ysr_e, dev, w)
        with torch.no_grad():
            p = torch.softmax(f(torch.tensor(Xtg_n, dtype=torch.float32, device=dev)), 1).cpu().numpy()
        probs += p; per.append(accuracy_score(ytg_e, p.argmax(1)))
    pred = (probs / len(SEEDS)).argmax(1)
    acc = accuracy_score(ytg_e, pred); mf1 = f1_score(ytg_e, pred, average="macro")
    log(f"\n[{src.upper()} -> {tgt.upper()}]  per-speaker z-norm + balanced + 3-seed")
    log(f"  SVM  acc={svm_acc*100:.1f}%")
    log(f"  MAML acc={acc*100:.1f}%  macroF1={mf1:.3f}  "
        f"(per-seed {np.mean(per)*100:.1f}% +/- {np.std(per)*100:.1f})")
    log(classification_report(ytg_e, pred, target_names=SHARED, zero_division=0))
    return svm_acc, acc, mf1


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    L = []
    def log(m=""): print(m); L.append(str(m))
    log("=" * 64)
    log("BIDIRECTIONAL, PER-SPEAKER z-norm (consistent recipe)")
    log("=" * 64)
    le = LabelEncoder().fit(SHARED)
    fwd = run_dir("savee", "embodb", le, dev, log)
    rev = run_dir("embodb", "savee", le, dev, log)
    log("\n" + "=" * 64)
    log("SUMMARY (per-speaker z-norm)")
    log("=" * 64)
    log(f"  SAVEE->EMO-DB : SVM {fwd[0]*100:.1f}%  MAML {fwd[1]*100:.1f}%")
    log(f"  EMO-DB->SAVEE : SVM {rev[0]*100:.1f}%  MAML {rev[1]*100:.1f}%")
    with open(os.path.join(OUT_DIR, "speaker_norm_bidirectional.txt"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
