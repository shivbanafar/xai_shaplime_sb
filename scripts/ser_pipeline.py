"""
Speech Emotion Recognition Pipeline
Dataset  : RAVDESS (8 classes)
Features : WavLM-base embeddings (768-dim) + handcrafted MFCC/chroma/pitch (227-dim) = 995-dim
Models   : SVM → XGBoost → MAML + fine-tune
Target   : ≥ 80% accuracy
"""

import os
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import librosa
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR      = "/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
RAVDESS_DIR   = os.path.join(BASE_DIR, "RAVDESS")
CACHE_WAVLM   = os.path.join(BASE_DIR, "ravdess_wavlm.npz")   # WavLM features
CACHE_HAND    = os.path.join(BASE_DIR, "ravdess_hand.npz")    # handcrafted (reuse)
CACHE_COMBINED = os.path.join(BASE_DIR, "ravdess_wavlm_combined.npz")

SR_MODEL = 16000    # WavLM requires 16 kHz
SR_HAND  = 22050
N_MFCC   = 40
SEED     = 42
TEST_SIZE = 0.2

# MAML
N_WAY       = 5
K_SHOT      = 5
Q_QUERY     = 10
INNER_LR    = 0.05
OUTER_LR    = 0.001
INNER_STEPS = 5
META_EPOCHS = 300
META_BATCH  = 4
FT_EPOCHS   = 300

EMOTION_MAP = {
    "01": "neutral",  "02": "calm",    "03": "happy",   "04": "sad",
    "05": "angry",    "06": "fearful", "07": "disgust",  "08": "surprised",
}

# ── 1a. Handcrafted Features ───────────────────────────────────────────────────
def hand_feats(y: np.ndarray, sr: int = SR_HAND) -> np.ndarray:
    """227-dim handcrafted features."""
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    f0     = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    voiced = f0[f0 > 0]
    pitch  = ([np.mean(voiced), np.std(voiced), np.max(voiced), len(voiced)/max(len(f0),1)]
              if len(voiced) > 0 else [0.0]*4)
    return np.concatenate([
        np.mean(mfcc, 1), np.std(mfcc, 1),   # 80
        np.mean(chroma, 1),                    # 12
        np.mean(mel, 1),                       # 128
        [np.mean(librosa.feature.zero_crossing_rate(y)),
         np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
         np.mean(librosa.feature.rms(y=y))],   # 3
        pitch,                                 # 4
    ])  # total = 227


# ── 1b. WavLM Embeddings ───────────────────────────────────────────────────────
def load_wavlm_model():
    from transformers import WavLMModel, AutoFeatureExtractor
    print("  Loading WavLM-base (~360 MB first time) ...")
    feat_ext = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model    = WavLMModel.from_pretrained("microsoft/wavlm-base")
    model.eval()
    return feat_ext, model


def wavlm_embed(y: np.ndarray, feat_ext, model, device) -> np.ndarray:
    """768-dim mean-pooled WavLM hidden state."""
    inputs = feat_ext(y, return_tensors="pt", sampling_rate=SR_MODEL, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


# ── 1c. Combined Feature Load ──────────────────────────────────────────────────
def load_all_features():
    """Returns X_combined (995-dim = 768 WavLM + 227 handcrafted), y (emotion strings)."""
    if os.path.exists(CACHE_COMBINED):
        print("  Loading cached WavLM + handcrafted features ...")
        d = np.load(CACHE_COMBINED, allow_pickle=True)
        return d["X"], d["y"]

    # Collect file paths & labels
    paths, labels = [], []
    for actor in sorted(os.listdir(RAVDESS_DIR)):
        adir = os.path.join(RAVDESS_DIR, actor)
        if not os.path.isdir(adir): continue
        for fname in os.listdir(adir):
            if not fname.endswith(".wav"): continue
            parts = fname.replace(".wav", "").split("-")
            emo   = EMOTION_MAP.get(parts[2] if len(parts) > 2 else "")
            if emo is None: continue
            paths.append(os.path.join(adir, fname))
            labels.append(emo)

    # WavLM embeddings
    if os.path.exists(CACHE_WAVLM):
        print("  Loading cached WavLM features ...")
        X_wavlm = np.load(CACHE_WAVLM)["X"]
    else:
        print("  Extracting WavLM embeddings ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feat_ext, wavlm = load_wavlm_model()
        wavlm = wavlm.to(device)
        X_wavlm = []
        for i, path in enumerate(paths):
            y_audio, _ = librosa.load(path, sr=SR_MODEL)
            X_wavlm.append(wavlm_embed(y_audio, feat_ext, wavlm, device))
            if (i+1) % 200 == 0:
                print(f"    {i+1}/{len(paths)}")
        X_wavlm = np.array(X_wavlm)
        np.savez(CACHE_WAVLM, X=X_wavlm)

    # Handcrafted features
    if os.path.exists(CACHE_HAND):
        print("  Loading cached handcrafted features ...")
        X_hand = np.load(CACHE_HAND)["X"]
    else:
        print("  Extracting handcrafted features ...")
        X_hand = []
        for i, path in enumerate(paths):
            y_audio, _ = librosa.load(path, sr=SR_HAND)
            X_hand.append(hand_feats(y_audio))
            if (i+1) % 200 == 0:
                print(f"    {i+1}/{len(paths)}")
        X_hand = np.array(X_hand)
        np.savez(CACHE_HAND, X=X_hand)

    X = np.hstack([X_wavlm, X_hand])   # 768 + 227 = 995
    y = np.array(labels)
    np.savez(CACHE_COMBINED, X=X, y=y)
    return X, y


# ── SVM ────────────────────────────────────────────────────────────────────────
def fit_svm(X, y):
    """LinearSVC is the standard choice for high-dimensional pre-trained embeddings."""
    lin = LinearSVC(C=0.05, max_iter=5000, random_state=SEED)
    m   = CalibratedClassifierCV(lin, cv=5)
    m.fit(X, y)
    return m


# ── XGBoost ────────────────────────────────────────────────────────────────────
def fit_xgboost(X_tr, y_tr, X_val, y_val, pca):
    """
    Tree-based models don't scale well to 995-dim dense embeddings.
    PCA to 200-dim first, then two-phase probe+retrain.
    """
    X_tr_p  = pca.transform(X_tr)
    X_val_p = pca.transform(X_val)

    # Phase 1: find optimal n_estimators via early stopping
    probe = xgb.XGBClassifier(
        n_estimators=800, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
        eval_metric="mlogloss", random_state=SEED, n_jobs=-1,
        early_stopping_rounds=50,
    )
    probe.fit(X_tr_p, y_tr, eval_set=[(X_val_p, y_val)], verbose=False)
    best_n = max(probe.best_iteration, 50)
    print(f"    Optimal n_estimators: {best_n}")

    # Phase 2: retrain on FULL data (tr + val) with that many trees
    X_full = np.vstack([X_tr_p, X_val_p])
    y_full = np.concatenate([y_tr, y_val])
    m = xgb.XGBClassifier(
        n_estimators=best_n, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
        eval_metric="mlogloss", random_state=SEED, n_jobs=-1,
    )
    m.fit(X_full, y_full)
    return m


# ── MAML ───────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, n_cls: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, n_cls)
        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.35)
        self.drop3 = nn.Dropout(0.3)
        # flat parameter list for MAML inner loop (no BN params in inner loop)
        self.linears = nn.ModuleList([self.fc1, self.fc2, self.fc3, self.fc4])

    def forward(self, x, weights=None):
        if weights is None:
            # Standard forward with BN + dropout
            out = self.drop1(F.relu(self.bn1(self.fc1(x))))
            out = self.drop2(F.relu(self.bn2(self.fc2(out))))
            out = self.drop3(F.relu(self.bn3(self.fc3(out))))
            return self.fc4(out)
        # Inner-loop forward: functional linear only (BN frozen)
        out = x
        n   = len(self.linears)
        for i in range(n - 1):
            out = F.relu(F.linear(out, weights[2*i], weights[2*i+1]))
        return F.linear(out, weights[2*(n-1)], weights[2*(n-1)+1])


def inner_loop(model, sx, sy, lr, steps, create_graph=True):
    w = []
    for lin in model.linears:
        w.extend([lin.weight, lin.bias])
    for _ in range(steps):
        loss  = F.cross_entropy(model(sx, w), sy)
        grads = torch.autograd.grad(loss, w, create_graph=create_graph)
        w     = [p - lr * g for p, g in zip(w, grads)]
    return w


def sample_episode(X_by_cls, n_way, k_shot, q_query, device):
    chosen = np.random.choice(list(X_by_cls.keys()), min(n_way, len(X_by_cls)), replace=False)
    sx, sy, qx, qy = [], [], [], []
    for i, cls in enumerate(chosen):
        arr  = X_by_cls[cls]
        need = k_shot + q_query
        idx  = np.random.choice(len(arr), min(need, len(arr)), replace=False)
        k    = min(k_shot, max(len(idx)//2, 1))
        sx.append(arr[idx[:k]]);   sy += [i] * k
        qx.append(arr[idx[k:]]);   qy += [i] * len(idx[k:])
    t = lambda a: torch.tensor(np.vstack(a), dtype=torch.float32, device=device)
    l = lambda a: torch.tensor(a,            dtype=torch.long,    device=device)
    return t(sx), l(sy), t(qx), l(qy)


def train_maml(X_tr, y_tr_str, n_cls, device):
    X_by_cls = defaultdict(list)
    for x, lbl in zip(X_tr, y_tr_str):
        X_by_cls[lbl].append(x)
    X_by_cls = {k: np.array(v) for k, v in X_by_cls.items()}

    model = MLP(X_tr.shape[1], n_cls).to(device)
    opt   = optim.Adam(model.parameters(), lr=OUTER_LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=META_EPOCHS)

    for ep in range(META_EPOCHS):
        model.train()
        opt.zero_grad()
        meta_loss = torch.tensor(0.0, device=device)
        for _ in range(META_BATCH):
            sx, sy, qx, qy = sample_episode(X_by_cls, N_WAY, K_SHOT, Q_QUERY, device)
            adapted   = inner_loop(model, sx, sy, INNER_LR, INNER_STEPS)
            meta_loss = meta_loss + F.cross_entropy(model(qx, adapted), qy)
        (meta_loss / META_BATCH).backward()
        opt.step(); sched.step()
        if (ep+1) % 100 == 0:
            print(f"    ep {ep+1}/{META_EPOCHS} | meta-loss {meta_loss.item()/META_BATCH:.4f}")

    return model


def fine_tune_maml(model, X_tr, y_tr, n_cls, device):
    """Fine-tune with label smoothing + cosine annealing for 300 fixed epochs."""
    ft    = copy.deepcopy(model)
    opt   = optim.Adam(ft.parameters(), lr=1e-4, weight_decay=1e-2)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS, eta_min=1e-6)
    ce    = nn.CrossEntropyLoss(label_smoothing=0.1)

    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.long,    device=device)
    ds  = TensorDataset(X_t, y_t)
    dl  = DataLoader(ds, batch_size=64, shuffle=True)

    for ep in range(FT_EPOCHS):
        ft.train()
        for xb, yb in dl:
            opt.zero_grad()
            ce(ft(xb), yb).backward()
            opt.step()
        sched.step()
        if (ep+1) % 50 == 0:
            ft.eval()
            with torch.no_grad():
                tr_acc = (torch.argmax(ft(X_t), 1) == y_t).float().mean().item()
            print(f"    ep {ep+1}/{FT_EPOCHS} | train acc {tr_acc:.4f}")

    return ft


def eval_mlp(model, X_te, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_te, dtype=torch.float32, device=device)
        return torch.argmax(model(X_t), 1).cpu().numpy()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1. Features
    print("[1] Features (WavLM-base 768-dim + handcrafted 227-dim = 995-dim)")
    X, y = load_all_features()
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_cls = len(le.classes_)
    print(f"    {X.shape[0]} samples | {X.shape[1]} features | {n_cls} classes\n")

    # Splits: full train/test, plus a val slice for XGBoost probe
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=SEED
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.15, stratify=y_tr_full, random_state=SEED
    )

    scaler       = StandardScaler()
    X_tr_sc      = scaler.fit_transform(X_tr)
    X_trfull_sc  = scaler.transform(X_tr_full)
    X_val_sc     = scaler.transform(X_val)
    X_te_sc      = scaler.transform(X_te)
    y_tr_str     = le.inverse_transform(y_tr)
    y_trfull_str = le.inverse_transform(y_tr_full)

    # PCA for XGBoost (200 components captures ~95% variance of WavLM features)
    pca = PCA(n_components=200, random_state=SEED)
    pca.fit(X_tr_sc)
    print(f"    PCA 200-dim explains {pca.explained_variance_ratio_.sum()*100:.1f}% variance\n")

    # 2. SVM
    print("[2] SVM  (LinearSVC C=0.05, full training set)")
    svm      = fit_svm(X_trfull_sc, y_tr_full)
    svm_pred = svm.predict(X_te_sc)
    svm_acc  = accuracy_score(y_te, svm_pred)
    print(f"    Test accuracy: {svm_acc:.4f}")
    print(classification_report(y_te, svm_pred, target_names=le.classes_))

    # 3. XGBoost
    print("[3] XGBoost  (PCA 200-dim + probe early stopping → retrain on full)")
    xgb_m = fit_xgboost(X_tr_sc, y_tr, X_val_sc, y_val, pca)
    X_te_pca = pca.transform(X_te_sc)
    xgb_pred = xgb_m.predict(X_te_pca)
    xgb_acc  = accuracy_score(y_te, xgb_pred)
    print(f"    Test accuracy: {xgb_acc:.4f}")
    print(classification_report(y_te, xgb_pred, target_names=le.classes_))

    # Meta-features for MAML stacking
    def meta(X_sc):
        return np.hstack([
            X_sc,
            svm.predict_proba(X_sc),
            xgb_m.predict_proba(pca.transform(X_sc)),
        ])

    X_trfull_meta = meta(X_trfull_sc)
    X_te_meta     = meta(X_te_sc)
    print(f"\n    Meta-feature dim: {X_trfull_meta.shape[1]}")

    # 4. MAML
    print("\n[4] MAML  (meta-train → fine-tune with label smoothing, 300 epochs)")
    print("  ▸ Meta-training ...")
    maml     = train_maml(X_trfull_meta, y_trfull_str, n_cls, device)

    print("  ▸ Fine-tuning (300 epochs, cosine annealing + label smoothing) ...")
    maml_ft   = fine_tune_maml(maml, X_trfull_meta, y_tr_full, n_cls, device)
    maml_pred = eval_mlp(maml_ft, X_te_meta, device)
    maml_acc  = accuracy_score(y_te, maml_pred)
    print(f"\n    Test accuracy: {maml_acc:.4f}")
    print(classification_report(y_te, maml_pred, target_names=le.classes_))

    # Summary
    print("=" * 48)
    print(f"  SVM        : {svm_acc:.4f}  {'✓' if svm_acc >= 0.80 else '✗'}")
    print(f"  XGBoost    : {xgb_acc:.4f}  {'✓' if xgb_acc >= 0.80 else '✗'}")
    print(f"  MAML (ft)  : {maml_acc:.4f}  {'✓' if maml_acc >= 0.80 else '✗'}")
    print("=" * 48)


if __name__ == "__main__":
    main()
