"""
XAI Analysis — SHAP + LIME on SER models
Datasets : RAVDESS, EMO-DB, SAVEE, TESS
Models   : SVM (LinearSVC) · XGBoost · MAML (MLP)
Outputs  : results/<DATASET>/xai/  (PNG plots + CSV feature importances)
"""

import os, re, copy, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42); torch.manual_seed(42)

BASE_DIR    = "/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
DATA_DIR    = os.path.join(BASE_DIR, "data")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SEED = 42

# MAML hyper-params (match run_all_datasets.py)
N_WAY=5; K_SHOT=5; Q_QUERY=10; INNER_LR=0.05; OUTER_LR=0.001
INNER_STEPS=5; META_EPOCHS=300; META_BATCH=4; FT_EPOCHS=300

# ── Feature names ──────────────────────────────────────────────────────────────
def get_feature_names(wavlm_dim=2048, hand_dim=227):
    """Names for the combined WavLM-large + handcrafted feature vector."""
    half = wavlm_dim // 2
    names  = [f"wavlm_mean_{i}" for i in range(half)]
    names += [f"wavlm_std_{i}"  for i in range(half)]
    names += [f"mfcc_mean_{i+1}" for i in range(40)]
    names += [f"mfcc_std_{i+1}"  for i in range(40)]
    names += [f"chroma_{i+1}"    for i in range(12)]
    names += [f"mel_{i+1}"       for i in range(128)]
    names += ["zcr", "spectral_centroid", "rms"]
    names += ["pitch_mean", "pitch_std", "pitch_max", "pitch_voiced_frac"]
    return names

# Feature groups for aggregate SHAP importance
FEATURE_GROUPS = {
    "WavLM (mean)":      (0,    1024),
    "WavLM (std)":       (1024, 2048),
    "MFCC (mean)":       (2048, 2088),
    "MFCC (std)":        (2088, 2128),
    "Chroma":            (2128, 2140),
    "Mel-spectrogram":   (2140, 2268),
    "ZCR":               (2268, 2269),
    "Spectral Centroid": (2269, 2270),
    "RMS Energy":        (2270, 2271),
    "Pitch (F0)":        (2271, 2275),
}

# ── Dataset loaders ────────────────────────────────────────────────────────────
def load_ravdess(ddir):
    EMAP = {"01":"neutral","02":"calm","03":"happy","04":"sad",
            "05":"angry","06":"fearful","07":"disgust","08":"surprised"}
    paths, labels = [], []
    for actor in sorted(os.listdir(ddir)):
        adir = os.path.join(ddir, actor)
        if not os.path.isdir(adir): continue
        for f in os.listdir(adir):
            if not f.endswith(".wav"): continue
            parts = f.replace(".wav","").split("-")
            emo = EMAP.get(parts[2] if len(parts)>2 else "")
            if emo: paths.append(os.path.join(adir, f)); labels.append(emo)
    return paths, labels, None

def load_emodb(ddir):
    EMAP = {"W":"anger","L":"boredom","E":"disgust","A":"fear",
            "F":"happiness","T":"sadness","N":"neutral"}
    paths, labels = [], []
    for f in os.listdir(ddir):
        if not f.endswith(".wav") or len(f)<6: continue
        emo = EMAP.get(f[5])
        if emo: paths.append(os.path.join(ddir, f)); labels.append(emo)
    return paths, labels, None

def load_savee(ddir):
    EMAP = {"a":"anger","d":"disgust","f":"fear","h":"happiness",
            "n":"neutral","sa":"sadness","su":"surprise"}
    paths, labels = [], []
    for f in os.listdir(ddir):
        if not f.endswith(".wav"): continue
        m = re.match(r"[A-Z]+_([a-z]+)\d+\.wav", f)
        if not m: continue
        emo = EMAP.get(m.group(1))
        if emo: paths.append(os.path.join(ddir, f)); labels.append(emo)
    return paths, labels, None

def load_tess(ddir):
    FOLDER_TO_EMO = {
        "angry":"anger","disgust":"disgust","fear":"fear","Fear":"fear",
        "happy":"happiness","neutral":"neutral","sad":"sadness","Sad":"sadness",
        "Pleasant_surprise":"surprise","pleasant_surprised":"surprise",
    }
    paths, labels, speakers = [], [], []
    for folder in os.listdir(ddir):
        fpath = os.path.join(ddir, folder)
        if not os.path.isdir(fpath): continue
        m = re.match(r"([A-Z]+)_(.*)", folder)
        if not m: continue
        speaker = m.group(1); emo = FOLDER_TO_EMO.get(m.group(2))
        if emo is None: continue
        for f in os.listdir(fpath):
            if f.endswith(".wav"):
                paths.append(os.path.join(fpath, f))
                labels.append(emo); speakers.append(speaker)
    return paths, labels, speakers

DATASETS = [
    ("RAVDESS", os.path.join(DATA_DIR,"RAVDESS"), load_ravdess),
    ("EMBODB",  os.path.join(DATA_DIR,"EMBODB"),  load_emodb),
    ("SAVEE",   os.path.join(DATA_DIR,"SAVEE"),   load_savee),
    ("TESS",    os.path.join(DATA_DIR,"TESS"),    load_tess),
]

# ── MLP (MAML architecture) ────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.fc1=nn.Linear(in_dim,512); self.bn1=nn.BatchNorm1d(512)
        self.fc2=nn.Linear(512,256);   self.bn2=nn.BatchNorm1d(256)
        self.fc3=nn.Linear(256,128);   self.bn3=nn.BatchNorm1d(128)
        self.fc4=nn.Linear(128,n_cls)
        self.d1=nn.Dropout(0.4); self.d2=nn.Dropout(0.35); self.d3=nn.Dropout(0.3)
        self.linears=nn.ModuleList([self.fc1,self.fc2,self.fc3,self.fc4])
    def forward(self, x, weights=None):
        if weights is None:
            x=self.d1(F.relu(self.bn1(self.fc1(x))))
            x=self.d2(F.relu(self.bn2(self.fc2(x))))
            x=self.d3(F.relu(self.bn3(self.fc3(x))))
            return self.fc4(x)
        for i,lin in enumerate(self.linears[:-1]):
            x=F.relu(F.linear(x,weights[2*i],weights[2*i+1]))
        return F.linear(x,weights[-2],weights[-1])

def inner_loop(model, sx, sy, lr, steps, create_graph=True):
    w=[]
    for lin in model.linears: w+=[lin.weight,lin.bias]
    for _ in range(steps):
        loss=F.cross_entropy(model(sx,w),sy)
        grads=torch.autograd.grad(loss,w,create_graph=create_graph)
        w=[p-lr*g for p,g in zip(w,grads)]
    return w

def sample_episode(X_by_cls, n_way, k_shot, q_query, device):
    chosen=np.random.choice(list(X_by_cls.keys()),min(n_way,len(X_by_cls)),replace=False)
    sx,sy,qx,qy=[],[],[],[]
    for i,cls in enumerate(chosen):
        arr=X_by_cls[cls]; need=k_shot+q_query
        idx=np.random.choice(len(arr),min(need,len(arr)),replace=False)
        k=min(k_shot,max(len(idx)//2,1))
        sx.append(arr[idx[:k]]); sy+=[i]*k
        qx.append(arr[idx[k:]]); qy+=[i]*len(idx[k:])
    t=lambda a:torch.tensor(np.vstack(a),dtype=torch.float32,device=device)
    l=lambda a:torch.tensor(a,dtype=torch.long,device=device)
    return t(sx),l(sy),t(qx),l(qy)

def train_maml(X_tr, y_tr_str, n_cls, device):
    X_by_cls=defaultdict(list)
    for x,lbl in zip(X_tr,y_tr_str): X_by_cls[lbl].append(x)
    X_by_cls={k:np.array(v) for k,v in X_by_cls.items()}
    model=MLP(X_tr.shape[1],n_cls).to(device)
    opt=optim.Adam(model.parameters(),lr=OUTER_LR,weight_decay=1e-4)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=META_EPOCHS)
    for ep in range(META_EPOCHS):
        model.train(); opt.zero_grad()
        ml=torch.tensor(0.,device=device)
        for _ in range(META_BATCH):
            sx,sy,qx,qy=sample_episode(X_by_cls,N_WAY,K_SHOT,Q_QUERY,device)
            ml=ml+F.cross_entropy(model(qx,inner_loop(model,sx,sy,INNER_LR,INNER_STEPS)),qy)
        (ml/META_BATCH).backward(); opt.step(); sched.step()
        if (ep+1)%100==0: print(f"    meta ep {ep+1}/{META_EPOCHS} loss={ml.item()/META_BATCH:.4f}")
    return model

def fine_tune_maml(model, X_tr, y_tr, device):
    ft=copy.deepcopy(model)
    opt=optim.Adam(ft.parameters(),lr=1e-4,weight_decay=1e-2)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=FT_EPOCHS,eta_min=1e-6)
    ce=nn.CrossEntropyLoss(label_smoothing=0.1)
    X_t=torch.tensor(X_tr,dtype=torch.float32,device=device)
    y_t=torch.tensor(y_tr,dtype=torch.long,device=device)
    dl=DataLoader(TensorDataset(X_t,y_t),batch_size=64,shuffle=True)
    for ep in range(FT_EPOCHS):
        ft.train()
        for xb,yb in dl: opt.zero_grad(); ce(ft(xb),yb).backward(); opt.step()
        sched.step()
    return ft

# ── Plotting helpers ───────────────────────────────────────────────────────────
COLORS = ["#2196F3","#4CAF50","#F44336","#FF9800","#9C27B0",
          "#00BCD4","#795548","#607D8B","#E91E63","#3F51B5"]

def bar_plot(values, labels, title, path, color="#2196F3", top_n=20):
    idx = np.argsort(values)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), values[idx][::-1], color=color)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([labels[i] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved: {path}")

def group_bar_plot(group_vals, title, path):
    names  = list(group_vals.keys())
    values = list(group_vals.values())
    colors = COLORS[:len(names)]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel("Sum of Mean |SHAP value|")
    ax.set_title(title)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved: {path}")

def lime_plot(exp, class_names, title, path):
    """Save a LIME explanation bar chart."""
    label = exp.top_labels[0]
    explanation = exp.as_list(label=label)
    feats  = [e[0] for e in explanation]
    values = [e[1] for e in explanation]
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in values]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(feats)), values, color=colors)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME weight")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved: {path}")

# ── Per-dataset XAI ────────────────────────────────────────────────────────────
def run_xai(key, ddir, loader_fn, device):
    xai_dir = os.path.join(RESULTS_DIR, key, "xai")
    os.makedirs(xai_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"XAI: {key}")
    print(f"{'='*60}")

    # Load cached features
    cache = os.path.join(CACHE_DIR, f"{key.lower()}_wavlm_large_combined.npz")
    if not os.path.exists(cache):
        print(f"  Cache not found: {cache}  — skipping {key}"); return

    d = np.load(cache, allow_pickle=True)
    X, y = d["X"], d["y"]
    le    = LabelEncoder(); y_enc = le.fit_transform(y)
    n_cls = len(le.classes_)
    feat_names = get_feature_names(wavlm_dim=X.shape[1] - 227)

    # Splits (mirror run_all_datasets.py)
    _, _, speakers_raw = loader_fn(ddir)
    speakers = np.array(speakers_raw) if speakers_raw is not None else None

    if speakers is not None:
        # TESS: drop handcrafted features (speaker-specific) for cross-speaker eval
        WLM_DIM = X.shape[1] - 227
        X = X[:, :WLM_DIM]
        feat_names = get_feature_names(wavlm_dim=WLM_DIM)[:WLM_DIM]
        train_mask = speakers == "OAF"; test_mask = speakers == "YAF"
        X_tr_full = X[train_mask]; y_tr_full = y_enc[train_mask]
        X_te      = X[test_mask];  y_te      = y_enc[test_mask]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15, stratify=y_tr_full, random_state=SEED)
    else:
        X_tr_full, X_te, y_tr_full, y_te = train_test_split(
            X, y_enc, test_size=0.2, stratify=y_enc, random_state=SEED)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15, stratify=y_tr_full, random_state=SEED)

    scaler = StandardScaler()
    X_tr_sc     = scaler.fit_transform(X_tr)
    X_trfull_sc = scaler.transform(X_tr_full)
    X_val_sc    = scaler.transform(X_val)
    X_te_sc     = scaler.transform(X_te)
    y_trfull_str = le.inverse_transform(y_tr_full)

    pca = PCA(n_components=min(300, X_tr_sc.shape[1]), random_state=SEED)
    pca.fit(X_tr_sc)
    pca_names = [f"PC{i+1}" for i in range(pca.n_components_)]

    # ── Train models ──────────────────────────────────────────────────────────
    print("  Training SVM ...")
    svm = CalibratedClassifierCV(LinearSVC(C=0.05, max_iter=5000, random_state=SEED), cv=5)
    svm.fit(X_trfull_sc, y_tr_full)
    print(f"    SVM test acc: {accuracy_score(y_te, svm.predict(X_te_sc)):.4f}")

    print("  Training XGBoost ...")
    Xtr_p = pca.transform(X_tr_sc); Xval_p = pca.transform(X_val_sc)
    Xte_p = pca.transform(X_te_sc)
    probe = xgb.XGBClassifier(n_estimators=800, max_depth=5, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                               gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
                               eval_metric="mlogloss", random_state=SEED, n_jobs=1,
                               early_stopping_rounds=50)
    probe.fit(Xtr_p, y_tr, eval_set=[(Xval_p, y_val)], verbose=False)
    best_n = max(probe.best_iteration, 50)
    Xfull_p = np.vstack([Xtr_p, Xval_p]); yfull = np.concatenate([y_tr, y_val])
    xgb_m = xgb.XGBClassifier(n_estimators=best_n, max_depth=5, learning_rate=0.03,
                                subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                                gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
                                eval_metric="mlogloss", random_state=SEED, n_jobs=1)
    xgb_m.fit(Xfull_p, yfull)
    print(f"    XGBoost test acc: {accuracy_score(y_te, xgb_m.predict(Xte_p)):.4f}")

    print("  Training RF ...")
    rf_m = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                   min_samples_leaf=1, random_state=SEED, n_jobs=1)
    rf_m.fit(pca.transform(X_trfull_sc), y_tr_full)
    print(f"    RF test acc: {accuracy_score(y_te, rf_m.predict(Xte_p)):.4f}")

    # Stacking ensemble: OOF probas → LogisticRegression meta-learner
    print("  Training Stacking ensemble ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    Xf_pca = pca.transform(X_trfull_sc)
    oof_svm = cross_val_predict(
        CalibratedClassifierCV(LinearSVC(C=0.05, max_iter=5000, random_state=SEED), cv=5),
        X_trfull_sc, y_tr_full, cv=skf, method="predict_proba", n_jobs=1)
    oof_rf  = cross_val_predict(
        RandomForestClassifier(n_estimators=500, max_features="sqrt",
                               min_samples_leaf=1, random_state=SEED, n_jobs=1),
        Xf_pca, y_tr_full, cv=skf, method="predict_proba", n_jobs=1)
    oof_xgb = cross_val_predict(
        xgb.XGBClassifier(n_estimators=best_n, max_depth=5, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.5, random_state=SEED,
                          n_jobs=1, eval_metric="mlogloss"),
        Xf_pca, y_tr_full, cv=skf, method="predict_proba", n_jobs=1)
    svm_proba_te  = svm.predict_proba(X_te_sc)
    rf_proba_te   = rf_m.predict_proba(Xte_p)
    xgb_proba_te  = xgb_m.predict_proba(Xte_p)
    meta_tr = np.hstack([oof_svm, oof_rf, oof_xgb])
    meta_te = np.hstack([svm_proba_te, rf_proba_te, xgb_proba_te])
    stack_m = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, n_jobs=-1)
    stack_m.fit(meta_tr, y_tr_full)
    print(f"    Stacking test acc: {accuracy_score(y_te, stack_m.predict(meta_te)):.4f}")
    # Meta-feature names for stacking
    stack_feat_names = (
        [f"svm_{c}" for c in le.classes_] +
        [f"rf_{c}"  for c in le.classes_] +
        [f"xgb_{c}" for c in le.classes_]
    )

    # MAML — raw scaled features only (independent of other models)
    print("  Training MAML (meta-train + fine-tune) ...")
    maml    = train_maml(X_trfull_sc, y_trfull_str, n_cls, device)
    maml_ft = fine_tune_maml(maml, X_trfull_sc, y_tr_full, device)
    maml_ft.eval()
    with torch.no_grad():
        preds = torch.argmax(maml_ft(torch.tensor(X_te_sc, dtype=torch.float32, device=device)), 1).cpu().numpy()
    print(f"    MAML test acc: {accuracy_score(y_te, preds):.4f}")

    # Background sets (small for speed)
    bg_idx  = np.random.choice(len(X_trfull_sc), min(100, len(X_trfull_sc)), replace=False)
    te_idx  = np.random.choice(len(X_te_sc),     min(100, len(X_te_sc)),     replace=False)
    X_bg    = X_trfull_sc[bg_idx]
    X_te_s  = X_te_sc[te_idx]
    X_bg_pca = pca.transform(X_bg)
    X_te_pca = pca.transform(X_te_s)
    # Stacking meta-features for background/test subsets
    X_bg_meta = np.hstack([svm.predict_proba(X_bg),
                           rf_m.predict_proba(X_bg_pca),
                           xgb_m.predict_proba(X_bg_pca)])
    X_te_meta_s = np.hstack([svm.predict_proba(X_te_s),
                              rf_m.predict_proba(pca.transform(X_te_s)),
                              xgb_m.predict_proba(pca.transform(X_te_s))])

    # ── SHAP: SVM (linear) ────────────────────────────────────────────────────
    print("  SHAP: SVM (linear explainer) ...")
    # Extract averaged coefficients from CalibratedClassifierCV
    coefs = np.mean([c.estimator.coef_ for c in svm.calibrated_classifiers_], axis=0)
    # Mean absolute coefficient per feature (aggregated over all classes)
    mean_abs_coef = np.mean(np.abs(coefs), axis=0)

    # Top-20 handcrafted features (skip WavLM dims for interpretability)
    hand_start = X.shape[1] - 227
    hand_coef  = mean_abs_coef[hand_start:]
    hand_names = feat_names[hand_start:]

    bar_plot(hand_coef, hand_names,
             f"SVM — Handcrafted Feature Importance (mean |coef|)\n{key}",
             os.path.join(xai_dir, "svm_feature_importance_handcrafted.png"),
             color="#2196F3")

    # Feature group importance (all features)
    group_vals = {}
    for gname, (g0, g1) in FEATURE_GROUPS.items():
        g1 = min(g1, len(mean_abs_coef))
        group_vals[gname] = float(mean_abs_coef[g0:g1].sum())
    group_bar_plot(group_vals,
                   f"SVM — Feature Group Importance\n{key}",
                   os.path.join(xai_dir, "svm_feature_groups.png"))

    # Per-class SHAP-like importances (top emotion drivers)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8)) if n_cls == 8 else plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for ci, cls_name in enumerate(le.classes_):
        if ci >= len(axes): break
        top_idx = np.argsort(np.abs(coefs[ci, hand_start:]))[::-1][:10]
        vals    = coefs[ci, hand_start:][top_idx]
        colors  = ["#4CAF50" if v>0 else "#F44336" for v in vals]
        axes[ci].barh([hand_names[i] for i in top_idx][::-1], vals[::-1], color=colors[::-1])
        axes[ci].set_title(cls_name, fontsize=10)
        axes[ci].axvline(0, color="black", linewidth=0.5)
    for ci in range(n_cls, len(axes)): axes[ci].axis("off")
    plt.suptitle(f"SVM — Top Handcrafted Features per Emotion Class\n{key}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(xai_dir, "svm_per_class_features.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    Saved: {path}")

    # ── SHAP: XGBoost (TreeExplainer) ────────────────────────────────────────
    print("  SHAP: XGBoost (TreeExplainer) ...")
    explainer = shap.TreeExplainer(xgb_m)
    shap_vals  = explainer.shap_values(X_te_pca)  # (n_samples, n_pca, n_cls) or (n_samples, n_pca)

    # Handle list / 2D / 3D shap_values
    if isinstance(shap_vals, list):
        sv = np.stack(shap_vals, axis=-1)   # (n_samples, n_pca, n_cls)
    elif shap_vals.ndim == 3:
        sv = shap_vals                       # already (n_samples, n_pca, n_cls)
    elif shap_vals.ndim == 2:
        sv = shap_vals[:, :, np.newaxis]     # (n_samples, n_pca, 1)
    else:
        sv = shap_vals.reshape(shap_vals.shape[0], -1, 1)
    mean_abs_shap = np.mean(np.abs(sv), axis=(0, 2))  # (n_pca,)

    bar_plot(mean_abs_shap, pca_names,
             f"XGBoost — PCA Feature Importance (SHAP)\n{key}",
             os.path.join(xai_dir, "xgboost_shap_pca.png"),
             color="#4CAF50")

    # SHAP summary beeswarm on top-20 PCA features
    top20 = np.argsort(mean_abs_shap)[::-1][:20]
    sv_top = sv[:, top20, :]
    sv_2d  = sv_top.mean(-1)   # average over classes for summary
    X_top  = X_te_pca[:, top20]
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv_2d, X_top,
                      feature_names=[pca_names[i] for i in top20],
                      plot_type="dot", show=False, max_display=20)
    plt.title(f"XGBoost SHAP Beeswarm (top 20 PCA components)\n{key}")
    plt.tight_layout()
    path = os.path.join(xai_dir, "xgboost_shap_beeswarm.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    Saved: {path}")

    # ── SHAP: MAML (KernelExplainer, subset) ─────────────────────────────────
    print("  SHAP: MAML (KernelExplainer, 50 background × 30 test) ...")
    def maml_predict(X_np):
        maml_ft.eval()
        with torch.no_grad():
            t = torch.tensor(X_np, dtype=torch.float32, device=device)
            probs = torch.softmax(maml_ft(t), dim=1).cpu().numpy()
        return probs

    n_bg_maml = min(50, len(X_bg))
    n_te_maml = min(30, len(X_te_s))
    explainer_maml = shap.KernelExplainer(maml_predict, X_bg[:n_bg_maml], link="identity")
    shap_maml = explainer_maml.shap_values(X_te_s[:n_te_maml], nsamples=200, silent=True)

    if isinstance(shap_maml, list):
        sv_maml = np.mean(np.abs(np.stack(shap_maml, axis=-1)), axis=(0, 2))
    elif shap_maml.ndim == 3:
        sv_maml = np.mean(np.abs(shap_maml), axis=(0, 2))
    else:
        sv_maml = np.mean(np.abs(shap_maml), axis=0)

    # Show handcrafted portion (or all features for TESS WavLM-only)
    n_feat = X_te_s.shape[1]
    hand_start = n_feat - 227 if n_feat > 227 else 0
    if hand_start > 0:
        focus_idx   = list(range(hand_start, n_feat))
        focus_names = feat_names[hand_start:]
        focus_vals  = sv_maml[focus_idx]
    else:
        focus_idx   = list(range(min(50, n_feat)))
        focus_names = feat_names[:len(focus_idx)]
        focus_vals  = sv_maml[focus_idx]

    bar_plot(focus_vals, focus_names,
             f"MAML — Feature Importance (SHAP)\n{key}",
             os.path.join(xai_dir, "maml_shap_features.png"),
             color="#9C27B0")

    # ── SHAP: Stacking (LinearExplainer on LogReg) ────────────────────────────
    print("  SHAP: Stacking (LinearExplainer) ...")
    explainer_stack = shap.LinearExplainer(stack_m, X_bg_meta, feature_perturbation="interventional")
    shap_stack = explainer_stack.shap_values(X_te_meta_s)  # list of (n_te, n_stack_feat) or 3D

    if isinstance(shap_stack, list):
        sv_stack = np.mean(np.abs(np.stack(shap_stack, axis=-1)), axis=(0, 2))
    elif shap_stack.ndim == 3:
        sv_stack = np.mean(np.abs(shap_stack), axis=(0, 2))
    else:
        sv_stack = np.mean(np.abs(shap_stack), axis=0)

    bar_plot(sv_stack, stack_feat_names,
             f"Stacking — Feature Importance (SHAP, base model probabilities)\n{key}",
             os.path.join(xai_dir, "stacking_shap_features.png"),
             color="#FF9800", top_n=len(stack_feat_names))

    # Grouped by base model: how much does each model contribute overall?
    n_cls_feat = n_cls
    model_contrib = {
        "SVM":     float(sv_stack[:n_cls_feat].sum()),
        "RF":      float(sv_stack[n_cls_feat:2*n_cls_feat].sum()),
        "XGBoost": float(sv_stack[2*n_cls_feat:].sum()),
    }
    group_bar_plot(model_contrib,
                   f"Stacking — Base Model Contribution (SHAP)\n{key}",
                   os.path.join(xai_dir, "stacking_shap_model_contribution.png"))

    # Per-class SHAP for stacking LogReg (coef_ shape: n_cls × n_stack_feat)
    coef_stack = stack_m.coef_  # (n_cls, n_stack_feat) for multiclass
    fig, axes = plt.subplots(2, (n_cls+1)//2, figsize=(14, 8))
    axes = axes.flatten()
    for ci, cls_name in enumerate(le.classes_):
        if ci >= len(axes): break
        vals   = coef_stack[ci]
        colors = ["#4CAF50" if v > 0 else "#F44336" for v in vals]
        axes[ci].barh(stack_feat_names[::-1], vals[::-1], color=colors[::-1])
        axes[ci].set_title(cls_name, fontsize=9)
        axes[ci].axvline(0, color="black", linewidth=0.5)
        axes[ci].tick_params(axis='y', labelsize=7)
    for ci in range(n_cls, len(axes)): axes[ci].axis("off")
    plt.suptitle(f"Stacking — LogReg Coefficients per Emotion Class\n{key}", fontsize=11)
    plt.tight_layout()
    path = os.path.join(xai_dir, "stacking_shap_per_class.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    Saved: {path}")

    # ── LIME ─────────────────────────────────────────────────────────────────
    print("  LIME: SVM, XGBoost, MAML ...")

    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        X_trfull_sc, feature_names=feat_names,
        class_names=list(le.classes_), mode="classification",
        random_state=SEED, discretize_continuous=False)

    lime_exp_pca = lime.lime_tabular.LimeTabularExplainer(
        pca.transform(X_trfull_sc), feature_names=pca_names,
        class_names=list(le.classes_), mode="classification",
        random_state=SEED, discretize_continuous=False)

    lime_exp_stack = lime.lime_tabular.LimeTabularExplainer(
        meta_tr, feature_names=stack_feat_names,
        class_names=list(le.classes_), mode="classification",
        random_state=SEED, discretize_continuous=False)

    # Explain 3 samples
    for idx_in_te, sample_idx in enumerate(te_idx[:3]):
        true_cls = le.inverse_transform([y_te[sample_idx]])[0]

        # SVM LIME
        exp_svm = lime_exp.explain_instance(
            X_te_sc[sample_idx], svm.predict_proba,
            num_features=15, top_labels=1)
        lime_plot(exp_svm, list(le.classes_),
                  f"LIME: SVM — sample {sample_idx} (true: {true_cls})\n{key}",
                  os.path.join(xai_dir, f"lime_svm_sample{idx_in_te}.png"))

        # XGBoost LIME
        exp_xgb = lime_exp_pca.explain_instance(
            Xte_p[idx_in_te], xgb_m.predict_proba,
            num_features=15, top_labels=1)
        lime_plot(exp_xgb, list(le.classes_),
                  f"LIME: XGBoost — sample {sample_idx} (true: {true_cls})\n{key}",
                  os.path.join(xai_dir, f"lime_xgboost_sample{idx_in_te}.png"))

        # MAML LIME (raw scaled features)
        exp_maml = lime_exp.explain_instance(
            X_te_s[idx_in_te], maml_predict,
            num_features=15, top_labels=1)
        lime_plot(exp_maml, list(le.classes_),
                  f"LIME: MAML — sample {sample_idx} (true: {true_cls})\n{key}",
                  os.path.join(xai_dir, f"lime_maml_sample{idx_in_te}.png"))

        # Stacking LIME
        exp_stack = lime_exp_stack.explain_instance(
            X_te_meta_s[idx_in_te], stack_m.predict_proba,
            num_features=len(stack_feat_names), top_labels=1)
        lime_plot(exp_stack, list(le.classes_),
                  f"LIME: Stacking — sample {sample_idx} (true: {true_cls})\n{key}",
                  os.path.join(xai_dir, f"lime_stacking_sample{idx_in_te}.png"))

    print(f"\n  XAI complete for {key}. Plots saved to {xai_dir}/")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    for key, ddir, loader_fn in DATASETS:
        run_xai(key, ddir, loader_fn, device)
    print("\nAll XAI analysis complete.")

if __name__ == "__main__":
    main()
