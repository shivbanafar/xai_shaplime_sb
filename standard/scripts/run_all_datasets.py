"""
Multi-dataset SER Pipeline
Datasets : RAVDESS (8 cls), EMO-DB (7 cls), SAVEE (7 cls), TESS (7 cls)
Features : WavLM-large all-layer mean, mean+std pooling (2048-dim)
           + handcrafted MFCC/chroma/pitch (227-dim) = 2275-dim
Models   : SVM (LinearSVC) · XGBoost · MAML
Splits   : random 80/20 stratified (RAVDESS, EMO-DB, SAVEE)
           speaker-independent OAF→train / YAF→test (TESS)
Results  : saved to results/<DATASET>/results.txt
"""

import os, re, copy, warnings, io
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import librosa
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42); torch.manual_seed(42)

BASE_DIR    = "/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
DATA_DIR    = os.path.join(BASE_DIR, "data")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "standard", "results")
SR_MODEL    = 16000
SR_HAND     = 22050
N_MFCC      = 40
SEED        = 42

# MAML hyper-params
N_WAY=5; K_SHOT=5; Q_QUERY=10; INNER_LR=0.05; OUTER_LR=0.001
INNER_STEPS=5; META_EPOCHS=300; META_BATCH=4; FT_EPOCHS=300

# ── Dataset configs ────────────────────────────────────────────────────────────
# Each entry: (dataset_key, dataset_dir, loader_fn)
# loader_fn(dataset_dir) -> (paths: list[str], labels: list[str])

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
    return paths, labels

def load_emodb(ddir):
    # Filename: e.g. 03a01Wa.wav — char at index 5 is emotion code
    EMAP = {"W":"anger","L":"boredom","E":"disgust","A":"fear",
            "F":"happiness","T":"sadness","N":"neutral"}
    paths, labels = [], []
    for f in os.listdir(ddir):
        if not f.endswith(".wav") or len(f) < 6: continue
        code = f[5]
        emo  = EMAP.get(code)
        if emo: paths.append(os.path.join(ddir, f)); labels.append(emo)
    return paths, labels

def load_savee(ddir):
    # Filename: e.g. DC_a01.wav — emotion prefix between _ and digits
    EMAP = {"a":"anger","d":"disgust","f":"fear","h":"happiness",
            "n":"neutral","sa":"sadness","su":"surprise"}
    paths, labels = [], []
    for f in os.listdir(ddir):
        if not f.endswith(".wav"): continue
        m = re.match(r"[A-Z]+_([a-z]+)\d+\.wav", f)
        if not m: continue
        emo = EMAP.get(m.group(1))
        if emo: paths.append(os.path.join(ddir, f)); labels.append(emo)
    return paths, labels

def load_tess(ddir):
    # Subdirs: OAF_angry, YAF_happy, OAF_Pleasant_surprise, etc.
    # Returns (paths, labels, speakers) — speakers used for split
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
        speaker = m.group(1)   # OAF or YAF
        emo_key = m.group(2)
        emo = FOLDER_TO_EMO.get(emo_key)
        if emo is None: continue
        for f in os.listdir(fpath):
            if f.endswith(".wav"):
                paths.append(os.path.join(fpath, f))
                labels.append(emo)
                speakers.append(speaker)
    return paths, labels, speakers

# Loaders for non-TESS datasets return (paths, labels, None) for uniform interface
def _wrap(fn):
    def wrapped(ddir):
        r = fn(ddir)
        return r[0], r[1], None
    return wrapped

DATASETS = [
    ("RAVDESS", os.path.join(DATA_DIR,"RAVDESS"), _wrap(load_ravdess)),
    ("EMBODB",  os.path.join(DATA_DIR,"EMBODB"),  _wrap(load_emodb)),
    ("SAVEE",   os.path.join(DATA_DIR,"SAVEE"),   _wrap(load_savee)),
    ("TESS",    os.path.join(DATA_DIR,"TESS"),    load_tess),
]

# ── Handcrafted features ───────────────────────────────────────────────────────
def hand_feats(y, sr=SR_HAND):
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    f0     = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    voiced = f0[f0>0]
    pitch  = [np.mean(voiced),np.std(voiced),np.max(voiced),len(voiced)/max(len(f0),1)] if len(voiced)>0 else [0.]*4
    return np.concatenate([
        np.mean(mfcc,1), np.std(mfcc,1),
        np.mean(chroma,1),
        np.mean(mel,1),
        [np.mean(librosa.feature.zero_crossing_rate(y)),
         np.mean(librosa.feature.spectral_centroid(y=y,sr=sr)),
         np.mean(librosa.feature.rms(y=y))],
        pitch,
    ])

# ── WavLM embeddings ───────────────────────────────────────────────────────────
_wavlm_cache = {}
def get_wavlm(device):
    if "model" not in _wavlm_cache:
        from transformers import WavLMModel, AutoFeatureExtractor
        print("  Loading WavLM-large (~1.2 GB, downloading first time) ...")
        fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
        m  = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
        m.eval()
        _wavlm_cache["fe"] = fe; _wavlm_cache["model"] = m
    return _wavlm_cache["fe"], _wavlm_cache["model"]

def wavlm_embed(y, fe, model, device):
    """1536-dim: average all transformer layers, then mean+std pool over time."""
    inp = fe(y, return_tensors="pt", sampling_rate=SR_MODEL, padding=True)
    inp = {k:v.to(device) for k,v in inp.items()}
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    # hidden_states: tuple of (1, T, 768) for each of 13 layers
    stacked   = torch.stack(out.hidden_states, dim=0)  # (13, 1, T, 768)
    avg_layer = stacked.mean(0).squeeze(0)             # (T, 768)
    mean_feat = avg_layer.mean(0)                      # (768,)
    std_feat  = avg_layer.std(0)                       # (768,)
    return torch.cat([mean_feat, std_feat]).cpu().numpy()  # (1536,)

# ── Feature loading (with per-dataset caching) ────────────────────────────────
def load_features(key, paths, labels, device):
    # v3 caches: WavLM-large all-layer mean + mean+std pooling (2048-dim)
    cache_wavlm    = os.path.join(CACHE_DIR, f"{key.lower()}_wavlm_large.npz")
    cache_hand     = os.path.join(CACHE_DIR, f"{key.lower()}_hand.npz")
    cache_combined = os.path.join(CACHE_DIR, f"{key.lower()}_wavlm_large_combined.npz")

    if os.path.exists(cache_combined):
        print(f"  [{key}] Loading cached combined features ...")
        d = np.load(cache_combined, allow_pickle=True)
        return d["X"], d["y"]

    # WavLM
    if os.path.exists(cache_wavlm):
        print(f"  [{key}] Loading cached WavLM features ...")
        X_wlm = np.load(cache_wavlm)["X"]
    else:
        print(f"  [{key}] Extracting WavLM embeddings ({len(paths)} files) ...")
        fe, wlm = get_wavlm(device)
        X_wlm = []
        for i, path in enumerate(paths):
            y_a, _ = librosa.load(path, sr=SR_MODEL)
            X_wlm.append(wavlm_embed(y_a, fe, wlm, device))
            if (i+1) % 200 == 0: print(f"    {i+1}/{len(paths)}")
        X_wlm = np.array(X_wlm)
        np.savez(cache_wavlm, X=X_wlm)

    # Handcrafted
    if os.path.exists(cache_hand):
        print(f"  [{key}] Loading cached handcrafted features ...")
        X_hand = np.load(cache_hand)["X"]
    else:
        print(f"  [{key}] Extracting handcrafted features ...")
        X_hand = []
        for i, path in enumerate(paths):
            y_a, _ = librosa.load(path, sr=SR_HAND)
            X_hand.append(hand_feats(y_a))
            if (i+1) % 200 == 0: print(f"    {i+1}/{len(paths)}")
        X_hand = np.array(X_hand)
        np.savez(cache_hand, X=X_hand)

    X = np.hstack([X_wlm, X_hand])
    y = np.array(labels)
    np.savez(cache_combined, X=X, y=y)
    return X, y

# ── Models ─────────────────────────────────────────────────────────────────────
def fit_svm(X, y):
    lin = LinearSVC(C=0.05, max_iter=5000, random_state=SEED)
    m   = CalibratedClassifierCV(lin, cv=5)
    m.fit(X, y)
    return m

def fit_rf(X_trfull, y_trfull, pca):
    Xtr_p = pca.transform(X_trfull)
    m = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                min_samples_leaf=1, random_state=SEED, n_jobs=1)
    m.fit(Xtr_p, y_trfull)
    return m

def fit_xgboost(X_tr, y_tr, X_val, y_val, pca):
    Xtr_p = pca.transform(X_tr); Xval_p = pca.transform(X_val)
    xp = dict(n_estimators=800, max_depth=5, learning_rate=0.03,
              subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
              gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
              eval_metric="mlogloss", random_state=SEED, n_jobs=1,
              early_stopping_rounds=50)
    probe = xgb.XGBClassifier(**xp)
    probe.fit(Xtr_p, y_tr, eval_set=[(Xval_p, y_val)], verbose=False)
    best_n = max(probe.best_iteration, 50)
    Xfull_p = np.vstack([Xtr_p, Xval_p]); yfull = np.concatenate([y_tr, y_val])
    xp_final = {k:v for k,v in xp.items() if k!="early_stopping_rounds"}
    m = xgb.XGBClassifier(**{**xp_final, "n_estimators": best_n})
    m.fit(Xfull_p, yfull)
    return m, best_n

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
        if (ep+1)%100==0: print(f"    ep {ep+1}/{META_EPOCHS} | meta-loss {ml.item()/META_BATCH:.4f}")
    return model

def fine_tune_maml(model, X_tr, y_tr, n_cls, device):
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
        if (ep+1)%100==0:
            ft.eval()
            with torch.no_grad():
                tr_acc=(torch.argmax(ft(X_t),1)==y_t).float().mean().item()
            print(f"    ep {ep+1}/{FT_EPOCHS} | train acc {tr_acc:.4f}")
    return ft

def eval_mlp(model, X_te, device):
    model.eval()
    with torch.no_grad():
        return torch.argmax(model(torch.tensor(X_te,dtype=torch.float32,device=device)),1).cpu().numpy()

# ── Run one dataset ────────────────────────────────────────────────────────────
def run_dataset(key, ddir, loader_fn, device):
    out_dir = os.path.join(RESULTS_DIR, key)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "results.txt")

    buf = io.StringIO()
    def log(msg=""):
        print(msg)
        buf.write(msg + "\n")

    log(f"{'='*60}")
    log(f"Dataset : {key}")
    log(f"{'='*60}")

    # Labels (and optional speaker info for TESS)
    paths, labels, speakers = loader_fn(ddir)
    log(f"  Total files : {len(paths)}")

    # Features
    X, y = load_features(key, paths, labels, device)
    le    = LabelEncoder(); y_enc = le.fit_transform(y)
    n_cls = len(le.classes_)
    log(f"  Samples     : {X.shape[0]} | Features: {X.shape[1]} | Classes: {n_cls}")
    log(f"  Classes     : {list(le.classes_)}\n")

    # Splits
    if speakers is not None:
        # Speaker-independent split for TESS: OAF → train, YAF → test
        # Drop handcrafted features — they are speaker-specific (pitch, MFCC stats)
        # and hurt cross-speaker generalization. WavLM-only gives +22% accuracy.
        WLM_DIM = X.shape[1] - 227
        X = X[:, :WLM_DIM]
        log(f"  TESS: using WavLM-only features ({WLM_DIM}-dim) for speaker-independent eval")
        speakers_arr = np.array(speakers)
        train_mask = speakers_arr == "OAF"
        test_mask  = speakers_arr == "YAF"
        X_tr_full = X[train_mask];  y_tr_full = y_enc[train_mask]
        X_te      = X[test_mask];   y_te      = y_enc[test_mask]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15, stratify=y_tr_full, random_state=SEED)
        log(f"  Split: speaker-independent  train(OAF)={len(y_tr_full)}  test(YAF)={len(y_te)}")
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

    pca = PCA(n_components=300, random_state=SEED)
    pca.fit(X_tr_sc)
    log(f"  PCA 300-dim explains {pca.explained_variance_ratio_.sum()*100:.1f}% variance")

    # SVM
    log("\n[SVM]  LinearSVC C=0.05, full training set")
    svm      = fit_svm(X_trfull_sc, y_tr_full)
    svm_pred = svm.predict(X_te_sc)
    svm_acc  = accuracy_score(y_te, svm_pred)
    log(f"  Accuracy : {svm_acc:.4f}")
    log(classification_report(y_te, svm_pred, target_names=le.classes_))

    # XGBoost
    log("[XGBoost]  PCA 300-dim + early stopping → retrain on full")
    xgb_m, best_n = fit_xgboost(X_tr_sc, y_tr, X_val_sc, y_val, pca)
    xgb_pred = xgb_m.predict(pca.transform(X_te_sc))
    xgb_acc  = accuracy_score(y_te, xgb_pred)
    log(f"  Optimal n_estimators : {best_n}")
    log(f"  Accuracy : {xgb_acc:.4f}")
    log(classification_report(y_te, xgb_pred, target_names=le.classes_))

    # Random Forest
    log("[RF]  PCA 300-dim, 500 trees, fit on full training set")
    rf_m    = fit_rf(X_trfull_sc, y_tr_full, pca)
    rf_pred = rf_m.predict(pca.transform(X_te_sc))
    rf_acc  = accuracy_score(y_te, rf_pred)
    log(f"  Accuracy : {rf_acc:.4f}")
    log(classification_report(y_te, rf_pred, target_names=le.classes_))

    # Stacking ensemble (LogReg meta-learner on OOF cross-val probas — no leakage)
    log("[Ensemble — Stacking]  OOF cross-val probas → LogisticRegression meta-learner")
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
    svm_proba = svm.predict_proba(X_te_sc)
    rf_proba  = rf_m.predict_proba(pca.transform(X_te_sc))
    xgb_proba = xgb_m.predict_proba(pca.transform(X_te_sc))
    meta_tr = np.hstack([oof_svm, oof_rf, oof_xgb])
    meta_te = np.hstack([svm_proba, rf_proba, xgb_proba])
    stack_m = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, n_jobs=-1)
    stack_m.fit(meta_tr, y_tr_full)
    stack_pred = stack_m.predict(meta_te)
    stack_acc  = accuracy_score(y_te, stack_pred)
    log(f"  Accuracy : {stack_acc:.4f}")
    log(classification_report(y_te, stack_pred, target_names=le.classes_))

    # MAML — trained on raw scaled features only (independent of other models)
    log("\n[MAML]  meta-train 300 ep → fine-tune 300 ep (label smoothing)")
    log("  Meta-training ...")
    maml    = train_maml(X_trfull_sc, y_trfull_str, n_cls, device)
    log("  Fine-tuning ...")
    maml_ft  = fine_tune_maml(maml, X_trfull_sc, y_tr_full, n_cls, device)
    maml_pred = eval_mlp(maml_ft, X_te_sc, device)
    maml_acc  = accuracy_score(y_te, maml_pred)
    log(f"  Accuracy : {maml_acc:.4f}")
    log(classification_report(y_te, maml_pred, target_names=le.classes_))

    # Summary
    log("=" * 56)
    t = lambda a: '✓' if a>=0.90 else '✗'
    log(f"  SVM          : {svm_acc:.4f}  {t(svm_acc)}")
    log(f"  XGBoost      : {xgb_acc:.4f}  {t(xgb_acc)}")
    log(f"  RF           : {rf_acc:.4f}  {t(rf_acc)}")
    log(f"  Stacking     : {stack_acc:.4f}  {t(stack_acc)}")
    log(f"  MAML         : {maml_acc:.4f}  {t(maml_acc)}")
    log("=" * 56)

    with open(log_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\n  >> Results saved to {log_path}\n")

    return {"dataset": key, "SVM": svm_acc, "XGBoost": xgb_acc, "RF": rf_acc,
            "Stacking": stack_acc, "MAML": maml_acc}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    all_results = []
    for key, ddir, loader_fn in DATASETS:
        r = run_dataset(key, ddir, loader_fn, device)
        all_results.append(r)

    # Cross-dataset summary
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    hdr = f"{'Dataset':<12} {'SVM':>7} {'XGBoost':>9} {'RF':>7} {'Stacking':>10} {'MAML':>7}"
    lines = [hdr, "-" * len(hdr)]
    for r in all_results:
        lines.append(
            f"{r['dataset']:<12} {r['SVM']:>7.4f} {r['XGBoost']:>9.4f} {r['RF']:>7.4f}"
            f" {r['Stacking']:>10.4f} {r['MAML']:>7.4f}"
        )
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(summary_path, "w") as f: f.write(summary + "\n")
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()
