"""
XAI for cross-corpus SER  —  SHAP + LIME on MAML + SVM
Train SAVEE (Eng) -> Test EMO-DB (Ger), WavLM-only 2048-dim, per-corpus z-norm.

Produces the same graph TYPES as scripts/xai_analysis.py (adapted to WavLM-only:
no handcrafted features, so feature-group/importance plots are over WavLM dims):
  SVM  : svm_feature_groups.png · svm_feature_importance.png · svm_per_class_features.png
  MAML : maml_shap_features.png  (KernelExplainer)
  LIME : lime_svm_sample{0,1,2}.png · lime_maml_sample{0,1,2}.png

Output: results/CROSS_CORPUS/xai/
"""
import os, copy, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42); torch.manual_seed(42)

BASE_DIR  = "/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
CACHE_DIR = os.path.join(BASE_DIR, "cache")
XAI_DIR   = os.path.join(BASE_DIR, "cross_corpus", "results", "xai")
SEED = 42
SHARED = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
WAVLM_DIM = 2048
SVM_C = 0.1; SVM_PCA = 300

N_WAY=5; K_SHOT=5; Q_QUERY=10; INNER_LR=0.05; OUTER_LR=0.001
INNER_STEPS=5; META_EPOCHS=300; META_BATCH=8; FT_EPOCHS=300
GRAD_CLIP=1.0; EMA_DECAY=0.995   # stabilizers (match stabilize_maml.py)

COLORS = ["#2196F3","#4CAF50","#F44336","#FF9800","#9C27B0",
          "#00BCD4","#795548","#607D8B","#E91E63","#3F51B5"]

def feat_names():
    half = WAVLM_DIM // 2
    return [f"wavlm_mean_{i}" for i in range(half)] + [f"wavlm_std_{i}" for i in range(half)]

FEATURE_GROUPS = {"WavLM (mean)": (0, 1024), "WavLM (std)": (1024, 2048)}

# ── plotting helpers (match xai_analysis.py) ─────────────────────────────────
def bar_plot(values, labels, title, path, color="#2196F3", top_n=20):
    values = np.asarray(values); top_n = min(top_n, len(values))
    idx = np.argsort(values)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), values[idx][::-1], color=color)
    ax.set_yticks(range(top_n)); ax.set_yticklabels([labels[i] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); print(f"    Saved: {path}")

def group_bar_plot(group_vals, title, path):
    names=list(group_vals.keys()); values=list(group_vals.values()); colors=COLORS[:len(names)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel("Sum of Mean |importance|"); ax.set_title(title)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); print(f"    Saved: {path}")

def lime_plot(exp, title, path):
    label = exp.top_labels[0]; explanation = exp.as_list(label=label)
    feats=[e[0] for e in explanation]; values=[e[1] for e in explanation]
    colors=["#4CAF50" if v>0 else "#F44336" for v in values]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(feats)), values, color=colors)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8); ax.set_xlabel("LIME weight"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); print(f"    Saved: {path}")

# ── MLP / MAML ───────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.fc1=nn.Linear(in_dim,512); self.bn1=nn.BatchNorm1d(512)
        self.fc2=nn.Linear(512,256);   self.bn2=nn.BatchNorm1d(256)
        self.fc3=nn.Linear(256,128);   self.bn3=nn.BatchNorm1d(128)
        self.fc4=nn.Linear(128,n_cls)
        self.d1=nn.Dropout(0.4); self.d2=nn.Dropout(0.35); self.d3=nn.Dropout(0.3)
        self.linears=nn.ModuleList([self.fc1,self.fc2,self.fc3,self.fc4])
    def forward(self, x, w=None):
        if w is None:
            x=self.d1(F.relu(self.bn1(self.fc1(x)))); x=self.d2(F.relu(self.bn2(self.fc2(x))))
            x=self.d3(F.relu(self.bn3(self.fc3(x)))); return self.fc4(x)
        for i,l in enumerate(self.linears[:-1]): x=F.relu(F.linear(x,w[2*i],w[2*i+1]))
        return F.linear(x,w[-2],w[-1])

def inner_loop(m,sx,sy,lr,steps):
    w=[]
    for l in m.linears: w+=[l.weight,l.bias]
    for _ in range(steps):
        g=torch.autograd.grad(F.cross_entropy(m(sx,w),sy),w,create_graph=True)
        w=[p-lr*gg for p,gg in zip(w,g)]
    return w

def sample_episode(D,dev):
    ch=np.random.choice(list(D.keys()),min(N_WAY,len(D)),replace=False); sx,sy,qx,qy=[],[],[],[]
    for i,c in enumerate(ch):
        a=D[c]; idx=np.random.choice(len(a),min(K_SHOT+Q_QUERY,len(a)),replace=False); k=min(K_SHOT,max(len(idx)//2,1))
        sx.append(a[idx[:k]]); sy+=[i]*k; qx.append(a[idx[k:]]); qy+=[i]*len(idx[k:])
    t=lambda z:torch.tensor(np.vstack(z),dtype=torch.float32,device=dev); l=lambda z:torch.tensor(z,dtype=torch.long,device=dev)
    return t(sx),l(sy),t(qx),l(qy)

def train_maml(X,ystr,n_cls,dev):
    D=defaultdict(list)
    for x,lb in zip(X,ystr): D[lb].append(x)
    D={k:np.array(v) for k,v in D.items()}
    m=MLP(X.shape[1],n_cls).to(dev); opt=optim.Adam(m.parameters(),lr=OUTER_LR,weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=META_EPOCHS)
    for ep in range(META_EPOCHS):
        m.train(); opt.zero_grad(); ml=torch.tensor(0.,device=dev)
        for _ in range(META_BATCH):
            sx,sy,qx,qy=sample_episode(D,dev); ml=ml+F.cross_entropy(m(qx,inner_loop(m,sx,sy,INNER_LR,INNER_STEPS)),qy)
        (ml/META_BATCH).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),GRAD_CLIP)   # stabilizer
        opt.step(); sch.step()
    return m

def fine_tune_maml(m,X,y,dev):
    f=copy.deepcopy(m); opt=optim.Adam(f.parameters(),lr=1e-4,weight_decay=1e-2)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=FT_EPOCHS,eta_min=1e-6)
    ce=nn.CrossEntropyLoss(label_smoothing=0.1)
    Xt=torch.tensor(X,dtype=torch.float32,device=dev); yt=torch.tensor(y,dtype=torch.long,device=dev)
    dl=DataLoader(TensorDataset(Xt,yt),batch_size=64,shuffle=True)
    ema={k:v.detach().clone() for k,v in f.state_dict().items()}   # EMA shadow (single model)
    for ep in range(FT_EPOCHS):
        f.train()
        for xb,yb in dl:
            opt.zero_grad(); ce(f(xb),yb).backward()
            torch.nn.utils.clip_grad_norm_(f.parameters(),GRAD_CLIP)
            opt.step()
            for k,v in f.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(EMA_DECAY).add_(v.detach(),alpha=1-EMA_DECAY)
                else: ema[k]=v.detach().clone()
        sch.step()
    f.load_state_dict(ema)   # use EMA-averaged weights as final single model
    return f

def load(key):
    d=np.load(os.path.join(CACHE_DIR,f"{key}_wavlm_large_combined.npz"),allow_pickle=True)
    X=d["X"]; y=np.array([str(v) for v in d["y"]]); m=np.isin(y,SHARED)
    return X[m][:, :WAVLM_DIM], y[m]

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(XAI_DIR, exist_ok=True)
    names=feat_names()
    print(f"Device: {device}\nXAI output: {XAI_DIR}")

    Xs,ys=load("savee"); Xt,yt=load("embodb")
    le=LabelEncoder().fit(SHARED); ys_e,yt_e=le.transform(ys),le.transform(yt)
    # per-corpus z-norm (the recipe)
    Zs=StandardScaler().fit_transform(Xs); Zt=StandardScaler().fit_transform(Xt)

    # ── SVM (LinearSVC on PCA300) ────────────────────────────────────────────
    print("\nTraining SVM ...")
    pca=PCA(n_components=SVM_PCA, random_state=SEED).fit(Zs)
    Ap,Bp=pca.transform(Zs), pca.transform(Zt)
    svm=LinearSVC(C=SVM_C, max_iter=5000, random_state=SEED).fit(Ap, ys_e)
    print(f"  SVM cross-corpus acc: {accuracy_score(yt_e, svm.predict(Bp)):.4f}")

    # Back-project PCA coefficients to WavLM space: (n_cls, 2048)
    coefs = svm.coef_ @ pca.components_                  # (n_cls, 2048)
    mean_abs_coef = np.mean(np.abs(coefs), axis=0)       # (2048,)

    # 1) SVM feature importance (top WavLM dims)
    bar_plot(mean_abs_coef, names,
             "SVM — WavLM Feature Importance (mean |coef|)\nCROSS_CORPUS SAVEE->EMO-DB",
             os.path.join(XAI_DIR, "svm_feature_importance.png"), color="#2196F3")

    # 2) SVM feature groups (WavLM mean vs std)
    group_vals={g:float(mean_abs_coef[a:b].sum()) for g,(a,b) in FEATURE_GROUPS.items()}
    group_bar_plot(group_vals,
                   "SVM — Feature Group Importance\nCROSS_CORPUS SAVEE->EMO-DB",
                   os.path.join(XAI_DIR, "svm_feature_groups.png"))

    # 3) SVM per-class top WavLM dims
    fig, axes = plt.subplots(2, 3, figsize=(16, 8)); axes=axes.flatten()
    for ci, cls in enumerate(le.classes_):
        top=np.argsort(np.abs(coefs[ci]))[::-1][:10]; vals=coefs[ci][top]
        colors=["#4CAF50" if v>0 else "#F44336" for v in vals]
        axes[ci].barh([names[i] for i in top][::-1], vals[::-1], color=colors[::-1])
        axes[ci].set_title(cls, fontsize=10); axes[ci].axvline(0,color="black",linewidth=0.5)
        axes[ci].tick_params(axis='y', labelsize=7)
    for ci in range(len(le.classes_), len(axes)): axes[ci].axis("off")
    plt.suptitle("SVM — Top WavLM Features per Emotion\nCROSS_CORPUS SAVEE->EMO-DB", fontsize=12)
    plt.tight_layout(); p=os.path.join(XAI_DIR,"svm_per_class_features.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"    Saved: {p}")

    # ── MAML ─────────────────────────────────────────────────────────────────
    # Reproduce the BEST stabilized instance (seed 123 -> 74.2% in stabilize_maml.py)
    print("\nTraining MAML (stabilized, seed 123 = best instance) ...")
    np.random.seed(123); torch.manual_seed(123)
    maml=train_maml(Zs, ys, len(SHARED), device)
    maml_ft=fine_tune_maml(maml, Zs, ys_e, device); maml_ft.eval()
    with torch.no_grad():
        acc=accuracy_score(yt_e, torch.argmax(maml_ft(torch.tensor(Zt,dtype=torch.float32,device=device)),1).cpu().numpy())
    print(f"  MAML cross-corpus acc: {acc:.4f}")

    def maml_predict(Xnp):
        maml_ft.eval()
        with torch.no_grad():
            return torch.softmax(maml_ft(torch.tensor(Xnp,dtype=torch.float32,device=device)),1).cpu().numpy()

    # 4) MAML SHAP (KernelExplainer)
    print("  SHAP: MAML (KernelExplainer 50 bg x 30 test) ...")
    bg_idx=np.random.choice(len(Zs), min(50,len(Zs)), replace=False)
    te_idx=np.random.choice(len(Zt), min(30,len(Zt)), replace=False)
    X_bg=Zs[bg_idx]; X_te_s=Zt[te_idx]
    expl=shap.KernelExplainer(maml_predict, X_bg, link="identity")
    sv=expl.shap_values(X_te_s, nsamples=200, silent=True)
    if isinstance(sv, list):       sv_imp=np.mean(np.abs(np.stack(sv,axis=-1)), axis=(0,2))
    elif np.ndim(sv)==3:           sv_imp=np.mean(np.abs(sv), axis=(0,2))
    else:                          sv_imp=np.mean(np.abs(sv), axis=0)
    bar_plot(sv_imp, names,
             "MAML — Feature Importance (SHAP)\nCROSS_CORPUS SAVEE->EMO-DB",
             os.path.join(XAI_DIR, "maml_shap_features.png"), color="#9C27B0")

    # ── LIME (3 EMO-DB test samples, SVM + MAML) ─────────────────────────────
    print("  LIME: SVM, MAML ...")
    def svm_proba(Xnp):  # softmax over decision_function (LinearSVC has no predict_proba)
        d=svm.decision_function(pca.transform(Xnp))
        e=np.exp(d-d.max(axis=1,keepdims=True)); return e/e.sum(axis=1,keepdims=True)
    lime_exp=lime.lime_tabular.LimeTabularExplainer(
        Zs, feature_names=names, class_names=list(le.classes_),
        mode="classification", random_state=SEED, discretize_continuous=False)
    for j, si in enumerate(te_idx[:3]):
        true_cls=le.inverse_transform([yt_e[si]])[0]
        e_svm=lime_exp.explain_instance(Zt[si], svm_proba, num_features=15, top_labels=1)
        lime_plot(e_svm, f"LIME: SVM — sample {si} (true: {true_cls})\nCROSS_CORPUS",
                  os.path.join(XAI_DIR, f"lime_svm_sample{j}.png"))
        e_maml=lime_exp.explain_instance(Zt[si], maml_predict, num_features=15, top_labels=1)
        lime_plot(e_maml, f"LIME: MAML — sample {si} (true: {true_cls})\nCROSS_CORPUS",
                  os.path.join(XAI_DIR, f"lime_maml_sample{j}.png"))

    print(f"\nXAI complete. Plots in {XAI_DIR}/")

if __name__ == "__main__":
    main()
