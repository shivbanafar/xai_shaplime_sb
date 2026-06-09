"""
Cross-Corpus Speech Emotion Recognition  —  MAML + SVM baseline
Train : SAVEE  (English, acted)
Test  : EMO-DB (German, acted)        cross-lingual cross-corpus transfer

Shared 6 classes: anger, disgust, fear, happiness, neutral, sadness
  (SAVEE 'surprise' and EMO-DB 'boredom' dropped — not in both corpora)

Features: WavLM-large, WavLM-only 2048-dim (handcrafted dropped — they hurt transfer).
Domain adaptation: per-corpus z-normalization (each corpus standardized by its OWN
  mean/std; uses target FEATURE statistics only, never target labels — standard,
  transductive, unsupervised DA for cross-corpus SER).

Two normalization conditions reported to show the effect:
  (A) source-scaler   : fit StandardScaler on SAVEE, apply to EMO-DB  (naive baseline)
  (B) per-corpus z    : standardize each corpus independently         (recommended)

Models: MAML (MLP, meta-train + fine-tune) · SVM (LinearSVC).  No ensembles.
Output: cross_corpus/results.txt
"""
import os, copy, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42); torch.manual_seed(42)

BASE_DIR  = "/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUT_DIR   = os.path.join(BASE_DIR, "cross_corpus", "results")
SEED = 42
SHARED = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
WAVLM_DIM = 2048
SVM_C = 0.1
SVM_PCA = 300

# MAML hyper-params (match project pipeline)
N_WAY=5; K_SHOT=5; Q_QUERY=10; INNER_LR=0.05; OUTER_LR=0.001
INNER_STEPS=5; META_EPOCHS=300; META_BATCH=4; FT_EPOCHS=300

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
            x=self.d1(F.relu(self.bn1(self.fc1(x))))
            x=self.d2(F.relu(self.bn2(self.fc2(x))))
            x=self.d3(F.relu(self.bn3(self.fc3(x))))
            return self.fc4(x)
        for i,l in enumerate(self.linears[:-1]): x=F.relu(F.linear(x,w[2*i],w[2*i+1]))
        return F.linear(x,w[-2],w[-1])

def inner_loop(m, sx, sy, lr, steps):
    w=[]
    for l in m.linears: w+=[l.weight,l.bias]
    for _ in range(steps):
        g=torch.autograd.grad(F.cross_entropy(m(sx,w),sy), w, create_graph=True)
        w=[p-lr*gg for p,gg in zip(w,g)]
    return w

def sample_episode(D, dev):
    ch=np.random.choice(list(D.keys()), min(N_WAY,len(D)), replace=False)
    sx,sy,qx,qy=[],[],[],[]
    for i,c in enumerate(ch):
        a=D[c]; idx=np.random.choice(len(a), min(K_SHOT+Q_QUERY,len(a)), replace=False)
        k=min(K_SHOT, max(len(idx)//2,1))
        sx.append(a[idx[:k]]); sy+=[i]*k; qx.append(a[idx[k:]]); qy+=[i]*len(idx[k:])
    t=lambda z: torch.tensor(np.vstack(z),dtype=torch.float32,device=dev)
    l=lambda z: torch.tensor(z,dtype=torch.long,device=dev)
    return t(sx),l(sy),t(qx),l(qy)

def train_maml(X, ystr, n_cls, dev):
    D=defaultdict(list)
    for x,lb in zip(X,ystr): D[lb].append(x)
    D={k:np.array(v) for k,v in D.items()}
    m=MLP(X.shape[1],n_cls).to(dev)
    opt=optim.Adam(m.parameters(),lr=OUTER_LR,weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=META_EPOCHS)
    for ep in range(META_EPOCHS):
        m.train(); opt.zero_grad(); ml=torch.tensor(0.,device=dev)
        for _ in range(META_BATCH):
            sx,sy,qx,qy=sample_episode(D,dev)
            ml=ml+F.cross_entropy(m(qx,inner_loop(m,sx,sy,INNER_LR,INNER_STEPS)),qy)
        (ml/META_BATCH).backward(); opt.step(); sch.step()
        if (ep+1)%100==0: print(f"      meta ep {ep+1}/{META_EPOCHS} loss={ml.item()/META_BATCH:.4f}")
    return m

def fine_tune_maml(m, X, y, dev):
    f=copy.deepcopy(m)
    opt=optim.Adam(f.parameters(),lr=1e-4,weight_decay=1e-2)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=FT_EPOCHS,eta_min=1e-6)
    ce=nn.CrossEntropyLoss(label_smoothing=0.1)
    Xt=torch.tensor(X,dtype=torch.float32,device=dev); yt=torch.tensor(y,dtype=torch.long,device=dev)
    dl=DataLoader(TensorDataset(Xt,yt),batch_size=64,shuffle=True)
    for ep in range(FT_EPOCHS):
        f.train()
        for xb,yb in dl: opt.zero_grad(); ce(f(xb),yb).backward(); opt.step()
        sch.step()
    return f

# ── Data ─────────────────────────────────────────────────────────────────────
def load(key):
    d=np.load(os.path.join(CACHE_DIR,f"{key}_wavlm_large_combined.npz"),allow_pickle=True)
    X=d["X"]; y=np.array([str(v) for v in d["y"]])
    m=np.isin(y,SHARED)
    return X[m][:, :WAVLM_DIM], y[m]      # WavLM-only

def plot_cm(yt_e, pred, le, acc, out_png):
    """Row-normalized (%) confusion-matrix heatmap, styled to match the flow diagram."""
    labels = list(le.classes_)
    cm = confusion_matrix(yt_e, pred, labels=range(len(labels)))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100.0
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"MAML — SAVEE$\\rightarrow$EMO-DB (acc={acc*100:.1f}%)", fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i,j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i,j] > 55 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("% of true class", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      confusion matrix saved -> {out_png}")

# ── Run one normalization condition ──────────────────────────────────────────
def run(cond, Xs_n, Xt_n, ys_e, yt_e, le, device, log, cm_path=None):
    log(f"\n{'='*60}\nNORMALIZATION: {cond}\n{'='*60}")

    # SVM baseline (z + PCA300, C=0.1)
    pca=PCA(n_components=min(SVM_PCA, Xs_n.shape[1]), random_state=SEED).fit(Xs_n)
    Ap,Bp=pca.transform(Xs_n), pca.transform(Xt_n)
    svm=LinearSVC(C=SVM_C, max_iter=5000, random_state=SEED).fit(Ap, ys_e)
    svm_pred=svm.predict(Bp)
    svm_acc=accuracy_score(yt_e,svm_pred); svm_f1=f1_score(yt_e,svm_pred,average="macro")
    log(f"\n[SVM]  LinearSVC C={SVM_C}, PCA{pca.n_components_}")
    log(f"  acc={svm_acc:.4f}  macroF1={svm_f1:.4f}")
    log(classification_report(yt_e,svm_pred,target_names=le.classes_,zero_division=0))

    # MAML (full 2048-dim, raw z-normed)
    log("[MAML]  meta-train 300ep + fine-tune 300ep (label smoothing)")
    m=train_maml(Xs_n, le.inverse_transform(ys_e), len(le.classes_), device)
    mf=fine_tune_maml(m, Xs_n, ys_e, device); mf.eval()
    with torch.no_grad():
        maml_pred=torch.argmax(mf(torch.tensor(Xt_n,dtype=torch.float32,device=device)),1).cpu().numpy()
    maml_acc=accuracy_score(yt_e,maml_pred); maml_f1=f1_score(yt_e,maml_pred,average="macro")
    log(f"  acc={maml_acc:.4f}  macroF1={maml_f1:.4f}")
    log(classification_report(yt_e,maml_pred,target_names=le.classes_,zero_division=0))
    if cm_path is not None:
        plot_cm(yt_e, maml_pred, le, maml_acc, cm_path)
    return {"SVM":(svm_acc,svm_f1), "MAML":(maml_acc,maml_f1)}

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    lines=[]
    def log(m=""): print(m); lines.append(str(m))

    log("="*60)
    log("CROSS-CORPUS SER:  MAML + SVM  |  train SAVEE (Eng) -> test EMO-DB (Ger)")
    log("="*60)
    log(f"Device: {device} | Features: WavLM-only {WAVLM_DIM}-dim | seed {SEED}")
    log(f"Shared classes ({len(SHARED)}): {SHARED}")

    Xs,ys=load("savee"); Xt,yt=load("embodb")
    le=LabelEncoder().fit(SHARED)
    ys_e,yt_e=le.transform(ys),le.transform(yt)
    log(f"SAVEE train: {Xs.shape[0]} | EMO-DB test: {Xt.shape[0]}")

    summary={}
    # (A) source-scaler baseline
    sc=StandardScaler().fit(Xs)
    summary["source-scaler (naive)"]=run("source-scaler (fit on SAVEE)",
        sc.transform(Xs), sc.transform(Xt), ys_e, yt_e, le, device, log)
    # (B) per-corpus z-norm (recommended)
    cm_png = os.path.join(OUT_DIR, "maml_confusion_matrix.png")
    summary["per-corpus z-norm"]=run("per-corpus z-norm (each corpus by own mean/std)",
        StandardScaler().fit_transform(Xs), StandardScaler().fit_transform(Xt),
        ys_e, yt_e, le, device, log, cm_path=cm_png)

    log(f"\n{'='*60}\nSUMMARY — accuracy / macro-F1 (SAVEE -> EMO-DB)\n{'='*60}")
    log(f"{'Normalization':<26}{'SVM':>16}{'MAML':>16}")
    log("-"*58)
    for cond,r in summary.items():
        log(f"{cond:<26}{r['SVM'][0]*100:>7.1f}% (F1 {r['SVM'][1]:.2f}){r['MAML'][0]*100:>7.1f}% (F1 {r['MAML'][1]:.2f})")
    log("\nNote: per-corpus z-norm = unsupervised target-domain feature normalization")
    log("(uses EMO-DB feature statistics, NOT labels). Standard for cross-corpus SER.")
    log("MAML single-run varies ~±5% across seeds; seed=42 reported.")

    with open(os.path.join(OUT_DIR,"results.txt"),"w") as f: f.write("\n".join(lines))
    log(f"\nResults written to {os.path.join(OUT_DIR,'results.txt')}")

if __name__ == "__main__":
    main()
