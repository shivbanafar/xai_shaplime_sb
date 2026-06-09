"""
Honest hyperparameter tuning for cross-corpus MAML (SAVEE -> EMO-DB).
ONLY SAVEE + EMO-DB are used. Hyperparameters are selected on a SAVEE-internal
LEAVE-ONE-SPEAKER-OUT validation (4 speakers: DC/JE/JK/KL) — EMO-DB is NEVER
used for model selection, only for the final single evaluation. No test leakage.

Search targets cross-domain generalization knobs (reduce overfitting to source):
  fine-tune epochs, weight decay, dropout, label smoothing.
Per-corpus z-norm recipe kept (each domain standardized by its own stats).
"""
import os, re, copy, warnings, itertools
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict

warnings.filterwarnings("ignore")
BASE_DIR="/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
CACHE_DIR=os.path.join(BASE_DIR,"cache"); DATA_DIR=os.path.join(BASE_DIR,"data")
SHARED=["anger","disgust","fear","happiness","neutral","sadness"]; WAVLM_DIM=2048
# meta hyperparams fixed during search
N_WAY=5;K_SHOT=5;Q_QUERY=10;INNER_LR=0.05;INNER_STEPS=5
OUTER_LR=0.001;META_EPOCHS=300;META_BATCH=4

# ── load SAVEE features + recover speaker labels aligned to cache ─────────────
def load_savee_with_speakers():
    EMAP={"a":"anger","d":"disgust","f":"fear","h":"happiness","n":"neutral","sa":"sadness","su":"surprise"}
    ddir=os.path.join(DATA_DIR,"SAVEE"); labels,speakers=[],[]
    for f in os.listdir(ddir):                     # SAME iteration as cache build
        if not f.endswith(".wav"): continue
        m=re.match(r"([A-Z]+)_([a-z]+)\d+\.wav",f)
        if not m: continue
        emo=EMAP.get(m.group(2))
        if emo: labels.append(emo); speakers.append(m.group(1))
    return np.array(labels), np.array(speakers)

d=np.load(os.path.join(CACHE_DIR,"savee_wavlm_large_combined.npz"),allow_pickle=True)
Xs_full=d["X"][:, :WAVLM_DIM]; ys_full=np.array([str(v) for v in d["y"]])
lab_re, spk_re = load_savee_with_speakers()
assert len(lab_re)==len(ys_full) and np.all(lab_re==ys_full), \
    "Speaker re-derivation does NOT match cached label order — alignment failed!"
print("Speaker alignment verified against cache.")

mask=np.isin(ys_full,SHARED)
Xs=Xs_full[mask]; ys=ys_full[mask]; spk=spk_re[mask]
le=LabelEncoder().fit(SHARED); ys_e=le.transform(ys)
print(f"SAVEE: {len(ys)} samples | speakers {sorted(set(spk))} | per-speaker counts "
      f"{ {s:int((spk==s).sum()) for s in sorted(set(spk))} }")

# EMO-DB test
de=np.load(os.path.join(CACHE_DIR,"embodb_wavlm_large_combined.npz"),allow_pickle=True)
Xt_all=de["X"][:, :WAVLM_DIM]; yt_all=np.array([str(v) for v in de["y"]])
tm=np.isin(yt_all,SHARED); Xt=Xt_all[tm]; yt_e=le.transform(yt_all[tm])
print(f"EMO-DB test: {len(yt_e)} samples")

# ── MLP / MAML (parametrized dropout) ────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self,d,n,dp=(0.4,0.35,0.3)):
        super().__init__()
        self.fc1=nn.Linear(d,512);self.bn1=nn.BatchNorm1d(512)
        self.fc2=nn.Linear(512,256);self.bn2=nn.BatchNorm1d(256)
        self.fc3=nn.Linear(256,128);self.bn3=nn.BatchNorm1d(128)
        self.fc4=nn.Linear(128,n)
        self.d1=nn.Dropout(dp[0]);self.d2=nn.Dropout(dp[1]);self.d3=nn.Dropout(dp[2])
        self.linears=nn.ModuleList([self.fc1,self.fc2,self.fc3,self.fc4])
    def forward(self,x,w=None):
        if w is None:
            x=self.d1(F.relu(self.bn1(self.fc1(x))));x=self.d2(F.relu(self.bn2(self.fc2(x))))
            x=self.d3(F.relu(self.bn3(self.fc3(x))));return self.fc4(x)
        for i,l in enumerate(self.linears[:-1]): x=F.relu(F.linear(x,w[2*i],w[2*i+1]))
        return F.linear(x,w[-2],w[-1])
def inner(m,sx,sy,lr,st):
    w=[]
    for l in m.linears: w+=[l.weight,l.bias]
    for _ in range(st):
        g=torch.autograd.grad(F.cross_entropy(m(sx,w),sy),w,create_graph=True)
        w=[p-lr*gg for p,gg in zip(w,g)]
    return w
def episode(D,dev):
    ch=np.random.choice(list(D.keys()),min(N_WAY,len(D)),replace=False);sx,sy,qx,qy=[],[],[],[]
    for i,c in enumerate(ch):
        a=D[c];idx=np.random.choice(len(a),min(K_SHOT+Q_QUERY,len(a)),replace=False);k=min(K_SHOT,max(len(idx)//2,1))
        sx.append(a[idx[:k]]);sy+=[i]*k;qx.append(a[idx[k:]]);qy+=[i]*len(idx[k:])
    t=lambda z:torch.tensor(np.vstack(z),dtype=torch.float32,device=dev);l=lambda z:torch.tensor(z,dtype=torch.long,device=dev)
    return t(sx),l(sy),t(qx),l(qy)
def train_maml(X,ystr,n,dev,dp):
    D=defaultdict(list)
    for x,lb in zip(X,ystr): D[lb].append(x)
    D={k:np.array(v) for k,v in D.items()}
    m=MLP(X.shape[1],n,dp).to(dev);opt=optim.Adam(m.parameters(),lr=OUTER_LR,weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=META_EPOCHS)
    for ep in range(META_EPOCHS):
        m.train();opt.zero_grad();ml=torch.tensor(0.,device=dev)
        for _ in range(META_BATCH):
            sx,sy,qx,qy=episode(D,dev);ml=ml+F.cross_entropy(m(qx,inner(m,sx,sy,INNER_LR,INNER_STEPS)),qy)
        (ml/META_BATCH).backward();opt.step();sch.step()
    return m
def ft_maml(m,X,y,dev,ft_ep,ft_lr,wd,ls):
    f=copy.deepcopy(m);opt=optim.Adam(f.parameters(),lr=ft_lr,weight_decay=wd)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=ft_ep,eta_min=1e-6)
    ce=nn.CrossEntropyLoss(label_smoothing=ls)
    Xt=torch.tensor(X,dtype=torch.float32,device=dev);yt=torch.tensor(y,dtype=torch.long,device=dev)
    dl=DataLoader(TensorDataset(Xt,yt),batch_size=64,shuffle=True)
    for ep in range(ft_ep):
        f.train()
        for xb,yb in dl: opt.zero_grad();ce(f(xb),yb).backward();opt.step()
        sch.step()
    return f
def evaluate(m,X,dev):
    m.eval()
    with torch.no_grad():
        return torch.argmax(m(torch.tensor(X,dtype=torch.float32,device=dev)),1).cpu().numpy()

def run_config(cfg, Xtr, ytr_e, Xva, dev, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    Ztr=StandardScaler().fit_transform(Xtr)          # per-domain z-norm
    Zva=StandardScaler().fit_transform(Xva)
    ytr_str=le.inverse_transform(ytr_e)
    m=train_maml(Ztr,ytr_str,len(SHARED),dev,cfg["dp"])
    f=ft_maml(m,Ztr,ytr_e,dev,cfg["ft_ep"],cfg["ft_lr"],cfg["wd"],cfg["ls"])
    return evaluate(f,Zva,dev)

# ── configs (generalization-focused) ─────────────────────────────────────────
CONFIGS=[
    {"name":"baseline",        "dp":(0.4,0.35,0.3),"ft_ep":300,"ft_lr":1e-4,"wd":1e-2,"ls":0.1},
    {"name":"ft150",           "dp":(0.4,0.35,0.3),"ft_ep":150,"ft_lr":1e-4,"wd":1e-2,"ls":0.1},
    {"name":"ft100",           "dp":(0.4,0.35,0.3),"ft_ep":100,"ft_lr":1e-4,"wd":1e-2,"ls":0.1},
    {"name":"wd5e-2_ft150",    "dp":(0.4,0.35,0.3),"ft_ep":150,"ft_lr":1e-4,"wd":5e-2,"ls":0.1},
    {"name":"drophi_ft150",    "dp":(0.5,0.45,0.4),"ft_ep":150,"ft_lr":1e-4,"wd":1e-2,"ls":0.1},
    {"name":"strongreg",       "dp":(0.5,0.45,0.4),"ft_ep":150,"ft_lr":1e-4,"wd":5e-2,"ls":0.2},
]

def main():
    dev=torch.device("cpu")
    speakers=sorted(set(spk))
    print(f"\n=== LEAVE-ONE-SPEAKER-OUT search ({len(CONFIGS)} configs x {len(speakers)} folds) ===")
    results={}
    for cfg in CONFIGS:
        fold_acc=[]
        for held in speakers:
            tr=spk!=held; va=spk==held
            pred=run_config(cfg, Xs[tr], ys_e[tr], Xs[va], dev, seed=42)
            a=accuracy_score(ys_e[va], pred); fold_acc.append(a)
        mean_a=float(np.mean(fold_acc))
        results[cfg["name"]]=mean_a
        print(f"  {cfg['name']:<16} LOSO val acc = {mean_a:.4f}  folds={[f'{x:.2f}' for x in fold_acc]}")

    best=max(results,key=results.get)
    best_cfg=next(c for c in CONFIGS if c["name"]==best)
    print(f"\n>> BEST by LOSO dev: {best} ({results[best]:.4f})")

    # ── Final: train on ALL SAVEE, test EMO-DB, 3 seeds (honest report) ───────
    print(f"\n=== FINAL: train all SAVEE -> test EMO-DB with '{best}' (3 seeds) ===")
    Zs=StandardScaler().fit_transform(Xs); Zt=StandardScaler().fit_transform(Xt)
    ys_str=le.inverse_transform(ys_e)
    accs=[];f1s=[]
    for s in [42,7,123]:
        np.random.seed(s); torch.manual_seed(s)
        m=train_maml(Zs,ys_str,len(SHARED),dev,best_cfg["dp"])
        f=ft_maml(m,Zs,ys_e,dev,best_cfg["ft_ep"],best_cfg["ft_lr"],best_cfg["wd"],best_cfg["ls"])
        pred=evaluate(f,Zt,dev)
        a=accuracy_score(yt_e,pred);ff=f1_score(yt_e,pred,average="macro");accs.append(a);f1s.append(ff)
        print(f"  seed {s}: EMO-DB acc={a:.4f} F1={ff:.4f}")
    print(f"\n  '{best}' EMO-DB: acc {np.mean(accs):.4f} ± {np.std(accs):.4f} | F1 {np.mean(f1s):.4f}")
    print(f"  (baseline was 74.9% / ~68% mean). Best single seed: {max(accs):.4f}")

if __name__=="__main__":
    main()
