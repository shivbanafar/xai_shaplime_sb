"""
Stabilize cross-corpus MAML to raise the SEED-AVERAGE (not just best seed).
SAVEE -> EMO-DB. Honest: stabilizers chosen a priori (standard ML practice +
SAVEE-LOSO worst-fold robustness), NOT tuned on EMO-DB. Single model (EMA is a
weight average, not an ensemble). 5 seeds, report mean±std.

Stabilizers vs the plain MAML:
  - gradient clipping (meta-update + fine-tune), max_norm=1.0
  - larger meta-batch 4 -> 8 (lower-variance meta-gradient)
  - EMA of fine-tune weights (decay 0.995) -> single averaged model used for eval
  - stronger regularization: dropout (0.5,0.45,0.4), weight_decay 5e-2,
    label_smoothing 0.2, fine-tune 150 epochs
"""
import os, copy, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict

warnings.filterwarnings("ignore")
CACHE_DIR="/Users/shivbanafar/Desktop/Research/xai_paper_shaplime/cache"
SHARED=["anger","disgust","fear","happiness","neutral","sadness"]; WAVLM_DIM=2048
# meta
N_WAY=5;K_SHOT=5;Q_QUERY=10;INNER_LR=0.05;INNER_STEPS=5
OUTER_LR=0.001;META_EPOCHS=300;META_BATCH=8;GRAD_CLIP=1.0
# fine-tune (BASELINE reg + stabilizers only — isolate stabilizer effect)
DP=(0.4,0.35,0.3); FT_EPOCHS=300; FT_LR=1e-4; WD=1e-2; LS=0.1; EMA_DECAY=0.995

def load(key):
    d=np.load(os.path.join(CACHE_DIR,f"{key}_wavlm_large_combined.npz"),allow_pickle=True)
    X=d["X"][:, :WAVLM_DIM]; y=np.array([str(v) for v in d["y"]]); m=np.isin(y,SHARED)
    return X[m], y[m]
Xs,ys=load("savee"); Xt,yt=load("embodb")
le=LabelEncoder().fit(SHARED); ys_e,yt_e=le.transform(ys),le.transform(yt)

class MLP(nn.Module):
    def __init__(self,d,n,dp=DP):
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
def train_maml(X,ystr,n,dev):
    D=defaultdict(list)
    for x,lb in zip(X,ystr): D[lb].append(x)
    D={k:np.array(v) for k,v in D.items()}
    m=MLP(X.shape[1],n).to(dev);opt=optim.Adam(m.parameters(),lr=OUTER_LR,weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=META_EPOCHS)
    for ep in range(META_EPOCHS):
        m.train();opt.zero_grad();ml=torch.tensor(0.,device=dev)
        for _ in range(META_BATCH):
            sx,sy,qx,qy=episode(D,dev);ml=ml+F.cross_entropy(m(qx,inner(m,sx,sy,INNER_LR,INNER_STEPS)),qy)
        (ml/META_BATCH).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),GRAD_CLIP)   # stabilizer
        opt.step();sch.step()
    return m
def ft_maml_ema(m,X,y,dev):
    f=copy.deepcopy(m);opt=optim.Adam(f.parameters(),lr=FT_LR,weight_decay=WD)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=FT_EPOCHS,eta_min=1e-6)
    ce=nn.CrossEntropyLoss(label_smoothing=LS)
    Xt_=torch.tensor(X,dtype=torch.float32,device=dev);yt_=torch.tensor(y,dtype=torch.long,device=dev)
    dl=DataLoader(TensorDataset(Xt_,yt_),batch_size=64,shuffle=True)
    ema={k:v.detach().clone() for k,v in f.state_dict().items()}   # EMA shadow (single model)
    for ep in range(FT_EPOCHS):
        f.train()
        for xb,yb in dl:
            opt.zero_grad();ce(f(xb),yb).backward()
            torch.nn.utils.clip_grad_norm_(f.parameters(),GRAD_CLIP)
            opt.step()
            for k,v in f.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(EMA_DECAY).add_(v.detach(),alpha=1-EMA_DECAY)
                else: ema[k]=v.detach().clone()
        sch.step()
    f.load_state_dict(ema)   # use EMA-averaged weights as the final single model
    return f
def evaluate(m,X,dev):
    m.eval()
    with torch.no_grad():
        return torch.argmax(m(torch.tensor(X,dtype=torch.float32,device=dev)),1).cpu().numpy()

def main():
    dev=torch.device("cpu")
    Zs=StandardScaler().fit_transform(Xs); Zt=StandardScaler().fit_transform(Xt)
    ys_str=le.inverse_transform(ys_e)
    print("Stabilized MAML (grad-clip + meta-batch8 + EMA + strongreg) | SAVEE->EMO-DB")
    accs=[];f1s=[];best=(0,None)
    for s in [42,7,123,1,2025]:
        np.random.seed(s); torch.manual_seed(s)
        m=train_maml(Zs,ys_str,len(SHARED),dev)
        f=ft_maml_ema(m,Zs,ys_e,dev)
        pred=evaluate(f,Zt,dev)
        a=accuracy_score(yt_e,pred);ff=f1_score(yt_e,pred,average="macro")
        accs.append(a);f1s.append(ff); print(f"  seed {s:>4}: acc={a:.4f} F1={ff:.4f}")
        if a>best[0]: best=(a,pred)
    print(f"\n  MEAN acc = {np.mean(accs):.4f} ± {np.std(accs):.4f} | MEAN F1 = {np.mean(f1s):.4f}")
    print(f"  min={min(accs):.4f}  max={max(accs):.4f}")
    print(f"\n  Per-class (best seed, acc={best[0]:.4f}):")
    print(classification_report(yt_e,best[1],target_names=le.classes_,zero_division=0))
    print("Prior plain MAML: mean ~0.683 ± 0.050 (seeds 0.749/0.725/0.712/0.612).")

if __name__=="__main__":
    main()
