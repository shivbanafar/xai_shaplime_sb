"""
Stacking ensemble for the FORWARD cross-corpus direction: SAVEE -> EMO-DB.

Base learners (trained on SAVEE, per-corpus z-norm):
  - SVM      : CalibratedClassifierCV(LinearSVC C=0.1) on PCA-300  -> probabilities
  - XGBoost  : on PCA-300                                          -> probabilities
  - MAML     : 4-layer MLP on full 2048-dim                        -> softmax probs

Meta-learner: LogisticRegression on out-of-fold (OOF) base-model probabilities.

Leakage-free protocol:
  1. 5-fold OOF on SAVEE: for each fold train base learners on the train part and
     predict probabilities on the held-out part -> OOF meta-features.
  2. Train the meta-learner on (SAVEE OOF probs, SAVEE labels).
  3. Retrain each base learner on ALL of SAVEE, predict EMO-DB probabilities,
     feed them to the meta-learner -> final EMO-DB prediction.
  The meta-learner never sees EMO-DB labels.

Compares the stack against each standalone base learner on EMO-DB.
"""
import os, sys, copy, warnings
import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb

warnings.filterwarnings("ignore")
BASE_DIR = "/Users/shivbanafar/Desktop/Research/xai_paper_shaplime"
sys.path.insert(0, os.path.join(BASE_DIR, "cross_corpus", "scripts"))
import cross_corpus_train as cc

OUT_DIR = os.path.join(BASE_DIR, "cross_corpus", "results")
SHARED = cc.SHARED
SEED = 42
N_FOLDS = 5
PCA_DIM = 300
K = len(SHARED)


def reseed(s=SEED):
    np.random.seed(s); torch.manual_seed(s)


# ── base-learner factories (fit -> proba) ────────────────────────────────────
def fit_svm(Xtr, ytr):
    pca = PCA(n_components=min(PCA_DIM, Xtr.shape[1]), random_state=SEED).fit(Xtr)
    clf = CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=5000, random_state=SEED),
                                 cv=3).fit(pca.transform(Xtr), ytr)
    return ("svm", pca, clf)


def fit_xgb(Xtr, ytr):
    pca = PCA(n_components=min(PCA_DIM, Xtr.shape[1]), random_state=SEED).fit(Xtr)
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                            random_state=SEED, num_class=K, verbosity=0)
    clf.fit(pca.transform(Xtr), ytr)
    return ("xgb", pca, clf)


def proba_linear(model, X):
    _, pca, clf = model
    return clf.predict_proba(pca.transform(X))


def fit_maml(Xtr, ytr, dev):
    reseed(SEED)
    m = cc.train_maml(Xtr, LabelEncoder().fit(SHARED).inverse_transform(ytr), K, dev)
    mf = cc.fine_tune_maml(m, Xtr, ytr, dev); mf.eval()
    return mf


def proba_maml(mf, X, dev):
    with torch.no_grad():
        return torch.softmax(mf(torch.tensor(X, dtype=torch.float32, device=dev)), 1).cpu().numpy()


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    L = []
    def log(m=""): print(m); L.append(str(m))

    log("=" * 64)
    log("STACKING ENSEMBLE  —  SAVEE -> EMO-DB  (per-corpus z-norm)")
    log("=" * 64)
    Xs, ys = cc.load("savee"); Xt, yt = cc.load("embodb")
    le = LabelEncoder().fit(SHARED)
    ys_e, yt_e = le.transform(ys), le.transform(yt)
    Zs = StandardScaler().fit_transform(Xs)
    Zt = StandardScaler().fit_transform(Xt)
    log(f"SAVEE {Zs.shape} | EMO-DB {Zt.shape} | base learners: SVM, XGBoost, MAML")

    # ── 1. OOF meta-features on SAVEE ────────────────────────────────────────
    log(f"\n[OOF] {N_FOLDS}-fold out-of-fold base predictions on SAVEE ...")
    names = ["svm", "xgb", "maml"]
    oof = {n: np.zeros((len(Zs), K)) for n in names}
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for i, (tr, va) in enumerate(skf.split(Zs, ys_e), 1):
        oof["svm"][va] = proba_linear(fit_svm(Zs[tr], ys_e[tr]), Zs[va])
        oof["xgb"][va] = proba_linear(fit_xgb(Zs[tr], ys_e[tr]), Zs[va])
        mf = fit_maml(Zs[tr], ys_e[tr], dev)
        oof["maml"][va] = proba_maml(mf, Zs[va], dev)
        log(f"  fold {i}/{N_FOLDS} done")

    # ── 2. meta-learner on SAVEE OOF probs ───────────────────────────────────
    Moof = np.hstack([oof[n] for n in names])
    meta = LogisticRegression(max_iter=2000, C=1.0).fit(Moof, ys_e)

    # ── 3. full base learners -> EMO-DB probs -> meta ────────────────────────
    log("\n[FULL] retraining base learners on all SAVEE, predicting EMO-DB ...")
    svm_f = fit_svm(Zs, ys_e)
    xgb_f = fit_xgb(Zs, ys_e)
    maml_f = fit_maml(Zs, ys_e, dev)
    pt = {"svm": proba_linear(svm_f, Zt),
          "xgb": proba_linear(xgb_f, Zt),
          "maml": proba_maml(maml_f, Zt, dev)}
    Mtest = np.hstack([pt[n] for n in names])
    stack_pred = meta.predict(Mtest)

    # ── report ───────────────────────────────────────────────────────────────
    def line(tag, pred):
        a = accuracy_score(yt_e, pred); f = f1_score(yt_e, pred, average="macro")
        log(f"  {tag:<22} acc={a*100:5.1f}%   macroF1={f:.3f}")
        return a
    log("\n" + "-" * 64)
    log("EMO-DB results: standalone base learners vs stacking")
    log("-" * 64)
    line("SVM (calibrated)", pt["svm"].argmax(1))
    line("XGBoost", pt["xgb"].argmax(1))
    line("MAML", pt["maml"].argmax(1))
    line("Soft-vote (mean prob)", (pt["svm"] + pt["xgb"] + pt["maml"]).argmax(1))
    log("-" * 64)
    a_stack = line("STACKING (meta-LR)", stack_pred)
    log("-" * 64)
    log(f"\n  meta-LR learned weights (mean |coef| per base learner):")
    coef = np.abs(meta.coef_).mean(0)
    for j, n in enumerate(names):
        log(f"    {n:<6} {coef[j*K:(j+1)*K].mean():.3f}")
    log("\n[STACKING] per-class report on EMO-DB:")
    log(classification_report(yt_e, stack_pred, target_names=SHARED, zero_division=0))

    with open(os.path.join(OUT_DIR, "cross_corpus_stacking.txt"), "w") as f:
        f.write("\n".join(L))
    log(f"Written to {os.path.join(OUT_DIR,'cross_corpus_stacking.txt')}")


if __name__ == "__main__":
    main()
