# Speech Emotion Recognition with XAI (SHAP + LIME) and MAML

Code and results for our work on **Speech Emotion Recognition (SER)** using
WavLM features, classical / meta-learning models, and post-hoc explainability
(SHAP + LIME).

The repo has two self-contained parts:

| Folder          | Setting                                   | Train → Test                     |
|-----------------|-------------------------------------------|----------------------------------|
| `standard/`     | Single-corpus SER (within each dataset)   | RAVDESS, EMO-DB, SAVEE, TESS     |
| `cross_corpus/` | Cross-lingual cross-corpus transfer       | SAVEE (English) → EMO-DB (German)|

> Only **scripts** and **results** are tracked. Audio datasets, the Python
> virtual environment, caches, and the large `.docx` report are not committed
> (the report is regenerable from the scripts).

---

## `standard/` — single-corpus SER

Trains and explains models **within** each dataset (random 80/20 splits, with a
speaker-independent OAF→YAF split for TESS).

**Features:** WavLM-large all-layer mean / mean+std pooling (2048-dim) plus
handcrafted MFCC / chroma / pitch (227-dim).
**Models:** SVM (LinearSVC), XGBoost, MAML (MLP), and a stacking ensemble.

### Scripts (`standard/scripts/`)
- `ser_pipeline.py` — single-dataset (RAVDESS) reference pipeline: feature
  extraction → SVM → XGBoost → MAML.
- `run_all_datasets.py` — runs the full pipeline across all four datasets and
  writes `results/<DATASET>/results.txt`.
- `xai_analysis.py` — SHAP + LIME explanations for each model; saves plots to
  `results/<DATASET>/xai/`.
- `generate_report.py` — trains everything, builds confusion matrices, and
  assembles the Word report (`SER_Report.docx`, not committed).

### Results (`standard/results/`)
- `<DATASET>/results.txt` — accuracy / per-class metrics per model.
- `<DATASET>/xai/` — SHAP & LIME plots (feature importance, beeswarm, per-class,
  per-sample LIME).
- `report/cm/` — confusion-matrix PNGs (`<DATASET>_<MODEL>_cm.png`).
- `RESULTS.md`, `summary.txt` — human-readable summaries.

---

## `cross_corpus/` — cross-lingual cross-corpus SER

Trains on **SAVEE (English)** and tests on **EMO-DB (German)** over the 6 shared
emotions (anger, disgust, fear, happiness, neutral, sadness).

**Features:** WavLM-large, WavLM-only 2048-dim (handcrafted features dropped — they
hurt transfer).
**Domain adaptation:** per-corpus z-normalization (each corpus standardized by its
own feature statistics — unsupervised, no target labels).
**Models:** MAML (MLP, meta-train + fine-tune) and an SVM (LinearSVC + PCA-300)
baseline.

### Scripts (`cross_corpus/scripts/`)
- `cross_corpus_train.py` — main MAML + SVM training/evaluation; compares naive
  source-scaler vs per-corpus z-norm; writes `results.txt` and the MAML
  confusion matrix.
- `cross_corpus_xai.py` — SHAP + LIME explanations for the cross-corpus MAML /
  SVM models → `results/xai/`.
- `tune_maml.py` — honest hyper-parameter search using SAVEE leave-one-speaker-out
  validation (EMO-DB never used for model selection).
- `stabilize_maml.py` — variance-reduction recipe (grad clipping, larger
  meta-batch, EMA, stronger regularization); reports mean±std over 5 seeds.
- `xai_feature_selection.py` — experiment using SHAP importances (from SVM, SAVEE
  only) to select top-k WavLM dims, then retrain MAML.
- `xai_feature_selection_combined.py` — same, with combined **SHAP + LIME**
  rankings.

> **Note on the feature-selection experiments:** these test whether ranking
> WavLM dims by SHAP/LIME importance and keeping a subset improves transfer.
> It does **not** — accuracy degrades monotonically as dims are removed, so the
> full 2048-dim representation is kept. SHAP/LIME are therefore used for
> **post-hoc interpretation**, not as a feature-selection step.

### Results (`cross_corpus/results/`)
- `results.txt` — SVM / MAML accuracy & per-class metrics for both normalization
  settings (per-corpus z-norm: SVM 69.6%, MAML 74.9%).
- `maml_confusion_matrix.png` — MAML confusion matrix (SAVEE → EMO-DB).
- `xai/` — SHAP & LIME plots for the cross-corpus models.
- `xai_feature_selection.txt`, `xai_feature_selection_combined.txt` — k-sweep
  results showing full features outperform any selected subset.
- `CROSS_CORPUS_RESULTS.md` — summary write-up.

---

## Reproducing

Each script has an absolute `BASE_DIR` / dataset path near the top — point these
at your local copy of the audio datasets, then run the script directly with
Python (PyTorch, scikit-learn, librosa, transformers, shap, lime, xgboost).
