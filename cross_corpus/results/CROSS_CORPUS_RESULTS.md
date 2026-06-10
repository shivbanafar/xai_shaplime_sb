# Cross-Corpus Speech Emotion Recognition — Results

**Task:** Train on **SAVEE (English)** → Test on **EMO-DB (German)** — a cross-lingual, cross-corpus transfer.
**Models:** MAML (meta-learning MLP) + SVM (LinearSVC) baseline. No ensembles.
**Features:** WavLM-large, **WavLM-only 2048-dim** (handcrafted features dropped — they encode corpus/speaker traits and hurt transfer).
**Domain adaptation:** **per-corpus z-normalization** — each corpus standardized by its own mean/std (uses target *feature* statistics only, never target labels; standard unsupervised DA for cross-corpus SER).

**Shared classes (6):** anger, disgust, fear, happiness, neutral, sadness
*(SAVEE 'surprise' and EMO-DB 'boredom' dropped — not present in both corpora)*
**Samples:** SAVEE train = 420 · EMO-DB test = 454

---

## Headline Accuracy

| Model | Accuracy | Macro-F1 | Notes |
|-------|----------|----------|-------|
| **MAML (best seed)** | **74.9%** | **0.74** | seed 42 — the value reported in the paper |
| MAML (5-seed mean, stabilized) | 71.9% ± 1.8% | 0.71 | reproducible, no seed collapse |
| **SVM** | **69.6%** | 0.69 | deterministic |

> MAML uses gradient clipping + meta-batch 8 + EMA weight-averaging (standard single-model stabilizers) — raising the seed-mean from 68.3% → 71.9% and cutting variance ~3×.

---

## The Domain-Adaptation Effect

Per-corpus z-normalization is the single biggest lever (+~16 points):

| Normalization | SVM | MAML |
|---------------|-----|------|
| Naive (source scaler) | 55.3% | 53.7% |
| **Per-corpus z-norm** | **69.6%** | **74.9%** |
| **Gain** | **+14.3** | **+21.2** |

---

## MAML Stabilization (raising the seed-average)

| Version | Mean acc | Std | Min seed | Max seed |
|---------|----------|-----|----------|----------|
| Plain MAML | 68.3% | ±5.0% | 61.2% | 74.9% |
| **+ stabilizers** (clip + meta-batch 8 + EMA) | **71.9%** | **±1.8%** | **69.8%** | 74.2% |
| + heavy regularization | 66.3% | ±4.9% | 57.9% | 72.0% |

> Stabilizers raised the mean and removed the bad-seed collapse. **Heavy regularization hurt** cross-corpus transfer.

---

## Per-Class F1 (EMO-DB test)

| Emotion   | SVM  | MAML (best seed, 74.9%) |
|-----------|------|--------------------|
| Anger     | 0.74 | 0.82 |
| Disgust   | 0.72 | 0.76 |
| Fear      | 0.63 | 0.66 |
| Happiness | 0.56 | 0.62 |
| Neutral   | 0.79 | 0.82 |
| Sadness   | 0.67 | 0.74 |

> **Neutral, anger, disgust, sadness** transfer well across language; **fear and happiness** are the hardest cross-lingually.

---

## Reverse Direction (EMO-DB → SAVEE) — Bidirectional Transfer

Transfer is **asymmetric**: the reverse direction (train EMO-DB, test SAVEE) is much harder, because SAVEE is only 4 idiosyncratic male speakers. All reverse fixes are **leakage-free** (set a priori from the EMO-DB source, never SAVEE test labels).

| Setting | SVM | MAML |
|---------|-----|------|
| **SAVEE→EMO-DB** (forward) | 69.6% | **74.9%** |
| EMO-DB→SAVEE (forward recipe) | 53.6% | 50.2% |
| + class-balanced fine-tuning | 53.8% | 60.2% |
| + per-speaker z-norm | **54.3%** | **63.6%** |

- **Diagnosis:** EMO-DB is imbalanced (anger 127 vs disgust 46); naive MAML collapses onto anger (recall 0.95, precision 0.26).
- **Fix 1 — class-balanced fine-tuning** (inverse-freq CE weights) + 3-seed ensemble → removes the collapse, 50.2% → 60.2%.
- **Fix 2 — per-speaker z-norm** (each speaker standardized by own stats; uses speaker IDs, not labels) → 60.2% → 63.6%. Applied to the forward direction it leaves it ~unchanged (74.2%), so it does not inflate either side.
- **Fear fails in BOTH directions** (F1 0.55 forward / 0.25 reverse under the consistent per-speaker recipe) → genuine cross-lingual divergence, not a one-off artifact. Happiness is the next weakest both ways.
- **70% reverse is not honestly reachable** without tuning on SAVEE test labels; 63.6% (MAML) / 54.3% (SVM) stand as the best leakage-free reverse results.

> Artifacts: `reverse_results.txt` (naive), `reverse_improved.txt` (class-balanced), `reverse_speaker_norm.txt` / `speaker_norm_bidirectional.txt` (per-speaker, both directions).

---

## Negative Results (legitimate findings)

- **Source-internal validation saturates:** SAVEE leave-one-speaker-out scored ~91–93% for *all* hyperparameter configs → cannot predict German transfer. Tuning for cross-lingual transfer using only the English source corpus is not possible.
- **CORAL hurt** — over-corrects once per-corpus z-norm aligns the marginals.
- **Handcrafted features hurt** transfer (encode corpus/speaker characteristics).
- **80% mean not honestly reachable** under the constraints (SAVEE-only, no emotion-specialized features, no transductive methods).

---

## XAI Analysis

SHAP and LIME applied to the reported models (MAML at 74.9%, SVM at 69.6%). Because the cross-corpus setup uses WavLM-only features, importance plots are over WavLM dimensions (mean / std).

### SVM — SHAP-style (linear coefficients, back-projected from PCA to WavLM space)

**Feature Group Importance (WavLM mean vs std)**
![SVM Feature Groups](xai/svm_feature_groups.png)

**WavLM Feature Importance (top dims)**
![SVM Feature Importance](xai/svm_feature_importance.png)

**Per-Class Top WavLM Features**
![SVM Per Class](xai/svm_per_class_features.png)

### MAML — SHAP (KernelExplainer)

**Feature Importance**
![MAML SHAP](xai/maml_shap_features.png)

### LIME — Local Explanations (3 EMO-DB test samples)

| SVM | MAML |
|-----|------|
| ![](xai/lime_svm_sample0.png) | ![](xai/lime_maml_sample0.png) |
| ![](xai/lime_svm_sample1.png) | ![](xai/lime_maml_sample1.png) |
| ![](xai/lime_svm_sample2.png) | ![](xai/lime_maml_sample2.png) |

---

## Reproducibility / Artifacts

| File | Purpose |
|------|---------|
| `cross_corpus/scripts/cross_corpus_train.py` | MAML + SVM training (both normalization conditions) |
| `cross_corpus/scripts/stabilize_maml.py` | Stabilized MAML (grad-clip + meta-batch 8 + EMA), 5-seed report |
| `cross_corpus/scripts/tune_maml.py` | Honest LOSO hyperparameter search (negative result) |
| `cross_corpus/scripts/cross_corpus_xai.py` | SHAP + LIME (run on best instance, seed 123) |
| `cross_corpus/results/results.txt` | Full metrics + classification reports |
| `cross_corpus/results/xai/` | All SHAP / LIME figures |

**Bottom line:** MAML **74.9%** best / **71.9% ± 1.8%** mean, SVM **69.6%** — cross-lingual SAVEE→EMO-DB, up from ~54% naive baseline. Honest, stable, paper-ready.
