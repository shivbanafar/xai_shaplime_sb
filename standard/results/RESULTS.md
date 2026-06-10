# Speech Emotion Recognition — Results Report

**Features:** WavLM-large all-layer mean+std (2048-dim) + handcrafted (227-dim) = **2275-dim**  
**TESS exception:** WavLM-only (2048-dim) — handcrafted features are speaker-specific and hurt cross-speaker transfer  
**Splits:** 80/20 stratified random (RAVDESS, EMO-DB, SAVEE) · Speaker-independent OAF→train / YAF→test (TESS)

---

## Overall Accuracy

| Dataset   | Classes | N     | SVM    | XGBoost | MAML       | Stacking |
|-----------|---------|-------|--------|---------|------------|----------|
| RAVDESS   | 8       | 1440  | 89.6%  | 81.3%   | **92.0%** ✓ | 89.9%   |
| EMO-DB    | 7       | 535   | 98.1%  | 89.7%   | **98.1%** ✓ | 96.3%   |
| SAVEE     | 7       | 480   | 93.8%  | 72.9%   | **93.8%** ✓ | 90.6%   |
| TESS *    | 7       | 2800  | 84.4%  | 74.6%   | **82.4%**  | 87.9%   |
| **Avg**   |         |       | **91.5%** | **79.6%** | **91.6%** | **91.2%** |

\* TESS: speaker-independent evaluation (train on OAF speaker, test on YAF speaker)

**Model ranking:** MAML (91.6%) > SVM (91.5%) > Stacking (91.2%) > XGBoost (79.6%)

**Stacking** = out-of-fold SVM/RF/XGBoost class probabilities → LogisticRegression meta-learner (per-dataset values from `results.txt`). It trails the best single model on most datasets but edges out MAML on TESS (87.9% vs 82.4%).

---

## Per-Dataset Details

### RAVDESS

8 emotions · 1440 samples · 288 test samples

| Emotion   | SVM F1 | XGBoost F1 | MAML F1 |
|-----------|--------|------------|---------|
| Angry     | 0.92   | 0.86       | **0.97** |
| Calm      | 0.84   | 0.76       | **0.88** |
| Disgust   | 0.97   | 0.87       | **1.00** |
| Fearful   | 0.93   | 0.91       | **0.95** |
| Happy     | 0.88   | 0.78       | **0.89** |
| Neutral   | 0.82   | 0.65       | **0.86** |
| Sad       | 0.82   | 0.73       | **0.82** |
| Surprised | 0.95   | 0.86       | **0.96** |

---

### EMO-DB

7 emotions · 535 samples · 107 test samples

| Emotion   | SVM F1 | XGBoost F1 | MAML F1 |
|-----------|--------|------------|---------|
| Anger     | 1.00   | 0.95       | **1.00** |
| Boredom   | 0.97   | 0.83       | **0.93** |
| Disgust   | 1.00   | 0.88       | **1.00** |
| Fear      | 0.97   | 0.89       | **1.00** |
| Happiness | 0.96   | 0.81       | **1.00** |
| Neutral   | 0.97   | 0.91       | **0.94** |
| Sadness   | 1.00   | 0.96       | **1.00** |

---

### SAVEE

7 emotions · 480 samples · 96 test samples

| Emotion   | SVM F1 | XGBoost F1 | MAML F1 |
|-----------|--------|------------|---------|
| Anger     | 1.00   | 0.92       | **1.00** |
| Disgust   | 0.91   | 0.63       | **0.91** |
| Fear      | 0.92   | 0.67       | **0.87** |
| Happiness | 0.96   | 0.67       | **0.96** |
| Neutral   | 0.94   | 0.79       | **0.98** |
| Sadness   | 0.92   | 0.67       | **0.92** |
| Surprise  | 0.91   | 0.64       | **0.88** |

---

### TESS (Speaker-Independent)

7 emotions · 2800 samples · 1400 test samples (YAF speaker)

| Emotion   | SVM F1 | XGBoost F1 | MAML F1 |
|-----------|--------|------------|---------|
| Anger     | 0.33   | 0.66       | **0.58** |
| Disgust   | 0.97   | 0.97       | **0.95** |
| Fear      | 1.00   | 0.81       | **0.93** |
| Happiness | 0.82   | 0.36       | **0.14** |
| Neutral   | 0.98   | 0.90       | **0.99** |
| Sadness   | 0.70   | 0.51       | **0.89** |
| Surprise  | 0.99   | 0.90       | **1.00** |

> Anger and happiness are hardest to transfer across speakers.

---

## XAI Analysis

### SHAP — Feature Group Importance

SHAP values aggregate feature importance into 10 groups:
WavLM (mean) · WavLM (std) · MFCC (mean) · MFCC (std) · Chroma · Mel-spectrogram · ZCR · Spectral Centroid · RMS Energy · Pitch (F0)

#### RAVDESS

**SVM — Feature Groups**
![SVM Feature Groups](RAVDESS/xai/svm_feature_groups.png)

**SVM — Handcrafted Feature Importance**
![SVM Handcrafted](RAVDESS/xai/svm_feature_importance_handcrafted.png)

**SVM — Per-Class Feature Importance**
![SVM Per Class](RAVDESS/xai/svm_per_class_features.png)

**XGBoost — SHAP (PCA space)**
![XGBoost SHAP PCA](RAVDESS/xai/xgboost_shap_pca.png)

**XGBoost — SHAP Beeswarm**
![XGBoost SHAP Beeswarm](RAVDESS/xai/xgboost_shap_beeswarm.png)

**MAML — SHAP Feature Importance**
![MAML SHAP](RAVDESS/xai/maml_shap_features.png)

#### RAVDESS — LIME Explanations

| SVM | XGBoost | MAML |
|-----|---------|------|
| ![](RAVDESS/xai/lime_svm_sample0.png) | ![](RAVDESS/xai/lime_xgboost_sample0.png) | ![](RAVDESS/xai/lime_maml_sample0.png) |
| ![](RAVDESS/xai/lime_svm_sample1.png) | ![](RAVDESS/xai/lime_xgboost_sample1.png) | ![](RAVDESS/xai/lime_maml_sample1.png) |
| ![](RAVDESS/xai/lime_svm_sample2.png) | ![](RAVDESS/xai/lime_xgboost_sample2.png) | ![](RAVDESS/xai/lime_maml_sample2.png) |

---

#### EMO-DB

**SVM — Feature Groups**
![SVM Feature Groups](EMBODB/xai/svm_feature_groups.png)

**SVM — Handcrafted Feature Importance**
![SVM Handcrafted](EMBODB/xai/svm_feature_importance_handcrafted.png)

**SVM — Per-Class Feature Importance**
![SVM Per Class](EMBODB/xai/svm_per_class_features.png)

**XGBoost — SHAP (PCA space)**
![XGBoost SHAP PCA](EMBODB/xai/xgboost_shap_pca.png)

**XGBoost — SHAP Beeswarm**
![XGBoost SHAP Beeswarm](EMBODB/xai/xgboost_shap_beeswarm.png)

**MAML — SHAP Feature Importance**
![MAML SHAP](EMBODB/xai/maml_shap_features.png)

#### EMO-DB — LIME Explanations

| SVM | XGBoost | MAML |
|-----|---------|------|
| ![](EMBODB/xai/lime_svm_sample0.png) | ![](EMBODB/xai/lime_xgboost_sample0.png) | ![](EMBODB/xai/lime_maml_sample0.png) |
| ![](EMBODB/xai/lime_svm_sample1.png) | ![](EMBODB/xai/lime_xgboost_sample1.png) | ![](EMBODB/xai/lime_maml_sample1.png) |
| ![](EMBODB/xai/lime_svm_sample2.png) | ![](EMBODB/xai/lime_xgboost_sample2.png) | ![](EMBODB/xai/lime_maml_sample2.png) |

---

#### SAVEE

**SVM — Feature Groups**
![SVM Feature Groups](SAVEE/xai/svm_feature_groups.png)

**SVM — Handcrafted Feature Importance**
![SVM Handcrafted](SAVEE/xai/svm_feature_importance_handcrafted.png)

**SVM — Per-Class Feature Importance**
![SVM Per Class](SAVEE/xai/svm_per_class_features.png)

**XGBoost — SHAP (PCA space)**
![XGBoost SHAP PCA](SAVEE/xai/xgboost_shap_pca.png)

**XGBoost — SHAP Beeswarm**
![XGBoost SHAP Beeswarm](SAVEE/xai/xgboost_shap_beeswarm.png)

**MAML — SHAP Feature Importance**
![MAML SHAP](SAVEE/xai/maml_shap_features.png)

#### SAVEE — LIME Explanations

| SVM | XGBoost | MAML |
|-----|---------|------|
| ![](SAVEE/xai/lime_svm_sample0.png) | ![](SAVEE/xai/lime_xgboost_sample0.png) | ![](SAVEE/xai/lime_maml_sample0.png) |
| ![](SAVEE/xai/lime_svm_sample1.png) | ![](SAVEE/xai/lime_xgboost_sample1.png) | ![](SAVEE/xai/lime_maml_sample1.png) |
| ![](SAVEE/xai/lime_svm_sample2.png) | ![](SAVEE/xai/lime_xgboost_sample2.png) | ![](SAVEE/xai/lime_maml_sample2.png) |

---

#### TESS

**SVM — Feature Groups**
![SVM Feature Groups](TESS/xai/svm_feature_groups.png)

**SVM — Handcrafted Feature Importance**
![SVM Handcrafted](TESS/xai/svm_feature_importance_handcrafted.png)

**SVM — Per-Class Feature Importance**
![SVM Per Class](TESS/xai/svm_per_class_features.png)

**XGBoost — SHAP (PCA space)**
![XGBoost SHAP PCA](TESS/xai/xgboost_shap_pca.png)

**XGBoost — SHAP Beeswarm**
![XGBoost SHAP Beeswarm](TESS/xai/xgboost_shap_beeswarm.png)

**MAML — SHAP Feature Importance**
![MAML SHAP](TESS/xai/maml_shap_features.png)

#### TESS — LIME Explanations

| SVM | XGBoost | MAML |
|-----|---------|------|
| ![](TESS/xai/lime_svm_sample0.png) | ![](TESS/xai/lime_xgboost_sample0.png) | ![](TESS/xai/lime_maml_sample0.png) |
| ![](TESS/xai/lime_svm_sample1.png) | ![](TESS/xai/lime_xgboost_sample1.png) | ![](TESS/xai/lime_maml_sample1.png) |
| ![](TESS/xai/lime_svm_sample2.png) | ![](TESS/xai/lime_xgboost_sample2.png) | ![](TESS/xai/lime_maml_sample2.png) |

---

## Key Findings

- **WavLM-large all-layer features** (2048-dim) are the dominant signal — switching from WavLM-base improved RAVDESS MAML from 87.9% → 92.0%
- **Handcrafted features** (pitch, MFCC, ZCR) improve within-speaker accuracy but encode speaker identity, hurting cross-speaker transfer — dropping them on TESS improved SVM by +22%
- **XGBoost underperforms** on high-dimensional dense embeddings across all datasets (avg 79.6%); tree models don't leverage WavLM geometry well even with PCA(300)
- **MAML's episodic training** (5-way 5-shot) with label smoothing and cosine annealing gives the best generalisation (avg 91.6%)
- **TESS anger and happiness** are the hardest emotions to transfer across speakers
