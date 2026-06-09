# Speech Emotion Recognition — Results Report

**Features:** WavLM-large all-layer mean+std (2048-dim) + handcrafted (227-dim) = **2275-dim**  
**TESS exception:** WavLM-only (2048-dim) — handcrafted features are speaker-specific and hurt cross-speaker transfer  
**Splits:** 80/20 stratified random (RAVDESS, EMO-DB, SAVEE) · Speaker-independent OAF→train / YAF→test (TESS)

---

## Overall Accuracy

| Dataset   | Classes | N     | SVM    | XGBoost | MAML       |
|-----------|---------|-------|--------|---------|------------|
| RAVDESS   | 8       | 1440  | 89.6%  | 81.6%   | **90.6%** ✓ |
| EMO-DB    | 7       | 535   | 98.1%  | 90.7%   | **100.0%** ✓ |
| SAVEE     | 7       | 480   | 93.8%  | 72.9%   | **93.8%** ✓ |
| TESS *    | 7       | 2800  | 84.4%  | 75.6%   | **86.2%**  |
| **Avg**   |         |       | **91.5%** | **80.2%** | **92.7%** |

\* TESS: speaker-independent evaluation (train on OAF speaker, test on YAF speaker)

**Model ranking:** MAML (92.7%) > SVM (91.5%) > XGBoost (80.2%)

---

## Per-Dataset Details

### RAVDESS

8 emotions · 1440 samples · 288 test samples

| Emotion   | SVM F1 | XGBoost F1 | MAML F1 |
|-----------|--------|------------|---------|
| Angry     | 0.92   | 0.86       | **0.96** |
| Calm      | 0.84   | 0.75       | **0.84** |
| Disgust   | 0.97   | 0.86       | **0.99** |
| Fearful   | 0.93   | 0.92       | **0.94** |
| Happy     | 0.88   | 0.79       | **0.87** |
| Neutral   | 0.82   | 0.65       | **0.86** |
| Sad       | 0.82   | 0.74       | **0.84** |
| Surprised | 0.95   | 0.86       | **0.94** |

---

### EMO-DB

7 emotions · 535 samples · 107 test samples

| Emotion   | SVM F1 | XGBoost F1 | MAML F1 |
|-----------|--------|------------|---------|
| Anger     | 1.00   | 0.95       | **1.00** |
| Boredom   | 0.97   | 0.84       | **1.00** |
| Disgust   | 1.00   | 0.88       | **1.00** |
| Fear      | 0.97   | 0.92       | **1.00** |
| Happiness | 0.96   | 0.86       | **1.00** |
| Neutral   | 0.97   | 0.91       | **1.00** |
| Sadness   | 1.00   | 0.96       | **1.00** |

---

### SAVEE

7 emotions · 480 samples · 96 test samples

| Emotion   | SVM F1 | XGBoost F1 | MAML F1 |
|-----------|--------|------------|---------|
| Anger     | 1.00   | 0.92       | **1.00** |
| Disgust   | 0.91   | 0.67       | **0.91** |
| Fear      | 0.92   | 0.67       | **0.88** |
| Happiness | 0.96   | 0.56       | **0.96** |
| Neutral   | 0.94   | 0.81       | **0.98** |
| Sadness   | 0.92   | 0.69       | **0.92** |
| Surprise  | 0.91   | 0.64       | **0.87** |

---

### TESS (Speaker-Independent)

7 emotions · 2800 samples · 1400 test samples (YAF speaker)

| Emotion   | SVM F1 | XGBoost F1 | MAML F1 |
|-----------|--------|------------|---------|
| Anger     | 0.33   | 0.67       | **0.64** |
| Disgust   | 0.97   | 0.96       | **0.95** |
| Fear      | 1.00   | 0.86       | **0.96** |
| Happiness | 0.82   | 0.32       | **0.48** |
| Neutral   | 0.98   | 0.90       | **0.99** |
| Sadness   | 0.70   | 0.59       | **0.89** |
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

- **WavLM-large all-layer features** (2048-dim) are the dominant signal — switching from WavLM-base improved RAVDESS MAML from 87.9% → 90.6%
- **Handcrafted features** (pitch, MFCC, ZCR) improve within-speaker accuracy but encode speaker identity, hurting cross-speaker transfer — dropping them on TESS improved SVM by +22%
- **XGBoost underperforms** on high-dimensional dense embeddings across all datasets (avg 80.2%); tree models don't leverage WavLM geometry well even with PCA(300)
- **MAML's episodic training** (5-way 5-shot) with label smoothing and cosine annealing gives the best generalisation (avg 92.7%)
- **TESS anger and happiness** are the hardest emotions to transfer across speakers
