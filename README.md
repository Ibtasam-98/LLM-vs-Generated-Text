# LLM-vs-Generated-Text

Lightweight machine learning models for detecting AI-generated student essays using **TF-IDF** features with Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Random Forest.

This repository supports the short paper **"Supporting Authentic Assessment in the Age of Generative AI: Evaluating Lightweight Models for AI-Generated Text Detection."**

## Dataset

- **Source:** `Training_Essay_Data.csv`
- **After cleaning:** 27,301 essays
- **Human-written:** 16,117 (59.0%)
- **AI-generated:** 11,184 (41.0%)
- **Split:** 70% training, 15% validation, 15% test
- **Features:** TF-IDF, 20,000 features, unigram + bigram (`ngram_range=(1,2)`)

## Models

| Model | Key Configuration |
|---|---|
| Logistic Regression | Balanced class weights, L2 |
| Multinomial Naive Bayes | `alpha=1.0` |
| Linear SVM | Balanced class weights |
| Random Forest | 100 trees, `max_depth=10` |

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9893 | 0.9910 | 0.9827 | 0.9868 | 0.9995 | 0.39 |
| Naive Bayes | 0.9753 | 0.9823 | 0.9571 | 0.9695 | 0.9943 | 0.02 |
| **Linear SVM** | **0.9973** | **0.9982** | **0.9952** | **0.9967** | **1.0000** | **0.76** |
| Random Forest | 0.9741 | 0.9781 | 0.9583 | 0.9681 | 0.9940 | 1.42 |

**Linear SVM** performed best, misclassifying only **11 of 4,096 test essays**.

## Repository

```text
LLM-vs-Generated-Text/
├── main.py
├── Training_Essay_Data.csv
└── ai_detection_results/
    ├── model_comparison.csv
    ├── best_model_confusion_matrix.png
    ├── roc_curves_comparison.png
    ├── performance_time_tradeoff.png
    ├── best_model_Linear_SVM.pkl
    └── tfidf_vectorizer.pkl
