# Part 1: Customer Churn Prediction — Neural Network Analysis

## Overview
Feed-forward neural network (Multi-Layer Perceptron) built to predict customer churn using the `customer_churn_nn.csv` dataset. Implemented using **scikit-learn's `MLPClassifier`**.

---

## Dataset Summary
| Property | Value |
|---|---|
| Rows | 2000 |
| Columns | 17 (16 features + 1 target) |
| Target | `churn` — 1 = churned, 0 = retained |
| Class balance | 98.4% retained / 1.6% churned (severely imbalanced) |
| Missing values | None |
| Categorical features | `region`, `plan_type`, `contract_type`, `payment_method` |
| Numerical features | tenure, charges, login days, tickets, delays, data usage, satisfaction, complaint recency, discounts, autopay, referrals |

---

## Model Architecture (Baseline)
```
Input Layer  →  [64 neurons, ReLU]  →  [32 neurons, ReLU]  →  Output (Sigmoid)
  16 features         Hidden L1               Hidden L2          Binary classification
```
- **Loss function:** Binary Cross-Entropy
- **Optimizer:** Adam (lr=0.001)
- **Batch size:** 32
- **Max epochs:** 300

---

## Results Summary

| Experiment | Hidden Layers | Activation | LR | Test Acc | AUC-ROC |
|---|---|---|---|---|---|
| Exp 1 – Baseline | (64, 32) | ReLU | 0.001 | 0.9750 | 0.728 |
| Exp 2 – Deeper | (128, 64, 32) | ReLU | 0.001 | 0.9800 | 0.726 |
| Exp 3 – High LR | (64, 32) | ReLU | 0.01 | 0.9725 | 0.782 |
| **Exp 4 – Low LR** | **(64, 32)** | **ReLU** | **0.0001** | **0.9800** | **0.811** ⭐ |
| **Exp 5 – tanh** | **(64, 32)** | **tanh** | **0.001** | **0.9700** | **0.813** ⭐ |
| Exp 6 – Large Batch | (64, 32) | ReLU | 0.001 | 0.9775 | 0.759 |

> ⭐ **Best configurations:** Exp 4 (Low LR) and Exp 5 (tanh) achieved the highest AUC-ROC scores.

---

## Key Findings
1. **Accuracy is misleading** for imbalanced datasets — AUC-ROC is the right metric here.
2. A **low learning rate (0.0001)** and **tanh activation** both improved generalisation.
3. The model shows mild overfitting due to class imbalance — future work should explore SMOTE or class weighting.

---

## Repository Structure
```
part-1-neural-network-analysis/
│
├── README.md                          ← This file
├── notebook.ipynb                     ← Full analysis notebook (Tasks 1–6)
├── requirements.txt                   ← Python dependencies
└── results/
    ├── eda_overview.png               ← Exploratory data analysis plots
    ├── evaluation_outputs.png         ← Confusion matrix + loss curve
    └── model_comparison_table.csv     ← Experiment comparison data
    └── model_comparison_table.png     ← AUC-ROC bar chart
```

---

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
