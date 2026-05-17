# Part 1: Neural Network Fundamentals and Training Behavior Analysis


## 📁 Repository Content
````
part-1-neural-network-analysis/
|
|── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_comparison_table.png or .csv
    └── confusion_matrix.png
    └── training_history.png
````

---

## 🎯 Problem Statement

You are given a structured dataset containing multiple input features and one target variable. Your task is to build and analyze a simple neural network model for a supervised learning problem.

The objective of this part is not only to train a model, but also to clearly demonstrate how neural networks learn through forward pass, loss calculation, backpropagation, and parameter updates.

---
## Dataset
Use the relevant Part 1 dataset from the shared folder:

https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

##  Dataset Overview

| Property | Value |
|----------|-------|
| Rows (customers) | 2,000 |
| Columns (features + target) | 17 |
| Missing values | None |
| Target column | `churn` |
| Categorical features | `region`, `plan_type`, `contract_type`, `payment_method` |
| Numerical features | tenure, charges, login days, tickets, delays, data usage, satisfaction, complaint recency, discounts, referrals |
| Identifier (excluded) | `customer_id` |

---

## 🔍 Key Observation: Class Imbalance

| Class | Count | Percentage |
|-------|-------|------------|
| No Churn (0) | 1,969 | 98.5% |
| Churn (1) | 31 | 1.5% |

> ⚠️ **This is a severely imbalanced dataset.** A model that always predicts "no churn" would score 98.5% accuracy but be completely useless. This shaped every decision made in preprocessing and evaluation.

---

##  Approach & Steps

### Task 1 — Dataset Understanding
- Loaded and inspected the dataset with `pandas`
- Confirmed 2000 rows × 17 columns
- Identified 4 categorical and 11 numerical feature columns
- Verified **zero missing values** - no imputation needed
- Plotted churn distribution (bar + pie chart) to visualize the imbalance

### Task 2 — Data Preprocessing

| Step | What | Why |
|------|------|-----|
| Drop `customer_id` | Remove identifier column | Not a predictive feature; may cause memorization |
| One-Hot Encoding | Convert categorical text → binary columns | Neural networks require numerical input |
| StandardScaler | Normalize numerical features to mean=0, std=1 | Prevents large-valued features from dominating |
| Train/Test Split | 80% train, 20% test | Evaluate on unseen data for honest performance |
| `stratify=y` in split | Preserve churn ratio in both sets | Critical for imbalanced datasets |
| Class weights | Weight churn=1 samples ~64× higher | Forces model to pay attention to the rare class |

> The scaler was **fit only on training data** and then applied to test data. Fitting on test data would be "data leakage."

### Task 3 — Model Architecture

```
Input Layer     → 19 features (after encoding)
Hidden Layer 1  → 16 neurons, ReLU activation
Hidden Layer 2  →  8 neurons, ReLU activation
Output Layer    →  1 neuron,  Sigmoid activation
```

| Component | Choice | Reason |
|-----------|--------|--------|
| Hidden activation | ReLU | Simple, fast, avoids vanishing gradients |
| Output activation | Sigmoid | Outputs probability [0,1] for binary classification |
| Loss function | Binary Crossentropy | Standard for binary classification problems |
| Optimizer | Adam | Adaptive learning rate; robust default choice |

### Task 4 — Training & Evaluation

- **Epochs:** 50
- **Batch size:** 32
- **Validation split:** 10% of training data monitored during training
- **Prediction threshold:** 0.3 (lowered from default 0.5 to catch more churners given the imbalance)

Evaluation metrics used:
- Confusion Matrix
- Precision, Recall, F1-Score (per class)
- Training vs. validation loss/accuracy curves

### Task 5 — Hyperparameter Experiments

Four configurations were tested and compared:

| Experiment | Architecture | Learning Rate | Batch Size | Epochs |
|------------|-------------|---------------|------------|--------|
| Exp 1 — Baseline | [16 → 8] | 0.001 | 32 | 50 |
| Exp 2 — Wider + Deeper | [32 → 16 → 8] | 0.001 | 32 | 50 |
| Exp 3 — High LR | [16 → 8] | 0.01 | 32 | 50 |
| Exp 4 — Small Batch | [16 → 8] | 0.001 | 16 | 50 |

---

## 📈 Results & Observations

### Overall Performance

- **Test Accuracy** was high across all experiments (~97–99%), but this is expected given the severe imbalance - not a reliable metric here.
- **Churn Recall** and **Churn F1-Score** are the meaningful metrics, as they measure how well the model actually identifies churners.

### Experiment Observations

**Exp 1 — Baseline [16→8]**  
A solid starting point. The model learned the dominant class well. With class weighting and a lowered threshold, it began catching some churners.

**Exp 2 — Wider + Deeper [32→16→8]**  
Adding more neurons and a third hidden layer gave the model more capacity. This can help when patterns are complex, but can also overfit on small datasets or rare classes.

**Exp 3 — High Learning Rate (lr=0.01)**  
A 10× higher learning rate causes the optimizer to take larger steps. This sometimes makes training faster but can cause the loss to bounce or overshoot the minimum. Expected to show less stable validation curves.

**Exp 4 — Small Batch (batch=16)**  
Smaller batches introduce more noise into each weight update, which can act as a form of regularization. Sometimes leads to better generalization, but training is slower.

### On Overfitting vs. Underfitting

| Signal | Meaning |
|--------|---------|
| Train accuracy >> Test accuracy (gap > 10%) | **Overfitting** — model memorized training data |
| Both train and test accuracy low | **Underfitting** — model hasn't learned enough |
| Train ≈ Test accuracy, both reasonable | **Good fit** |

Given the tiny number of positive samples (31 churners), some instability in churn-specific metrics across experiments is expected and normal.

---

## 💡 Key Concepts Explained 

### Weights & Biases
Every neuron computes: `output = activation(W·input + b)`  
- **Weights (W)** — control how much each input feature matters  
- **Biases (b)** — allow the neuron to shift its activation threshold  
- Both are learned automatically through backpropagation

### Why Activation Functions?
Without them, stacking layers is mathematically equivalent to a single linear equation — no matter how deep the network is. Activations like **ReLU** introduce non-linearity, enabling the network to model complex, curved decision boundaries.

### Learning Rate Effects

```
Too HIGH  →  Large steps → overshoots minimum → unstable loss ❌
Too LOW   →  Tiny steps  → very slow training → may get stuck ⏳
Just right → Steady convergence to minimum ✅
```

Adam optimizer helps by adapting the effective learning rate per weight automatically.

### Why Recall > Accuracy for Churn?
Missing a customer who is about to churn (False Negative) is far more costly to a business than a false alarm (False Positive). **Recall for the churn class** - "of all customers who actually churned, how many did we catch?" is therefore the primary metric of interest.

---

## 🛠️ How to Run

1. Place `customer_churn_nn.csv` in the same folder as the notebook
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```
3. Open the notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
4. Run cells from top to bottom,each cell builds on the previous one

---

## 📦 Dependencies

| Library | Version (recommended) | Purpose |
|---------|-----------------------|---------|
| `pandas` | ≥ 1.5 | Data loading and manipulation |
| `numpy` | ≥ 1.23 | Numerical operations |
| `matplotlib` | ≥ 3.6 | Plotting |
| `seaborn` | ≥ 0.12 | Enhanced visualizations |
| `scikit-learn` | ≥ 1.2 | Preprocessing, metrics |
| `tensorflow` | ≥ 2.10 | Neural network (Keras API) |

---
## Results files
| File | What it shows | Which Task |
|------|--------------|------------|
| `churn_distribution.png` | Bar + Pie chart of how many customers churned vs stayed (340 vs 54 etc.) | **Task 1** - Dataset Understanding |
| `confusion_matrix.png` | How well the model predicted churn vs no churn after training | **Task 4** - Evaluation |
| `training_history.png` | Loss and accuracy curves over 50 epochs | **Task 4** - Training |
| `model_comparison_table.png` | Bar charts comparing all 4 hyperparameter experiments | **Task 5** - Experiments |




