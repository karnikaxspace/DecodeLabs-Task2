
---


# 🌸 Project 2: Data Classification Using AI

**Author:** Karnika Kumari  
**Batch:** 2026 | DecodeLabs Industrial Training Kit  
**Role:** Artificial Intelligence Engineer

---

## 📌 Overview

A supervised learning model that classifies iris flowers into three species (Setosa, Versicolour, Virginica) using the **K‑Nearest Neighbors (KNN)** algorithm. This project covers the full ML pipeline: data loading, train‑test split, feature scaling, model training, and performance evaluation.

---

## 🎯 Learning Objectives

- Load and explore a real dataset (Iris)
- Apply **train‑test split** (80/20)
- Use **StandardScaler** to normalise features
- Train a **KNN classifier** with k=5
- Evaluate with **confusion matrix**, **accuracy**, and **F1 score**
- Build an interactive **Streamlit UI** for real‑time prediction

---

## 🧮 Algorithm & Math

### K‑Nearest Neighbors (KNN)
- **Proximity principle:** Similar things exist close together in feature space.
- **Majority vote:** The class of a new point is determined by the most common class among its `k` nearest neighbours.

### StandardScaler
- Transforms features to have **mean = 0** and **variance = 1**.
- Essential for distance‑based algorithms (prevents features with larger scales from dominating).

---

## 📁 Project Structure
iris_classifier/
├── README.md
├── requirements.txt
├── app.py # Streamlit UI
└── train_evaluate.py # CLI training + evaluation


---

## 🚀 Setup & Execution

### Prerequisites
- Python 3.7+

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Web UI (recommended)
```bash
streamlit run app.py
```
---

### 📊 Dataset: Iris (from scikit‑learn)
Feature	Range (cm)
Sepal Length	4.3 – 7.9
Sepal Width	2.0 – 4.4
Petal Length	1.0 – 6.9
Petal Width	0.1 – 2.5
Classes: Setosa, Versicolour, Virginica (50 samples each – perfectly balanced)

---

📈 Model Performance
After training on 120 samples and testing on 30:

Accuracy: ~96‑100% (depends on random seed)

Confusion matrix: Visible in the UI sidebar

F1 score (macro): ≥ 0.96


---

### 🧪 Test the Model
Try these known examples (should predict correctly with >95% confidence):
| Sepal Length (cm) | Sepal Width (cm) | Petal Length (cm) | Petal Width (cm) | Expected Species |
|-------------------|------------------|-------------------|------------------|------------------|
| 5.1               | 3.5              | 1.4               | 0.2              | Setosa           |
| 6.5               | 3.0              | 5.2               | 2.0              | Virginica        |
| 5.9               | 3.0              | 4.2               | 1.5              | Versicolour      |

---

### ✅ Completion Badge
Earned: Supervised Learning | KNN | Feature Scaling | Model Evaluation
