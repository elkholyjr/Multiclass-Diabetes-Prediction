
# 🩺 Multiclass Diabetes Prediction System

👉 [**Live Demo**](https://multiclass-diabetes-prediction.streamlit.app/)
![0d5dc2c7-6468-4db6-8170-332a95666500](https://github.com/user-attachments/assets/8f38ec99-3821-41a1-9a3f-7e86fca543bd)


This project aims to build a **Multiclass Diabetes Prediction System** that classifies patients into one of the following categories based on medical diagnostic features:

- **0 → Non-Diabetic**
- **1 → Diabetic**
- **2 → Predicted Diabetic (At Risk)**

---

## 🧪 Dataset Overview

The dataset used is a multi-class diabetes dataset that includes various biochemical and physical examination parameters such as:

- Age
- Urea
- Creatinine (Cr)
- HbA1c
- Triglycerides (TG)
- HDL Cholesterol
- BMI
- Class (Target)

---

## 🔍 Data Preprocessing

### Feature Engineering & Cleaning:
- Removed non-informative or redundant features:
  - `Gender`, `Chol`, `LDL`, `VLDL`
- Created a new derived feature:
  - `TG_to_HDL = TG / HDL`
- Final features selected for training:
  - `AGE`, `Urea`, `Cr`, `HbA1c`, `TG`, `TG_to_HDL`, `BMI`, `Class`

### Visualization:
- `sns.pairplot()` and `sns.boxplot()` used to inspect distributions and outliers.
- `sns.heatmap()` used to check feature correlations.

### Scaling & Balancing:
- **StandardScaler** used to normalize features.
- **SMOTE** used to oversample underrepresented class `1 (Diabetic)` for class balance.

---

## 🤖 Model Training

Three models were trained and compared:

| Model              | Notes |
|-------------------|-------|
| **Logistic Regression** | Good baseline, but misclassified diabetics as non-diabetics. |
| **K-Nearest Neighbors (k=6)** | Still showed misclassification on key diabetic patients. |
| **Random Forest Classifier** | Best performance with balanced training & testing accuracy. Overfitting reduced by tuning `n_estimators`, `max_depth`, `min_samples_leaf`. |

> ✅ **Final Model:** `RandomForestClassifier` with optimized parameters.

---

## 🧠 Evaluation Metrics

Final Random Forest Model Metrics:

- **Accuracy:** ~98%
- **Confusion Matrix:** Few false negatives (critical cases)
- **Precision & Recall:** Balanced across all three classes

---

## 💾 Saved Artifacts

- `random_forest_model.pkl` → Trained model
- `scaler.pkl` → Trained StandardScaler

---

## 🌐 Streamlit App

The trained model is deployed using **Streamlit**, allowing users to input medical information and get a real-time prediction of diabetes class.

### App Features:
- Inputs: Age, Urea, Cr, HbA1c, TG, HDL, BMI
- Auto computes `TG_to_HDL`
- Outputs: One of three classes with emoji and description

> 🚀 [Launch App](https://multiclass-diabetes-prediction.streamlit.app/)

---

## 📌 Conclusion

This system is effective in early screening for diabetes and risk prediction based on easily measurable medical attributes. The optimized Random Forest model provides high accuracy and low false negatives, making it suitable for real-world medical applications.

---
