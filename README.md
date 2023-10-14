# Email Spam Detection and Filtering 📧

> A machine learning project that detects and filters spam emails using TF-IDF feature extraction and six classification models with K-Fold cross-validation, built on the SMS Spam Collection dataset.

---

## 📌 Overview

Spam detection is a classic yet critical text classification problem. This project builds and compares six machine learning models to classify messages as spam or ham, using TF-IDF vectorization for feature extraction and K-Fold cross-validation to ensure robust and unbiased evaluation.

The project also includes a data-driven approach to selecting the optimal number of folds and hyperparameters for each model before final evaluation.

---

## 🔍 Key Results

| Model | Validation Method | Overall Accuracy |
|-------|------------------|-----------------|
| SVC | 10-Fold Cross Validation | ~98% |
| KNN | 10-Fold Cross Validation | ~92% |
| Decision Tree | 10-Fold Cross Validation | ~97% |
| Logistic Regression | 10-Fold Cross Validation | ~98% |
| Random Forest | 10-Fold Cross Validation | ~98% |
| Gradient Boosting | Train/Test Split (80/20) | ~97% |

---

## ⚙️ Project Pipeline

### 1. Data Cleaning
- Dropped irrelevant unnamed columns
- Renamed columns to `Category` and `Message`
- Removed duplicate entries
- Encoded target variable: `spam = 0`, `ham = 1`

### 2. Data Visualization
- Count plot and pie charts showing class distribution
- Dataset contains ~87% ham and ~13% spam messages

### 3. Feature Extraction
- Applied **TF-IDF Vectorization** with English stop words removal

### 4. Optimal Fold Selection
- Tested K-Fold cross-validation from 2 to 10 folds using Random Forest
- Plotted overall accuracy per number of folds to determine the optimal split

### 5. Model Training and Evaluation
- Trained and evaluated SVC, KNN, Decision Tree, Logistic Regression, and Random Forest using K-Fold cross-validation
- Tuned hyperparameters for KNN (best n_neighbors), Decision Tree (best max_depth), and Random Forest (best n_estimators) using error rate plots
- Trained Gradient Boosting separately using train/test split with confusion matrix and classification report

---

## 🛠️ Tools and Libraries

| Tool | Usage |
|------|-------|
| **pandas** | Data loading, cleaning, and manipulation |
| **numpy** | Numerical operations |
| **matplotlib / seaborn / plotly** | Data visualization |
| **scikit-learn** | TF-IDF, models, cross-validation, and metrics |

---

## 📂 Dataset

This project uses the **SMS Spam Collection Dataset** from Kaggle.

🔗 [View Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
