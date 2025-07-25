
# 📨 Spam Message Classifier (NLP + ML)

This project builds a **Spam Detection System** that classifies text messages as either **spam** or **ham (not spam)** using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. It includes detailed preprocessing, visualization, feature engineering, model training, and evaluation — with a focus on **imbalanced data handling** using SMOTE.

---

## 📁 Project Structure

- `Spam_Classifier.ipynb` — Complete Jupyter Notebook with code and explanations.
- `spam.csv` — The dataset used for training and testing.
- `Spam_detection.pdf` — Report summarizing the full project.
- `README.md` — This documentation file.

---

## 🔍 Steps and Methodology

### 1. Data Loading & Exploration
- Dataset loaded from `spam.csv` using `latin-1` encoding.
- Inspected structure, shape, and initial entries.

### 2. Data Cleaning
- Dropped unnamed/redundant columns.
- Renamed:
  - `'v1'` → `'label'` (spam/ham)
  - `'v2'` → `'text'` (message content)
- Converted labels to binary:
  - `spam` → `1`
  - `ham` → `0`

### 3. Data Visualization 📊
- Pie and bar plots to show class imbalance (more ham than spam).

### 4. Text Preprocessing 💬
Applied standard NLP steps:
- Lowercasing
- Removing punctuation, tags, and special characters (regex)
- Tokenization
- Stopword removal
- Stemming (with optional lemmatization)

### 5. Feature Engineering 🛠
- Added message length as a feature.
- Transformed cleaned text into numeric form using **TF-IDF vectorization**.

### 6. Model Training & Evaluation
- **Classifiers used**:
  - Logistic Regression
  - Linear SVM (LinearSVC)
  - Random Forest
- **Metrics**:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix

#### 📈 Sample Results (Before SMOTE):
| Model             | Accuracy | Spam Recall | F1-score (Spam) |
|------------------|----------|-------------|-----------------|
| LogisticRegression | 97%      | 76%         | 87%             |
| LinearSVC          | 98%      | 87%         | 93%             |
| Random Forest      | 98%      | 82%         | 90%             |

### 7. Imbalanced Data Handling ⚖
- Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance spam and ham samples.

### 8. Results After SMOTE:
| Model             | Accuracy | Spam Recall | F1-score (Spam) |
|------------------|----------|-------------|-----------------|
| LogisticRegression | 98%      | 88%         | 90%             |
| LinearSVC          | 98%      | 89%         | 93%             |
| Random Forest      | 98%      | 82%         | 90%             |

---

## ✅ Final Conclusion

- **LinearSVC after SMOTE** delivered the **best performance**, combining:
  - High overall accuracy (98%)
  - High precision and recall for spam
  - Excellent handling of imbalanced data
- Recommended model for deployment.

---

## 🔧 Requirements

Install dependencies using:

```bash
pip install pandas numpy matplotlib scikit-learn seaborn nltk imbalanced-learn
