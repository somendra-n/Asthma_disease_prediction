# 🫁 Asthma Disease Prediction Using Regression Models

This project focuses on predicting asthma severity based on medical symptoms and demographic attributes using **Logistic Regression** and **Linear Regression**.

---

## 📂 Dataset

The dataset is read from Google Drive and contains **316,800 records** with **19 features**, such as:

* **Symptoms**: Tiredness, Dry Cough, Difficulty in Breathing, etc.
* **Demographics**: Age groups, Gender
* **Target Labels**: Severity\_Mild, Severity\_Moderate, Severity\_None

After removing duplicates, **5,760 unique entries** were used for model training.

---

## 🔍 Exploratory Data Analysis (EDA)

* Visualized nasal congestion impact on asthma using histograms
* Inspected data distribution, missing values, and feature balance
* Found no missing values
* Removed duplicate records to avoid bias

---

## 🧼 Data Preprocessing

* Features normalized using `sklearn.preprocessing.normalize`
* Target variable: `Severity_None` (Binary classification: 1 = No asthma, 0 = Asthma)
* Data split using `train_test_split` with a 70:30 ratio, stratified on `Severity_None`

---

## 📊 Models Used

### 🔹 Logistic Regression

```python
model = LogisticRegression()
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
accuracy = accuracy_score(test_Y, prediction)
```

* **Accuracy**: `0.748`
* **Precision & Recall**:

  * Class 0 (Asthma): Precision = 0.80, Recall = 0.89
  * Class 1 (No Asthma): Precision = 0.49, Recall = 0.32

### 🔹 Linear Regression

```python
model = LinearRegression()
model.fit(train_X, train_Y)
prediction = model.predict(test_X).round()
accuracy = accuracy_score(test_Y, prediction)
```

* **Accuracy**: `0.7481`
* Note: Linear Regression is not ideal for classification tasks but performs similarly to logistic regression here.

---

## 📈 Evaluation Metrics

* `accuracy_score`
* `classification_report` for Logistic Regression
* Rounded predictions used for Linear Regression to simulate classification

---

## 📊 Visualization

* Line and histogram plots to explore feature distribution and class correlation
* Checked age group contributions and nasal congestion frequency

---

## ✅ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🚀 How to Run

```python
# Mount Drive in Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Load data
import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/asthma.csv')

# Run preprocessing, model training, and evaluation
```

---

## 📌 Notes

* Consider using **classification algorithms** like **Random Forest**, **SVM**, or **XGBoost** for improved performance.
* Feature selection or dimensionality reduction (e.g., PCA) could be applied to reduce multicollinearity and improve generalization.

---

## 📬 Contact

For questions or contributions, feel free to open an issue or submit a pull request.
