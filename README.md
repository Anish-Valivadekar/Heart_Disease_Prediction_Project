🫀 Heart Disease Risk Prediction – Machine Learning Project

This project develops an end-to-end machine learning classification pipeline to predict the likelihood of heart disease using clinical patient data. The goal is to support early risk identification and assist in preventive healthcare decision-making.

📊 Dataset

1025 patient records
14 clinical features (age, chest pain type, cholesterol, thalach, ca, oldpeak, etc.)
Binary target (0 = No Disease, 1 = Disease)
No missing values

🔎 Methodology

1. Performed Exploratory Data Analysis (EDA) including statistical summary, class distribution, and correlation heatmap.
2. Applied Stratified Train-Test Split (80:20) and feature scaling using StandardScaler.
3. Trained and compared:
-Logistic Regression
-K-Nearest Neighbors (KNN)
-Random Forest
4. Used GridSearchCV for hyperparameter tuning.
5. Evaluated models using Accuracy, Precision, Recall, F1-score, Confusion Matrix, Cross-Validation, and ROC-AUC.

📈 Results

Logistic Regression: 80.9%
KNN: 86.3%
Random Forest: 100%
Cross-Validation Accuracy: 98.17%
ROC-AUC Score: 1.0
Random Forest achieved the best performance and demonstrated strong class separation ability.

🧠 Key Insights

The most influential predictors were:
Chest pain type (cp)
Number of major vessels (ca)
Maximum heart rate (thalach)
ST depression (oldpeak)
Age

🛠 Technologies Used

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
