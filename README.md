
# Diabetes Risk Prediction Using Machine Learning

## Description
This project aims to build a machine learning model that predicts the likelihood of diabetes in patients based on medical data. Using advanced algorithms such as Gradient Boosting (XGBoost), Support Vector Machines (SVM), and Neural Networks, we analyze key predictors of diabetes and provide actionable insights for healthcare professionals. The goal is to enable early detection and prevention of diabetes through data-driven predictions.

The project leverages the **PIMA Diabetes Dataset**, which contains medical predictor variables such as glucose levels, BMI, age, and insulin levels, along with a binary outcome variable (`Outcome`) indicating whether a patient has diabetes.

---

## Table of Contents
1. [Dataset Description](#1-dataset-description)
2. [Preprocessing Steps](#2-preprocessing-steps)
3. [Models Implemented](#3-models-implemented)
4. [Key Insights and Visualizations](#4-key-insights-and-visualizations)
5. [Challenges Faced and Solutions](#5-challenges-faced-and-solutions)
6. [How to Run the Code](#6-how-to-run-the-code)
7. [Conclusion](#7-conclusion)

---

### 1. Dataset Description
The **PIMA Diabetes Dataset** is a publicly available dataset containing medical records of female patients aged 21 years or older. It includes the following features:
- **Pregnancies**: Number of times pregnant.
- **Glucose**: Plasma glucose concentration after a 2-hour oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)^2).
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history.
- **Age**: Age in years.
- **Outcome**: Binary variable (0 = Non-diabetic, 1 = Diabetic).

The dataset is imbalanced, with more non-diabetic patients than diabetic ones, which requires careful handling during modeling.

---

### 2. Preprocessing Steps
To ensure the dataset is ready for modeling, the following preprocessing steps were performed:
- Checked for missing values and outliers.
- Analyzed feature distributions and correlations.
- Used `StandardScaler` to normalize the features, ensuring all variables are on the same scale.
- Divided the dataset into training (80%) and testing (20%) sets to evaluate model performance.
- Addressed class imbalance by using metrics like F1 Score and AUC-ROC, which are robust to imbalanced datasets.

---

### 3. Models Implemented
Three machine learning models were implemented to predict diabetes risk:

1. **Gradient Boosting (XGBoost)**:
   - Rationale: XGBoost is highly effective for tabular data and handles non-linear relationships well. It also provides feature importance, which is valuable for understanding key predictors.
2. **Support Vector Machine (SVM)**:
   - Rationale: SVM is suitable for high-dimensional data and works well with scaled features. It was included to compare performance with tree-based methods.
3. **Neural Network (MLPClassifier)**:
   - Rationale: Neural networks can capture complex patterns in the data, making them a good candidate for comparison with traditional models.

Each model was evaluated using **F1 Score** and **AUC-ROC** metrics to ensure robustness against class imbalance.

---

### 4. Key Insights and Visualizations
- **Feature Importance**:
  - Key predictors of diabetes include **Glucose**, **BMI**, and **Age**, as identified by the XGBoost model.
  - A bar chart visualizes the importance of each feature, helping healthcare professionals focus on critical factors.
- **Class Distribution**:
  - A count plot shows the imbalance between diabetic and non-diabetic patients.
- **Model Performance**:
  - ROC curves and confusion matrices provide insights into model accuracy and trade-offs between true positives and false positives.
- **Actionable Insights**:
  - Patients with high glucose levels, elevated BMI, or advanced age are at higher risk of diabetes.
  - Recommendations include lifestyle changes such as weight management and regular glucose monitoring.

---

### 5. Challenges Faced and Solutions
1. **Class Imbalance**:
   - Challenge: The dataset is imbalanced, with fewer diabetic patients than non-diabetic ones.
   - Solution: Used metrics like F1 Score and AUC-ROC, which are less sensitive to class imbalance.
2. **Feature Scaling**:
   - Challenge: Features had varying scales, which could affect model performance.
   - Solution: Applied `StandardScaler` to normalize the features.
3. **Model Selection**:
   - Challenge: Choosing the right model for the dataset.
   - Solution: Compared multiple models (XGBoost, SVM, Neural Network) and selected the best-performing one based on evaluation metrics.

---

### 6. How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
2. Install required libraries
   pip install -r requirements.txt
3. Run the main script:
   python main.py
4.  View the results:
    -Model performance metrics will be printed in the console.
    -Visualizations will be displayed in separate windows.
Note : Ensure you have Python 3.x installed and the necessary libraries (pandas, numpy, scikit-learn, xgboost, etc.).

   ## Conclusion
This project demonstrates the power of machine learning in predicting diabetes risk based on medical data. By leveraging advanced algorithms and providing actionable insights, this tool can assist healthcare professionals in early detection and prevention. Future work could involve expanding the dataset, exploring additional models, and deploying the solution in a clinical setting.
