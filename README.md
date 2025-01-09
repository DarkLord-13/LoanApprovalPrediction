# Loan Approval Prediction

This repository contains a project for predicting the loan status (approved or not approved) using machine learning techniques. The project demonstrates the workflow from data preprocessing to model training, evaluation, and prediction. The dataset used is the "Loan Prediction" dataset.

## Table of Contents
- [Introduction](#introduction)
- [Workflow](#workflow)
- [Setup Instructions](#setup-instructions)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Dependencies](#dependencies)

## Introduction

The objective of this project is to predict the loan status based on various features such as gender, marital status, dependents, education, self-employment, income, loan amount, loan term, credit history, and property area.

## Workflow

1. Data acquisition
2. Data preprocessing
3. Data visualization
4. Train-test split
5. Model training
6. Model evaluation
7. Prediction

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/DarkLord-13/Machine-Learning-01.git
    cd Machine-Learning-01
    ```

2. Open the Jupyter Notebook `LoanStatusPrediction.ipynb` in your preferred environment (e.g., Jupyter Notebook, Google Colab).

## Data Preparation

1. Import the required libraries:
    ```python
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    ```

2. Load the dataset:
    ```python
    dataset = pd.read_csv('/content/Loan Prediction.csv')
    ```

3. Handle missing values by dropping any rows with missing values:
    ```python
    dataset = dataset.dropna()
    ```

4. Perform label encoding on categorical features:
    ```python
    dataset.replace({'Loan_Status':{'N':0, 'Y':1}, 'Married':{'No':0,'Yes':1}, 'Gender':{'Male':1,'Female':0}, 'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2}, 'Education':{'Graduate':1,'Not Graduate':0}}, inplace=True)
    ```

5. Replace the "3+" value in the "Dependents" feature with 4:
    ```python
    dataset = dataset.replace({"Dependents": {"3+": 4}})
    ```

## Model Training

1. Split the data into features (X) and target (y):
    ```python
    X = dataset.drop(columns=['Loan_Status', 'Loan_ID'], axis=1)
    y = dataset['Loan_Status']
    ```

2. Split the data into training and testing sets:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2, stratify=y)
    ```

3. Train a Support Vector Machine (SVM) model:
    ```python
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    ```

## Evaluation

Evaluate the model on the training and testing sets:
```python
# Training accuracy
X_train_prediction = model.predict(X_train)
train_accuracy = accuracy_score(X_train_prediction, y_train)
print(f'Training Accuracy: {train_accuracy}')

# Testing accuracy
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, y_test)
print(f'Testing Accuracy: {test_accuracy}')
```

## Prediction

Predict the loan status for a new input:

```python
input_data = X_test.iloc[1]
input_data = np.array(input_data).reshape(1, -1)
prediction = model.predict(input_data)
print(f'Prediction: {"Approved" if prediction[0] == 1 else "Not Approved"}')
```

## Dependencies
1. NumPy
2. Pandas
3. Seaborn
4. Scikit-learn
To install the dependencies, run:
```python
pip install numpy pandas seaborn scikit-learn
```
