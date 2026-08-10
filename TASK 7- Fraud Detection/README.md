# Credit Card Fraud Detection Using Machine Learning

# Internship Task 7 – Credit Card Fraud Detection

This project was completed as the seventh task of my data analytics internship.

The main objective of this task was to identify fraudulent credit card transactions using machine learning.

Fraud detection is different from many normal classification problems because fraudulent transactions are very rare compared with legitimate transactions. Because of this, simply looking at accuracy can give a misleading result.

In this project, I first explored the class distribution and then used SMOTE (Synthetic Minority Oversampling Technique) to deal with the class imbalance. After that, I trained and compared two machine learning models:

* Logistic Regression
* Random Forest Classifier

The models were evaluated using Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Project Objective

The main objectives of this task were:

* Understand the credit card transaction dataset.
* Check the structure and data types.
* Check for missing values.
* Analyze the number of fraudulent and legitimate transactions.
* Calculate the percentage of fraudulent transactions.
* Perform exploratory data analysis.
* Understand the transaction amount distribution.
* Study the effect of class imbalance.
* Understand why accuracy alone is not suitable for fraud detection.
* Split the data into training and testing sets.
* Apply SMOTE to the training data.
* Train Logistic Regression.
* Train Random Forest.
* Evaluate both models using appropriate classification metrics.
* Compare Precision, Recall, F1-Score, and ROC-AUC.
* Analyze the confusion matrices.
* Study Random Forest feature importance.
* Select the more suitable model for fraud detection.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dataset

The dataset used in this project is a Credit Card Fraud Detection dataset.

It contains credit card transactions made by European cardholders.

The dataset contains:

* 284,807 transactions
* 31 columns
* 30 numerical input variables
* 1 target variable (`Class`)

The columns include `Time`, `V1` to `V28`, `Amount`, and `Class`.

Most of the `V1` to `V28` variables are anonymized features, so their original business meaning is not directly available in the dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Target Variable

The target variable is:

```text
Class
```

It contains two values:

| Class | Meaning                |
| ----- | ---------------------- |
| 0     | Legitimate transaction |
| 1     | Fraudulent transaction |

The dataset contains:

* 284,315 legitimate transactions
* 492 fraudulent transactions

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Class Imbalance

One of the most important observations from this project was the severe class imbalance.

The percentage distribution was:

| Transaction Type | Percentage |
| ---------------- | ---------: |
| Legitimate       |   99.8273% |
| Fraudulent       |    0.1727% |

This means that fraudulent transactions represent less than 1% of the entire dataset.

This creates a major problem for machine learning models because a model can achieve very high accuracy simply by predicting almost every transaction as legitimate.

For example, a model that predicts every transaction as legitimate would get approximately 99.83% accuracy but would fail to detect fraud.

Therefore, accuracy should not be the only metric used for this problem.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Tools and Technologies

The project was completed using:

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn

Machine learning techniques used:

* SMOTE
* Logistic Regression
* Random Forest Classifier

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. Importing Libraries

I started by importing the libraries required for data analysis, visualization, and machine learning.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

These libraries were used for data handling, numerical operations, and visualization.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 2. Loading the Dataset

The dataset was loaded using Pandas.

```python
df = pd.read_csv("creditcard.csv")
```

I then displayed the first few records to understand the structure of the dataset.

The dataset contains 284,807 rows and 31 columns.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3. Understanding the Dataset

I used:

```python
df.info()
```

to check the data types and number of non-null observations.

All 31 columns contained 284,807 non-null observations.

The dataset contains:

* 30 floating-point columns
* 1 integer column (`Class`)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 4. Checking Missing Values

I checked for missing values using:

```python
df.isna().sum()
```

The result showed zero missing values for all columns.

Therefore, no missing-value treatment was required for this dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 5. Checking the Target Classes

I checked the unique values of the target variable:

```python
df['Class'].unique()
```

The result was:

```text
[0, 1]
```

This confirmed that the problem is a binary classification problem.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 6. Class Distribution Analysis

I calculated the number of transactions in each class using:

```python
class_counts = df['Class'].value_counts()
print(class_counts)
```

The result was:

```text
Class
0    284315
1       492
```

This clearly shows the large difference between legitimate and fraudulent transactions.

I also calculated the percentage:

```python
class_percentage = df['Class'].value_counts(
    normalize=True
) * 100
```

The result showed approximately:

```text
Legitimate: 99.83%
Fraud:       0.17%
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 7. Exploratory Data Analysis

I performed exploratory analysis to understand the transaction data and the difference between legitimate and fraudulent transactions.

The EDA included:

* Class distribution
* Transaction amount distribution
* Fraud vs legitimate transaction comparison
* Time-related analysis
* Visualizations using Matplotlib and Seaborn

The class distribution plot clearly showed the severe imbalance in the dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 8. Transaction Amount Analysis

The `Amount` variable represents the transaction amount.

I analyzed transaction amounts separately for fraudulent and legitimate transactions to understand whether there were visible differences in transaction values.

This type of analysis is useful because transaction amount can be an important factor when identifying unusual transactions.

However, the model does not rely on transaction amount alone. It uses the available transaction features together.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 9. Time Analysis

The dataset also contains a `Time` variable.

I explored transaction timing as part of the EDA to understand whether transaction behavior changes over time.

Time-based analysis can be useful in fraud detection because unusual transaction timing may sometimes provide additional information about suspicious activity.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 10. Why Accuracy Is Not Enough

Accuracy is normally a useful classification metric, but in this project it can be misleading.

Because only0.1727% of transactions are fraudulent, a model can achieve approximately 99.83% accuracy by predicting almost everything as legitimate.

Such a model would not be useful for a real fraud detection system.

For this reason, I focused on:

* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

instead of judging the models using accuracy alone.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 11. Train-Test Split

The data was divided into training and testing datasets.

The test set contained:

56,962 transactions

The testing set included:

* 56,864 legitimate transactions
* 98 fraudulent transactions

The test data was kept separate from the SMOTE process so that synthetic observations were generated only from the training data.

This is important because the test set should represent unseen transactions.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 12. Handling Class Imbalance Using SMOTE

To address the severe class imbalance, I used SMOTE.

SMOTE stands for:

Synthetic Minority Oversampling Technique

It creates synthetic examples of the minority class instead of simply duplicating existing fraud observations.

The implementation used was:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

x_train_smote, y_train_smote = smote.fit_resample(
    x_train,
    y_train
)
```

After applying SMOTE, the training data became balanced:

```text
Class 0    227451
Class 1    227451
```

This gave the models a much better opportunity to learn the characteristics of fraudulent transactions.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 13. Logistic Regression

The first machine learning model used was Logistic Regression.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    random_state=42,
    max_iter=1000
)

lr.fit(x_train_smote, y_train_smote)

lr_pred = lr.predict(x_test)
lr_prob = lr.predict_proba(x_test)[:,1]
```

The model was trained using the SMOTE-balanced training data.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 14. Random Forest Classifier

The second model used was Random Forest.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

rf.fit(x_train_smote, y_train_smote)

rf_pred = rf.predict(x_test)
rf_prob = rf.predict_proba(x_test)[:,1]
```

The Random Forest model used 50 trees.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 15. Evaluation Metrics

The models were evaluated using the following metrics.

# Precision

Precision tells us how many transactions predicted as fraudulent were actually fraudulent.

A high precision means fewer legitimate transactions are incorrectly flagged as fraud.


# Recall

Recall tells us how many of the actual fraudulent transactions were correctly identified.

For fraud detection, recall is particularly important because missing a fraudulent transaction can result in financial loss.


# F1-Score

F1-Score combines Precision and Recall into a single metric.

It is useful when both false positives and false negatives matter.

# ROC-AUC

ROC-AUC measures how well the model separates fraudulent transactions from legitimate transactions across different classification thresholds.

# Confusion Matrix

The confusion matrix provides four important values:

* True Positive
* True Negative
* False Positive
* False Negative

These values help understand the types of mistakes made by the model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 16. Logistic Regression Results

The Logistic Regression model produced the following results:

| Metric    | Result |
| --------- | -----: |
| Precision |  0.077 |
| Recall    |  0.908 |
| F1-Score  |  0.142 |
| ROC-AUC   |  0.973 |

The model achieved very high recall, meaning it detected most fraudulent transactions.

However, its precision was very low.

This means that the model also classified many legitimate transactions as fraudulent.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 17. Logistic Regression Confusion Matrix

The Logistic Regression confusion matrix showed:

* 55,794 legitimate transactions correctly classified.
* 89 fraudulent transactions correctly detected.
* 1,070 legitimate transactions incorrectly classified as fraud.
* 9 fraudulent transactions missed.

The main advantage of Logistic Regression was its high fraud detection rate.

The main disadvantage was the large number of false positives.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 18. Random Forest Results

The Random Forest model produced:

| Metric    | Result |
| --------- | -----: |
| Precision |  0.875 |
| Recall    |  0.857 |
| F1-Score  |  0.866 |
| ROC-AUC   |  0.970 |

The Random Forest model achieved a much better balance between Precision and Recall.

It correctly identified a large percentage of fraudulent transactions while keeping false positives much lower than Logistic Regression.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 19. Model Comparison

The final comparison was:

| Model               | Precision |    Recall |  F1-Score |   ROC-AUC |
| ------------------- | --------: | --------: | --------: | --------: |
| Logistic Regression |   0.077   |   0.908   |   0.142   |   0.973   |
| Random Forest       |   0.875   |   0.857   |   0.866   |   0.970   |

The comparison shows an important trade-off.

Logistic Regression has slightly higher Recall and ROC-AUC, but its Precision is very low.

Random Forest has slightly lower Recall and ROC-AUC, but its Precision and F1-Score are much higher.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 20. Best Model

Based on the overall results, Random Forest was selected as the better model for this project.

The main reason is its balance between Precision and Recall.

Random Forest achieved:

* Precision = 87.5%
* Recall = 85.7%
* F1-Score = 86.6%
* ROC-AUC = 97.0%

Compared with Logistic Regression, Random Forest produced a much stronger F1-Score and much higher Precision.

This makes it more practical when the goal is to detect fraud while avoiding too many false alarms.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 21. Recall vs Precision in Fraud Detection

There is an important trade-off between Recall and Precision in this project.

# High Recall

A high Recall means the model catches more fraudulent transactions.

This is important because missing a fraudulent transaction can cause:

* Financial losses
* Customer complaints
* Loss of trust
* Additional security problems

# High Precision

High Precision means that when the model flags a transaction as fraud, it is more likely to actually be fraudulent.

This reduces unnecessary investigations and false alerts.

In this project, Logistic Regression prioritized Recall, while Random Forest provided a much better balance between Recall and Precision.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 22. Feature Importance

I also examined the feature importance generated by Random Forest.

The top features included:

| Rank | Feature | Importance |
| ---: | ------- | ---------: |
|    1 | V14     |   0.195626 |
|    2 | V4      |   0.129471 |
|    3 | V10     |   0.114604 |
|    4 | V11     |   0.106697 |
|    5 | V17     |   0.086510 |
|    6 | V12     |   0.071724 |
|    7 | V3      |   0.035460 |
|    8 | V7      |   0.034472 |
|    9 | V16     |   0.033671 |
|   10 | V2      |   0.025466 |

Among the features shown, V14 had the highest Random Forest importance.

Since the V1–V28 variables are anonymized, the analysis can identify which variables are useful for prediction but cannot directly explain their real-world meaning.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 23. Key Findings

The main findings from this project were:

1. The dataset contains 284,807 transactions and 31 columns.

2. Only 492 transactions were fraudulent, compared with 284,315 legitimate transactions.

3. Fraudulent transactions represented only 0.1727% of the complete dataset.

4. The extreme class imbalance makes accuracy a poor standalone metric.

5. SMOTE was used to balance the training data, resulting in 227,451 observations in each class.

6. Logistic Regression achieved higher Recall (90.8%) but very low Precision (7.7%).

7. Random Forest achieved 87.5% Precision and 85.7% Recall.

8. Random Forest achieved a much stronger F1-Score (86.6%) than Logistic Regression (14.2%).

9. V14 was the most important feature in the Random Forest model among the features analyzed.

10. Random Forest was selected as the more practical model because of its better balance between fraud detection and false alarms.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# What I Learned

This task helped me understand one of the important challenges in machine learning: class imbalance.

During this project, I learned how to:

* Explore a large transaction dataset.
* Check data quality.
* Analyze class imbalance.
* Understand why accuracy can be misleading.
* Use SMOTE for minority class oversampling.
* Train Logistic Regression.
* Train Random Forest.
* Evaluate classification models using Precision, Recall, F1-Score, and ROC-AUC.
* Interpret a confusion matrix.
* Understand the Precision-Recall trade-off.
* Compare machine learning models based on the problem requirements.
* Analyze feature importance.
* Think about model performance from a real-world fraud detection perspective.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Conclusion

In this project, I worked on credit card fraud detection using machine learning.

The dataset contained 284,807 transactions, but only 492 were fraudulent. This severe class imbalance was the main challenge of the project. Because of this, I did not rely only on accuracy and instead focused on Precision, Recall, F1-Score, ROC-AUC, and the Confusion Matrix.

I used SMOTE to balance the training data and then trained Logistic Regression and Random Forest models.

Logistic Regression achieved a high Recall of 90.8%, but its Precision was only 7.7%, which resulted in many false positives. Random Forest achieved a Recall of 85.7% and a much higher Precision of 87.5%, giving it an F1-Score of 86.6%.

Based on these results, I selected Random Forest as the better model for this project because it provides a much better balance between detecting fraudulent transactions and avoiding unnecessary false alerts.

This task gave me practical experience with imbalanced datasets, SMOTE, binary classification, model evaluation, confusion matrices, Precision-Recall trade-offs, and fraud detection using Python.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Author

Krishna Deshmukh

M.Sc. Statistics

Skills: Python | SQL | Excel | Power BI | Machine Learning | Data Analysis

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Internship

This task is a part of my Data Analysis Internship.

Project: Credit Card Fraud Detection Using Machine Learning
