# Wine Quality Classification Using Machine Learning

# Internship Task 6 – Wine Quality Prediction

This project was completed as the sixth task of my data analytics internship.

The purpose of this project was to use machine learning to classify wine into three quality categories based on its chemical properties.

The original dataset contains a numerical `quality` score ranging from 3 to 9. For this project, I grouped the quality scores into three categories:

* Low
* Medium
* High

I then compared three classification algorithms:

* Random Forest
* Stochastic Gradient Descent (SGD)
* Support Vector Classifier (SVC)

The models were evaluated using accuracy, precision, recall, F1-score, classification reports, and confusion matrices.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Project Objective

The main objectives of this task were:

* Load and understand the Wine Quality dataset.
* Check the structure and data types.
* Identify missing values.
* Study the distribution of wine quality scores.
* Perform exploratory data analysis.
* Analyze the chemical features of wine.
* Convert the numerical quality score into three classes.
* Encode categorical variables.
* Split the data into training and testing sets.
* Maintain class proportions using stratified sampling.
* Train three different classification models.
* Evaluate and compare model performance.
* Identify the best-performing model.
* Understand the limitations of the classification approach.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dataset

The dataset contains information about the chemical properties of wine along with its quality score.

The original dataset contains:

* 6,497 observations
* 13 columns

The dataset includes both white and red wine observations through the `type` column.

# Main Variables

| Column               | Description                        |
| -------------------- | ---------------------------------- |
| type                 | Type of wine                       |
| fixed acidity        | Amount of fixed acids in the wine  |
| volatile acidity     | Amount of volatile acids           |
| citric acid          | Citric acid content                |
| residual sugar       | Sugar remaining after fermentation |
| chlorides            | Chloride concentration             |
| free sulfur dioxide  | Free sulfur dioxide level          |
| total sulfur dioxide | Total sulfur dioxide level         |
| density              | Density of the wine                |
| pH                   | Acidity level                      |
| sulphates            | Sulphate concentration             |
| alcohol              | Alcohol percentage                 |
| quality              | Original wine quality score        |

The target variable used for classification was created from the `quality` score.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Quality Classification

The original `quality` variable contains scores from 3 to 9.

The quality scores were converted into three groups:

| Quality Score | New Class |
| ------------- | --------- |
| 3–5           | Low       |
| 6             | Medium    |
| 7–9           | High      |

This created a new column:

```text
quality_3class
```

The purpose of grouping the scores was to convert the original numerical quality problem into a three-class classification problem.

Examples from the dataset show:

* Quality 5 → Low
* Quality 6 → Medium
* Quality 7 → High
* Quality 8 → High

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

Machine learning algorithms used:

* Random Forest Classifier
* SGD Classifier
* Support Vector Classifier

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. Importing Libraries

I started by importing the libraries required for data manipulation, visualization, and machine learning.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Pandas was used for data handling, NumPy for numerical operations, and Matplotlib and Seaborn for visualization.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 2. Loading the Dataset

The dataset was loaded using Pandas.

```python
df = pd.read_csv("winequality.csv")
```

I then checked the first few rows using:

```python
df.head()
```

The dataset contains 6,497 rows and 13 columns.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3. Checking Dataset Information

I used `df.info()` to check the structure and data types.

```python
df.info()
```

The dataset contained:

* 6,497 observations
* 13 columns
* 11 floating-point variables
* 1 integer variable
* 1 string/categorical variable

Some chemical variables contained missing values.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 4. Checking Missing Values

I checked missing values using:

```python
df.isna().sum()
```

The missing values were:

| Feature              | Missing Values |
| -------------------- | -------------: |
| fixed acidity        |             10 |
| volatile acidity     |              8 |
| citric acid          |              3 |
| residual sugar       |              2 |
| chlorides            |              2 |
| free sulfur dioxide  |              0 |
| total sulfur dioxide |              0 |
| density              |              0 |
| pH                   |              9 |
| sulphates            |              4 |
| alcohol              |              0 |
| quality              |              0 |

The `type` and `quality` columns did not contain missing values.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 5. Descriptive Statistics

I used:

```python
df.describe()
```

to understand the distribution of the numerical variables.

Some observations from the dataset were:

| Variable         |  Mean | Minimum | Maximum |
| ---------------- | ----: | ------: | ------: |
| Fixed acidity    |  7.22 |     3.8 |    15.9 |
| Volatile acidity |  0.34 |    0.08 |    1.58 |
| Citric acid      |  0.32 |    0.00 |    1.66 |
| Residual sugar   |  5.44 |     0.6 |    65.8 |
| Alcohol          | 10.49 |     8.0 |    14.9 |
| Quality          |  5.82 |       3 |       9 |

# The average quality score was approximately 5.82, while the median quality score was 6.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 6. Quality Score Distribution

I checked the number of observations for each quality score.

The distribution was:

| Quality | Count |
| ------: | ----: |
|       3 |    30 |
|       4 |   216 |
|       5 | 2,138 |
|       6 | 2,836 |
|       7 | 1,079 |
|       8 |   193 |
|       9 |     5 |

The largest number of wines had a quality score of 6, followed by scores of 5 and 7.

Very few observations belonged to quality scores 3 and 9.

This shows that the original quality variable was not evenly distributed.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 7. Creating Three Quality Classes

To make the classification problem easier to interpret, I created three quality categories:

```text
Low
Medium
High
```

The new variable was:

```python
quality_3class
```

This converted the original numerical quality score into a categorical target variable.

The resulting categories were then used for machine learning classification.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 8. Exploratory Data Analysis

I performed exploratory data analysis on the chemical variables.

The main areas examined were:

* Distribution of chemical features
* Quality score distribution
* Chemical characteristics of wine
* Correlation between variables
* Relationship between chemical properties and wine quality

The chemical features were separated from the quality variable for analysis.

```python
Chemical_features = df.drop(columns=['quality'])
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 9. Encoding Categorical Variables

Machine learning models require numerical input.

Therefore, I used `LabelEncoder` to convert categorical variables into numerical values.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['quality_3class'] = le.fit_transform(
    df['quality_3class']
)

df['type'] = le.fit_transform(
    df['type']
)
```

Both the wine type and the three-class quality variable were encoded before model training.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 10. Defining Features and Target

The original `quality` column and the new `quality_3class` column were separated from the predictors.

```python
x = df.drop(
    columns=["quality", "quality_3class"]
)

y = df["quality_3class"]
```

Here:

* `x` contains the input features.
* `y` contains the target class.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 11. Train-Test Split

I divided the dataset into training and testing sets using an 80/20 split.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The `stratify=y` parameter was used to maintain the class distribution in both training and testing datasets.

The test set contained 1,300 observations.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 12. Feature Scaling

For models that are sensitive to feature scales, I applied StandardScaler.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
```

The scaled data was used for the SGD and SVC models.

The Random Forest model was trained using the original feature values.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 13. Random Forest Classifier

The first classification model was Random Forest.

I used:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(x_train, y_train)

rf_pred = rf.predict(x_test)
```

The model used 100 decision trees.

Random Forest produced the best accuracy among the three models tested.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 14. SGD Classifier

The second model was Stochastic Gradient Descent.

```python
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(
    random_state=42,
    max_iter=1000
)

sgd.fit(x_train_scaled, y_train)

sgd_pred = sgd.predict(x_test_scaled)
```

The SGD model was trained using the standardized features.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 15. Support Vector Classifier

The third model was Support Vector Classifier using an RBF kernel.

```python
from sklearn.svm import SVC

svc = SVC(
    kernel='rbf',
    random_state=42
)

svc.fit(x_train_scaled, y_train)

svc_pred = svc.predict(x_test_scaled)
```

The SVC model was also trained using the standardized features.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 16. Model Evaluation

I compared the three models using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix

The accuracy results were:

| Model         |   Accuracy |
| ------------- | ---------: |
| Random Forest |   73.77%   |
| SVC           |   61.85%   |
| SGD           |   51.00%   |

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 17. Random Forest Performance

Random Forest achieved the best overall accuracy.

# Accuracy

73.77%

# Classification Report

| Class       | Precision |   Recall | F1-score |
| ----------- | --------: | -------: | -------: |
| Class 0     |   0.79    |     0.59 |   0.67   |
| Class 1     |   0.80    |     0.76 |   0.78   |
| Class 2     |   0.68    |     0.78 |   0.73   |
| Overall     |   0.74    |     0.74 |   0.74   |

The macro-average F1-score was approximately 0.73.

Random Forest performed relatively well across all three classes compared with the other models.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 18. SGD Performance

SGD achieved an accuracy of: 51.00%

Its classification results were:

| Class       | Precision |   Recall | F1-score |
| ----------- | --------: | -------: | -------: |
| Class 0     |      0.44 |     0.65 |     0.52 |
| Class 1     |      0.58 |     0.63 |     0.60 |
| Class 2     |      0.49 |     0.35 |     0.41 |
| Overall     |      0.51 |     0.51 |     0.50 |

The macro-average F1-score was approximately 0.51.

Compared with Random Forest and SVC, SGD had the weakest overall performance.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 19. SVC Performance

SVC achieved an accuracy of: 61.85%

Its classification results were:

| Class       | Precision |   Recall | F1-score |
| ----------- | --------: | -------: | -------: |
| Class 0     |      0.66 |     0.35 |     0.46 |
| Class 1     |      0.69 |     0.68 |     0.69 |
| Class 2     |      0.56 |     0.69 |     0.62 |
| Overall     |      0.62 |     0.62 |     0.61 |

The macro-average F1-score was approximately 0.59.

SVC performed better than SGD but lower than Random Forest.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 20. Confusion Matrix

Confusion matrices were used to understand the classification errors made by each model.

A confusion matrix helps identify:

* Correct predictions for each class.
* Classes that were confused with each other.
* Performance differences between minority and majority classes.

This was particularly useful because the quality classes were not equally represented in the dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 21. Model Comparison

The three models can be compared as follows:

| Model             |   Accuracy | Macro F1-score | Overall Performance |
| ----------------- | ---------: | -------------: | ------------------- |
| Random Forest     | 73.77%     |       0.73     | Best                |
| SVC               | 61.85%     |       0.59     | Moderate            |
| SGD               | 51.00%     |       0.51     | Lowest              |

Random Forest clearly performed better than the other two models in this experiment.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 22. Best Model

Based on the results, Random Forest was the best-performing model.

It achieved:

Accuracy = 73.77%

and an overall weighted F1-score of approximately: 0.74

It also provided more balanced results across the three classes compared with SGD and SVC.

Therefore, Random Forest was selected as the preferred model for this task.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 23. Key Findings

The main findings from this project were:

1. The dataset contains 6,497 wine observations and 13 original variables

2. The dataset contains chemical measurements such as acidity, sugar, chlorides, sulfur dioxide, density, pH, sulphates, and alcohol.

3. There were some missing values in the chemical variables, while the quality target did not contain missing values.

4. Quality score 6 was the most common quality rating, with 2,836 observations.

5. Quality scores 3 and 9 were very rare, with only 30 and 5 observations respectively.

6. The original quality score was converted into three categories: Low, Medium, and High.

7. Random Forest achieved the highest accuracy at 73.77%.

8. SVC achieved 61.85%, while SGD achieved 51.00%.

9. Random Forest also achieved the strongest overall F1 performance among the three models.

10. The results suggest that Random Forest was better suited to this dataset than the other two models used in this experiment.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#  What I Learned

This task helped me understand how classification algorithms can be applied to a real-world dataset.

During this project, I learned how to:

* Explore a dataset before modelling.
* Check missing values.
* Analyze class distributions.
* Convert a numerical target into meaningful categories.
* Encode categorical variables.
* Use stratified train-test splitting.
* Apply feature scaling.
* Train Random Forest, SGD, and SVC models.
* Evaluate classification models using multiple metrics.
* Read classification reports.
* Analyze confusion matrices.
* Compare different machine learning algorithms.
* Understand the effect of class imbalance on model performance.
* Identify the most suitable model based on evaluation results.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Conclusion

In this project, I used machine learning to classify wines into Low, Medium, and High quality categories based on their chemical properties.

The original dataset contained 6,497 observations and 13 variables. After examining the dataset and creating the three quality categories, I prepared the data for classification using label encoding and an 80/20 stratified train-test split.

I trained three models: Random Forest, SGD Classifier, and SVC. Random Forest achieved the best result with an accuracy of 73.77%, followed by SVC at 61.85% and SGD at 51.00%.

Based on the comparison, Random Forest was selected as the best-performing model for this task. The project also showed that class imbalance and the grouping of quality scores are important factors to consider when building a wine quality classification model.

Overall, this task gave me practical experience in exploratory data analysis, classification, feature preprocessing, model comparison, evaluation metrics, and machine learning using Python.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Author

Krishna Deshmukh

M.Sc. Statistics

Skills: Python | SQL | Excel | Power BI | Machine Learning | Data Analysis

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Internship

This task is a part of Data Analysis Internship.

Project: Wine Quality Classification Using Machine Learning
