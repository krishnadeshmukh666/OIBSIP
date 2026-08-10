# Cafe Sales Data Cleaning and Quality Analysis

# Internship Task 3 – Data Cleaning

This project was completed as the third task of my data analytics internship.

The main purpose of this task was to clean a raw cafe sales dataset and prepare it for further analysis. The dataset contained missing values, inconsistent entries such as `ERROR` and `UNKNOWN`, incorrect data types, and some potential outliers.

I used Python and Pandas to inspect the data, identify data quality problems, handle the issues, and save a cleaned version of the dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Project Objective

The main objectives of this task were:

* Understand the structure of the raw café sales dataset.
* Check the dataset for missing values.
* Identify duplicate records.
* Find inconsistent values such as `ERROR` and `UNKNOWN`.
* Handle missing values using suitable methods.
* Correct the data types of different columns.
* Detect outliers using the IQR method.
* Compare the dataset before and after cleaning.
* Save the cleaned dataset as a new CSV file.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dataset Description

The dataset contains café transaction and sales information.

It includes 10,000 records and 8 columns before cleaning.

The columns are:

| Column           | Description                            |
| ---------------- | -------------------------------------- |
| Transaction ID   | Unique identifier for each transaction |
| Item             | Name of the café item purchased        |
| Quantity         | Number of items purchased              |
| Price Per Unit   | Price of one unit of the item          |
| Total Spent      | Total amount spent on the transaction  |
| Payment Method   | Method used for payment                |
| Location         | Location of the transaction            |
| Transaction Date | Date of the transaction                |

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Tools and Technologies

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* CSV

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. Importing Libraries

I started by importing the libraries required for data cleaning and basic visualization.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

%matplotlib inline
```

Pandas was mainly used for data manipulation and cleaning, while NumPy was used for handling missing and numerical values. Matplotlib was used for the boxplots during outlier analysis.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 2. Loading the Dataset

The raw café sales dataset was loaded using Pandas.

```python
df = pd.read_csv("dirty_cafe_sales.csv")
```

I created a copy of the original dataset before making any changes.

```python
df_before = df.copy()
```

This allowed me to compare the dataset before and after cleaning.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3. Initial Dataset Inspection

I checked the first and last few records to understand how the data was structured.

```python
df.head()
df.tail()
```

I also checked the shape of the dataset.

```python
df.shape
```

The original dataset contained:

* Rows: 10,000
* Columns: 8

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 4. Checking Duplicate Records

Duplicate rows were checked using Pandas.

```python
df.duplicated().sum()
```

The result was:0 duplicate rows

Therefore, no duplicate records needed to be removed from the dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 5. Checking Data Types

I used `info()` to check the data types and number of non-null values.

```python
df.info()
```

Initially, all eight columns were stored as string/object-type values.

This was not appropriate for columns such as:

* Quantity
* Price Per Unit
* Total Spent
* Transaction Date

These columns required data type correction before further analysis.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 6. Checking Missing Values

I checked the number of missing values in each column.

```python
df.isna().sum()
```

The original missing values were:

| Column           | Missing Values |
| ---------------- | -------------: |
| Transaction ID   |              0 |
| Item             |            333 |
| Quantity         |            138 |
| Price Per Unit   |            179 |
| Total Spent      |            173 |
| Payment Method   |          2,579 |
| Location         |          3,265 |
| Transaction Date |            159 |

The dataset contained a total of 6,826 missing values before cleaning.

The largest number of missing values was found in:

* Location
* Payment Method

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 7. Handling ERROR and UNKNOWN Values

The dataset contained some entries such as:

```text
ERROR
UNKNOWN
```

These values cannot be treated as valid numerical or categorical observations in their original form.

I replaced them with `NaN` so that they could be handled consistently during the cleaning process.

```python
df[col] = df[col].replace(
    ['ERROR', 'UNKNOWN'],
    np.nan
)
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 8. Handling Missing Numerical Values

The numerical columns were:

```python
numeric_cols = [
    'Quantity',
    'Price Per Unit',
    'Total Spent'
]
```

The values were first converted to numeric format.

```python
df[col] = pd.to_numeric(
    df[col],
    errors='coerce'
)
```

Missing values in these numerical columns were then replaced using the column mean.

```python
df[col] = df[col].fillna(
    df[col].mean()
)
```

I used mean imputation because these columns represent numerical sales-related measurements.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 9. Handling Missing Categorical Values

The categorical columns were:

```python
categorical_cols = [
    'Transaction ID',
    'Item',
    'Payment Method',
    'Location'
]
```

Missing values in these columns were filled using the mode.

```python
df[col] = df[col].fillna(
    df[col].mode()[0]
)
```

The mode represents the most frequently occurring value in the column.

This was used for categorical variables where calculating a mean is not meaningful.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 10. Handling Missing Date Values

The transaction date was treated separately.

Rows that still contained missing values in fields not handled by the numerical or categorical strategies were removed.

After the cleaning process, the dataset contained 9,540 rows.

Therefore:

Rows removed = 10,000 − 9,540 = 460

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 11. Correcting the Transaction Date

The `Transaction Date` column was originally stored as a string.

I converted it into datetime format using:

```python
df['Transaction Date'] = pd.to_datetime(
    df['Transaction Date'],
    format='%m/%d/%Y',
    errors='coerce'
)
```

After conversion, the column had the correct datetime data type.

This makes the column suitable for future time-based analysis.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 12. Correcting Numerical Data Types

The numerical columns were converted into integer format.

```python
df[numeric_cols] = df[numeric_cols].astype(int)
```

After cleaning, the dataset contained:

* 3 integer columns
* 4 string columns
* 1 datetime column

The final data types were therefore more appropriate for further analysis.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 13. Outlier Detection

I used the Interquartile Range (IQR) method to identify possible outliers in the numerical columns.

The IQR is calculated as:

```text
IQR = Q3 - Q1
```

The lower and upper limits were calculated using:

```text
Lower Limit = Q1 - 1.5 × IQR

Upper Limit = Q3 + 1.5 × IQR
```

I applied this method to:

* Quantity
* Price Per Unit
* Total Spent

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 14. Outlier Results

The IQR analysis produced the following results:

| Column         | IQR | Lower Limit | Upper Limit | Outliers |
| -------------- | --: | ----------: | ----------: | -------: |
| Quantity       |   2 |          -1 |           7 |        0 |
| Price Per Unit |   2 |          -1 |           7 |        0 |
| Total Spent    |   8 |          -8 |          24 |      250 |

The `Total Spent` column had 250 observations outside the calculated IQR limits.

Instead of deleting these observations, I used clipping to limit the values to the calculated lower and upper boundaries.

```python
df[col] = df[col].clip(
    lower,
    upper
)
```

This approach keeps the records while reducing the effect of extreme values.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 15. Visualizing Outliers

I created boxplots for the numerical variables to visually inspect their distributions.

```python
for col in numeric_cols:
    plt.figure(figsize=(5,4))
    plt.boxplot(
        df[col],
        patch_artist=True
    )
    plt.title(col)
    plt.ylabel(col)
    plt.show()
```

The boxplots were used to visually check the numerical columns after the outlier treatment.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 16. Before vs After Cleaning

To understand the effect of the cleaning process, I created a summary table comparing the dataset before and after cleaning.

The comparison included:

* Row count
* Duplicate rows
* Total missing values
* Data types

# Before Cleaning

* Rows: 10,000
* Duplicate rows: 0
* Total missing values: 6,826

# After Cleaning

* Rows: 9,540
* Duplicate rows: 0
* Total missing values: 0

This shows that the cleaning process removed the missing-value problem while retaining most of the original records.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Data Quality Summary

| Metric               | Before Cleaning | After Cleaning |
| -------------------- | --------------: | -------------: |
| Row Count            |          10,000 |          9,540 |
| Duplicate Rows       |               0 |              0 |
| Total Missing Values |           6,826 |              0 |
| Columns              |               8 |              8 |
| Transaction Date     |          String |       Datetime |
| Numerical Variables  |          String |        Integer |

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 17. Saving the Cleaned Dataset

After completing the cleaning process, I saved the final dataset as a new CSV file.

```python
df.to_csv(
    "Cleaned_Cafe_Sales.csv",
    index=False
)
```

The cleaned dataset was saved as:

```text
Cleaned_Cafe_Sales.csv
```

This file can now be used for further exploratory analysis or visualization.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Key Findings

The main observations from the data cleaning process were:

1. The original dataset contained 10,000 records and 8 columns.

2. There were no duplicate records in the dataset.

3. The dataset contained 6,826 missing values before cleaning.

4. `Location` and `Payment Method` had the highest number of missing values.

5. The dataset contained inconsistent values such as `ERROR` and `UNKNOWN`, which were converted to missing values before treatment.

6. The `Transaction Date` column was converted from string to datetime format.

7. `Quantity`, `Price Per Unit`, and `Total Spent` were converted into numerical data types.

8. The IQR method identified 250 potential outliers in Total Spent.

9. Instead of deleting the outliers, I capped them using the IQR boundaries.

10. After cleaning, there were 9,540 records and no missing values.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# What I Learned

This task helped me understand that data cleaning is an important step before performing any analysis or machine learning.

During this project, I learned how to:

* Inspect a raw dataset.
* Generate a basic data quality report.
* Check for duplicate records.
* Identify missing values.
* Handle inconsistent entries.
* Choose different missing-value strategies for different types of variables.
* Convert columns into appropriate data types.
* Work with datetime variables.
* Detect outliers using the IQR method.
* Use clipping to control extreme values.
* Compare data quality before and after cleaning.
* Export a cleaned dataset for further use.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Conclusion

This project focused on preparing a raw café sales dataset for further analysis.

The dataset initially contained 10,000 records with several missing values and inconsistent entries. After cleaning, the dataset contained 9,540 records with no missing values or duplicate records. The numerical and date columns were also converted into more appropriate data types.

Outlier analysis using the IQR method identified potential extreme values in the `Total Spent` column, which were handled using value clipping.

Overall, this task gave me practical experience in data cleaning, missing-value handling, data type conversion, outlier treatment, and data quality checking using Python.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Author

Krishna Deshmukh

M.Sc. Statistics

Skills: Python | SQL | Excel | Power BI | Data Analysis

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Internship

This project was completed as part of my Data Analytics Internship.

Project: Café Sales Data Cleaning and Quality Analysis
