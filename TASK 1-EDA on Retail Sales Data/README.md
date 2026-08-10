# Amazon Sales Data Analysis

# Internship Task 1 – Data Analysis

This project was completed as part of my data analytics internship. The main purpose of this task was to work with an Amazon sales dataset and understand the sales data through data cleaning, basic statistical analysis, and visualization.

The analysis was performed using Python in Jupyter Notebook. I started by inspecting the dataset, checking its structure and missing values, cleaning the data, and then performing exploratory and time-based analysis.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Project Objective

The main objectives of this project were:

* Understand the structure and characteristics of the Amazon sales dataset.
* Check the dataset for missing values and duplicate records.
* Clean the data so that it can be used for further analysis.
* Calculate descriptive statistics for important numerical variables.
* Study sales trends over time.
* Analyze available customer-related information such as shipping location and B2B classification.
* Create visualizations to make the findings easier to understand.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dataset Description

The dataset contains Amazon sales/order information from 2022. It includes details related to orders, products, order status, fulfilment, sales channel, quantity, amount, shipping location, promotions, and B2B transactions.

After loading the dataset, I found:

* Rows: 128,975
* Columns: 24
* Sales Channel: Amazon.in
* Currency: INR
* Data period: Mainly April to June 2022

Some of the important columns in the dataset are:

| Column             | Description                                |
| ------------------ | ------------------------------------------ |
| Order ID           | Unique order identifier                    |
| Date               | Date of the order                          |
| Status             | Current status of the order                |
| Fulfilment         | Fulfilment method                          |
| Sales Channel      | Channel through which the order was placed |
| ship-service-level | Shipping service type                      |
| Style              | Product style                              |
| SKU                | Product stock keeping unit                 |
| Category           | Product category                           |
| Size               | Product size                               |
| Qty                | Quantity ordered                           |
| Amount             | Order amount                               |
| ship-city          | Customer shipping city                     |
| ship-state         | Customer shipping state                    |
| ship-postal-code   | Shipping postal code                       |
| promotion-ids      | Promotion used for the order               |
| B2B                | Whether the order was a B2B transaction    |

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Tools and Technologies

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Microsoft Excel

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. Importing Libraries

I used Pandas and NumPy for data manipulation and numerical operations. Matplotlib and Seaborn were used for creating visualizations.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 2. Loading the Dataset

The dataset was provided in Excel format, so I loaded it using Pandas.

```python
df = pd.read_excel("Amazon Sale Report.xlsx")
```

After loading the data, I used `head()`, `tail()`, `columns`, and `shape` to understand the dataset.

The dataset contains 128,975 records and 24 columns.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3. Data Inspection

I checked the column names, data types, number of records, and missing values.

```python
df.shape
df.columns
df.info()
df.isna().sum()
```

The dataset contained missing values in several columns, including:

* Courier Status
* Currency
* Amount
* Ship City
* Ship State
* Ship Postal Code
* Ship Country
* Promotion IDs
* Fulfilled-by
* Unnamed: 22

There were no duplicate rows according to the duplicate check performed in the notebook.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 4. Handling Missing Values

Different approaches were used depending on the column.

For categorical variables such as `Courier Status`, `ship-city`, `ship-state`, `ship-postal-code`, and `promotion-ids`, missing values were replaced using the mode.

For example:

```python
df['Courier Status'] = df['Courier Status'].fillna(
    df[df['Courier Status'].notnull()]['Courier Status'].mode()[0]
)
```

For the `currency` column, missing values were replaced with `INR`.

```python
df['currency'] = df['currency'].fillna('INR')
```

Missing values in `Amount` were replaced using the mean:

```python
df['Amount'] = df['Amount'].fillna(df['Amount'].mean())
```

The shipping country was filled with `IN` where it was missing.

Two columns, `Unnamed: 22` and `fulfilled-by`, were removed because they were not useful for the analysis.

```python
df = df.drop(columns=['Unnamed: 22', 'fulfilled-by'])
```

---

# 5. Descriptive Statistics

I calculated descriptive statistics for the numerical variables `Qty` and `Amount`.

```python
df[["Qty", "Amount"]].describe()
```

Some important values were:

## Quantity

* Mean: 0.904
* Median: 1
* Minimum: 0
* Maximum: 15

## Amount

* Mean: 648.56 INR
* Median: 605 INR
* Minimum: 0 INR
* Maximum: 5,584 INR

The median order quantity is 1, which indicates that most records contain a quantity of one item. The order amount has a wider range, showing variation in the value of different orders.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 6. Time-Based Analysis

To study sales over time, I extracted the day, month, and year from the `Date` column.

```python
df['Day'] = df["Date"].dt.day
df['Month'] = df["Date"].dt.strftime('%b')
df['Year'] = df["Date"].dt.year
```

I then grouped the data by month and calculated total sales.

```python
Monthly_sales = df.groupby('Month')['Amount'].sum()
```

The monthly sales obtained from the analysis were approximately:

| Month |   Total Sales |
| ----- | ------------: |
| March |       107,521 |
| April | 30.78 Million |
| May   | 27.85 Million |
| June  | 24.91 Million |

April recorded the highest sales among the months available in the dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 7. Quarterly Sales Analysis

I also created a quarterly sales trend using a line chart.

The analysis showed that most of the sales were recorded during Q2 (April–June 2022), while Q1 had very little representation.

This is mainly because the dataset contains transactions primarily from April to June 2022. Therefore, monthly analysis gives more useful information for this particular dataset than quarterly comparison.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 8. Customer Demographic Analysis

The original task included analysis of customer age groups and gender.

However, the dataset does not contain columns for customer age or gender. Therefore, these two variables could not be analyzed directly.

Instead, customer-related information was considered using variables that were actually available in the dataset, such as:

* Shipping city
* Shipping state
* Product category
* B2B classification

I did not create age or gender analysis because those variables were not present in the dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Key Findings

Based on the analysis, some observations were:

1. The dataset contains 128,975 Amazon order records across 24 columns.

2. The majority of the data is related to transactions from April to June 2022.

3. April recorded the highest total sales among the months analyzed.

4. The median order quantity is 1, suggesting that most order records contain one item.

5. The average order amount after handling missing values is around 648.56 rs.

6. Several columns contained missing values, so data cleaning was an important part of the analysis.

7. Customer age and gender could not be analyzed because those attributes were not available in the dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# What I Learned

Through this task, I got practical experience in working with a real-world sales dataset.

I learned how to:

* Load Excel data using Pandas.
* Inspect a dataset before starting analysis.
* Identify and handle missing values.
* Check for duplicate records.
* Calculate descriptive statistics.
* Work with date and time information.
* Group data to calculate monthly and quarterly sales.
* Create charts using Matplotlib.
* Interpret results instead of only generating visualizations.
* Identify limitations in a dataset before drawing conclusions.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Conclusion

This project helped me understand the complete basic workflow of a data analysis project, starting from data loading and inspection and moving toward cleaning, statistical analysis, visualization, and interpretation.

The analysis provided useful information about sales amounts, order quantities, and monthly sales trends. At the same time, it showed the importance of understanding the available data and its limitations before making conclusions.

This was my first internship task and gave me practical experience with Python-based data analysis and working with an actual sales dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Author

Krishna Deshmukh

M.Sc.(Statistics)

Skills: Python | SQL | Excel | Power BI | Data Analysis

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Internship

This project was completed as part of my **Data Analytics Internship**.

Task 1 – Amazon Sales Data Analysis
