# House Price Prediction Using Linear Regression

# Internship Task 5 – House Price Prediction

This project was completed as the fifth task of my data analytics internship.

The main aim of this task was to build a machine learning model that can predict house prices based on different property-related features.

I used a house price dataset containing information such as number of bedrooms, bathrooms, living area, lot area, house condition, grade, location, renovation details, and other property characteristics.

For this project, I performed exploratory data analysis, checked the dataset, studied relationships between variables, trained a Multiple Linear Regression model, and evaluated its performance using MSE, RMSE, and R².

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Project Objective

The main objectives of this project were:

* Load and understand the house price dataset.
* Perform basic exploratory data analysis.
* Check for missing values.
* Calculate descriptive statistics.
* Understand the distribution of house prices.
* Study the relationship between different features and price.
* Analyze correlations between variables.
* Select the features required for prediction.
* Split the dataset into training and testing data.
* Train a Linear Regression model.
* Evaluate the model using MSE, RMSE, and R².
* Compare actual prices with predicted prices.
* Analyze residuals.
* Study the effect of individual features using regression coefficients.
* Understand the limitations of the model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dataset

The dataset contains information about residential properties and their selling prices.

The dataset used in the notebook contains:

* 14,620 observations
* 23 columns

The dataset contains features related to the size, condition, location, age, renovation, and other characteristics of the houses.

Some of the important variables are:

| Feature                               | Description                                |
| ------------------------------------- | ------------------------------------------ |
| id                                    | Unique property identifier                 |
| Date                                  | Date associated with the property sale     |
| number of bedrooms                    | Number of bedrooms                         |
| number of bathrooms                   | Number of bathrooms                        |
| living area                           | Living area of the house                   |
| lot area                              | Area of the land                           |
| number of floors                      | Number of floors                           |
| waterfront present                    | Whether the property has waterfront access |
| number of views                       | Number of views                            |
| condition of the house                | Condition rating of the property           |
| grade of the house                    | Grade/rating of the house                  |
| Area of the house(excluding basement) | Above-ground house area                    |
| Area of the basement                  | Basement area                              |
| Built Year                            | Year the house was built                   |
| Renovation Year                       | Year of renovation                         |
| Postal Code                           | Postal code                                |
| Lattitude                             | Latitude of the property                   |
| Longitude                             | Longitude of the property                  |
| living_area_renov                     | Renovated living area                      |
| lot_area_renov                        | Renovated lot area                         |
| Number of schools nearby              | Number of nearby schools                   |
| Distance from the airport             | Distance from the airport                  |
| Price                                 | Target variable                            |

The target variable for this project is Price.

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
* Multiple Linear Regression

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. Importing Libraries

I started the project by importing the Python libraries required for data analysis and visualization.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Pandas was used for working with the dataset, NumPy for numerical operations, and Matplotlib and Seaborn for visualization.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 2. Loading the Dataset

The dataset was loaded using Pandas.

```python
df = pd.read_csv("MY_HOUSE_PROJECT.csv")
```

I then checked the first few records using:

```python
df.head()
```

# The dataset contained 14,620 rows and 23 columns.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3. Checking the Dataset Information

I used `df.info()` to understand the structure and data types of the dataset.

```python
df.info()
```

The dataset contained:

* 19 integer columns
* 4 floating-point columns
* 14,620 observations

There were no missing values according to the `info()` output.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 4. Checking Missing Values

I checked the dataset for null values using:

```python
df.isna().sum()
```

The result showed zero missing values across all 23 columns.

Therefore, no missing-value treatment was required before building the regression model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 5. Descriptive Statistics

I used the `describe()` function to understand the numerical characteristics of the dataset.

```python
df.describe()
```

Some important statistics from the analysis were:

| Variable            |      Mean | Minimum |   Maximum |
| ------------------- | --------: | ------: | --------: |
| Number of bedrooms  |      3.38 |       1 |        33 |
| Number of bathrooms |      2.13 |     0.5 |         8 |
| Living area         |   2098.26 |     370 |    13,540 |
| Lot area            | 15,093.28 |     520 | 1,074,218 |
| Number of floors    |      1.50 |       1 |       3.5 |
| House Price         |   538,932 |  78,000 | 7,700,000 |

The average house price in the dataset was approximately 538,932.
The descriptive statistics also showed that some variables had very large maximum values compared with their median values, especially lot area and price.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 6. Exploratory Data Analysis

I performed exploratory analysis to understand the distribution of the target variable and the relationships between the features and house price.

The main areas examined were:

* Distribution of house prices
* Feature relationships
* Correlation between variables
* Scatter plots
* Heatmap
* Relationship between important house characteristics and price

The target variable used for prediction was:

```python
y = df['Price']
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 7. Target Variable – Price

The target variable in this project is Price.

The descriptive statistics showed:

* Mean price: approximately 538,932
* Median price: 450,000
* Minimum price: 78,000
* Maximum price: 7,700,000

The difference between the mean and median indicates that the price distribution is not perfectly symmetric and contains some high-priced properties.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 8. Correlation Analysis

Correlation analysis was performed to understand the relationship between the numerical variables.

A correlation matrix helps identify which features have stronger positive or negative relationships with house price.

The correlation analysis was useful for understanding which variables may have a greater influence on the target variable before fitting the regression model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 9. Feature Selection

For the regression model, I separated the target variable from the input features.

The target variable was:

```python
y = df['Price']
```

The following columns were removed from the predictors:

```python
['Price', 'id', 'Sale_Year', 'Sale_Month', 'Sale_Day']
```

The remaining variables were used as predictor variables.

```python
x = df.drop(
    columns=['Price', 'id', 'Sale_Year', 'Sale_Month', 'Sale_Day']
)
```

This prevented the target variable from being included as an input feature and also excluded the ID and date-derived variables from the regression model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 10. Train-Test Split

The data was divided into training and testing sets.

I used an 80/20 split.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=43
)
```

This means:

* 80% of the data was used for training.
* 20% of the data was used for testing.

The training data was used to fit the regression model, while the test data was used to evaluate how well the model performed on unseen observations.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 11. Building the Linear Regression Model

I used Linear Regression from Scikit-learn.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

The model was fitted using the training data.

```python
model.fit(x_train, y_train)
```

# The model estimates house price based on the selected property features.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 12. Making Predictions

After training the model, predictions were generated for the test dataset.

```python
y_pred = model.predict(x_test)
```

The predicted values were then compared with the actual house prices.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 13. Model Evaluation

I evaluated the model using three regression metrics:

# Mean Squared Error – MSE

MSE measures the average squared difference between actual and predicted values.

The obtained MSE was:

36,979,926,385.36

```text
MSE = 36,979,926,385.36
```

---

# Root Mean Squared Error – RMSE

RMSE is the square root of MSE.

The obtained RMSE was: 192,301.65

```text
RMSE = 192,301.65
```

RMSE is expressed in the same unit as the target variable, so it gives an approximate idea of the typical prediction error in terms of house price.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# R² Score

The R² score measures the proportion of variation in the target variable explained by the regression model.

The obtained R² score was: 0.7052

or approximately: 70.52%

This means that the model explains around 70.52% of the variation in house prices in the test data.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 14. Actual Price vs Predicted Price

I created a scatter plot comparing the actual house prices with the prices predicted by the model.

The purpose of this plot was to check how closely the predictions follow the actual values.

Ideally, the points should be close to a diagonal line representing:

```text
Actual Price = Predicted Price
```

A closer concentration around the diagonal indicates better prediction performance.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 15. Residual Analysis

I also analyzed the residuals of the regression model.

A residual is calculated as:

```text
Residual = Actual Price − Predicted Price
```

The residual plot was used to check whether the errors were randomly distributed.

A good regression model should ideally have residuals scattered around zero without a strong systematic pattern.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 16. Feature Coefficient Analysis

The regression coefficients were analyzed to understand how each feature affects the predicted house price while keeping the other variables constant.

The coefficients obtained from the model included:

| Feature                          | Coefficient |
| -------------------------------- | ----------: |
| Waterfront present               |  578,037.99 |
| Lattitude                        |  556,007.26 |
| Grade of the house               |   99,387.90 |
| Number of views                  |   50,243.49 |
| Number of bathrooms              |   38,753.97 |
| Condition of the house           |   32,305.13 |
| Number of floors                 |    3,081.30 |
| Number of schools nearby         |    1,699.79 |
| Postal Code                      |      327.16 |
| Living area                      |      114.88 |
| Area of house excluding basement |       77.31 |
| Area of basement                 |       37.58 |
| Renovation Year                  |       25.29 |
| Living area renovation           |       13.40 |
| Lot area                         |     -0.0038 |
| Lot area renovation              |       -0.39 |
| Distance from airport            |      -23.13 |
| Built Year                       |   -2,490.29 |
| Number of bedrooms               |  -34,618.38 |
| Longitude                        |  -97,364.43 |

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 17. Important Coefficient Observations

The largest positive coefficients in the model were associated with:

* Waterfront presence
* Latitude
* Grade of the house
* Number of views
* Number of bathrooms

Among these, waterfront presence had the largest positive coefficient in the fitted model.

Some variables had negative coefficients, including:

* Longitude
* Number of bedrooms
* Built Year
* Distance from the airport
* Lot area

These coefficients should be interpreted carefully because a regression coefficient represents the relationship with price after considering the other variables included in the model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 18. Main Results

The final Linear Regression model produced the following results:

| Metric        |            Result |
| ------------- | ----------------: |
| MSE           | 36,979,926,385.36 |
| RMSE          |        192,301.65 |
| R²            |            0.7052 |
| R² Percentage |            70.52% |

The R² value indicates that the model explains approximately 70.52% of the variation in house prices.

The RMSE of approximately 192,302 means that the typical prediction error is around 192,302 price units, based on the RMSE metric.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 19. Key Findings

The main findings from this project were:

1. The dataset contains 14,620 observations and 23 variables.

2. There were no missing values in the dataset.

3. The average house price was approximately 538,932, while the median was 450,000.

4. The house price variable had a wide range, from 78,000 to 7,700,000.

5. The Linear Regression model achieved an R² score of 0.7052.

6. The model's RMSE was approximately 192,301.65.

7. Waterfront presence had the largest positive coefficient among the features in the fitted model.

8. Grade of the house, number of views, and number of bathrooms also had relatively large positive coefficients.

9. Some features had negative coefficients, showing that their relationship with price was negative after accounting for the other predictors.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# What I Learned

This task helped me understand how regression can be used to solve a real-world prediction problem.

During this project, I learned how to:

* Explore a house price dataset.
* Check data quality and missing values.
* Calculate descriptive statistics.
* Analyze relationships between variables.
* Study correlations.
* Select features for a regression model.
* Split data into training and testing sets.
* Build a Linear Regression model using Scikit-learn.
* Make predictions on unseen data.
* Calculate MSE, RMSE, and R².
* Interpret regression coefficients.
* Analyze actual versus predicted values.
* Use residual plots to evaluate model behavior.
* Understand the limitations of a basic linear regression model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Conclusion

In this project, I built a Multiple Linear Regression model to predict house prices using different property-related features.

The dataset contained 14,620 observations and 23 columns. After performing exploratory analysis and checking the data, I selected the required features and divided the data into training and testing sets using an 80/20 split.

The Linear Regression model achieved an R² score of 0.7052, meaning that it explained approximately 70.52% of the variation in house prices in the test dataset. The RMSE was 192,301.65, which represents the typical size of prediction error in the target variable's units.

The coefficient analysis showed that features such as waterfront presence, latitude, house grade, number of views, and number of bathrooms had relatively large positive coefficients.

Overall, this task gave me practical experience with exploratory data analysis, regression modelling, model evaluation, prediction, residual analysis, and coefficient interpretation using Python.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Author

Krishna Deshmukh

M.Sc. Statistics

Skills: Python | SQL | Excel | Power BI | Machine Learning | Data Analysis

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Internship

Internship Task: Task 5 – House Price Prediction

Project: House Price Prediction Using Linear Regression

Status: Completed
