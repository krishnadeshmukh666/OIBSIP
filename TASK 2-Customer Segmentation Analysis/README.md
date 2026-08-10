# Customer Segmentation Using RFM Analysis and K-Means

# Internship Task 2 – Customer Segmentation

This project was completed as the second task of my data analytics internship.

The main objective of this task was to analyze customer purchasing behavior and divide customers into different groups based on their buying patterns. I used RFM analysis to calculate customer-level behavioral features and then applied the K-Means clustering algorithm to identify customer segments.

The analysis was performed using Python and the Online Retail dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Project Objective

The main objectives of this project were:

* Understand customer purchasing behavior.
* Clean and prepare the retail transaction data.
* Calculate important customer-level metrics.
* Perform RFM (Recency, Frequency, Monetary) analysis.
* Select behavioral features for customer segmentation.
* Use K-Means clustering to divide customers into groups.
* Determine the appropriate number of clusters using the Elbow Method.
* Visualize the resulting customer segments.
* Suggest suitable marketing strategies for each customer group.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dataset

The project uses the Online Retail dataset, which contains transaction-level information about customer purchases.

The important variables used in this analysis include:

| Column      | Description                           |
| ----------- | ------------------------------------- |
| InvoiceNo   | Unique invoice/transaction number     |
| StockCode   | Product identification code           |
| Description | Description of the product            |
| Quantity    | Number of products purchased          |
| InvoiceDate | Date and time of the transaction      |
| UnitPrice   | Price per unit                        |
| CustomerID  | Unique customer identification number |
| Country     | Customer's country                    |

The dataset was provided in Excel format and was loaded into Python using Pandas.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Tools and Technologies

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* K-Means Clustering
* StandardScaler

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. Importing Libraries

I first imported the libraries required for data manipulation, visualization, scaling, and clustering.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 2. Loading the Dataset

The Online Retail dataset was loaded using Pandas.

```python
df = pd.read_excel("Online Retail.xlsx")
```

I then inspected the first and last few records to understand the structure of the dataset.

```python
df.head()
df.tail()
```

I also checked the shape and information of the dataset.

```python
df.shape
df.info()
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3. Checking Missing Values

Before performing the analysis, I checked the dataset for missing values.

```python
df.isna().sum()
```

The `Description` and `CustomerID` columns contained missing values.

For the `Description` column, I filled the missing values using the most frequent description.

```python
mode = df[df["Description"].notnull()]["Description"].mode()[0]

df["Description"] = df["Description"].fillna(mode)
```

For customer-level analysis, `CustomerID` is required. Therefore, rows where CustomerID was missing were removed.

```python
df = df.dropna(subset=['CustomerID'])
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 4. Removing Duplicate Records

Duplicate records were checked and removed.

```python
df = df.drop_duplicates()
```

After removing duplicates, I checked again to make sure that duplicate records were no longer present.

```python
df.duplicated().sum()
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 5. Handling Cancelled Orders

The `InvoiceNo` column contains invoices beginning with the letter C, which represent cancelled orders.

Since the objective of this project is to analyze actual customer purchases, these cancelled transactions were removed.

```python
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
```

This helps prevent cancelled transactions from affecting the customer segmentation results.

---

# 6. Checking Quantity and Unit Price

I checked whether the dataset contained negative or zero values in the `Quantity` and `UnitPrice` columns.

```python
is_all_positive = (df['Quantity'] > 0).all()
is_all_positive = (df['UnitPrice'] > 0).all()
```

Records with non-positive unit prices were removed.

```python
df = df[df['UnitPrice'] > 0]
```

This was done because negative or zero prices would not be appropriate when calculating customer spending.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 7. Converting Invoice Date

The `InvoiceDate` column was converted into datetime format.

```python
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
```

This allowed me to perform date-based calculations required for the Recency metric.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 8. Creating Total Price

To calculate the amount spent by each customer, I created a new variable called `TotalPrice`.

```python
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
```

This represents the total value of each transaction line.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 9. Descriptive Analysis

Before clustering customers, I calculated some basic customer purchasing metrics.

# Average Purchase Value

Average Purchase Value was calculated as:```textTotal Revenue / Number of Transactions```

The calculated average purchase value was approximately: 373.07

This means that the average transaction generated approximately 373.07 in revenue.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Purchase Frequency

Purchase frequency was calculated as: ```text Number of Transactions / Number of Customers```

This provides an estimate of the average number of transactions made by each customer.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Customer Lifetime Value

An estimated customer lifetime value was calculated using:

```text
Average Purchase Value × Purchase Frequency
```

The estimated value obtained from the analysis was approximately: 1893.53

This gives an approximate indication of the revenue generated by an average customer based on the calculated metrics.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 10. RFM Analysis

The main part of this project was customer segmentation using RFM analysis.

RFM stands for:

* Recency
* Frequency
* Monetary

These three metrics help describe customer purchasing behavior.

# Recency

Recency measures how recently a customer made a purchase.

A lower Recency value means that the customer purchased more recently.

# Frequency

Frequency represents the number of unique transactions made by a customer.

A higher Frequency indicates that the customer purchases more often.

# Monetary

Monetary represents the total amount spent by a customer.

A higher Monetary value indicates a customer who contributes more revenue.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 11. Calculating RFM Features

I created a reference date based on the last transaction date in the dataset.

```python
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
```

Then I calculated Recency, Frequency, and Monetary values for each customer.

```python
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']
```

The resulting dataset contains one row for each customer with their RFM values.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 12. Feature Scaling

Since Recency, Frequency, and Monetary have different ranges, I standardized them before applying K-Means.

```python
scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm)
```

The scaled values were then converted back into a DataFrame.

```python
rfm_scaled = pd.DataFrame(
    rfm_scaled,
    columns=rfm.columns,
    index=rfm.index
)
```

Scaling was important because K-Means uses distances to form clusters.

---

# 13. Finding the Number of Clusters

I used the Elbow Method to determine a suitable number of clusters.

I tested values of K from 1 to 10.

```python
wcss = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)
```

The WCSS values were then plotted.

The resulting Elbow plot showed a noticeable bend around K = 3.

Therefore, I selected 3 clusters for the final K-Means model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 14. Applying K-Means Clustering

I created the final K-Means model using three clusters.

```python
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
```

Each customer was assigned to one of the three clusters.

---

# 15. Customer Segments

The three clusters were interpreted based on their average RFM values.

# Cluster 0 – Lost Customers

These customers have relatively high Recency values and lower purchasing activity.

They have not purchased recently and may be at risk of being lost customers.

Possible strategy:

* Send re-engagement offers.
* Provide discounts or cashback.
* Ask for feedback.
* Use email or social media campaigns to bring them back.

# Cluster 1 – Regular Customers

This was the largest customer segment.

These customers show regular purchasing behavior but have potential to become higher-value customers.

Possible strategy:

* Introduce loyalty programs.
* Provide seasonal offers.
* Give exclusive promotions.
* Encourage customers to increase their purchase frequency.

# Cluster 2 – Premium Customers

This was the smallest segment but represented the most valuable customers.

These customers have more recent purchases, higher purchase frequency, and higher monetary value.

Possible strategy:

* Provide VIP membership benefits.
* Give early access to new products.
* Provide personalized recommendations.
* Offer exclusive discounts and rewards.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 16. Cluster Visualization

I created several scatter plots to understand the separation between the customer segments.

# Recency vs Monetary

```python
plt.scatter(
    rfm['Recency'],
    rfm['Monetary'],
    c=rfm['Cluster'],
    cmap='viridis',
    s=50
)
```

The visualization shows that customers with lower Recency and higher Monetary values are associated with the premium segment, while customers with higher Recency and lower spending are associated with the lost customer segment.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Frequency vs Monetary

I also plotted Frequency against Monetary value.

This visualization showed a positive relationship between purchase frequency and total spending. Customers who purchase more frequently generally contribute more revenue.

# Recency vs Frequency

The third visualization compared Recency and Frequency.

This helped identify differences between customers who purchase frequently and customers who have not purchased recently.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 17. Number of Customers in Each Cluster

I also calculated the number of customers belonging to each cluster.

The results were approximately:

| Cluster | Customer Segment  | Customers | Approx. Share |
| ------- | ----------------- | --------: | ------------: |
| 0       | Lost Customers    |     1,082 |           25% |
| 1       | Regular Customers |     3,230 |           74% |
| 2       | Premium Customers |        26 |            1% |

The Regular Customer segment is clearly the largest group, while the Premium Customer segment contains only a small number of customers.

This shows that the business has a relatively small group of highly valuable customers that may require special attention and retention strategies.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Marketing Recommendations

The customer segments can be used to create different marketing strategies rather than treating all customers in the same way.

# Cluster 0 – Lost Customers

* Send reactivation campaigns.
* Offer discounts or cashback.
* Collect feedback.
* Use targeted email and social media campaigns.

# Cluster 1 – Regular Customers

* Introduce loyalty programs.
* Provide seasonal promotions.
* Offer rewards for repeat purchases.
* Encourage customers to increase their spending.

# Cluster 2 – Premium Customers

* Create VIP membership programs.
* Provide exclusive discounts.
* Give early access to new products.
* Provide personalized product recommendations.
* Focus on customer retention.

 --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Key Findings

The main findings from this analysis are:

1. RFM analysis provided a useful way to summarize customer purchasing behavior using Recency, Frequency, and Monetary value.

2. The Elbow Method indicated that three clusters were suitable for this dataset.

3. The largest segment was the Regular Customer group, containing approximately 74% of customers.

4. Approximately 25% of customers were identified as Lost Customers, indicating a potential group for re-engagement campaigns.

5. The Premium Customer segment contained only around 1% of customers, but these customers showed the strongest purchasing behavior.

6. Customers with higher purchase frequency generally generated higher total spending.

7. The segmentation can help a business use different marketing strategies for different types of customers.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# What I Learned

This task helped me understand how customer segmentation can be performed using real transaction data.

During this project, I learned how to:

* Clean transaction-level data.
* Handle missing values and duplicate records.
* Identify cancelled transactions.
* Create new business-related variables.
* Calculate Average Purchase Value and Purchase Frequency.
* Perform RFM analysis.
* Standardize features before clustering.
* Use the Elbow Method to select the number of clusters.
* Apply K-Means clustering.
* Interpret customer segments.
* Convert analytical results into marketing recommendations.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Conclusion

This project demonstrated how customer transaction data can be used to understand different types of customers.

Using RFM analysis and K-Means clustering, I divided customers into three groups: Lost Customers, Regular Customers, and Premium Customers. The results showed that most customers belong to the Regular segment, while a very small group of Premium customers has much stronger purchasing behavior.

The segmentation can help a business focus its marketing efforts according to customer value and purchasing behavior. Lost customers can be targeted with re-engagement campaigns, regular customers can be encouraged to increase their spending, and premium customers can be given personalized rewards and retention benefits.

Overall, this task gave me practical experience in customer analytics, feature engineering, unsupervised machine learning, data visualization, and business interpretation.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Author

Krishna Deshmukh

M.Sc. Statistics

Skills: Python | SQL | Excel | Power BI | Data Analysis | Machine Learning

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Internship

This project was completed as part of my **Data Analytics Internship**.

Project: RFM Analysis and K-Means Customer Segmentation


