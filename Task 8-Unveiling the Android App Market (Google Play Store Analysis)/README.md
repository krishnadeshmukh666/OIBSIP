# Google Play Store Apps Analysis and Sentiment Analysis

# Internship Task 8

This project was completed as the eighth task of my data analytics internship.

The objective of this task was to analyze Google Play Store applications and understand different factors related to app categories, ratings, installations, pricing, app size, and user reviews.

I worked with two datasets:

* Google Play Store Apps dataset
* Google Play Store User Reviews dataset

The analysis included data cleaning, exploratory data analysis, pricing analysis, estimated revenue calculation, sentiment analysis of user reviews, category-wise sentiment analysis, and interactive visualizations using Plotly.

The final aim was to identify useful insights that could help a developer make better decisions when planning and launching a new application.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Project Objectives

The main objectives of this project were:

* Load and understand the Google Play Store datasets.
* Inspect the structure and data types of both datasets.
* Identify missing values and duplicate records.
* Clean columns containing incorrectly formatted values.
* Analyze the distribution of apps across categories.
* Analyze app ratings.
* Compare average ratings across categories.
* Study the relationship between app size and number of installs.
* Analyze free and paid applications.
* Study the pricing of paid applications.
* Estimate revenue by category.
* Perform sentiment analysis on user reviews.
* Classify reviews as Positive, Negative, or Neutral.
* Compare sentiment across different app categories.
* Create interactive visualizations using Plotly.
* Generate data-driven recommendations for a developer planning to launch a new app.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Datasets

Two datasets were used in this project.

# 1. Google Play Store Apps Dataset

This dataset contains information about applications available on the Google Play Store.

Important columns include:

| Column         | Description                 |
| -------------- | --------------------------- |
| App            | Application name            |
| Category       | App category                |
| Rating         | User rating                 |
| Reviews        | Number of reviews           |
| Size           | Application size            |
| Installs       | Number of installations     |
| Type           | Free or Paid                |
| Price          | Application price           |
| Content Rating | Target audience             |
| Genres         | App genre                   |
| Last Updated   | Last update date            |
| Current Ver    | Current application version |
| Android Ver    | Required Android version    |

# 2. Google Play Store User Reviews Dataset

The second dataset contains user reviews and sentiment-related information.

Important columns include:

| Column                 | Description                    |
| ---------------------- | ------------------------------ |
| App                    | Application name               |
| Translated_Review      | User review text               |
| Sentiment              | Existing sentiment information |
| Sentiment_Polarity     | Polarity score                 |
| Sentiment_Subjectivity | Subjectivity score             |

The two datasets were later joined using the `App` column for category-level sentiment analysis.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Tools and Technologies

The project was completed using:

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* TextBlob
* Plotly

Main techniques used:

* Data cleaning
* Missing-value treatment
* Duplicate removal
* Exploratory Data Analysis
* Correlation analysis
* Sentiment analysis
* Data visualization
* Interactive visualization

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. Importing Libraries

I started by importing the Python libraries required for the analysis.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Plotly was imported later for interactive visualizations.

```python
import plotly.express as px
```

TextBlob was used for sentiment analysis.

```python
from textblob import TextBlob
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 2. Loading the Datasets

The two datasets were loaded separately.

```python
app = pd.read_csv("googleplaystore.csv")

reviews = pd.read_csv(
    "googleplaystore_user_reviews.csv"
)
```

I then used `head()` to view the first few records of both datasets and understand their structure.

The app dataset was used for application-level analysis, while the reviews dataset was used for sentiment analysis.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3. Understanding the Data

I checked:

* Dataset shape
* Column names
* Data types
* Missing values
* Duplicate records

For example:

```python
print(app.shape)
print(reviews.shape)
```

and:

```python
print(app.columns)
print(reviews.columns)
```

I also used:

```python
app.info()
reviews.info()
```

to understand the data types and number of non-null values.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 4. Checking Missing Values

Missing values were checked using:

```python
app.isna().sum()
reviews.isna().sum()
```

The missing values were handled during the data-cleaning stage.

For numerical columns such as:

* Rating
* Reviews
* Size
* Installs
* Price

I used the median to fill missing values.

```python
num_cols = [
    "Rating",
    "Reviews",
    "Size",
    "Installs",
    "Price"
]

for col in num_cols:
    app[col] = app[col].fillna(
        app[col].median()
    )
```

For categorical columns, I used the mode.

```python
cat_cols = [
    "Type",
    "Content Rating",
    "Current Ver",
    "Android Ver",
    "Category",
    "Genres",
    "Last Updated"
]

for col in cat_cols:
    app[col] = app[col].fillna(
        app[col].mode()[0]
    )
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 5. Checking Duplicate Records

I checked duplicate records in both datasets.

```python
app.duplicated().sum()

reviews.duplicated().sum()
```

Duplicate records were considered during the cleaning process so that repeated observations would not unnecessarily affect the analysis.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 6. Cleaning the Installs Column

The `Installs` column originally contained values such as:

```text
10,000+
50,000+
1,000,000+
```

These values were stored as strings.

I removed the comma and `+` symbol and converted the column into a numerical format.

```python
app["Installs"] = (
    app["Installs"]
    .str.replace(",", "", regex=False)
    .str.replace("+", "", regex=False)
)

app["Installs"] = pd.to_numeric(
    app["Installs"],
    errors="coerce"
)
```

This allowed me to perform numerical calculations and comparisons using installation counts.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 7. Cleaning the Price Column

The `Price` column contained dollar signs for paid applications.

I removed the `$` symbol and converted the column into a numerical variable.

```python
app["Price"] = (
    app["Price"]
    .str.replace("$", "", regex=False)
)

app["Price"] = pd.to_numeric(
    app["Price"],
    errors="coerce"
)
```

This made it possible to calculate prices and estimate revenue.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 8. Cleaning the Reviews Column

The `Reviews` column was also converted into a numerical format.

```python
app["Reviews"] = pd.to_numeric(
    app["Reviews"],
    errors="coerce"
)
```

This was necessary because the number of reviews is used as a quantitative variable during analysis.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 9. App Category Analysis

I analyzed the number of applications available in each category.

```python
Category = app["Category"].value_counts()
```

I then created a bar chart to visualize the distribution.

The analysis showed that some categories contain significantly more applications than others.

The largest categories in the analysis were:

1. Family
2. Game
3. Tools

The Family category contained 1,943 apps, followed by Game with 1,121 apps and Tools with 843 apps.

This indicates that these categories are relatively saturated and may have higher competition for new applications.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 10. App Ratings Analysis

I analyzed the distribution of application ratings using a histogram.

```python
plt.hist(
    app["Rating"],
    bins=20,
    edgecolor="black"
)
```

I also calculated the average rating for each category.

```python
avg_rating = (
    app.groupby("Category")["Rating"]
    .mean()
    .sort_values(ascending=False)
)
```

This allowed me to compare the overall user rating across different app categories.

Ratings provide an indication of user satisfaction, although rating alone does not explain the complete user experience.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 11. App Size vs. Number of Installs

I investigated whether larger applications receive more installations.

A scatter plot was created using:

```python
plt.scatter(
    app["Size"],
    app["Installs"],
    alpha=0.4
)
```

I also calculated the correlation between app size and installs.

```python
correlation = app["Size"].corr(
    app["Installs"]
)
```

The scatter plot showed a wide spread of installation counts across different app sizes.

The analysis did not show a strong linear relationship between app size and number of installs.

This suggests that app size alone is not a reliable indicator of popularity. Other factors such as ratings, category, features, user demand, and marketing can have a stronger influence on installations.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 12. Free vs Paid Applications

I analyzed the distribution of free and paid applications.

The analysis showed that the Google Play Store dataset is dominated by free applications, while paid apps represent a much smaller portion.

This suggests that developers commonly use free downloads to attract a larger number of users.

Revenue can then potentially come from:

* Advertisements
* In-app purchases
* Subscriptions
* Other monetization methods

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 13. Paid App Price Analysis

For paid applications, I analyzed the distribution of prices.

The price distribution was positively skewed, with most paid applications concentrated toward the lower price range.

Only a smaller number of applications were priced at higher levels.

This suggests that developers generally keep application prices relatively affordable to remain competitive and attract users.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 14. Estimated Revenue by Category

I created an estimated revenue variable using:

```python
app["Estimated_Revenue"] = (
    app["Price"] * app["Installs"]
)
```

I then calculated estimated revenue by category.

```python
revenue_category = (
    app.groupby("Category")["Estimated_Revenue"]
    .sum()
    .sort_values(ascending=False)
)
```

A bar chart was created to compare estimated revenue across categories.

The analysis showed that revenue can vary significantly between categories.

Categories with a combination of high installation numbers and higher prices can produce larger estimated revenue.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 15. Sentiment Analysis of User Reviews

I performed sentiment analysis on the user reviews using TextBlob.

The purpose was to classify reviews into:

* Positive
* Negative
* Neutral

TextBlob calculates a polarity score for the review text.

I created a function:

```python
def get_sentiment(review):

    if pd.isna(review):
        return "Neutral"

    polarity = TextBlob(
        str(review)
    ).sentiment.polarity

    if polarity > 0:
        return "Positive"

    elif polarity < 0:
        return "Negative"

    else:
        return "Neutral"
```

The function was applied to the review text:

```python
reviews["Predicted_Sentiment"] = (
    reviews["Translated_Review"]
    .apply(get_sentiment)
)
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 16. Handling Missing Reviews

Missing values in the review dataset were handled before sentiment analysis.

For example:

```python
reviews["Translated_Review"] = (
    reviews["Translated_Review"]
    .fillna("No Review")
)
```

The existing sentiment-related columns were also handled during preprocessing.

This allowed the sentiment analysis to run without errors caused by missing review text.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 17. Sentiment Distribution

After calculating the predicted sentiment, I checked the number of reviews in each sentiment category.

```python
sentiment_counts = (
    reviews["Predicted_Sentiment"]
    .value_counts()
)
```

I then visualized the results using a bar chart.

The analysis showed that positive reviews formed a large proportion of the available reviews, while negative reviews represented a smaller portion.

This provides a general indication of user satisfaction, although automated sentiment analysis can sometimes misunderstand context, sarcasm, or mixed opinions.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 18. Sentiment by App Category

To understand how user sentiment differs between categories, I merged the two datasets using the `App` column.

```python
merged_df = pd.merge(
    app,
    reviews,
    on="App",
    how="inner"
)
```

I then created a cross-tabulation:

```python
sentiment_category = pd.crosstab(
    merged_df["Category"],
    merged_df["Predicted_Sentiment"]
)
```

This allowed me to compare Positive, Negative, and Neutral reviews across different app categories.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 19. Category-wise Sentiment Analysis

I created a stacked bar chart showing sentiment distribution across app categories.

The analysis showed that positive reviews were dominant across most categories.

The Game category had the highest volume of reviews and also a large number of positive reviews.

Other categories such as:

* Family
* Productivity
* Tools
* Sports

also showed predominantly positive sentiment.

However, negative reviews are still useful because they can point developers toward possible issues with:

* App performance
* Bugs
* Features
* User experience
* Customer expectations

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 20. Positive Reviews by Category

I calculated the number of positive reviews for each category.

```python
positive_category = (
    merged_df.groupby("Category")
    ["Predicted_Sentiment"]
    .apply(
        lambda x: (x == "Positive").sum()
    )
    .sort_values(ascending=False)
)
```

I then selected the top categories with the highest number of positive reviews.

This helped identify categories with a large amount of favorable user feedback.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 21. Negative Reviews by Category

I also calculated the number of negative reviews for each category.

```python
negative_category = (
    merged_df.groupby("Category")
    ["Predicted_Sentiment"]
    .apply(
        lambda x: (x == "Negative").sum()
    )
    .sort_values(ascending=False)
)
```

This analysis can help developers identify categories where users are reporting more negative experiences.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 22. Neutral Reviews by Category

Similarly, I calculated the number of neutral reviews.

```python
neutral_category = (
    merged_df.groupby("Category")
    ["Predicted_Sentiment"]
    .apply(
        lambda x: (x == "Neutral").sum()
    )
    .sort_values(ascending=False)
)
```

Neutral reviews may contain factual statements or opinions without strong positive or negative sentiment.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 23. Interactive Visualization Using Plotly

I also created interactive visualizations using Plotly.

The first interactive visualization was an app size versus installation scatter plot.

```python
fig = px.scatter(
    plot_data,
    x="Size",
    y="Installs",
    color="Category",
    size="Rating",
    hover_name="App",
    hover_data=[
        "Category",
        "Rating",
        "Price"
    ],
    title="Interactive Scatter Plot: App Size vs. Installs"
)
```

I used a logarithmic scale for installations because the number of installs varies considerably between applications.

The interactive chart allows the user to hover over individual applications and view information such as:

* App name
* Category
* Rating
* Price
* Number of installs
* App size

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 24. Interactive Category Visualization

I also created an interactive bar chart for the top 10 app categories.

```python
category_counts = (
    app["Category"]
    .value_counts()
    .head(10)
    .reset_index()
)
```

The chart was created using Plotly:

```python
fig = px.bar(
    category_counts,
    x="Category",
    y="Number of Apps",
    color="Number of Apps",
    text="Number of Apps",
    title="Top 10 App Categories"
)
```

This makes it easier to interact with the category-level results.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 25. Main Findings

The main findings from the analysis were:

# 1. Some categories are highly saturated

Family, Game, and Tools contain the largest number of applications in the analysis.

Family has 1,943 apps, Game has 1,121, and Tools has 843.

This indicates strong competition in these categories.


# 2. App size does not appear to strongly determine installs

The scatter plot showed a wide range of installation counts for both small and large applications.

Therefore, app size alone does not appear to be a strong indicator of popularity.


# 3. Free apps dominate

The Play Store dataset contains substantially more free applications than paid applications.

This suggests that free distribution is a common strategy for attracting users.


# 4. Most paid apps have relatively low prices

The price distribution showed that most paid applications are concentrated toward the lower end of the price range.

Only a small number of apps have relatively high prices.


# 5. User sentiment is generally positive

The sentiment analysis showed that positive reviews form a large portion of the analyzed reviews.

This suggests generally favorable user feedback across many app categories.


# 6. Game has a large volume of user reviews

The Game category has a particularly large number of reviews.

This indicates high user activity and engagement within this category.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 26. Recommendations for a New App Developer

Based on the analysis, I would consider the following points before launching a new application.

# 1. Select the category carefully

Family, Game, and Tools have a large number of applications, which indicates strong competition.

A developer entering these categories should have a clear difference in features or user experience.

Less saturated categories may provide opportunities, but market demand should also be considered.

# 2. Focus on user experience

Ratings and reviews provide direct feedback from users.

Developers should regularly monitor negative reviews and use them to identify bugs, missing features, performance problems, or other issues.

# 3. Consider a free-first strategy

Since free applications dominate the dataset, a developer could consider offering the initial app for free and using other monetization methods such as:

* Advertisements
* In-app purchases
* Subscriptions

The appropriate model would depend on the type of application and target audience.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 27. Three Data-Driven Insights for a Developer

# Insight 1 – Choose the category strategically

Family, Game, and Tools have the largest number of applications.

This suggests high competition. A new developer entering these categories should offer unique features or a better user experience to stand out.


# Insight 2 – User experience should be a priority

The analysis of ratings and user reviews shows generally positive sentiment across many categories.

Developers should continue monitoring reviews because negative feedback can help identify areas that need improvement.

Regular updates, bug fixes, and responding to user needs can help maintain good ratings.


# Insight 3 – Consider an effective monetization strategy

Free applications dominate the dataset.

Instead of depending only on an initial purchase price, developers can consider a free application combined with advertisements, subscriptions, or in-app purchases depending on the product.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# What I Learned

This task helped me gain practical experience in analyzing real-world application and user-review data.

During this project, I learned how to:

* Work with multiple datasets.
* Inspect and clean real-world data.
* Handle missing values.
* Convert string-based numerical columns into proper numeric formats.
* Analyze categorical data.
* Compare ratings across categories.
* Study relationships between numerical variables.
* Calculate estimated revenue.
* Perform basic sentiment analysis using TextBlob.
* Merge datasets using a common column.
* Analyze sentiment across categories.
* Create interactive visualizations using Plotly.
* Convert analysis results into practical recommendations.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Conclusion

In this project, I analyzed Google Play Store applications along with user reviews to understand app categories, ratings, installations, pricing, revenue potential, and user sentiment.

The analysis showed that the Play Store is highly concentrated in a few categories, particularly Family, Game, and Tools. These categories have a large number of applications and therefore represent competitive markets.

The analysis of app size and installations did not show a strong relationship, suggesting that app size alone is not a major factor in determining popularity. The dataset was also dominated by free applications, showing that free distribution is a common approach for reaching a larger user base.

I also performed sentiment analysis using TextBlob. Positive reviews were dominant across most categories, while negative reviews provided useful information about areas where developers could improve the user experience.

Finally, I created interactive Plotly visualizations to make the analysis easier to explore.

Overall, this task helped me understand how data cleaning, exploratory analysis, visualization, sentiment analysis, and business insights can be combined to support decisions related to mobile application development.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Author

Krishna Deshmukh

M.Sc. Statistics

Skills: Python | SQL | Excel | Power BI | Machine Learning | Data Analysis

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Internship

This task is a part of my Data Analysis Interrnship

Project: Google Play Store Apps Analysis and User Review Sentiment Analysis

