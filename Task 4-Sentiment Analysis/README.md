# Yelp Review Sentiment Analysis Using NLP

# Internship Task 4 – Sentiment Analysis

This project was completed as the fourth task of my data analytics internship.

The objective of this task was to analyze customer reviews and classify them into three sentiment categories: Positive, Neutral, and Negative.

I used Natural Language Processing (NLP) techniques to clean and prepare the review text and then converted the text into numerical features using TF-IDF. Three machine learning models were trained and compared:

* Logistic Regression
* Multinomial Naive Bayes
* Linear Support Vector Machine (SVM)

The models were evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Project Objective

The main objectives of this project were:

* Load and inspect the Yelp review dataset.
* Understand the distribution of positive, neutral, and negative reviews.
* Convert star ratings into sentiment labels.
* Clean and preprocess the review text.
* Apply tokenization, stopword removal, and lemmatization.
* Convert text into numerical features using TF-IDF.
* Split the dataset into training and testing sets.
* Train multiple classification models.
* Compare model performance using different evaluation metrics.
* Visualize sentiment distributions and important words.
* Analyze incorrectly classified reviews.
* Identify the best-performing model.
* Discuss possible real-world applications of sentiment analysis.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dataset

The project uses a Yelp customer review dataset containing review text and star ratings.

The main columns used in this project were:

| Column     | Description                                     |
| ---------- | ----------------------------------------------- |
| Text       | Customer review                                 |
| Stars      | Rating given by the customer                    |
| Sentiment  | Sentiment category created from the star rating |
| Clean Text | Preprocessed review text                        |

The original star ratings were converted into three sentiment classes.

# Sentiment Mapping

| Stars | Sentiment |
| ----- | --------- |
| 1     | Negative  |
| 2     | Negative  |
| 3     | Neutral   |
| 4     | Positive  |
| 5     | Positive  |

This mapping was used to create the target variable for the machine learning models.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Tools and Technologies

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* NLTK
* Contractions
* Scikit-learn
* TF-IDF
* Logistic Regression
* Multinomial Naive Bayes
* Linear SVM
* WordCloud

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. Importing Libraries

I started by importing the basic Python libraries required for data manipulation and visualization.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

Additional libraries were imported later for NLP preprocessing and machine learning.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 2. Loading the Dataset

The Yelp review dataset was loaded using Pandas.

```python
df = pd.read_csv("yelp_dataset_reviews.csv")
```

I then used functions such as `head()`, `tail()`, `shape()`, and `info()` to understand the dataset.

```python
df.head()
df.tail()
df.shape
df.info()
```

I also checked the unique values in the `stars` column.

```python
df['stars'].unique()
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3. Checking Missing Values

Before starting the NLP process, I checked the dataset for missing values.

```python
df.isna().sum()
```

This helped identify whether missing values needed to be handled before creating the sentiment labels and processing the review text.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 4. Creating Sentiment Labels

The Yelp star ratings were converted into three sentiment categories.

```python
df["sentiment"] = df["stars"].map({
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive"
})
```

The resulting target variable contains:

* Positive
* Neutral
* Negative

This made the problem a three-class sentiment classification problem.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 5. Sentiment Class Distribution

I calculated the number of reviews in each sentiment class.

```python
sentiment_count = df['sentiment'].value_counts()
sentiment_count
```

I also created a bar chart to visualize the distribution.

```python
sentiment_count.plot(kind='bar')
plt.title("Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.show()
```

# Observation

The dataset is imbalanced.

* Positive reviews form the majority class.
* Neutral reviews are the minority class.
* Negative reviews are fewer than positive reviews.

Because of this imbalance, accuracy alone may not give a complete picture of model performance. Therefore, precision, recall, and F1-score were also considered.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 6. Text Preprocessing

Raw customer reviews cannot be directly given to most traditional machine learning algorithms.

Therefore, I created a text preprocessing pipeline.

The preprocessing steps included:

1. Lowercasing
2. Expanding contractions
3. Removing punctuation
4. Removing numbers
5. Removing extra spaces
6. Tokenization
7. Removing very short words
8. Removing repeated-character words
9. Stopword removal
10. Lemmatization

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 7. Lowercasing

All text was converted to lowercase.

For example:

```text
"This Restaurant Is AMAZING"
```

becomes:

```text
"this restaurant is amazing"
```

This prevents words with different capitalization from being treated as different features.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 8. Expanding Contractions

The `contractions` library was used to expand shortened forms.

For example:

```text
"don't" → "do not"
"can't" → "cannot"
```

This is useful for sentiment analysis because negation words such as `not` can affect the meaning of a sentence.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 9. Removing Punctuation and Numbers

Punctuation and numeric characters were removed from the text.

```python
text = text.translate(
    str.maketrans('', '', string.punctuation)
)

text = re.sub(r'\d+', '', text)
```

Extra spaces were also removed.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 10. Tokenization

The review text was split into individual words using NLTK's `word_tokenize()`.

For example:

```text
"The food was excellent"
```

becomes approximately:

```text
["the", "food", "was", "excellent"]
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 11. Stopword Removal

Common English words were removed using NLTK stopwords.

However, I intentionally kept important negative words such as:

```text
not
no
nor
won't
hasn't
hadn't
haven't
doesn't
isn't
aren't
```

This is important because removing words related to negation can change the meaning of a review.

For example:

```text
"I do not recommend this restaurant."
```

contains a very different sentiment from:

```text
"I recommend this restaurant."
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 12. Lemmatization

I used `WordNetLemmatizer` to convert words to their base form.

For example, different forms of a word can be reduced to a common base form.

```python
lemmatizer = WordNetLemmatizer()
```

The complete preprocessing function was then applied to the review text.

```python
df["clean_text"] = df["text"].apply(preprocess_text)
```

The cleaned reviews were stored in a new column called:

```text
clean_text
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 13. TF-IDF Feature Extraction

Machine learning models cannot directly work with raw text.

Therefore, I converted the cleaned reviews into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency).

The vectorizer was configured as:

```python
tfidf = TfidfVectorizer(
    stop_words="english",
    min_df=5,
    max_df=0.8,
    max_features=20000,
    ngram_range=(1,2),
    token_pattern=r"\b[a-zA-Z]{3,}\b"
)
```

The important settings were:

* `min_df=5` – ignore extremely rare terms.
* `max_df=0.8` – ignore terms appearing in a very large percentage of documents.
* `max_features=20000` – limit the total number of features.
* `ngram_range=(1,2)` – use both unigrams and bigrams.

The TF-IDF matrix was created using:

```python
x = tfidf.fit_transform(df['clean_text'])
```

The sentiment column was used as the target variable.

```python
y = df['sentiment']
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 14. Train-Test Split

The dataset was divided into training and testing data using an 80/20 split.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The `stratify=y` parameter was used to maintain a similar sentiment distribution in both the training and testing datasets.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 15. Model 1 – Linear Support Vector Machine

The first model used was a Linear Support Vector Classifier.

I used `GridSearchCV` to search for a suitable value of `C`.

```python
param_grid = {
    'C': [0.01, 0.1, 1, 5, 10]
}
```

The model was configured with:

```python
LinearSVC(
    class_weight='balanced',
    random_state=42
)
```

Using `class_weight='balanced'` helps give more importance to minority classes.

I used macro F1-score as the GridSearchCV scoring metric because the dataset is imbalanced.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 16. Model 2 – Logistic Regression

The second model was Logistic Regression.

```python
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(x_train, y_train)
```

The model was then used to predict the sentiment of the test reviews.

```python
y_pred = lr_model.predict(x_test)
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 17. Model 3 – Multinomial Naive Bayes

The third model was Multinomial Naive Bayes.

```python
nb_model = MultinomialNB(alpha=1.0)

nb_model.fit(x_train, y_train)

nb_y_pred = nb_model.predict(x_test)
```

Multinomial Naive Bayes is commonly used for text classification because it works well with word-frequency-based features such as TF-IDF.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 18. Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

The evaluation results were stored in a comparison DataFrame.

```python
Evaluation_of_Models = pd.DataFrame({
    "Model": [
        'Logistic Regression',
        'Multinomial Naive Bayes',
        'Linear SVM'
    ],
    "Accuracy": [
        LR_Accuracy,
        nb_accuracy,
        svm_accuracy
    ],
    "Precision_Score": [
        LR_pre,
        nb_pre,
        svm_pre
    ],
    "Recall_Score": [
        LR_rec,
        nb_rec,
        svm_rec
    ],
    "f1_score": [
        LR_f1_score,
        nb_f1_score,
        svm_f1_score
    ]
})
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 19. Model Performance

The results obtained from the notebook were:

| Model                   |   Accuracy |  Precision |     Recall |   F1-Score |
| ----------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression     |   84.51%   |   84.51%   |   64.28    |   66.00    |
| Multinomial Naive Bayes |   80.79%   |   79.12%   |   53.89    |   53.41    |
| Linear SVM              |   84.23%   |   71.13    |   70.00%   |   70.39%   |

The main performance observations were:

# Logistic Regression

Logistic Regression achieved the highest accuracy at 84.51%.

However, its recall and F1-score were lower than Linear SVM.

# Multinomial Naive Bayes

Multinomial Naive Bayes achieved the highest precision at 79.12%.

However, it had the lowest recall and F1-score among the three models.

# Linear SVM

Linear SVM achieved:

* Accuracy: 84.23%
* Recall: 69.96%
* F1-score: 70.39%

Although its accuracy was slightly lower than Logistic Regression, it achieved the best F1-score and recall.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 20. Confusion Matrix

Confusion matrices were created for each model.

A confusion matrix helps understand:

* Which sentiment classes were predicted correctly.
* Which classes were confused with each other.
* How well the model handles the minority classes.

This was particularly important for the Neutral class because it had fewer observations than the Positive class.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 21. Word Cloud Analysis

I also created word clouds for each sentiment category.

Three separate word clouds were created for:

* Positive reviews
* Negative reviews
* Neutral reviews

Before generating the word clouds, some common domain-specific words were removed, such as:

```text
food
restaurant
place
one
go
get
would
could
really
```

This helped make the word clouds more focused on sentiment-related vocabulary.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 22. Positive Review Word Cloud

A word cloud was generated using reviews classified as Positive.

```python
positive_text = " ".join(
    df[df["sentiment"]=="Positive"]["clean_text"]
)
```

The visualization provides a quick view of frequently occurring terms in positive reviews.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 23. Negative Review Word Cloud

The same approach was used for negative reviews.

```python
negative_text = " ".join(
    df[df["sentiment"]=="Negative"]["clean_text"]
)
```

The negative word cloud helps identify commonly used terms associated with negative customer experiences.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 24. Neutral Review Word Cloud

A separate word cloud was created for neutral reviews.

```python
neutral_text = " ".join(
    df[df["sentiment"]=="Neutral"]["clean_text"]
)
```

This helps identify the common vocabulary used in reviews with a neutral rating.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 25. Error Analysis

After model evaluation, I performed error analysis to understand why some reviews were classified incorrectly.

I created a DataFrame containing:

* Review
* Actual sentiment
* Predicted sentiment

```python
predictions = pd.DataFrame({
    "Review": df.loc[test_idx, "text"],
    "Actual": y_test.values,
    "Predicted": y_pred
})
```

I then selected reviews where the actual and predicted sentiment were different.

```python
misclassified = predictions[
    predictions["Actual"] != predictions["Predicted"]
]
```

Five misclassified reviews were examined manually.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 26. Examples of Misclassification

# Example 1 – Positive predicted as Negative

One review contained several strong negative words such as disappointment and bad business practices. However, the reviewer later explained that the issue had been resolved and expressed satisfaction.

The model focused heavily on the negative words and did not fully understand the change in sentiment throughout the review.


# Example 2 – Neutral predicted as Positive

One 3-star review contained several positive expressions such as praise for the food and service but also included some criticism.

Since the sentiment label was created from the star rating, the review was labeled Neutral even though much of its text sounded positive.

This shows a possible difference between rating-based labels and actual textual sentiment.

# Example 3 – Negative predicted as Positive

A review mentioned serious problems and included the statement that the customer would never return. It also contained sarcasm.

The TF-IDF + SVM model could not fully understand the sarcasm and contextual meaning, which contributed to the incorrect Positive prediction.


# Example 4 – Neutral predicted as Positive

Another review criticized one particular food item but also included positive comments about the restaurant and other food.

The positive words received strong TF-IDF representation, causing the model to predict Positive even though the rating-based label was Neutral.


# Example 5 – Negative predicted as Positive

One review contained positive expressions about the chef and lunch but clearly stated that the restaurant was not authentic and was not recommended for the intended cuisine.

The model was influenced by the positive vocabulary and had difficulty understanding the overall recommendation and negation.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 27. Key Findings

The main findings from the project were:

1. The Yelp review dataset contains an imbalanced sentiment distribution, with Positive reviews being the majority class.

2. TF-IDF with unigrams and bigrams provided a useful numerical representation of the review text.

3. Logistic Regression achieved the highest accuracy of 84.51%.

4. Multinomial Naive Bayes achieved the highest precision of 79.12%.

5. Linear SVM achieved the highest recall (69.96%) and F1-score (70.39%).

6. Although Logistic Regression had slightly higher accuracy, Linear SVM was selected as the preferred model because F1-score and recall are more useful when evaluating an imbalanced multi-class problem.

7. Error analysis showed that sarcasm, negation, context, and differences between star ratings and textual sentiment were common reasons for incorrect predictions.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 28. Final Model Selection

Based on the evaluation results, Linear SVM was selected as the preferred model.

Logistic Regression achieved the highest accuracy: 84.51%

However, Linear SVM achieved: F1-score: 70.39% and Recall: 69.96%

Since the dataset is imbalanced, F1-score gives a more balanced view of the model's performance across the sentiment classes.

Therefore, Linear SVM provided the most suitable overall performance for this project.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 29. Real-World Applications

The developed sentiment analysis approach can be used in several practical situations.

# Restaurant Review Analysis

Businesses can automatically classify customer reviews as Positive, Neutral, or Negative.

# Business Reputation Management

Companies can identify negative reviews and respond to dissatisfied customers.

# Product Review Analysis

The same approach can be applied to customer reviews on platforms such as Amazon or Flipkart.

# Social Media Monitoring

Companies can analyze public opinions about their brands, products, or campaigns.

# Customer Support

Negative reviews can be automatically identified and prioritized for faster response.

# Market Research

Sentiment analysis can help businesses understand customer opinions and identify areas that need improvement.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# What I Learned

This task helped me understand how Natural Language Processing can be used to analyze customer opinions.

During this project, I learned how to:

* Convert star ratings into sentiment classes.
* Analyze class imbalance.
* Clean customer review text.
* Use tokenization and stopword removal.
* Preserve important negation words.
* Apply lemmatization.
* Convert text into numerical features using TF-IDF.
* Use unigrams and bigrams for text representation.
* Perform a stratified train-test split.
* Train Logistic Regression, Naive Bayes, and Linear SVM models.
* Tune the SVM hyperparameter using GridSearchCV.
* Evaluate classification models using multiple metrics.
* Use confusion matrices for model analysis.
* Create sentiment-based word clouds.
* Investigate model errors through manual error analysis.
* Connect machine learning results with real-world business applications.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Conclusion

This project focused on classifying Yelp customer reviews into Positive, Neutral, and Negative sentiment categories.

The review text was cleaned using several NLP preprocessing techniques and converted into numerical features using TF-IDF. Three machine learning models were then trained and compared.

Logistic Regression achieved the highest accuracy at 84.51%, while Linear SVM achieved the highest recall and F1-score. Because the dataset is imbalanced, I selected Linear SVM as the preferred model based mainly on its F1-score of 70.39% and recall of 69.96%.

The error analysis also showed that traditional TF-IDF-based models can struggle with sarcasm, negation, context, and cases where the star rating does not completely match the language used in the review.

Overall, this task gave me practical experience in NLP, text preprocessing, TF-IDF feature extraction, machine learning classification, model evaluation, and sentiment analysis.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Author

Krishna Deshmukh

M.Sc.(Statistics)

Skills: Python | NLP | Machine Learning | SQL | Excel | Power BI | Data Analysis

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Internship

This project was completed as part of my Data Analytics Internship.

Project: Yelp Review Sentiment Analysis Using NLP
