# Email Autocomplete and Autocorrect using NLP

# Internship Task 9

This project was completed as the ninth task of my data analytics internship.

The main objective of this task was to build a simple text prediction and spelling correction system using email data.

For this project, I worked with the Enron email dataset and used a sample of 10,000 emails. The email messages were first cleaned and converted into a suitable format for Natural Language Processing (NLP).

I then built:

* A frequency-based Bigram model
* A frequency-based Trigram model
* An **Email Autocomplete system
* An Autocorrect system using PySpellChecker

Finally, I tested both autocomplete approaches and compared their Precision and Recall.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Project Objective

The main objectives of this task were:

* Work with a large text corpus.
* Extract useful information from raw email messages.
* Perform basic NLP preprocessing.
* Tokenize email text.
* Convert text to lowercase.
* Remove punctuation and numbers.
* Remove English stopwords.
* Build a frequency-based n-gram model.
* Implement an autocomplete system.
* Predict the next word from a given prefix.
* Implement an autocorrect system.
* Test autocorrect on deliberately misspelled words.
* Calculate correction accuracy.
* Calculate Precision and Recall.
* Compare Bigram and Trigram approaches.
* Visualize the most frequently used words in the corpus.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dataset

The project uses the Enron email dataset.

The original CSV file contains email records with two main columns:

* `file`
* `message`

The `message` column contains the complete raw email content, including information such as:

* Message ID
* Date
* Sender
* Receiver
* Subject
* Email body

# In this project, I worked with the first 10,000 email messages from the dataset.

# Tools and Technologies

The project was implemented using Python.

# Libraries used

* Pandas
* NumPy
* Matplotlib
* Seaborn
* NLTK
* Regular Expressions (`re`)
* `email`
* Collections
* PySpellChecker
* Contractions

The main concepts used were:

* Natural Language Processing
* Text preprocessing
* Tokenization
* Stopword removal
* N-gram language modeling
* Autocomplete
* Spell correction
* Precision and Recall

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. Importing Libraries

I started by importing the libraries required for data handling and visualization.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

# For NLP preprocessing, I used NLTK along with regular expressions and the Python email module.

# 2. Loading the Email Dataset

The dataset was loaded using Pandas.

```python
df = pd.read_csv("emails.csv")
```

I checked the first few records to understand how the raw email data was stored.

The `message` column contained the complete email content.

For example, a message contained information such as the date, sender, receiver and email body.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3. Extracting Email Information

Since the complete email was stored in a single `message` column, I used Python's `email` module to extract individual fields.

I created a function:

```python
from email import message_from_string

def extract_email(text):

    msg = message_from_string(text)

    return pd.Series({
        'Message_ID': msg['Message-ID'],
        'Date': msg['Date'],
        'From': msg['From'],
        'To': msg['To'],
        'Cc': msg['Cc'],
        'Bcc': msg['Bcc'],
        'Subject': msg['Subject'],
        'Body': msg.get_payload()
    })
```

I then applied the function to the first 10,000 emails.
This converted the raw email messages into separate columns.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 4. Extracted Email Fields

After extraction, the dataset contained:

| Column     | Description                    |
| ---------- | ------------------------------ |
| Message_ID | Unique identifier of the email |
| Date       | Date and time of the email     |
| From       | Sender                         |
| To         | Receiver                       |
| Cc         | Carbon-copy recipients         |
| Bcc        | Blind carbon-copy recipients   |
| Subject    | Email subject                  |
| Body       | Main email content             |

The extracted dataset initially contained 8 columns and 10,000 rows.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 5. Checking Duplicate Records

I checked whether duplicate email records were present.

```python
emails.duplicated().sum()
```

The result was:

```text
0
```

So no duplicate rows were found in the extracted 10,000-email dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 6. Checking Missing Values

I checked missing values in the extracted email data.

```python
emails.isna().sum()
```

The result showed missing values mainly in the `To`, `Cc`, and `Bcc` fields.

```text
To      329
Cc     9075
Bcc    9075
```

The other main fields did not have missing values.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 7. Cleaning Email Metadata

Since `Cc` and `Bcc` contained a large number of missing values and were not required for the NLP task, I removed these columns.

```python
emails = emails.drop(
    columns=['Cc', 'Bcc']
)
```

For missing receiver information, I replaced the missing values with `"Unknown"`.

```python
emails['To'] = emails['To'].fillna(
    'Unknown'
)
```

After this step, the working dataset contained:

* Message_ID
* Date
* From
* To
* Subject
* Body

All 10,000 records had values in these six columns.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 8. Converting Date

The email date was originally stored as text.

I converted it into a datetime format:

```python
emails['Date'] = pd.to_datetime(
    emails['Date'],
    utc=True,
    errors='coerce'
)
```

I then converted it to a date and again to datetime format for further handling.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 9. NLP Text Preprocessing

The main NLP preprocessing was performed on the email `Body`.

The preprocessing steps were:

1. Lowercasing
2. Punctuation removal
3. Number removal
4. Tokenization
5. Stopword removal

I created a preprocessing function:

```python
def preprocess(text):

    text = str(text).lower()

    text = re.sub(
        r'[^\w\s]',
        '',
        text
    )

    text = re.sub(
        r'\d+',
        '',
        text
    )

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)
```

The function was applied to the email body:

```python
emails['Clean_Text'] = (
    emails['Body'].apply(preprocess)
)
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 10. Tokenization

After cleaning the email text, I combined the processed text into one corpus.

```python
text_data = emails['Clean_Text']

text_data = text_data[
    text_data.str.strip() != ''
]

corpus = " ".join(text_data)
```

I then split the corpus into individual words.

```python
tokens = corpus.split()
```

This created the sequence of words that was later used for building the n-gram models.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 11. Building the Trigram Model

For the main autocomplete system, I used a Trigram model.

A trigram consists of three consecutive words.

For example:

```text
"please let me"
```

can be treated as:

```text
please → let → me
```

The model learns which third word is most likely to occur after a particular pair of words.

I generated trigrams using:

```python
trigrams = [
    (
        tokens[i],
        tokens[i+1],
        tokens[i+2]
    )
    for i in range(len(tokens)-2)
]
```

I then counted their frequencies using `Counter`.

```python
trigram_counts = Counter(
    trigrams
)
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 12. Creating the Next-Word Dictionary

I created a dictionary where each pair of words is connected with possible next words.

```python
next_words = defaultdict(Counter)

for (word1, word2, word3), count \
        in trigram_counts.items():

    next_words[(word1, word2)][word3] = count
```

For example, if the input is:

```text
please let
```

the model checks which words commonly occurred after that pair in the training corpus.

The output from the notebook showed:

```text
know → 332 times
us → 30 times
cathy → 3 times
```

This means `"know"` was the most frequent next word for the prefix `"please let"` in the corpus.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 13. Autocomplete Function

I created an autocomplete function that accepts a text prefix and returns the most frequent possible next words.

```python
def autocomplete(prefix, n=5):

    words = prefix.lower().split()

    if len(words) < 2:
        return "Please enter at least two words."

    last_two = (
        words[-2],
        words[-1]
    )

    predictions = next_words.get(
        last_two
    )

    if not predictions:
        return "No prediction found."

    return predictions.most_common(n)
```

The function requires at least two words because the Trigram model uses the previous two words to predict the next word.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 14. Testing Autocomplete

I tested the autocomplete function with different prefixes.

Some of the prefixes tested were:

```text
please let
let me
look forward
thank you
i would
we need
as soon
if you
do not
in order
```

The system returned the top three possible predictions for each prefix.

For example:

```text
Prefix: please let

know → 332 times
us → 30 times
cathy → 3 times
```

For some prefixes, no prediction was returned because the particular word pair was not available in the learned dictionary.

This is one limitation of a frequency-based n-gram model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 15. Autocorrect System

For the autocorrect part, I used the PySpellChecker library.

```python
from spellchecker import SpellChecker

spell = SpellChecker()
```

The model was first tested on a single misspelled word:

```text
travling
```

The system corrected it to:

```text
traveling
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 16. Autocorrecting Complete Sentences

I created a function that checks every word in a sentence.

```python
def autocorrect_sentence(text):

    words = str(text).split()

    corrected_words = []

    for word in words:

        corrected = spell.correction(word)

        if corrected is None:
            corrected = word

        corrected_words.append(
            corrected
        )

    return " ".join(corrected_words)
```

For example:

# Original

```text
I am travling to the offce tomorow
```

# Corrected

```text
I am traveling to the office tomorrow
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 17. Testing Autocorrect

I created a test set containing 20 deliberately misspelled words.

Some examples were:

| Misspelled  | Correct Word |
| ----------- | ------------ |
| travling    | traveling    |
| offce       | office       |
| tomorow     | tomorrow     |
| recieve     | receive      |
| managment   | management   |
| meetng      | meeting      |
| busines     | business     |
| comittee    | committee    |
| seperate    | separate     |
| definately  | definitely   |
| adress      | address      |
| calender    | calendar     |
| availble    | available    |
| schedul     | schedule     |
| importent   | important    |
| infromation | information  |
| oppertunity | opportunity  |
| departmant  | department   |
| recomend    | recommend    |
| discussin   | discussion   |

These were compared with the predictions generated by the spell checker.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 18. Autocorrect Accuracy

Out of the 20 deliberately misspelled words, the system correctly corrected 17.

The resulting correction accuracy was:

```text
85.00%
```

The incorrect predictions included examples such as:

```text
adress → dress
calender → calender
discussin → discussing
```

This shows that a general-purpose spell checker does not always select the intended word.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 19. Autocorrect Precision and Recall

I also calculated Precision and Recall for the autocorrect test.

The results were:

| Metric    | Result |
| --------- | -----: |
| Precision |   0.85 |
| Recall    |   0.85 |

The values are equal in this test because the evaluation treats each correctly corrected word as a true positive and each incorrect correction as an error.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 20. Bigram Model

To compare another approach, I also created a Bigram model.

A bigram contains two consecutive words.

For example:

```text
"thank you"
```

is one bigram.

The Bigram model predicts the next word using only the immediately preceding word.

I created bigram frequencies using:

```python
bigram_counts = Counter()

bigram_next = defaultdict(
    Counter
)

for i in range(len(tokens) - 1):

    w1 = tokens[i]
    w2 = tokens[i + 1]

    bigram_counts[
        (w1, w2)
    ] += 1

    bigram_next[w1][w2] += 1
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 21. Bigram Prediction

The Bigram prediction function was:

```python
def bigram_predict(word, n=3):

    return bigram_next[
        word
    ].most_common(n)
```

For example, for:

```text
thank
```

the model returned:

```text
phillip → 109
jeff → 44
help → 38
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 22. Bigram Performance

The Bigram model produced:

| Metric    | Result |
| --------- | -----: |
| Precision | 12.60% |
| Recall    | 37.80% |

The relatively low values show that using only one previous word gives less context for predicting the next word.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 23. Trigram Performance

The Trigram model uses two previous words to predict the next word.

The measured results in the notebook were:

| Metric    |  Result |
| --------- | ------: |
| Precision |  33.33% |
| Recall    | 100.00% |

For the specific top-3 test used in the notebook, the Trigram model performed better than the Bigram model.

However, these results are based on the particular evaluation procedure used in the notebook and should not be interpreted as general performance on all possible text.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 24. Bigram vs Trigram Comparison

The final comparison was:

| Model       |  Precision |      Recall |
| ----------- | ---------: | ----------: |
| Bigram      |   12.60%   |    37.80%   |
| Trigram     |   33.33%   |    100.00%  |

# Interpretation

The Trigram model performed better in the notebook's evaluation.

The main reason is that the Trigram model uses two previous words as context, while the Bigram model only uses one.

For example:

```text
Bigram:
"thank" → next word

Trigram:
"thank you" → next word
```

The additional context can help the model make more specific predictions.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 25. Most Frequent Words

I also analyzed the most frequently occurring words in the cleaned email corpus.

A bar chart was created for the top 20 words.

The chart showed that "enron" had the highest frequency, followed by words such as "pm" and "subject".

Other frequent terms included words related to email communication, companies, messages, and business activities.

This makes sense because the dataset consists of business emails.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 26. Visualization

The project includes a bar chart showing the top 20 most frequent words.

The visualization was created using Matplotlib:

```python
plt.figure(figsize=(12, 6))

plt.bar(words, counts)

plt.title(
    "Top 20 Most Frequent Words"
)

plt.xlabel("Words")
plt.ylabel("Frequency")

plt.xticks(rotation=45)

plt.show()
```

A heatmap-style visualization was also created to compare correct and incorrect autocorrection results before and after correction.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 27. Overall Workflow

The complete project workflow can be summarized as:

```text
Enron Email Dataset
    ↓
Load Dataset
    ↓
Extract Email Fields
    ↓
Check Missing Values
    ↓
Remove Unnecessary Columns
    ↓
Clean Email Body
    ↓
Lowercase
    ↓
Remove Punctuation & Numbers
    ↓
Tokenization
    ↓
Stopword Removal
    ↓
Create Text Corpus
    ↓
Build N-gram Models
    ↓
 ┌───────────────┐
 │               │
Bigram        Trigram
 │               │
 ↓               ↓
Autocomplete   Autocomplete
 │
 ↓
Performance Comparison
    ↓
Autocorrect using PySpellChecker
    ↓
Accuracy / Precision / Recall
    ↓
Final Analysis
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 28. Key Results

The important results from the project are:

# Autocorrect

* Tested on 20 misspelled words
* Correction accuracy = 85%
* Precision = 85%
* Recall = 85%

# Autocomplete

| Model   | Precision |  Recall |
| ------- | --------: | ------: |
| Bigram  |    12.60% |  37.80% |
| Trigram |    33.33% | 100.00% |

Based on the evaluation performed in this project, the Trigram model gave better autocomplete results than the Bigram model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 29. Main Observations

# 1. More context improved autocomplete

The Trigram model performed better than the Bigram model in the test used in this project.

Using two previous words gives the model more context for predicting the next word.

# 2. Autocorrect worked well for common spelling mistakes

The spell checker correctly handled most of the deliberately introduced spelling mistakes.

The overall test accuracy was 85%.

# 3. Some words were difficult to correct

The spell checker made mistakes for words such as:

```text
adress
calender
discussin
```

This shows that spelling correction can depend on the surrounding context and the available dictionary.

# 4. The corpus contains strong business-related vocabulary

Words such as `enron`, `subject`, and `pm` appeared frequently.

This reflects the nature of the Enron business email corpus.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 31. Possible Improvements

The system could be improved in several ways:

* Use a larger and more diverse text corpus.
* Use word frequency together with sentence context.
* Implement smoothing for unseen n-grams.
* Compare different n-gram sizes.
* Use a neural language model.
* Try Word2Vec or other word embeddings.
* Use transformer-based language models.
* Improve spelling correction using contextual information.
* Create a larger manually verified test dataset.
* Evaluate the system using Top-1 and Top-K accuracy.
* Build a simple user interface where users can type text and receive suggestions.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 32. What I Learned

This task helped me understand the basic working of NLP systems used for text prediction and spelling correction.

During this project, I learned how to:

* Work with raw email data.
* Extract structured information from email messages.
* Clean text data.
* Perform tokenization.
* Remove stopwords.
* Create a text corpus.
* Build Bigram and Trigram models.
* Use word frequencies for next-word prediction.
* Implement an autocomplete function.
* Implement an autocorrect function.
* Evaluate text prediction using Precision and Recall.
* Analyze spelling correction accuracy.
* Compare two NLP approaches.
* Create visualizations for text-frequency analysis.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Conclusion

In this project, I built a simple NLP-based autocomplete and autocorrect system using the Enron email dataset.

I first extracted useful information from the raw email messages and prepared the email body for NLP processing. The text was converted to lowercase, punctuation and numbers were removed, and stopwords were removed after tokenization.

For autocomplete, I implemented both Bigram and Trigram frequency-based models. The Trigram model performed better than the Bigram model in the evaluation used in this project, with a Precision of 33.33% and Recall of 100%, compared with 12.60% Precision and 37.80% Recall for the Bigram model.

For autocorrect, I used PySpellChecker and tested it on 20 deliberately misspelled words. The system achieved an overall correction accuracy of 85%.

Overall, this task gave me practical experience with NLP preprocessing, n-gram language models, autocomplete, autocorrect, text-frequency analysis, and evaluation metrics.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Author

Krishna Deshmukh

M.Sc. Statistics

Skills: Python | SQL | Excel | Power BI | Machine Learning | NLP | Data Analysis

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Internship

This is an task of my Data Analysis Internship.

Project: Email Autocomplete and Autocorrect using NLP
