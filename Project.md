---

# Guinness World Records Search Engine

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset Extraction](#2-dataset-extraction)
3. [Loading the Documents](#3-loading-the-documents)
4. [Text Preprocessing](#4-text-preprocessing)
5. [TF-IDF Computation](#5-tf-idf-computation)
6. [User Query Processing](#6-user-query-processing)
7. [Document Scoring](#7-document-scoring)
8. [Ranking and Retrieving Results](#8-ranking-and-retrieving-results)
9. [Full Search Workflow Example](#9-full-search-workflow-example)

---

## 1. Introduction

This notebook demonstrates building a simple search engine for Guinness World Records Instagram posts.  
We will:

- Load the dataset
- Preprocess the text
- Compute TF-IDF scores
- Retrieve the most relevant records for a user query

---

## 2. Dataset Extraction

The `guinnessWorldRecords` folder contains many documents, each representing the text content of an Instagram post by the official Guinness World Records account (@guinnessworldrecords).

To begin, extract the dataset using the following command:

```python
!unzip guinnessWorldRecords.zip
```

This command decompresses the folder and makes the records available for processing.

---

## 3. Loading the Documents

We will write a helper function to retrieve all `.txt` files from the specified folder and load their content into a list.

```python
from os import listdir
from os.path import isfile, join

def get_all_files(folder_path):
    """Returns a list of file paths for all files in the given folder."""
    return [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

def load_data(folder_path):
    """Reads all .txt files in the given folder and returns a list of texts."""
    file_paths = get_all_files(folder_path)
    texts = []
    for file_path in file_paths:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

# Example usage:
folder_path = "guinnessWorldRecords"
docs = load_data(folder_path)
print("Input:", folder_path)
print("Output:", docs[:2])  # Show only first 2 for brevity
```

---

## 4. Text Preprocessing

We will preprocess the text using the following steps:

- Convert to lowercase
- Remove punctuation
- Tokenize (using `WordPunctTokenizer`)
- Remove English stopwords
- Apply stemming (`PorterStemmer`)
- Apply lemmatization (`WordNetLemmatizer`)

```python
import string
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = WordPunctTokenizer().tokenize(text)
    # Remove punctuation
    tokens = [t.translate(str.maketrans('', '', string.punctuation)) for t in tokens]
    tokens = [t for t in tokens if t]  # Remove empty tokens
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    # Stemming
    ps = PorterStemmer()
    tokens = [ps.stem(t) for t in tokens]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

# Example:
print(preprocess_text("The BOYS are jumping on the trampoline."))
# Output: ['boy', 'jump', 'trampolin']
```

To preprocess all documents:

```python
def preprocess_docs(docs):
    """Apply preprocess_text to each document in docs."""
    return [preprocess_text(doc) for doc in docs]

processed_docs = preprocess_docs(docs)
```

---

## 5. TF-IDF Computation

Here, we’ll calculate TF, IDF, and TF-IDF scores for each token in each document.

```python
import math

def calc_tf(token, tokens_of_document):
    """Term Frequency for a token in a document."""
    return tokens_of_document.count(token) / len(tokens_of_document)

def calc_idf(token, list_of_tokens_docs):
    """Inverse Document Frequency for a token across all documents."""
    freq_token = sum(1 for doc in list_of_tokens_docs if token in doc)
    return math.log10(len(list_of_tokens_docs) / freq_token) if freq_token else 0

def calc_tf_idf(token, tokens_doc, list_of_tokens_docs):
    return calc_tf(token, tokens_doc) * calc_idf(token, list_of_tokens_docs)

def calc_tf_idf_doc(tokens_doc, list_of_tokens_docs):
    """TF-IDF dictionary for a single document."""
    tf_idf_doc = {}
    for token in set(tokens_doc):
        tf_idf_doc[token] = calc_tf_idf(token, tokens_doc, list_of_tokens_docs)
    return tf_idf_doc

def calc_tf_idf_all_docs(list_of_tokens_docs):
    """TF-IDF dictionaries for all documents."""
    return [calc_tf_idf_doc(tokens_doc, list_of_tokens_docs) for tokens_doc in list_of_tokens_docs]

tfidf_docs = calc_tf_idf_all_docs(processed_docs)
```

---

## 6. User Query Processing

Preprocess the user’s search query using our existing pipeline.

```python
def preprocess_user_query(query):
    return preprocess_text(query)

# Example:
print(preprocess_user_query("Who is the tallest DOG in the world?"))
# Output: ['tallest', 'dog', 'world']
```

---

## 7. Document Scoring

Calculate the score of each document for the query tokens by summing the TF-IDF scores for the tokens present.

```python
def calc_score_doc(tf_idf_doc, query_tokens):
    """Sum the TF-IDF scores for the query tokens in a single document."""
    return sum(tf_idf_doc.get(token, 0) for token in query_tokens)

def calc_scores_docs(tf_idf_docs, query_tokens):
    """Return a list of scores for all documents."""
    return [calc_score_doc(tf_idf_doc, query_tokens) for tf_idf_doc in tf_idf_docs]

# Example:
tf_idf_docs_example = [{"test": 0.22, "doc": 0.013, "1": 0.1}, {"test": 0.4, "doc": 0.02, "2": 0.9}]
query_tokens_example = ["best", "doc"]
print(calc_scores_docs(tf_idf_docs_example, query_tokens_example))
# Output: [0.013, 0.02]
```

---

## 8. Ranking and Retrieving Results

Return the top 5 documents with the highest scores for the query.

```python
def rank_docs(docs, scores_docs, top_n=5):
    """Return the top_n documents with the highest scores."""
    combined = list(zip(docs, scores_docs))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_combined[:top_n]]
```

---

## 9. Full Search Workflow Example

Now, we’ll combine all the steps into a single `search` function, which takes a search query, the tf-idf dictionaries, and the original documents, and returns the top 5 results.

```python
def search(search_question, tf_idf_per_doc, docs):
    query_tokens = preprocess_user_query(search_question)
    scores = calc_scores_docs(tf_idf_per_doc, query_tokens)
    return rank_docs(docs, scores, top_n=5)

# Example usage:
search_question = "Who is the tallest DOG in the world?"
top_results = search(search_question, tfidf_docs, docs)
for idx, res in enumerate(top_results, 1):
    print(f"{idx}. {res}\n")
```

---
