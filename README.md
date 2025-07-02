# NLP Search Engine — Contextual Query Matching

A simple NLP-based search engine that uses semantic similarity (TF-IDF and NLP preprocessing) to match user queries to Guinness World Records Instagram post documents.

## Overview
This project demonstrates building a basic search engine pipeline using Python and NLP libraries. It covers:
- Loading and preprocessing text documents
- Computing TF-IDF representations
- Matching and ranking search results for user queries

## Dataset
- Folder: `guinnessWorldRecords/` (contains `.txt` files, each an Instagram post)
- To extract:  
  ```python
  !unzip guinnessWorldRecords.zip
  ```

## Usage

1. **Install dependencies**
   ```bash
   pip install nltk
   ```

2. **Run the search engine**
   - Edit and run `search-engine-code.py`:
     ```python
     docs = load_data("guinnessWorldRecords")
     processed_docs = preprocess_docs(docs)
     tfidf_docs = calc_tf_idf_all_docs(processed_docs)

     query = "Who is the tallest dog in the world?"
     top_results = search(query, tfidf_docs, docs)
     for result in top_results:
         print(result)
     ```

3. **Customizing**
   - You can change preprocessing or the ranking function in `search-engine-code.py`.

## Project Structure

- `search-engine-code.py` — Main code (loading, preprocessing, search)
- `README.md` — Project introduction and quickstart
- `Project.md` — Detailed notebook-style explanation (motivation, step-by-step, code snippets)

## For More Details

See [Project.md](./Project.md) for a methodical walkthrough, expanded code, and a full workflow example.

## Requirements

- Python 3.x
- nltk
- (Optional) Additional data science libraries (pandas, numpy, etc.)

## License

MIT
