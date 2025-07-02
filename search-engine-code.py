from os import listdir
from os.path import isfile, join
import string
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import math

# -- 3. Loading the Documents --

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
# folder_path = "guinnessWorldRecords"
# docs = load_data(folder_path)
# print("Input:", folder_path)
# print("Output:", docs[:2])  # Show only first 2 for brevity


# -- 4. Text Preprocessing --

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
# print(preprocess_text("The BOYS are jumping on the trampoline."))
# Output: ['boy', 'jump', 'trampolin']

def preprocess_docs(docs):
    """Apply preprocess_text to each document in docs."""
    return [preprocess_text(doc) for doc in docs]

# processed_docs = preprocess_docs(docs)


# -- 5. TF-IDF Computation --

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

# tfidf_docs = calc_tf_idf_all_docs(processed_docs)


# -- 6. User Query Processing --

def preprocess_user_query(query):
    return preprocess_text(query)

# Example:
# print(preprocess_user_query("Who is the tallest DOG in the world?"))
# Output: ['tallest', 'dog', 'world']


# -- 7. Document Scoring --

def calc_score_doc(tf_idf_doc, query_tokens):
    """Sum the TF-IDF scores for the query tokens in a single document."""
    return sum(tf_idf_doc.get(token, 0) for token in query_tokens)

def calc_scores_docs(tf_idf_docs, query_tokens):
    """Return a list of scores for all documents."""
    return [calc_score_doc(tf_idf_doc, query_tokens) for tf_idf_doc in tf_idf_docs]

# Example:
# tf_idf_docs_example = [{"test": 0.22, "doc": 0.013, "1": 0.1}, {"test": 0.4, "doc": 0.02, "2": 0.9}]
# query_tokens_example = ["best", "doc"]
# print(calc_scores_docs(tf_idf_docs_example, query_tokens_example))
# Output: [0.013, 0.02]


# -- 8. Ranking and Retrieving Results --

def rank_docs(docs, scores_docs, top_n=5):
    """Return the top_n documents with the highest scores."""
    combined = list(zip(docs, scores_docs))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_combined[:top_n]]


# -- 9. Full Search Workflow Example --

def search(search_question, tf_idf_per_doc, docs):
    query_tokens = preprocess_user_query(search_question)
    scores = calc_scores_docs(tf_idf_per_doc, query_tokens)
    return rank_docs(docs, scores, top_n=5)

# Example usage:
# search_question = "Who is the tallest DOG in the world?"
# top_results = search(search_question, tfidf_docs, docs)
# for idx, res in enumerate(top_results, 1):
#     print(f"{idx}. {res}\n")

# --- Real Input/Output Example ---

# search_question = "What is the name of the person with the largest collection of Pepsi cans in the world?" #@param {type:"string"}
# output = search(search_question, tfidf_docs, docs)
# print(output)
# Expected Output:
# [
#     "Gary Feng from Canada has a colossal collection of 11308 rare Coca-Cola cans from around the world. Check out some of his favourites 🥤  #cola #cocacola #soda #sodacans #beverages #collector #collection #collectable #guinnessworldrecords #officiallyamazing https://www.instagram.com/p/CVNgDDtDxLr",
#     "Davide Andreani from Italy has a collection of 10,558 @cocacola cans from 87 countries. Davide received his first Coca Cola can back in 1982 when he was just 5 years old, beginning a lifetime's obsession with the soft drink. His record was confirmed on this day in 2013.  Soon after, Davide began collecting the distinctive tins, with his father bringing him home unusual designs when returning from his European business trips.  Today, he searches the globe for can designs which only appeared in shops for a limited time or sometimes never even released to the public, including rare gold and silver coloured cans released in various countries for Christmas and special sporting events🥤 \"The most valuable cans are those produced from the factory for a special moment. Like gold cans produced for plant openings or special anniversaries. But these cans are very limited and very rare,\" the passionate collector explained, ahead of his appearance in our #GWR2015 book with the record title 'Largest collection of soft drink cans - same brand'. _________________________________________  #cocacola #collection #collectable #collector #guinnessworldrecords #officiallyamazing #italy #coke #soda #onthisday https://www.instagram.com/p/BmeZN4xnYDr",
#     "The largest toothpaste collection belongs to a passionate dentist in Alpharetta, Georgia (USA). With over 2,037 kinds of toothpaste, Val Kolpakov has collected tubes from all over the world, including Korea, China, India, Russia, and Japan. His brilliant and exotic collection earned him the GWR title in 2012  #guinnessworldrecords #officiallyamazing #teeth #toothpaste #dental #dentist #collection https://www.instagram.com/p/BHP28tcgYLN",
#     "Meet the @disney superfan who holds the record for the largest collection of Mickey Mouse memorabilia! Today we're exploring the Janet Esteve's collection of 10,210 items on Facebook LIVE. Tune in from 1pm (EST) 10am (PST) 6pm (GMT) - we welcome your questions  #disney #mickeymouse #guinnessworldrecords #officiallyamazing #collectibles #collection #facebooklive https://www.instagram.com/p/BPIS7EaABsq",
#     "The largest collection of @transformersofficial memorabilia is 2,111 items and was achieved by Louis Georgiou from Manchester, UK - as featured in the GWR 2020 book.\u2063 \u2063 Louis started his collection in 2011 after buying some Transformers toys for his son. His collection steadily grew and when he realised the scale of it, he applied for the record 🤖\u2063 \u2063 _____________________________________________\u2063 \u2063 #transformers #autobots #decepticons #collectibles #collection #collector #GWR2020 #guinnessworldrecords #officiallyamazing https://www.instagram.com/p/B6nSOPZhsnZ"
# ]
