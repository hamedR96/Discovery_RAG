import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources if they haven't been downloaded already
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_documents(documents):
    """
    Preprocesses a list of documents by lowercasing, removing punctuation,
    and stopwords, and lemmatizing the words.

    :param documents: A list of strings where each string is a document.
    :return: A list of preprocessed documents.
    """
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Define stopwords list
    stop_words = set(stopwords.words('english'))

    # Preprocess each document
    preprocessed_docs = []
    for doc in documents:
        # Lowercase
        doc = doc.lower()

        # Remove punctuation
        doc = doc.translate(str.maketrans('', '', string.punctuation))

        # Tokenize
        tokens = word_tokenize(doc)

        # Remove stopwords and lemmatize
        cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

        # Join tokens back into a string
        preprocessed_doc = ' '.join(cleaned_tokens)
        preprocessed_docs.append(preprocessed_doc)

    return preprocessed_docs
