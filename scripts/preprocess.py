import os
import json
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from sklearn.decomposition import PCA
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from transformers import AutoTokenizer

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and spell checker
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

tokenizer = AutoTokenizer.from_pretrained("teknium/Llama-3.1-AlternateTokenizer")

# Loading datasets
def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return pd.json_normalize(data)

# Cleaning Text Data
def clean_text(text):
    # Convert to lowercase
    if isinstance(text, str):  # Ensure the text is a string
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Correct misspellings
        corrected_words = [spell.correction(word) for word in text.split()]
        # Lemmatize words
        lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in corrected_words if word])
        return lemmatized_text
    else:
        return ''  # Return an empty string if the text is not a string

# Function to tokenize with Llama 3.1 Tokenizer
def tokenize_with_llama(text_data):
    # Tokenize each text entry using Llama's tokenizer
    tokens = [tokenizer.encode(text, truncation=True, padding='max_length') for text in text_data.astype(str)]
    return tokens

# Tokenization with TF-IDF
def tokenize_and_vectorize(text_data):
    # Convert all entries to strings to avoid errors
    text_data = text_data.astype(str)
    
    # Clean the text data
    text_data = text_data.apply(clean_text)

    # Tokenize the text
    tokens = [word_tokenize(text) for text in text_data]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [[word for word in token_list if word not in stop_words] for token_list in tokens]

    # Initialize and fit the TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=500)
    tfidf_matrix = tfidf.fit_transform([' '.join(token) for token in tokens])

    # Save the TF-IDF model for later use
    joblib.dump(tfidf, 'models/tfidf_model.pkl')

    return tokens, tfidf_matrix

# Word2Vec Embedding
def word2vec_embedding(tokens):
    # Train Word2Vec model
    model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=1, workers=4)
    model.save("models/word2vec.model")

    # Create word vectors
    vectors = [np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(100)], axis=0) for words in tokens]

    return np.array(vectors)

# PCA for dimensionality reduction
def reduce_dimensions(vectors):
    pca = PCA(n_components=50)
    reduced_vectors = pca.fit_transform(vectors)

    # Save the PCA model
    joblib.dump(pca, 'models/pca_model.pkl')

    return reduced_vectors

# Main preprocessing function
def preprocess_data(filepath):
    data = load_data(filepath)

    # Drop rows with missing values in critical columns
    data.dropna(subset=['Natural language explanation'], inplace=True)
    
    # Use Llama 3.1 tokenizer for tokenization
    llama_tokens = tokenize_with_llama(data['Natural language explanation'])  # Use Llama 3.1 tokenizer

    # Ensure the column 'Natural language explanation' is converted to strings
    tokens, tfidf_matrix = tokenize_and_vectorize(data['Natural language explanation'])
    word_vectors = word2vec_embedding(tokens)
    reduced_vectors = reduce_dimensions(word_vectors)
    
    # Combine features from TF-IDF and reduced Word2Vec vectors
    combined_features = np.hstack((tfidf_matrix.toarray(), reduced_vectors))
    
    np.save('data/processed/combined_features.npy', combined_features)
    
    return combined_features

# Call the preprocess function with the dataset path
if __name__ == "__main__":
    combined_features = preprocess_data('data/raw/expunations_annotated_full.json')
    print("Preprocessing complete. Combined features shape:", combined_features.shape)
