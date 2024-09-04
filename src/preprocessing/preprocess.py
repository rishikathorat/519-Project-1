import json
import pandas as pd 
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np 
from sklearn.decomposition import PCA
import joblib

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Loading datasets
def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return pd.json_normalize(data)

# Tokenization with TF-IDF
def tokenize_and_vectorize(text_data):
    # Convert all entries to strings to avoid errors
    text_data = text_data.astype(str)
    
    tokens = [word_tokenize(text) for text in text_data]

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
    
    # Ensure the column 'Natural language explanation' is converted to strings
    tokens, tfidf_matrix = tokenize_and_vectorize(data['Natural language explanation'])
    word_vectors = word2vec_embedding(tokens)
    reduced_vectors = reduce_dimensions(word_vectors)
    
    # Combine features from TF-IDF and reduced Word2Vec vectors
    combined_features = np.hstack((tfidf_matrix.toarray(), reduced_vectors))
    
    np.save('root/data/processed/combined_features.npy', combined_features)
    
    return combined_features

# Call the preprocess function with the dataset path
if __name__ == "__main__":
    combined_features = preprocess_data('data/raw/expunations_annotated_full.json')
    print("Preprocessing complete. Combined features shape:", combined_features.shape)
