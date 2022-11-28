import nltk

from sentence_transformers import SentenceTransformer


class Vectorizer:
    """Class to expose tokenization and embedding functionality."""
    def __init__(self):
        """Downloads and initializes pre trained models."""
        nltk.download('punkt')
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
    def tokenize_sentences(self, transcript):
        """Tokenizes transcript into sentences."""
        sentences = nltk.tokenize.sent_tokenize(transcript)
        return sentences
  
    def embed(self, sentences):
        """Returns semantic embeddings for sentences."""
        return self.model.encode(sentences)

    def get_embeddings(self, transcript):
        """Calling function to get sentences and embeddings."""
        sentences = self.tokenize_sentences(transcript)
        return sentences, self.embed(sentences)
