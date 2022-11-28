"""
Author: Swapnil S Shah
Email: sss4174@rit.edu
"""

#imports
from hdbscan import HDBSCAN
import yaml

from pathlib import Path

from modules.vectorize import Vectorizer
from modules.chunk import ParagraphChunker
from modules.cluster import HDBScanClustering
from modules.select import Selector


class Highlighter:
    """Class to provide highlight functionality."""
    def __init__(self):
        """Initializes modules."""
        self.vect = Vectorizer()
        self.chunker = ParagraphChunker()
        self.clusterer = HDBScanClustering()
        self.selector = Selector()
    
    def get_highlights(self, transcript):
        """Calling function for pipeline."""
        sentences, embeddings = self.vect.get_embeddings(transcript)
        sent_df = self.chunker.get_para_splits(sentences, embeddings)
        sent_df = self.clusterer.get_clusters(sent_df, embeddings)
        result = self.selector.create_highlights(sent_df, sentences)
        return result
    