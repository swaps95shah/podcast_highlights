import math
import yaml
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema


class ParagraphChunker:
    """Class to analyse text and determine appropriate paragraph splits."""
    def __init__(self):
        """Loads parameters."""
        with open('config.yaml') as f:
            configs = yaml.safe_load(f)
            self.params = configs['parameters']['chunker']
  
    def activate_similarities(self, similarities, p_size):
        """Get list of weighted sums of activated sentence similarities."""
        def rev_sigmoid(x):
            return (1 / (1 + math.exp(0.5*x)))
        x = np.linspace(-10,10,p_size)
        y = np.vectorize(rev_sigmoid) 
        activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
        diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
        diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
        diagonals = np.stack(diagonals)
        diagonals = diagonals * activation_weights.reshape(-1,1)
        activated_similarities = np.sum(diagonals, axis=0)
        return activated_similarities

    def get_para_splits(self, sentences, embeddings):
        """Returns paragraph splits and sentence location within it's paragraph."""
        similarities = cosine_similarity(embeddings)
        activated_similarities = self.activate_similarities(similarities, p_size=10)
        minimas = argrelextrema(activated_similarities, np.less, order=4)
        sent_details = []
        split_points = set([each for each in minimas[0]])
        para_num = 0
        para_pos = 0
        for num, sent in enumerate(sentences):
            curr = {}
            curr['sentence'] = sent
            curr['pos'] = num
            curr['para_num'] = para_num
            curr['para_pos'] = para_pos 
            if num in split_points:
                para_num += 1
                para_pos = 0
                curr['para_num'] = para_num
                curr['para_pos'] = para_pos 
            sent_details.append(curr)
            para_pos += 1
        sent_df = pd.json_normalize(sent_details)
        return sent_df
