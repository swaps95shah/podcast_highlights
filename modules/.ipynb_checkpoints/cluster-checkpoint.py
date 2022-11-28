import yaml
import hdbscan
import pandas as pd
import numpy as np
import umap.umap_ as umap

class HDBScanClustering:
    
    def __init__(self):
        with open('config.yaml') as f:
            configs = yaml.safe_load(f)
            self.params = configs['parameters']['clustering']
  
    def generate_umap_embeddings(self, sentence_embeddings):
        """Performs dimensionality reduction."""
        umap_embeddings = (umap.UMAP(n_neighbors = self.params['n_neighbors'], 
                                     n_components = self.params['n_components'],
                                     random_state = self.params['random_state'],
                                     metric= self.params['umap_metric']).fit_transform(sentence_embeddings))
        return umap_embeddings

    def generate_clusters(self, embeddings):
        """Generate HDBSCAN cluster object."""
        clusters = hdbscan.HDBSCAN(min_cluster_size = self.params['min_cluster_size'],
                                   metric=self.params['cluster_metric'], 
                                   cluster_selection_method=self.params['cluster_selection_method']).fit(embeddings)
        return clusters
  
    def get_clusters(self, sent_df, embeddings):
        """Calling function for dimensionality reduction and clustering."""
        umap_embeddings = self.generate_umap_embeddings(embeddings)
        vectorized = pd.get_dummies(sent_df, columns=['para_num', 'para_pos'])
        vectorized = vectorized.drop('sentence', axis=1)
        cl_embeddings = np.concatenate((vectorized.to_numpy(), umap_embeddings),axis=1)
        clusters = self.generate_clusters(cl_embeddings)
        cluster_details = []
        for idx in range(len(sent_df)):
            cluster_dict = {'st1_cluster': clusters.labels_[idx], 
                            'st1_cluster_prob': clusters.probabilities_[idx]}
            cluster_details.append(cluster_dict)
        sent_df = sent_df.join(pd.json_normalize(cluster_details))
        return sent_df
    

  

  
