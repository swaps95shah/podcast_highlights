import yaml
import pandas as pd

class Selector:
    def __init__(self):
        """Loads configuration."""
        with open('config.yaml') as f:
            configs = yaml.safe_load(f)
            self.params = configs['parameters']['selector']
  
    def create_highlights(self, sent_df, sentences):
        """Function to choose highlight sentences from clustered sentences."""
        def get_sentence(position):
            return sentences[position]

        total_sentences = len(sent_df)
        clusters = sent_df.query('st1_cluster != -1').groupby("st1_cluster")["sentence"].count().sort_values(ascending = False).tolist()
        para_beginnings = sent_df.query('para_pos < 2 & st1_cluster_prob > 0.85')
        highlights = set([0,1,2])
        chunks = {0 : [0, 1, 2]}
        change = True
        count = 1
        while len(highlights) <= 20:
            if change == False:
                break
            change = False
        for cluster in clusters:
            candidates = para_beginnings.query('st1_cluster == '+ str(cluster)).sort_values(by = ['pos'])['pos'].tolist()
            for candidate in candidates:
                if len(sentences[candidate]) < self.params['min_candidate_length']:
                    continue
                if candidate not in highlights:
                    highlights.add(candidate)
                    chunks[count] = [candidate]
                if (candidate + 1) <= (total_sentences-1):
                    highlights.add(candidate + 1)
                    chunks[count].append(candidate+1)
                count += 1
                change = True
                break
    
        chunks = sorted(chunks.values(), key=lambda item: item[0])
        final_hl = {}
        count = 0
        for chunk in chunks:
            final_hl[count] = " ".join(map(get_sentence, chunk))
        count += 1
        return final_hl
  
  
  
