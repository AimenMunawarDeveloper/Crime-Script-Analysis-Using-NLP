import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    nltk_stopwords = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk_stopwords = set(stopwords.words('english'))


class TFIDFExtractor:
    
    def __init__(self, additional_stopwords: List[str] = None):
        self.stopwords = nltk_stopwords.copy()
        if additional_stopwords:
            self.stopwords.update(additional_stopwords)
    
    def remove_stopwords_from_documents(self, doc_list: List[str]) -> List[str]:
        without_stopwords = []
        for doc in doc_list:
            tokens = word_tokenize(doc.lower())
            filtered_tokens = [token for token in tokens if token not in self.stopwords]
            text = ' '.join(filtered_tokens)
            text = text.replace('< ', '<').replace(' >', '>').replace(' ,', ',').replace(' .', '.')
            without_stopwords.append(text)
        
        return without_stopwords
    
    def extract_ngrams(self, doc_list: List[str], 
                      n_min: int = 1, 
                      n_max: int = 1, 
                      top_n: int = 20,
                      remove_stopwords: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if remove_stopwords:
            processed_docs = self.remove_stopwords_from_documents(doc_list)
        else:
            processed_docs = doc_list
        
        vectorizer = TfidfVectorizer(ngram_range=(n_min, n_max))
        vectorizer_vectors = vectorizer.fit_transform(processed_docs)
        
        features = vectorizer.get_feature_names_out()
        
        term_doc_df = pd.DataFrame(vectorizer_vectors.toarray(), columns=features)
        
        term_scores = []
        for term in features:
            score = term_doc_df[term].sum()
            term_scores.append((term, score))
        
        ranking_df = pd.DataFrame(term_scores, columns=['term', 'tfidf_score'])
        ranking_df = ranking_df.nlargest(top_n, 'tfidf_score').reset_index(drop=True)
        
        return term_doc_df, ranking_df
    
    def arrange_ngrams_by_sequence(self, ranking_df: pd.DataFrame, 
                                  doc_list: List[str]) -> pd.DataFrame:
        ranking_df = ranking_df.copy()
        ranking_df['sequence'] = 0.0
        
        for idx, row in ranking_df.iterrows():
            term = row['term']
            start_positions = []
            
            for doc in doc_list:
                pattern = r'\b' + re.escape(term) + r'\b'
                match = re.search(pattern, doc, re.IGNORECASE)
                if match:
                    start_positions.append(match.span()[0])
            
            if len(start_positions) > 0:
                median_pos = np.median(start_positions)
                ranking_df.loc[idx, 'sequence'] = round(median_pos, 2)
            else:
                ranking_df.loc[idx, 'sequence'] = float('inf')
        
        ranking_df = ranking_df.sort_values('sequence').reset_index(drop=True)
        
        return ranking_df
    
    def create_sequence_graph(self, ranking_df: pd.DataFrame) -> pd.DataFrame:
        df = ranking_df.copy()
        df['next_term'] = ''
        
        for idx in range(len(df) - 1):
            try:
                df.loc[idx, 'next_term'] = df.loc[idx + 1, 'term']
            except (KeyError, IndexError):
                continue
        
        return df
    
    def extract_key_terms_from_similar_scams(self,
                                            similar_scam_texts: List[str],
                                            n_min: int = 1,
                                            n_max: int = 1,
                                            top_n: int = 20,
                                            arrange_by_sequence: bool = True) -> pd.DataFrame:
        term_doc_df, ranking_df = self.extract_ngrams(
            similar_scam_texts,
            n_min=n_min,
            n_max=n_max,
            top_n=top_n,
            remove_stopwords=True
        )
        
        if arrange_by_sequence:
            ranking_df = self.arrange_ngrams_by_sequence(ranking_df, similar_scam_texts)
            ranking_df = self.create_sequence_graph(ranking_df)
        
        return ranking_df
