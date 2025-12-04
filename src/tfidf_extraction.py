"""
TF-IDF Key Term Extraction Module
Extracts key terms from similar scam reports using TF-IDF
"""

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
    """Class for extracting key terms using TF-IDF"""
    
    def __init__(self, additional_stopwords: List[str] = None):
        """
        Initialize TF-IDF extractor
        
        Args:
            additional_stopwords: Additional stopwords to exclude (e.g., 'ask', 'said', 'claimed')
        """
        self.stopwords = nltk_stopwords.copy()
        if additional_stopwords:
            self.stopwords.update(additional_stopwords)
    
    def remove_stopwords_from_documents(self, doc_list: List[str]) -> List[str]:
        """
        Remove stopwords from a list of documents
        
        Args:
            doc_list: List of text documents
            
        Returns:
            List of documents with stopwords removed
        """
        without_stopwords = []
        for doc in doc_list:
            tokens = word_tokenize(doc.lower())
            filtered_tokens = [token for token in tokens if token not in self.stopwords]
            text = ' '.join(filtered_tokens)
            # Clean up spacing around punctuation
            text = text.replace('< ', '<').replace(' >', '>').replace(' ,', ',').replace(' .', '.')
            without_stopwords.append(text)
        
        return without_stopwords
    
    def extract_ngrams(self, doc_list: List[str], 
                      n_min: int = 1, 
                      n_max: int = 1, 
                      top_n: int = 20,
                      remove_stopwords: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract top N n-grams based on TF-IDF scores
        
        Args:
            doc_list: List of documents
            n_min: Minimum n-gram size (1 for unigrams, 2 for bigrams, etc.)
            n_max: Maximum n-gram size
            top_n: Number of top n-grams to return
            remove_stopwords: Whether to remove stopwords before extraction
            
        Returns:
            Tuple of (term-document matrix DataFrame, ranking DataFrame with top terms)
        """
        # Remove stopwords if requested
        if remove_stopwords:
            processed_docs = self.remove_stopwords_from_documents(doc_list)
        else:
            processed_docs = doc_list
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(ngram_range=(n_min, n_max))
        vectorizer_vectors = vectorizer.fit_transform(processed_docs)
        
        # Get feature names (terms)
        features = vectorizer.get_feature_names_out()
        
        # Create term-document matrix
        term_doc_df = pd.DataFrame(vectorizer_vectors.toarray(), columns=features)
        
        # Calculate TF-IDF scores (sum across all documents)
        term_scores = []
        for term in features:
            score = term_doc_df[term].sum()
            term_scores.append((term, score))
        
        # Create ranking DataFrame
        ranking_df = pd.DataFrame(term_scores, columns=['term', 'tfidf_score'])
        ranking_df = ranking_df.nlargest(top_n, 'tfidf_score').reset_index(drop=True)
        
        return term_doc_df, ranking_df
    
    def arrange_ngrams_by_sequence(self, ranking_df: pd.DataFrame, 
                                  doc_list: List[str]) -> pd.DataFrame:
        """
        Arrange n-grams by their median index positions in documents
        
        This helps identify the temporal ordering of terms in scam reports
        
        Args:
            ranking_df: DataFrame with ranked n-grams (must have 'term' column)
            doc_list: List of documents to search for term positions
            
        Returns:
            DataFrame with 'sequence' column added, sorted by median position
        """
        ranking_df = ranking_df.copy()
        ranking_df['sequence'] = 0.0
        
        for idx, row in ranking_df.iterrows():
            term = row['term']
            start_positions = []
            
            # Find positions of term in each document
            for doc in doc_list:
                # Use regex to find term (case-insensitive, word boundary)
                pattern = r'\b' + re.escape(term) + r'\b'
                match = re.search(pattern, doc, re.IGNORECASE)
                if match:
                    start_positions.append(match.span()[0])
            
            # Calculate median position
            if len(start_positions) > 0:
                median_pos = np.median(start_positions)
                ranking_df.loc[idx, 'sequence'] = round(median_pos, 2)
            else:
                ranking_df.loc[idx, 'sequence'] = float('inf')
        
        # Sort by sequence (median position)
        ranking_df = ranking_df.sort_values('sequence').reset_index(drop=True)
        
        return ranking_df
    
    def create_sequence_graph(self, ranking_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a sequence graph by adding 'next_term' column
        
        This creates edges for temporal ordering visualization
        
        Args:
            ranking_df: DataFrame with terms sorted by sequence
            
        Returns:
            DataFrame with 'next_term' column added
        """
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
        """
        Extract key terms from a group of similar scam reports
        
        Args:
            similar_scam_texts: List of text from similar scam reports
            n_min: Minimum n-gram size
            n_max: Maximum n-gram size
            top_n: Number of top terms to extract
            arrange_by_sequence: Whether to arrange terms by temporal sequence
            
        Returns:
            DataFrame with key terms, TF-IDF scores, and optionally sequence positions
        """
        # Extract n-grams
        term_doc_df, ranking_df = self.extract_ngrams(
            similar_scam_texts,
            n_min=n_min,
            n_max=n_max,
            top_n=top_n,
            remove_stopwords=True
        )
        
        # Arrange by sequence if requested
        if arrange_by_sequence:
            ranking_df = self.arrange_ngrams_by_sequence(ranking_df, similar_scam_texts)
            ranking_df = self.create_sequence_graph(ranking_df)
        
        return ranking_df
    
    def extract_unigrams(self, doc_list: List[str], top_n: int = 20) -> pd.DataFrame:
        """Extract top unigrams"""
        _, ranking_df = self.extract_ngrams(doc_list, n_min=1, n_max=1, top_n=top_n)
        return ranking_df
    
    def extract_bigrams(self, doc_list: List[str], top_n: int = 20) -> pd.DataFrame:
        """Extract top bigrams"""
        _, ranking_df = self.extract_ngrams(doc_list, n_min=2, n_max=2, top_n=top_n)
        return ranking_df
    
    def extract_trigrams(self, doc_list: List[str], top_n: int = 20) -> pd.DataFrame:
        """Extract top trigrams"""
        _, ranking_df = self.extract_ngrams(doc_list, n_min=3, n_max=3, top_n=top_n)
        return ranking_df
    
    def extract_combined_ngrams(self, doc_list: List[str], 
                               top_n_unigrams: int = 10,
                               top_n_bigrams: int = 10,
                               top_n_trigrams: int = 10) -> pd.DataFrame:
        """
        Extract combination of unigrams, bigrams, and trigrams
        
        Args:
            doc_list: List of documents
            top_n_unigrams: Number of top unigrams
            top_n_bigrams: Number of top bigrams
            top_n_trigrams: Number of top trigrams
            
        Returns:
            Combined DataFrame with all n-grams
        """
        unigrams = self.extract_unigrams(doc_list, top_n_unigrams)
        unigrams['ngram_type'] = 'unigram'
        
        bigrams = self.extract_bigrams(doc_list, top_n_bigrams)
        bigrams['ngram_type'] = 'bigram'
        
        trigrams = self.extract_trigrams(doc_list, top_n_trigrams)
        trigrams['ngram_type'] = 'trigram'
        
        # Combine and sort by TF-IDF score
        combined = pd.concat([unigrams, bigrams, trigrams], ignore_index=True)
        combined = combined.sort_values('tfidf_score', ascending=False).reset_index(drop=True)
        
        return combined

