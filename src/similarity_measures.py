"""
Similarity Measures Module
Implements Jaccard similarity and combined similarity measures for finding similar scam reports
"""

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import spacy
from spacy.matcher import Matcher
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
    nlp = None


class SimilarityMeasures:
    """Class for computing similarity measures between documents"""
    
    def __init__(self):
        """Initialize the similarity measures class"""
        if nlp is not None:
            self.matcher = Matcher(nlp.vocab)
            # Pattern for noun phrases: one or more nouns
            pattern_1 = [{"POS": "NOUN", "OP": "+"}]
            # Pattern for adjective + one or more nouns
            pattern_2 = [{"POS": "ADJ"}, {"POS": "NOUN", "OP": "+"}]
            # Add patterns (spaCy 3.x+ API: patterns must be in a list)
            # Multiple patterns can be added with the same match_id
            self.matcher.add("Noun_phrases", [pattern_1, pattern_2])
        else:
            self.matcher = None
    
    def get_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text using spaCy
        
        Args:
            text: Input text string
            
        Returns:
            List of noun phrases
        """
        if nlp is None or self.matcher is None:
            # Fallback: return tokenized words if spaCy not available
            return word_tokenize(text.lower())
        
        doc = nlp(text)
        matches = self.matcher(doc)
        
        span = []
        for match_id, start, end in matches:
            span.append(doc[start:end].text.lower())
        
        # If no noun phrases found, return tokenized words
        if len(span) == 0:
            return word_tokenize(text.lower())
        
        return span
    
    def get_jaccard_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Compute Jaccard similarity between two sets of tokens
        
        Jaccard similarity = |A ∩ B| / |A ∪ B|
        
        Args:
            tokens1: First set of tokens
            tokens2: Second set of tokens
            
        Returns:
            Jaccard similarity score (0 to 1)
        """
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if len(union) == 0:
            return 0.0
        
        return float(len(intersection)) / len(union)
    
    def compute_jaccard_similarity_matrix(self, texts: List[str], 
                                         use_noun_phrases: bool = True) -> np.ndarray:
        """
        Compute Jaccard similarity matrix for all document pairs
        
        Args:
            texts: List of preprocessed text strings
            use_noun_phrases: If True, use noun phrases; if False, use all tokens
            
        Returns:
            Numpy array of Jaccard similarity scores (n_documents x n_documents)
        """
        n_docs = len(texts)
        similarity_matrix = np.zeros((n_docs, n_docs))
        
        # Extract tokens/noun phrases for each document
        doc_tokens = []
        for text in texts:
            if use_noun_phrases:
                tokens = self.get_noun_phrases(text)
            else:
                tokens = word_tokenize(text.lower())
            doc_tokens.append(tokens)
        
        # Compute pairwise Jaccard similarity
        for i in range(n_docs):
            for j in range(n_docs):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = self.get_jaccard_similarity(
                        doc_tokens[i], doc_tokens[j]
                    )
        
        return similarity_matrix
    
    def find_similar_documents_hybrid(self, 
                                     cosine_similarities: np.ndarray,
                                     jaccard_similarities: np.ndarray,
                                     doc_ids: List,
                                     query_idx: int,
                                     top_n: int = 10,
                                     cosine_weight: float = 0.5,
                                     jaccard_weight: float = 0.5) -> pd.DataFrame:
        """
        Find similar documents using both cosine and Jaccard similarity
        
        Args:
            cosine_similarities: Cosine similarity matrix
            jaccard_similarities: Jaccard similarity matrix
            doc_ids: List of document IDs
            query_idx: Index of query document
            top_n: Number of similar documents to return
            cosine_weight: Weight for cosine similarity (default 0.5)
            jaccard_weight: Weight for Jaccard similarity (default 0.5)
            
        Returns:
            DataFrame with similar documents and their similarity scores
        """
        # Normalize weights
        total_weight = cosine_weight + jaccard_weight
        cosine_weight = cosine_weight / total_weight
        jaccard_weight = jaccard_weight / total_weight
        
        # Get similarity scores for query document
        cosine_scores = cosine_similarities[query_idx, :]
        jaccard_scores = jaccard_similarities[query_idx, :]
        
        # Compute weighted combined score
        combined_scores = (cosine_weight * cosine_scores + 
                          jaccard_weight * jaccard_scores)
        
        # Create results list (excluding the query document itself)
        results = []
        for i, doc_id in enumerate(doc_ids):
            if i != query_idx:
                results.append({
                    'document_id': doc_id,
                    'cosine_similarity': cosine_scores[i],
                    'jaccard_similarity': jaccard_scores[i],
                    'combined_similarity': combined_scores[i]
                })
        
        # Sort by combined similarity
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('combined_similarity', ascending=False)
        
        return results_df.head(top_n)
    
    def find_similar_documents_by_threshold(self,
                                           cosine_similarities: np.ndarray,
                                           jaccard_similarities: np.ndarray,
                                           doc_ids: List,
                                           query_idx: int,
                                           cosine_threshold: float = None,
                                           jaccard_threshold: float = None,
                                           quantile: float = 0.995) -> pd.DataFrame:
        """
        Find similar documents using similarity thresholds
        
        Args:
            cosine_similarities: Cosine similarity matrix
            jaccard_similarities: Jaccard similarity matrix
            doc_ids: List of document IDs
            query_idx: Index of query document
            cosine_threshold: Cosine similarity threshold (if None, uses quantile)
            jaccard_threshold: Jaccard similarity threshold (if None, uses quantile)
            quantile: Quantile to use for threshold (default 0.995 = top 0.5%)
            
        Returns:
            DataFrame with similar documents
        """
        cosine_scores = cosine_similarities[query_idx, :]
        jaccard_scores = jaccard_similarities[query_idx, :]
        
        # Determine thresholds
        if cosine_threshold is None:
            cosine_threshold = np.quantile(cosine_scores, quantile)
        if jaccard_threshold is None:
            jaccard_threshold = np.quantile(jaccard_scores, quantile)
        
        # Find documents above thresholds
        results = []
        for i, doc_id in enumerate(doc_ids):
            if i != query_idx:
                cosine_above = cosine_scores[i] >= cosine_threshold
                jaccard_above = jaccard_scores[i] >= jaccard_threshold
                
                if cosine_above or jaccard_above:
                    results.append({
                        'document_id': doc_id,
                        'cosine_similarity': cosine_scores[i],
                        'jaccard_similarity': jaccard_scores[i]
                    })
        
        results_df = pd.DataFrame(results)
        
        # Sort by cosine similarity if using cosine threshold, else by Jaccard
        if cosine_threshold is not None:
            results_df = results_df.sort_values('cosine_similarity', ascending=False)
        else:
            results_df = results_df.sort_values('jaccard_similarity', ascending=False)
        
        return results_df
    
    def find_similar_docs_for_new_document(self,
                                          model,
                                          corpus_texts: List[str],
                                          corpus_doc_ids: List,
                                          new_doc: str,
                                          top_n: int = 10,
                                          use_noun_phrases: bool = True) -> pd.DataFrame:
        """
        Find similar documents for a new input document using both Cosine and Jaccard similarity
        
        This is useful for case studies where you want to analyze a new scam report
        
        Args:
            model: Trained Doc2Vec model (gensim Doc2Vec model)
            corpus_texts: List of preprocessed texts from corpus
            corpus_doc_ids: List of document IDs from corpus
            new_doc: New document text to find similar documents for
            top_n: Number of similar documents to return
            use_noun_phrases: Whether to use noun phrases for Jaccard similarity
            
        Returns:
            DataFrame with similar documents, cosine and Jaccard similarity scores
        """
        from nltk.tokenize import word_tokenize
        
        # Infer vector for new document
        new_doc_tokens = word_tokenize(new_doc.lower())
        infer_vector = model.infer_vector(new_doc_tokens)
        
        # Get noun phrases for new document
        if use_noun_phrases:
            candidate_noun_phrases = self.get_noun_phrases(new_doc.lower())
        else:
            candidate_noun_phrases = word_tokenize(new_doc.lower())
        
        # Find similar documents using cosine similarity
        similar_documents = model.dv.most_similar([infer_vector], topn=len(corpus_doc_ids))
        
        # Compute Jaccard similarity for each similar document
        similarity_scores_list = []
        for tag_id, cosine_score in similar_documents:
            # Get document index
            doc_idx = corpus_doc_ids.index(tag_id)
            corpus_text = corpus_texts[doc_idx]
            
            # Get noun phrases for corpus document
            if use_noun_phrases:
                corpus_noun_phrases = self.get_noun_phrases(corpus_text.lower())
            else:
                corpus_noun_phrases = word_tokenize(corpus_text.lower())
            
            # Compute Jaccard similarity
            jaccard_score = self.get_jaccard_similarity(
                candidate_noun_phrases,
                corpus_noun_phrases
            )
            
            similarity_scores_list.append({
                'document_id': tag_id,
                'cosine_similarity': cosine_score,
                'jaccard_similarity': jaccard_score,
                'text': corpus_text
            })
        
        results_df = pd.DataFrame(similarity_scores_list)
        
        # Sort by cosine similarity (or could sort by combined score)
        results_df = results_df.sort_values('cosine_similarity', ascending=False)
        
        return results_df.head(top_n)
    
    def save_jaccard_similarity_matrix(self, similarity_matrix: np.ndarray,
                                     filepath: str, doc_ids: List = None):
        """
        Save Jaccard similarity matrix to CSV file
        
        Args:
            similarity_matrix: Numpy array of similarity scores
            filepath: Path to save CSV file
            doc_ids: List of document IDs for row/column labels (optional)
        """
        if doc_ids is not None:
            df = pd.DataFrame(similarity_matrix, index=doc_ids, columns=doc_ids)
        else:
            df = pd.DataFrame(similarity_matrix)
        
        df.to_csv(filepath, index=(doc_ids is not None))
        print(f"Jaccard similarity matrix saved to {filepath}")

