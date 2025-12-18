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
    
    def __init__(self):
        if nlp is not None:
            self.matcher = Matcher(nlp.vocab)
            pattern_1 = [{"POS": "NOUN", "OP": "+"}]
            pattern_2 = [{"POS": "ADJ"}, {"POS": "NOUN", "OP": "+"}]
            self.matcher.add("Noun_phrases", [pattern_1, pattern_2])
        else:
            self.matcher = None
    
    def get_noun_phrases(self, text: str) -> List[str]:
        if nlp is None or self.matcher is None:
            return word_tokenize(text.lower())
        
        doc = nlp(text)
        matches = self.matcher(doc)
        
        span = []
        for match_id, start, end in matches:
            span.append(doc[start:end].text.lower())
        
        if len(span) == 0:
            return word_tokenize(text.lower())
        
        return span
    
    def get_jaccard_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if len(union) == 0:
            return 0.0
        
        return float(len(intersection)) / len(union)
    
    def compute_jaccard_similarity_matrix(self, texts: List[str], 
                                         use_noun_phrases: bool = True) -> np.ndarray:
        n_docs = len(texts)
        similarity_matrix = np.zeros((n_docs, n_docs))
        
        doc_tokens = []
        for text in texts:
            if use_noun_phrases:
                tokens = self.get_noun_phrases(text)
            else:
                tokens = word_tokenize(text.lower())
            doc_tokens.append(tokens)
        
        for i in range(n_docs):
            for j in range(n_docs):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = self.get_jaccard_similarity(
                        doc_tokens[i], doc_tokens[j]
                    )
        
        return similarity_matrix
    
    def find_similar_docs_for_new_document(self,
                                          model=None,
                                          corpus_texts: List[str] = None,
                                          corpus_doc_ids: List = None,
                                          new_doc: str = None,
                                          top_n: int = 10,
                                          use_noun_phrases: bool = True,
                                          transformer_model=None,
                                          corpus_embeddings: np.ndarray = None,
                                          use_transformer: bool = True) -> pd.DataFrame:
        from nltk.tokenize import word_tokenize
        from sklearn.metrics.pairwise import cosine_similarity
        
        if corpus_texts is None or corpus_doc_ids is None or new_doc is None:
            raise ValueError("corpus_texts, corpus_doc_ids, and new_doc are required")
        
        if use_transformer and transformer_model is not None:
            new_doc_embedding = transformer_model.model.encode(
                [new_doc],
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            
            if corpus_embeddings is None:
                corpus_embeddings = transformer_model.generate_embeddings(corpus_texts, normalize=True)
            
            similarities = cosine_similarity(new_doc_embedding, corpus_embeddings)[0]
            
            similarity_scores_list = []
            for i, sim_score in enumerate(similarities):
                similarity_scores_list.append({
                    'document_id': corpus_doc_ids[i],
                    'transformer_similarity': float(sim_score),
                    'text': corpus_texts[i]
                })
            
            results_df = pd.DataFrame(similarity_scores_list)
            results_df = results_df.sort_values('transformer_similarity', ascending=False)
            
            if use_noun_phrases:
                candidate_noun_phrases = self.get_noun_phrases(new_doc.lower())
            else:
                candidate_noun_phrases = word_tokenize(new_doc.lower())
            
            jaccard_scores = []
            for idx, row in results_df.iterrows():
                corpus_text = row['text']
                if use_noun_phrases:
                    corpus_noun_phrases = self.get_noun_phrases(corpus_text.lower())
                else:
                    corpus_noun_phrases = word_tokenize(corpus_text.lower())
                
                jaccard_score = self.get_jaccard_similarity(
                    candidate_noun_phrases,
                    corpus_noun_phrases
                )
                jaccard_scores.append(jaccard_score)
            
            results_df['jaccard_similarity'] = jaccard_scores
            results_df['cosine_similarity'] = results_df['transformer_similarity']
            
            return results_df.head(top_n)
        
        elif model is not None:
            new_doc_tokens = word_tokenize(new_doc.lower())
            infer_vector = model.infer_vector(new_doc_tokens)
            
            if use_noun_phrases:
                candidate_noun_phrases = self.get_noun_phrases(new_doc.lower())
            else:
                candidate_noun_phrases = word_tokenize(new_doc.lower())
            
            similar_documents = model.dv.most_similar([infer_vector], topn=len(corpus_doc_ids))
            
            similarity_scores_list = []
            for tag_id, cosine_score in similar_documents:
                doc_idx = corpus_doc_ids.index(tag_id)
                corpus_text = corpus_texts[doc_idx]
                
                if use_noun_phrases:
                    corpus_noun_phrases = self.get_noun_phrases(corpus_text.lower())
                else:
                    corpus_noun_phrases = word_tokenize(corpus_text.lower())
                
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
            results_df = results_df.sort_values('cosine_similarity', ascending=False)
            
            return results_df.head(top_n)
        else:
            raise ValueError("Either transformer_model or model (Doc2Vec) must be provided")
    
    def save_jaccard_similarity_matrix(self, similarity_matrix: np.ndarray,
                                     filepath: str, doc_ids: List = None):
        if doc_ids is not None:
            df = pd.DataFrame(similarity_matrix, index=doc_ids, columns=doc_ids)
        else:
            df = pd.DataFrame(similarity_matrix)
        
        df.to_csv(filepath, index=(doc_ids is not None))
        print(f"Jaccard similarity matrix saved to {filepath}")
