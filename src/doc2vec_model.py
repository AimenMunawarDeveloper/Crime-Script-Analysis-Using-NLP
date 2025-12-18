import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class Doc2VecModel:
    
    def __init__(self, vector_size: int = 50, min_count: int = 2, 
                 epochs: int = 100, dm: int = 1, alpha: float = 0.025, 
                 min_alpha: float = 0.00025):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.dm = dm
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.model = None
        self.corpus = None
    
    def create_tagged_documents(self, texts: List[str], ids: List = None) -> List[TaggedDocument]:
        if ids is None:
            ids = list(range(len(texts)))
        
        tagged_docs = []
        for i, text in enumerate(texts):
            tokens = word_tokenize(text.lower())
            tagged_docs.append(TaggedDocument(words=tokens, tags=[ids[i]]))
        
        return tagged_docs
    
    def train(self, corpus: List[TaggedDocument], verbose: bool = True):
        if verbose:
            print(f"Training Doc2Vec model...")
            print(f"  Vector size: {self.vector_size}")
            print(f"  Min count: {self.min_count}")
            print(f"  Epochs: {self.epochs}")
            print(f"  DM mode: {'PV-DM' if self.dm == 1 else 'PV-DBOW'}")
            print(f"  Corpus size: {len(corpus)} documents")
        
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            min_count=self.min_count,
            epochs=self.epochs,
            dm=self.dm,
            alpha=self.alpha,
            min_alpha=self.min_alpha
        )
        
        if verbose:
            print("Building vocabulary...")
        self.model.build_vocab(corpus)
        
        if verbose:
            print(f"  Vocabulary size: {len(self.model.wv)}")
        
        if verbose:
            print("Training model...")
        self.model.train(corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        
        self.corpus = corpus
        
        if verbose:
            print("Training complete!")
    
    def generate_embeddings(self, texts: List[str] = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if texts is None:
            embeddings = []
            for doc in self.corpus:
                doc_id = doc.tags[0]
                embedding = self.model.dv[doc_id]
                embeddings.append(embedding)
            return np.array(embeddings)
        else:
            embeddings = []
            for text in texts:
                tokens = word_tokenize(text.lower())
                embedding = self.model.infer_vector(tokens)
                embeddings.append(embedding)
            return np.array(embeddings)
    
    def compute_similarity_matrix(self, embeddings: np.ndarray = None) -> np.ndarray:
        if embeddings is None:
            embeddings = self.generate_embeddings()
        
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    
    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        self.model = Doc2Vec.load(filepath)
        print(f"Model loaded from {filepath}")
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        df = pd.DataFrame(embeddings)
        df.to_csv(filepath, index=False)
        print(f"Embeddings saved to {filepath}")
    
    def save_similarity_matrix(self, similarity_matrix: np.ndarray, 
                              filepath: str, doc_ids: List = None):
        if doc_ids is not None:
            df = pd.DataFrame(similarity_matrix, index=doc_ids, columns=doc_ids)
        else:
            df = pd.DataFrame(similarity_matrix)
        
        df.to_csv(filepath, index=(doc_ids is not None))
        print(f"Similarity matrix saved to {filepath}")
