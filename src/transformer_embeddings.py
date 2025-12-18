import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional, Dict
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Please install it using: pip install sentence-transformers")


class TransformerEmbeddings:
    
    MODELS = {
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
        'mpnet': 'sentence-transformers/all-mpnet-base-v2',
        'minilm_l12': 'sentence-transformers/all-MiniLM-L12-v2',
        'mpnet_large': 'sentence-transformers/all-mpnet-base-v2'
    }
    
    def __init__(self, model_name: str = 'minilm', device: str = None, 
                 batch_size: int = 32, show_progress: bool = True):
        if not TRANSFORMER_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it using: pip install sentence-transformers"
            )
        
        if model_name not in self.MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Available models: {list(self.MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.model_path = self.MODELS[model_name]
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
        
        self.device = device
        
        print(f"Loading transformer model: {model_name} ({self.model_path})")
        print(f"Device: {device}")
        
        try:
            self.model = SentenceTransformer(self.model_path, device=device)
            print(f"Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: List[str], 
                           normalize: bool = True,
                           convert_to_numpy: bool = True) -> np.ndarray:
        if not texts:
            raise ValueError("Text list is empty")
        
        print(f"Generating embeddings for {len(texts)} documents...")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Normalize: {normalize}")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=convert_to_numpy
            )
            
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            
            print(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {e}")
    
    def compute_similarity_matrix(self, embeddings: np.ndarray = None,
                                 texts: List[str] = None) -> np.ndarray:
        if embeddings is None:
            if texts is None:
                raise ValueError("Either embeddings or texts must be provided")
            embeddings = self.generate_embeddings(texts)
        
        print("Computing cosine similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"  Score range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
        print(f"  Mean score: {similarity_matrix.mean():.4f}")
        
        return similarity_matrix
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str, 
                       doc_ids: List = None):
        if doc_ids is not None:
            df = pd.DataFrame(embeddings, index=doc_ids)
            df.index.name = 'document_id'
        else:
            df = pd.DataFrame(embeddings)
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        df.to_csv(filepath, index=(doc_ids is not None))
        print(f"Embeddings saved to {filepath}")
    
    def save_similarity_matrix(self, similarity_matrix: np.ndarray,
                              filepath: str, doc_ids: List = None):
        if doc_ids is not None:
            df = pd.DataFrame(similarity_matrix, index=doc_ids, columns=doc_ids)
        else:
            df = pd.DataFrame(similarity_matrix)
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        df.to_csv(filepath, index=(doc_ids is not None))
        print(f"Similarity matrix saved to {filepath}")
    
    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("No model to save. Initialize the model first.")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        self.model.save(filepath)
        print(f"Transformer model saved to {filepath}")
    
    def load_model(self, filepath: str):
        if not TRANSFORMER_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it using: pip install sentence-transformers"
            )
        
        try:
            self.model = SentenceTransformer(filepath)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Transformer model loaded from {filepath}")
            print(f"Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {filepath}: {e}")


class EmbeddingComparison:
    
    def __init__(self):
        self.results = {}
    
    def compare_similarity_matrices(self,
                                   transformer_sim: np.ndarray,
                                   doc2vec_sim: np.ndarray,
                                   jaccard_sim: np.ndarray,
                                   doc_ids: List = None) -> pd.DataFrame:
        n_docs = transformer_sim.shape[0]
        assert doc2vec_sim.shape == (n_docs, n_docs), "Doc2Vec matrix shape mismatch"
        assert jaccard_sim.shape == (n_docs, n_docs), "Jaccard matrix shape mismatch"
        
        mask = np.triu(np.ones((n_docs, n_docs)), k=1).astype(bool)
        
        transformer_scores = transformer_sim[mask]
        doc2vec_scores = doc2vec_sim[mask]
        jaccard_scores = jaccard_sim[mask]
        
        comparison_stats = {
            'Method': ['Transformer', 'Doc2Vec', 'Jaccard'],
            'Mean': [
                transformer_scores.mean(),
                doc2vec_scores.mean(),
                jaccard_scores.mean()
            ],
            'Std': [
                transformer_scores.std(),
                doc2vec_scores.std(),
                jaccard_scores.std()
            ],
            'Min': [
                transformer_scores.min(),
                doc2vec_scores.min(),
                jaccard_scores.min()
            ],
            'Max': [
                transformer_scores.max(),
                doc2vec_scores.max(),
                jaccard_scores.max()
            ],
            'Median': [
                np.median(transformer_scores),
                np.median(doc2vec_scores),
                np.median(jaccard_scores)
            ],
            'Q25': [
                np.percentile(transformer_scores, 25),
                np.percentile(doc2vec_scores, 25),
                np.percentile(jaccard_scores, 25)
            ],
            'Q75': [
                np.percentile(transformer_scores, 75),
                np.percentile(doc2vec_scores, 75),
                np.percentile(jaccard_scores, 75)
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_stats)
        
        correlations = {
            'Transformer-Doc2Vec': np.corrcoef(transformer_scores, doc2vec_scores)[0, 1],
            'Transformer-Jaccard': np.corrcoef(transformer_scores, jaccard_scores)[0, 1],
            'Doc2Vec-Jaccard': np.corrcoef(doc2vec_scores, jaccard_scores)[0, 1]
        }
        
        print("\n" + "=" * 60)
        print("Embedding Method Comparison")
        print("=" * 60)
        print("\nSimilarity Score Statistics:")
        print(comparison_df.to_string(index=False))
        
        print("\nCorrelations between methods:")
        for pair, corr in correlations.items():
            print(f"  {pair}: {corr:.4f}")
        
        self.results = {
            'statistics': comparison_df,
            'correlations': correlations
        }
        
        return comparison_df
    
    def save_comparison_results(self, comparison_df: pd.DataFrame,
                               filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        comparison_df.to_csv(filepath, index=False)
        print(f"Comparison results saved to {filepath}")
