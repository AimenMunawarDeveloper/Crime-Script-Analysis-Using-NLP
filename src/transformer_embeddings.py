"""
Transformer Embeddings Module
Implements sentence-transformer embeddings (MiniLM/MPNet) for semantic similarity
Includes batch processing, similarity computation, and performance comparison
"""

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
    """
    Class for generating semantic embeddings using sentence-transformers
    Supports MiniLM and MPNet models with batch processing
    """
    
    # Available models
    MODELS = {
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
        'mpnet': 'sentence-transformers/all-mpnet-base-v2',
        'minilm_l12': 'sentence-transformers/all-MiniLM-L12-v2',
        'mpnet_large': 'sentence-transformers/all-mpnet-base-v2'
    }
    
    def __init__(self, model_name: str = 'minilm', device: str = None, 
                 batch_size: int = 32, show_progress: bool = True):
        """
        Initialize Transformer Embeddings model
        
        Args:
            model_name: Name of the model to use ('minilm', 'mpnet', 'minilm_l12', 'mpnet_large')
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
            batch_size: Batch size for encoding (larger = faster but more memory)
            show_progress: Whether to show progress bar during encoding
        """
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
        
        # Auto-detect device if not specified
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
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: List[str], 
                           normalize: bool = True,
                           convert_to_numpy: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to encode
            normalize: Whether to normalize embeddings (L2 normalization)
            convert_to_numpy: Whether to convert to numpy array
            
        Returns:
            Numpy array of embeddings (n_documents x embedding_dim)
        """
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
        """
        Compute cosine similarity matrix for embeddings
        
        Args:
            embeddings: Pre-computed embeddings (n_documents x embedding_dim)
            texts: List of texts (will generate embeddings if embeddings not provided)
            
        Returns:
            Numpy array of similarity scores (n_documents x n_documents)
        """
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
    
    def find_similar_documents(self, query_text: str, 
                              corpus_texts: List[str],
                              corpus_ids: List = None,
                              top_n: int = 10) -> pd.DataFrame:
        """
        Find similar documents for a query text
        
        Args:
            query_text: Query text string
            corpus_texts: List of corpus texts
            corpus_ids: List of corpus document IDs (optional)
            top_n: Number of similar documents to return
            
        Returns:
            DataFrame with similar documents and similarity scores
        """
        # Generate embeddings for query and corpus
        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        corpus_embeddings = self.generate_embeddings(corpus_texts, normalize=True)
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
        
        # Create results
        results = []
        for i, sim_score in enumerate(similarities):
            doc_id = corpus_ids[i] if corpus_ids else i
            results.append({
                'document_id': doc_id,
                'similarity_score': float(sim_score),
                'text': corpus_texts[i]
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('similarity_score', ascending=False)
        
        return results_df.head(top_n)
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str, 
                       doc_ids: List = None):
        """
        Save embeddings to CSV file
        
        Args:
            embeddings: Numpy array of embeddings
            filepath: Path to save CSV file
            doc_ids: List of document IDs (optional)
        """
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
        """
        Save similarity matrix to CSV file
        
        Args:
            similarity_matrix: Numpy array of similarity scores
            filepath: Path to save CSV file
            doc_ids: List of document IDs for row/column labels (optional)
        """
        if doc_ids is not None:
            df = pd.DataFrame(similarity_matrix, index=doc_ids, columns=doc_ids)
        else:
            df = pd.DataFrame(similarity_matrix)
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        df.to_csv(filepath, index=(doc_ids is not None))
        print(f"Similarity matrix saved to {filepath}")


class EmbeddingComparison:
    """
    Class for comparing different embedding methods (Transformer, Doc2Vec, Jaccard)
    """
    
    def __init__(self):
        """Initialize comparison class"""
        self.results = {}
    
    def compare_similarity_matrices(self,
                                   transformer_sim: np.ndarray,
                                   doc2vec_sim: np.ndarray,
                                   jaccard_sim: np.ndarray,
                                   doc_ids: List = None) -> pd.DataFrame:
        """
        Compare similarity matrices from different methods
        
        Args:
            transformer_sim: Transformer-based similarity matrix
            doc2vec_sim: Doc2Vec-based similarity matrix
            jaccard_sim: Jaccard similarity matrix
            doc_ids: List of document IDs (optional)
            
        Returns:
            DataFrame with comparison statistics
        """
        # Ensure all matrices have the same shape
        n_docs = transformer_sim.shape[0]
        assert doc2vec_sim.shape == (n_docs, n_docs), "Doc2Vec matrix shape mismatch"
        assert jaccard_sim.shape == (n_docs, n_docs), "Jaccard matrix shape mismatch"
        
        # Extract upper triangle (excluding diagonal) for comparison
        mask = np.triu(np.ones((n_docs, n_docs)), k=1).astype(bool)
        
        transformer_scores = transformer_sim[mask]
        doc2vec_scores = doc2vec_sim[mask]
        jaccard_scores = jaccard_sim[mask]
        
        # Compute statistics
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
        
        # Compute correlations between methods
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
        
        # Store results
        self.results = {
            'statistics': comparison_df,
            'correlations': correlations
        }
        
        return comparison_df
    
    def find_agreement_pairs(self,
                            transformer_sim: np.ndarray,
                            doc2vec_sim: np.ndarray,
                            jaccard_sim: np.ndarray,
                            doc_ids: List = None,
                            threshold: float = 0.7,
                            top_n: int = 20) -> pd.DataFrame:
        """
        Find document pairs that all three methods agree are similar
        
        Args:
            transformer_sim: Transformer similarity matrix
            doc2vec_sim: Doc2Vec similarity matrix
            jaccard_sim: Jaccard similarity matrix
            doc_ids: List of document IDs
            threshold: Similarity threshold for agreement
            top_n: Number of top pairs to return
            
        Returns:
            DataFrame with agreed similar pairs
        """
        n_docs = transformer_sim.shape[0]
        
        # Find pairs above threshold in all three methods
        agreement_pairs = []
        
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                trans_score = transformer_sim[i, j]
                doc2vec_score = doc2vec_sim[i, j]
                jaccard_score = jaccard_sim[i, j]
                
                if (trans_score >= threshold and 
                    doc2vec_score >= threshold and 
                    jaccard_score >= threshold):
                    
                    doc_id_i = doc_ids[i] if doc_ids else i
                    doc_id_j = doc_ids[j] if doc_ids else j
                    
                    agreement_pairs.append({
                        'document_1': doc_id_i,
                        'document_2': doc_id_j,
                        'transformer_similarity': trans_score,
                        'doc2vec_similarity': doc2vec_score,
                        'jaccard_similarity': jaccard_score,
                        'average_similarity': (trans_score + doc2vec_score + jaccard_score) / 3
                    })
        
        if not agreement_pairs:
            print(f"No pairs found with all similarities >= {threshold}")
            return pd.DataFrame()
        
        agreement_df = pd.DataFrame(agreement_pairs)
        agreement_df = agreement_df.sort_values('average_similarity', ascending=False)
        
        print(f"\nFound {len(agreement_df)} pairs with all similarities >= {threshold}")
        print(f"Top {min(top_n, len(agreement_df))} pairs:")
        print(agreement_df.head(top_n).to_string(index=False))
        
        return agreement_df.head(top_n)
    
    def save_comparison_results(self, comparison_df: pd.DataFrame,
                               filepath: str):
        """Save comparison results to CSV"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        comparison_df.to_csv(filepath, index=False)
        print(f"Comparison results saved to {filepath}")

