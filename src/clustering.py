"""
Clustering Module
Groups similar scam reports using similarity measures
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ScamClustering:
    """Class for clustering similar scam reports"""
    
    def __init__(self):
        """Initialize clustering class"""
        pass
    
    def cluster_by_similarity_threshold(self,
                                       cosine_similarities: np.ndarray,
                                       jaccard_similarities: np.ndarray,
                                       doc_ids: List,
                                       cosine_threshold: float = 0.7,
                                       jaccard_threshold: float = 0.3,
                                       min_cluster_size: int = 2) -> pd.DataFrame:
        """
        Group documents into clusters based on similarity thresholds
        
        Args:
            cosine_similarities: Cosine similarity matrix
            jaccard_similarities: Jaccard similarity matrix
            doc_ids: List of document IDs
            cosine_threshold: Minimum cosine similarity for grouping
            jaccard_threshold: Minimum Jaccard similarity for grouping
            min_cluster_size: Minimum number of documents in a cluster
            
        Returns:
            DataFrame with document_id and cluster_id columns
        """
        n_docs = len(doc_ids)
        clusters = {}
        cluster_id = 0
        assigned = set()
        
        # Create combined similarity matrix (average of cosine and Jaccard)
        combined_similarities = (cosine_similarities + jaccard_similarities) / 2.0
        
        for i in range(n_docs):
            if i in assigned:
                continue
            
            # Find all documents similar to document i
            similar_docs = []
            for j in range(n_docs):
                if i != j:
                    cosine_sim = cosine_similarities[i, j]
                    jaccard_sim = jaccard_similarities[i, j]
                    
                    # Check if both similarities meet thresholds
                    if cosine_sim >= cosine_threshold and jaccard_sim >= jaccard_threshold:
                        similar_docs.append(j)
            
            # If enough similar documents found, create cluster
            if len(similar_docs) >= (min_cluster_size - 1):
                cluster_members = [i] + similar_docs
                clusters[cluster_id] = cluster_members
                
                # Mark all as assigned
                for doc_idx in cluster_members:
                    assigned.add(doc_idx)
                
                cluster_id += 1
        
        # Create result DataFrame
        results = []
        for cid, members in clusters.items():
            for doc_idx in members:
                results.append({
                    'document_id': doc_ids[doc_idx],
                    'cluster_id': cid,
                    'cluster_size': len(members)
                })
        
        # Add unassigned documents as singleton clusters
        for i in range(n_docs):
            if i not in assigned:
                results.append({
                    'document_id': doc_ids[i],
                    'cluster_id': cluster_id,
                    'cluster_size': 1
                })
                cluster_id += 1
        
        return pd.DataFrame(results)
    
    def cluster_using_kmeans(self,
                           embeddings: np.ndarray,
                           doc_ids: List,
                           n_clusters: int = None,
                           max_clusters: int = 10) -> pd.DataFrame:
        """
        Cluster documents using K-means on embeddings
        
        Args:
            embeddings: Document embeddings (n_documents x embedding_dim)
            doc_ids: List of document IDs
            n_clusters: Number of clusters (if None, uses elbow method)
            max_clusters: Maximum clusters to try for elbow method
            
        Returns:
            DataFrame with document_id and cluster_id columns
        """
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(embeddings_scaled, max_clusters)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)
        
        # Create result DataFrame
        results = pd.DataFrame({
            'document_id': doc_ids,
            'cluster_id': cluster_labels
        })
        
        # Add cluster sizes
        cluster_sizes = results['cluster_id'].value_counts().to_dict()
        results['cluster_size'] = results['cluster_id'].map(cluster_sizes)
        
        return results
    
    def cluster_using_dbscan(self,
                            embeddings: np.ndarray,
                            doc_ids: List,
                            eps: float = 0.5,
                            min_samples: int = 2) -> pd.DataFrame:
        """
        Cluster documents using DBSCAN
        
        Args:
            embeddings: Document embeddings
            doc_ids: List of document IDs
            eps: Maximum distance between samples in same cluster
            min_samples: Minimum number of samples in a cluster
            
        Returns:
            DataFrame with document_id and cluster_id columns
        """
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(embeddings_scaled)
        
        # Create result DataFrame
        results = pd.DataFrame({
            'document_id': doc_ids,
            'cluster_id': cluster_labels
        })
        
        # Add cluster sizes (excluding noise points with cluster_id=-1)
        cluster_sizes = results[results['cluster_id'] != -1]['cluster_id'].value_counts().to_dict()
        results['cluster_size'] = results['cluster_id'].map(cluster_sizes).fillna(1)
        
        return results
    
    def _find_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int = 10) -> int:
        """
        Find optimal number of clusters using elbow method
        
        Args:
            embeddings: Scaled embeddings
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Optimal number of clusters
        """
        inertias = []
        k_range = range(2, min(max_clusters + 1, len(embeddings)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection: find point with maximum curvature
        if len(inertias) < 2:
            return 2
        
        # Calculate rate of change
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        
        if len(second_diffs) > 0:
            elbow_idx = np.argmax(second_diffs) + 2
            return k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[-1]
        else:
            return k_range[len(k_range) // 2]
    
    def get_cluster_statistics(self, cluster_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics for each cluster
        
        Args:
            cluster_df: DataFrame with cluster assignments
            
        Returns:
            DataFrame with cluster statistics
        """
        stats = cluster_df.groupby('cluster_id').agg({
            'document_id': 'count',
            'cluster_size': 'first'
        }).rename(columns={'document_id': 'num_documents'})
        
        stats = stats.reset_index()
        stats = stats.sort_values('num_documents', ascending=False)
        
        return stats
    
    def get_cluster_members(self, cluster_df: pd.DataFrame, cluster_id: int) -> List:
        """
        Get list of document IDs in a specific cluster
        
        Args:
            cluster_df: DataFrame with cluster assignments
            cluster_id: ID of cluster
            
        Returns:
            List of document IDs in the cluster
        """
        cluster_members = cluster_df[cluster_df['cluster_id'] == cluster_id]['document_id'].tolist()
        return cluster_members

