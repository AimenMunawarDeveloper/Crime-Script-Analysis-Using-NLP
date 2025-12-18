import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ScamClustering:
    
    def __init__(self):
        pass
    
    def cluster_by_similarity_threshold(self,
                                       similarity_matrix: np.ndarray,
                                       doc_ids: List,
                                       cosine_threshold: float = 0.7,
                                       jaccard_threshold: float = 0.1,
                                       jaccard_matrix: np.ndarray = None,
                                       min_cluster_size: int = 2) -> pd.DataFrame:
        n_docs = len(doc_ids)
        visited = np.zeros(n_docs, dtype=bool)
        cluster_id = 0
        cluster_labels = np.full(n_docs, -1, dtype=int)
        
        for i in range(n_docs):
            if visited[i]:
                continue
            
            cluster_members = [i]
            visited[i] = True
            
            for j in range(i + 1, n_docs):
                if visited[j]:
                    continue
                
                cosine_sim = similarity_matrix[i, j]
                
                meets_cosine = cosine_sim >= cosine_threshold
                meets_jaccard = True
                
                if jaccard_matrix is not None:
                    jaccard_sim = jaccard_matrix[i, j]
                    meets_jaccard = jaccard_sim >= jaccard_threshold
                
                if meets_cosine and meets_jaccard:
                    cluster_members.append(j)
                    visited[j] = True
            
            if len(cluster_members) >= min_cluster_size:
                for member_idx in cluster_members:
                    cluster_labels[member_idx] = cluster_id
                cluster_id += 1
        
        unclustered = cluster_labels == -1
        for idx in np.where(unclustered)[0]:
            cluster_labels[idx] = cluster_id
            cluster_id += 1
        
        results = pd.DataFrame({
            'document_id': doc_ids,
            'cluster_id': cluster_labels
        })
        
        cluster_sizes = results['cluster_id'].value_counts().to_dict()
        results['cluster_size'] = results['cluster_id'].map(cluster_sizes)
        
        return results
    
    def cluster_using_similarity_graph(self,
                                      similarity_matrix: np.ndarray,
                                      doc_ids: List,
                                      threshold: float = 0.7,
                                      jaccard_matrix: np.ndarray = None,
                                      jaccard_threshold: float = 0.1,
                                      min_cluster_size: int = 2) -> pd.DataFrame:
        import networkx as nx
        
        n_docs = len(doc_ids)
        G = nx.Graph()
        G.add_nodes_from(range(n_docs))
        
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                cosine_sim = similarity_matrix[i, j]
                
                if cosine_sim >= threshold:
                    if jaccard_matrix is not None:
                        jaccard_sim = jaccard_matrix[i, j]
                        if jaccard_sim >= jaccard_threshold:
                            G.add_edge(i, j, weight=cosine_sim)
                    else:
                        G.add_edge(i, j, weight=cosine_sim)
        
        clusters = list(nx.connected_components(G))
        
        cluster_labels = np.full(n_docs, -1, dtype=int)
        cluster_id = 0
        
        for cluster in clusters:
            if len(cluster) >= min_cluster_size:
                for node in cluster:
                    cluster_labels[node] = cluster_id
                cluster_id += 1
        
        unclustered = cluster_labels == -1
        for idx in np.where(unclustered)[0]:
            cluster_labels[idx] = cluster_id
            cluster_id += 1
        
        results = pd.DataFrame({
            'document_id': doc_ids,
            'cluster_id': cluster_labels
        })
        
        cluster_sizes = results['cluster_id'].value_counts().to_dict()
        results['cluster_size'] = results['cluster_id'].map(cluster_sizes)
        
        return results
    
    def get_cluster_statistics(self, cluster_df: pd.DataFrame) -> pd.DataFrame:
        stats = cluster_df.groupby('cluster_id').agg({
            'document_id': 'count',
            'cluster_size': 'first'
        }).rename(columns={'document_id': 'num_documents'})
        
        stats = stats.reset_index()
        stats = stats.sort_values('num_documents', ascending=False)
        
        return stats
    
    def get_cluster_members(self, cluster_df: pd.DataFrame, cluster_id: int) -> List:
        cluster_members = cluster_df[cluster_df['cluster_id'] == cluster_id]['document_id'].tolist()
        return cluster_members

