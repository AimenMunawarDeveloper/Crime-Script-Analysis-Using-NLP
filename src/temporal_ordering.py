"""
Temporal Ordering Module
Identifies temporal ordering of actions in scam reports to generate crime scripts
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TemporalOrdering:
    """Class for identifying temporal ordering of actions in scam reports"""
    
    def __init__(self):
        """Initialize temporal ordering class"""
        pass
    
    def calculate_term_positions(self, terms: List[str], documents: List[str]) -> Dict[str, List[int]]:
        """
        Calculate positions of terms in documents
        
        Args:
            terms: List of terms to find
            documents: List of document texts
            
        Returns:
            Dictionary mapping terms to lists of positions
        """
        import re
        
        term_positions = {term: [] for term in terms}
        
        for doc in documents:
            for term in terms:
                # Find all occurrences of term in document
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = re.finditer(pattern, doc, re.IGNORECASE)
                positions = [match.start() for match in matches]
                
                if positions:
                    # Use median position if multiple occurrences
                    median_pos = np.median(positions)
                    term_positions[term].append(median_pos)
                else:
                    # If not found, use a large value
                    term_positions[term].append(float('inf'))
        
        return term_positions
    
    def calculate_median_positions(self, term_positions: Dict[str, List[int]]) -> Dict[str, float]:
        """
        Calculate median positions for each term across documents
        
        Args:
            term_positions: Dictionary mapping terms to lists of positions
            
        Returns:
            Dictionary mapping terms to median positions
        """
        median_positions = {}
        
        for term, positions in term_positions.items():
            # Filter out infinite values
            valid_positions = [p for p in positions if p != float('inf')]
            
            if len(valid_positions) > 0:
                median_positions[term] = np.median(valid_positions)
            else:
                median_positions[term] = float('inf')
        
        return median_positions
    
    def generate_sequence_from_terms(self, 
                                   terms_df: pd.DataFrame,
                                   position_column: str = 'sequence') -> pd.DataFrame:
        """
        Generate sequence from terms ordered by their positions
        
        Args:
            terms_df: DataFrame with terms and their positions
            position_column: Name of column containing positions
            
        Returns:
            DataFrame sorted by position with sequence information
        """
        df = terms_df.copy()
        
        # Sort by position
        df = df.sort_values(position_column).reset_index(drop=True)
        
        # Add sequence index
        df['sequence_index'] = range(len(df))
        
        # Add next term column
        df['next_term'] = ''
        for idx in range(len(df) - 1):
            df.loc[idx, 'next_term'] = df.loc[idx + 1, 'term']
        
        return df
    
    def create_crime_script(self,
                           terms_df: pd.DataFrame,
                           position_column: str = 'sequence',
                           tfidf_column: str = 'tfidf_score') -> pd.DataFrame:
        """
        Create crime script from terms with temporal ordering
        
        Args:
            terms_df: DataFrame with terms, positions, and TF-IDF scores
            position_column: Name of column containing positions
            tfidf_column: Name of column containing TF-IDF scores
            
        Returns:
            DataFrame representing the crime script sequence
        """
        # Generate sequence
        script_df = self.generate_sequence_from_terms(terms_df, position_column)
        
        # Add step descriptions
        script_df['step'] = script_df['sequence_index'] + 1
        script_df['action'] = script_df['term']
        script_df['importance'] = script_df[tfidf_column]
        
        # Select relevant columns
        crime_script = script_df[['step', 'action', 'importance', 'next_term']].copy()
        
        return crime_script
    
    def visualize_sequence_graph(self,
                                sequence_df: pd.DataFrame,
                                term_column: str = 'term',
                                next_term_column: str = 'next_term',
                                weight_column: str = 'tfidf_score',
                                figsize: Tuple[int, int] = (12, 8),
                                save_path: Optional[str] = None):
        """
        Visualize temporal sequence as a directed graph (matching reference implementation)
        
        Args:
            sequence_df: DataFrame with sequence information
            term_column: Name of column containing terms
            next_term_column: Name of column containing next terms
            weight_column: Name of column for edge weights
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        # Create a copy to avoid modifying original
        df = sequence_df.copy()
        
        # Drop the last row (as in reference code)
        if len(df) == 0:
            print("Warning: No data to visualize")
            return
        
        score_last = df.iloc[len(df)-1].get(weight_column, 0.0)
        df = df.iloc[:len(df)-1].copy()
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Create connections between nodes
        for idx, row in df.iterrows():
            term = row[term_column]
            next_term = row[next_term_column]
            weight = row.get(weight_column, 1.0)
            
            if pd.notna(next_term) and next_term != '':
                G.add_edge(term, next_term, weight=weight)
        
        if len(G.edges()) == 0:
            print("Warning: No edges to visualize")
            return
        
        # Calculate node sizes (matching reference: 100 * exp(weight))
        weight_last = 100 * math.exp(score_last)
        node_size = [100 * math.exp(G[u][v]['weight']) for u, v in G.edges()]
        node_size.append(weight_last)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout (circular layout as in reference)
        pos = nx.circular_layout(G)
        # Alternative layouts (commented out in reference):
        # pos = nx.spring_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.spectral_layout(G)
        
        # Plot networks (matching reference style exactly)
        nx.draw_networkx(G, pos,
                        font_size=16,
                        width=2,
                        edge_color='#000000',
                        node_color='#CCCCCC',
                        node_size=node_size[:len(G.nodes())] if len(node_size) >= len(G.nodes()) else [300] * len(G.nodes()),
                        with_labels=False,  # Labels added separately with offset
                        ax=ax,
                        arrowstyle='-|>',
                        arrowsize=30)
        
        # Set background color
        ax.set_facecolor("#FFFFFF")
        
        # Create offset labels with bboxes (matching reference style)
        for key, value in pos.items():
            x, y = value[0] + 0.12, value[1] + 0.1
            ax.text(x, y, s=key,
                   bbox=dict(facecolor='#FFCC99', alpha=0.95),
                   horizontalalignment='center',
                   fontsize=13)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sequence graph saved to {save_path}")
            plt.close()  # Close figure to free memory
        else:
            plt.show()
    
    def generate_consensus_script(self,
                                 similar_scam_texts: List[str],
                                 terms_df: pd.DataFrame,
                                 position_column: str = 'sequence') -> pd.DataFrame:
        """
        Generate consensus crime script from multiple similar scam reports
        
        Args:
            similar_scam_texts: List of texts from similar scam reports
            terms_df: DataFrame with key terms and their TF-IDF scores
            position_column: Name of column containing positions
            
        Returns:
            DataFrame representing consensus crime script
        """
        # Calculate positions for all terms
        terms = terms_df['term'].tolist()
        term_positions = self.calculate_term_positions(terms, similar_scam_texts)
        
        # Calculate median positions
        median_positions = self.calculate_median_positions(term_positions)
        
        # Add median positions to terms DataFrame
        terms_df = terms_df.copy()
        terms_df['median_position'] = terms_df['term'].map(median_positions)
        
        # Generate sequence
        script_df = self.generate_sequence_from_terms(terms_df, 'median_position')
        
        # Create crime script
        crime_script = self.create_crime_script(script_df, 'median_position')
        
        return crime_script
    
    def export_crime_script(self,
                          crime_script_df: pd.DataFrame,
                          filepath: str):
        """
        Export crime script to CSV
        
        Args:
            crime_script_df: DataFrame with crime script
            filepath: Path to save CSV file
        """
        crime_script_df.to_csv(filepath, index=False)
        print(f"Crime script saved to {filepath}")

