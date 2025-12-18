import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TemporalOrdering:
    
    def __init__(self):
        pass
    
    def calculate_term_positions(self, terms: List[str], documents: List[str]) -> Dict[str, List[int]]:
        import re
        
        term_positions = {term: [] for term in terms}
        
        for doc in documents:
            for term in terms:
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = re.finditer(pattern, doc, re.IGNORECASE)
                positions = [match.start() for match in matches]
                
                if positions:
                    median_pos = np.median(positions)
                    term_positions[term].append(median_pos)
                else:
                    term_positions[term].append(float('inf'))
        
        return term_positions
    
    def calculate_median_positions(self, term_positions: Dict[str, List[int]]) -> Dict[str, float]:
        median_positions = {}
        
        for term, positions in term_positions.items():
            valid_positions = [p for p in positions if p != float('inf')]
            
            if len(valid_positions) > 0:
                median_positions[term] = np.median(valid_positions)
            else:
                median_positions[term] = float('inf')
        
        return median_positions
    
    def generate_sequence_from_terms(self, 
                                   terms_df: pd.DataFrame,
                                   position_column: str = 'sequence') -> pd.DataFrame:
        df = terms_df.copy()
        
        df = df.sort_values(position_column).reset_index(drop=True)
        
        df['sequence_index'] = range(len(df))
        
        df['next_term'] = ''
        for idx in range(len(df) - 1):
            df.loc[idx, 'next_term'] = df.loc[idx + 1, 'term']
        
        return df
    
    def create_crime_script(self,
                           terms_df: pd.DataFrame,
                           position_column: str = 'sequence',
                           tfidf_column: str = 'tfidf_score') -> pd.DataFrame:
        script_df = self.generate_sequence_from_terms(terms_df, position_column)
        
        script_df['step'] = script_df['sequence_index'] + 1
        script_df['action'] = script_df['term']
        script_df['importance'] = script_df[tfidf_column]
        
        crime_script = script_df[['step', 'action', 'importance', 'next_term']].copy()
        
        return crime_script
    
    def visualize_sequence_graph(self,
                                sequence_df: pd.DataFrame,
                                term_column: str = 'term',
                                next_term_column: str = 'next_term',
                                weight_column: str = 'tfidf_score',
                                figsize: Tuple[int, int] = (12, 8),
                                save_path: Optional[str] = None,
                                max_terms: int = 30):
        df = sequence_df.copy()
        
        if len(df) == 0:
            print("Warning: No data to visualize")
            return
        
        if len(df) > max_terms:
            df = df.head(max_terms)
        
        score_last = df.iloc[len(df)-1].get(weight_column, 0.0)
        df_edges = df.iloc[:len(df)-1].copy()
        
        G = nx.DiGraph()
        
        term_to_score = {}
        for idx, row in df.iterrows():
            term = row[term_column]
            score = row.get(weight_column, 0.0)
            term_to_score[term] = score
        
        for idx, row in df_edges.iterrows():
            term = row[term_column]
            next_term = row[next_term_column]
            weight = row.get(weight_column, 1.0)
            
            if pd.notna(next_term) and next_term != '' and next_term in term_to_score:
                G.add_edge(term, next_term, weight=weight)
        
        if len(G.edges()) == 0:
            print("Warning: No edges to visualize")
            return
        
        node_size = []
        for node in G.nodes():
            if node in term_to_score:
                node_size.append(100 * math.exp(term_to_score[node]))
            else:
                node_size.append(300)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        pos = nx.circular_layout(G)
        
        nx.draw_networkx(G, pos,
                        font_size=16,
                        width=2,
                        edge_color='#000000',
                        node_color='#CCCCCC',
                        node_size=node_size,
                        with_labels=False,
                        ax=ax,
                        arrowstyle='-|>',
                        arrowsize=30)
        
        ax.set_facecolor("#FFFFFF")
        
        for key, value in pos.items():
            x, y = value[0] + 0.12, value[1] + 0.1
            ax.text(x, y, s=key,
                   bbox=dict(facecolor='#FFCC99', alpha=0.95),
                   horizontalalignment='center',
                   fontsize=13)
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
                plt.close(fig)
            except Exception as e:
                plt.close(fig)
                raise e
        else:
            plt.show()
    
    def generate_consensus_script(self,
                                 similar_scam_texts: List[str],
                                 terms_df: pd.DataFrame,
                                 position_column: str = 'sequence') -> pd.DataFrame:
        terms = terms_df['term'].tolist()
        term_positions = self.calculate_term_positions(terms, similar_scam_texts)
        
        median_positions = self.calculate_median_positions(term_positions)
        
        terms_df = terms_df.copy()
        terms_df['median_position'] = terms_df['term'].map(median_positions)
        
        script_df = self.generate_sequence_from_terms(terms_df, 'median_position')
        
        crime_script = self.create_crime_script(script_df, 'median_position')
        
        return crime_script
    
    def export_crime_script(self,
                          crime_script_df: pd.DataFrame,
                          filepath: str):
        crime_script_df.to_csv(filepath, index=False)
        print(f"Crime script saved to {filepath}")
