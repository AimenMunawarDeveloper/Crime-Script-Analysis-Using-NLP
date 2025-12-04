"""
Case Study Module
Demonstrates how to analyze a new scam report by finding similar reports,
extracting key terms, and generating crime scripts
"""

import os
import pandas as pd
import numpy as np
from doc2vec_model import Doc2VecModel
from similarity_measures import SimilarityMeasures
from tfidf_extraction import TFIDFExtractor
from temporal_ordering import TemporalOrdering
from preprocessing import TextPreprocessor
from typing import List, Optional


class CaseStudy:
    """Class for performing case study analysis on new scam reports"""
    
    def __init__(self, model_path: str, preprocessed_data_path: str):
        """
        Initialize case study with trained model and preprocessed data
        
        Args:
            model_path: Path to trained Doc2Vec model
            preprocessed_data_path: Path to preprocessed dataset CSV
        """
        # Load model
        self.doc2vec = Doc2VecModel()
        self.doc2vec.load_model(model_path)
        
        # Load preprocessed data
        self.df = pd.read_csv(preprocessed_data_path)
        
        # Initialize components
        self.similarity_measures = SimilarityMeasures()
        self.tfidf_extractor = TFIDFExtractor(
            additional_stopwords=['ask', 'said', 'say', 'asked', 'claimed', 'told', 'got', 'tell', 'get']
        )
        self.temporal_ordering = TemporalOrdering()
        self.preprocessor = TextPreprocessor()
        
        # Prepare corpus
        if 'submission_id' in self.df.columns:
            self.doc_ids = self.df['submission_id'].tolist()
        else:
            self.doc_ids = self.df.index.tolist()
        
        text_col = 'lemmatised' if 'lemmatised' in self.df.columns else 'preprocessed_text'
        self.corpus_texts = self.df[text_col].fillna('').astype(str).tolist()
    
    def analyze_new_scam_report(self,
                               new_doc: str,
                               similarity_metric: str = 'cosine',
                               quantile: float = 0.995,
                               top_n: int = 20,
                               n_min: int = 1,
                               n_max: int = 1,
                               top_ngrams: int = 20) -> dict:
        """
        Analyze a new scam report: find similar reports, extract key terms, generate script
        
        Args:
            new_doc: New scam report text
            similarity_metric: 'cosine', 'jaccard', or 'hybrid'
            quantile: Quantile threshold for filtering similar documents (0.995 = top 0.5%)
            top_n: Number of top similar documents to use
            n_min: Minimum n-gram size for TF-IDF
            n_max: Maximum n-gram size for TF-IDF
            top_ngrams: Number of top n-grams to extract
            
        Returns:
            Dictionary with analysis results
        """
        # Preprocess new document
        preprocessed_new_doc = self.preprocessor.preprocess(new_doc)
        preprocessed_new_doc = self.preprocessor.remove_stopwords(preprocessed_new_doc)
        preprocessed_new_doc = self.preprocessor.lemmatise(preprocessed_new_doc)
        
        # Find similar documents
        similar_docs_df = self.similarity_measures.find_similar_docs_for_new_document(
            self.doc2vec.model,
            self.corpus_texts,
            self.doc_ids,
            preprocessed_new_doc,
            top_n=len(self.doc_ids),
            use_noun_phrases=True
        )
        
        # Filter by quantile threshold
        if similarity_metric.lower() == 'cosine':
            threshold = similar_docs_df['cosine_similarity'].quantile(quantile)
            filtered_docs = similar_docs_df[
                similar_docs_df['cosine_similarity'] >= threshold
            ].sort_values('cosine_similarity', ascending=False)
        elif similarity_metric.lower() == 'jaccard':
            threshold = similar_docs_df['jaccard_similarity'].quantile(quantile)
            filtered_docs = similar_docs_df[
                similar_docs_df['jaccard_similarity'] >= threshold
            ].sort_values('jaccard_similarity', ascending=False)
        else:  # hybrid
            # Use combined score
            filtered_docs = similar_docs_df.sort_values(
                ['jaccard_similarity', 'cosine_similarity'],
                ascending=[False, False]
            ).head(top_n)
        
        # Limit to top_n
        filtered_docs = filtered_docs.head(top_n)
        
        if len(filtered_docs) == 0:
            return {
                'similar_documents': pd.DataFrame(),
                'key_terms': pd.DataFrame(),
                'crime_script': pd.DataFrame(),
                'message': 'No similar documents found above threshold'
            }
        
        # Get texts of similar documents
        similar_texts = filtered_docs['text'].tolist()
        
        # Extract key terms
        key_terms = self.tfidf_extractor.extract_key_terms_from_similar_scams(
            similar_texts,
            n_min=n_min,
            n_max=n_max,
            top_n=top_ngrams,
            arrange_by_sequence=True
        )
        
        # Generate crime script
        crime_script = self.temporal_ordering.generate_consensus_script(
            similar_texts,
            key_terms,
            position_column='sequence'
        )
        
        return {
            'similar_documents': filtered_docs,
            'key_terms': key_terms,
            'crime_script': crime_script,
            'num_similar_docs': len(filtered_docs)
        }
    
    def visualize_case_study(self,
                           analysis_results: dict,
                           save_dir: str = None):
        """
        Visualize case study results
        
        Args:
            analysis_results: Results from analyze_new_scam_report()
            save_dir: Directory to save visualizations (optional)
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Visualize sequence graph
        if 'key_terms' in analysis_results and len(analysis_results['key_terms']) > 0:
            key_terms = analysis_results['key_terms']
            if 'next_term' in key_terms.columns:
                viz_path = os.path.join(save_dir, "case_study_sequence_graph.png") if save_dir else None
                self.temporal_ordering.visualize_sequence_graph(
                    key_terms,
                    term_column='term',
                    next_term_column='next_term',
                    weight_column='tfidf_score',
                    figsize=(14, 10),
                    save_path=viz_path
                )
    
    def export_case_study_results(self,
                                 analysis_results: dict,
                                 output_dir: str):
        """
        Export case study results to CSV files
        
        Args:
            analysis_results: Results from analyze_new_scam_report()
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if 'similar_documents' in analysis_results:
            similar_path = os.path.join(output_dir, "similar_documents.csv")
            analysis_results['similar_documents'].to_csv(similar_path, index=False)
            print(f"Similar documents saved to {similar_path}")
        
        if 'key_terms' in analysis_results:
            key_terms_path = os.path.join(output_dir, "key_terms.csv")
            analysis_results['key_terms'].to_csv(key_terms_path, index=False)
            print(f"Key terms saved to {key_terms_path}")
        
        if 'crime_script' in analysis_results:
            script_path = os.path.join(output_dir, "crime_script.csv")
            analysis_results['crime_script'].to_csv(script_path, index=False)
            print(f"Crime script saved to {script_path}")


def run_case_study_example():
    """
    Example function demonstrating case study usage
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    model_path = os.path.join(project_root, "Trained Models", "scam_doc2vec_model.model")
    data_path = os.path.join(project_root, "Data Set", "scam_data_preprocessed.csv")
    output_dir = os.path.join(project_root, "Analysis Results", "case_study_example")
    
    # Initialize case study
    case_study = CaseStudy(model_path, data_path)
    
    # Example new scam report
    new_scam = """
    I received a scam call. It was an automated voice from the Singapore High Court, 
    stating that I have an outstanding summon. I was asked to pay it.
    """
    
    # Analyze
    results = case_study.analyze_new_scam_report(
        new_scam,
        similarity_metric='cosine',
        quantile=0.995,
        top_n=20
    )
    
    # Export results
    case_study.export_case_study_results(results, output_dir)
    
    # Visualize
    case_study.visualize_case_study(results, output_dir)
    
    print(f"\nCase study complete! Results saved to {output_dir}")
    print(f"Found {results['num_similar_docs']} similar documents")
    
    return results


if __name__ == "__main__":
    run_case_study_example()

