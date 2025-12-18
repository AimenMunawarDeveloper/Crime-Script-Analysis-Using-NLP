import os
import pandas as pd
import numpy as np
from doc2vec_model import Doc2VecModel
from similarity_measures import SimilarityMeasures
from tfidf_extraction import TFIDFExtractor
from temporal_ordering import TemporalOrdering
from preprocessing import TextPreprocessor
from transformer_embeddings import TransformerEmbeddings
from typing import List, Optional


class CaseStudy:
    
    def __init__(self, preprocessed_data_path: str, 
                 transformer_model_path: str = None,
                 doc2vec_model_path: str = None,
                 use_transformer: bool = True):
        self.df = pd.read_csv(preprocessed_data_path)
        
        self.similarity_measures = SimilarityMeasures()
        self.tfidf_extractor = TFIDFExtractor(
            additional_stopwords=['ask', 'said', 'say', 'asked', 'claimed', 'told', 'got', 'tell', 'get']
        )
        self.temporal_ordering = TemporalOrdering()
        self.preprocessor = TextPreprocessor()
        self.use_transformer = use_transformer
        
        if use_transformer:
            try:
                self.transformer = TransformerEmbeddings(
                    model_name='minilm',
                    batch_size=32,
                    show_progress=False
                )
                text_col = 'lemmatised' if 'lemmatised' in self.df.columns else 'preprocessed_text'
                corpus_texts = self.df[text_col].fillna('').astype(str).tolist()
                self.corpus_embeddings = self.transformer.generate_embeddings(corpus_texts, normalize=True)
                print("Transformer embeddings loaded successfully")
            except Exception as e:
                print(f"Warning: Could not initialize transformer embeddings: {e}")
                print("Falling back to Doc2Vec...")
                use_transformer = False
                self.use_transformer = False
        
        if not use_transformer and doc2vec_model_path:
            self.doc2vec = Doc2VecModel()
            self.doc2vec.load_model(doc2vec_model_path)
        else:
            self.doc2vec = None
        
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
        preprocessed_new_doc = self.preprocessor.preprocess(new_doc)
        preprocessed_new_doc = self.preprocessor.remove_stopwords(preprocessed_new_doc)
        preprocessed_new_doc = self.preprocessor.lemmatise(preprocessed_new_doc)
        
        if self.use_transformer and hasattr(self, 'transformer'):
            similar_docs_df = self.similarity_measures.find_similar_docs_for_new_document(
                corpus_texts=self.corpus_texts,
                corpus_doc_ids=self.doc_ids,
                new_doc=preprocessed_new_doc,
                top_n=len(self.doc_ids),
                use_noun_phrases=True,
                transformer_model=self.transformer,
                corpus_embeddings=self.corpus_embeddings,
                use_transformer=True
            )
        elif self.doc2vec is not None:
            similar_docs_df = self.similarity_measures.find_similar_docs_for_new_document(
                model=self.doc2vec.model,
                corpus_texts=self.corpus_texts,
                corpus_doc_ids=self.doc_ids,
                new_doc=preprocessed_new_doc,
                top_n=len(self.doc_ids),
                use_noun_phrases=True,
                use_transformer=False
            )
        else:
            raise ValueError("No embedding model available. Initialize with use_transformer=True or provide doc2vec_model_path")
        
        if similarity_metric.lower() == 'cosine' or similarity_metric.lower() == 'transformer':
            sim_col = 'transformer_similarity' if 'transformer_similarity' in similar_docs_df.columns else 'cosine_similarity'
            threshold = similar_docs_df[sim_col].quantile(quantile)
            filtered_docs = similar_docs_df[
                similar_docs_df[sim_col] >= threshold
            ].sort_values(sim_col, ascending=False)
        elif similarity_metric.lower() == 'jaccard':
            threshold = similar_docs_df['jaccard_similarity'].quantile(quantile)
            filtered_docs = similar_docs_df[
                similar_docs_df['jaccard_similarity'] >= threshold
            ].sort_values('jaccard_similarity', ascending=False)
        else:
            sim_col = 'transformer_similarity' if 'transformer_similarity' in similar_docs_df.columns else 'cosine_similarity'
            if 'jaccard_similarity' in similar_docs_df.columns:
                similar_docs_df['combined_similarity'] = (
                    similar_docs_df[sim_col] * 0.7 + similar_docs_df['jaccard_similarity'] * 0.3
                )
                filtered_docs = similar_docs_df.sort_values('combined_similarity', ascending=False).head(top_n)
            else:
                filtered_docs = similar_docs_df.sort_values(sim_col, ascending=False).head(top_n)
        
        filtered_docs = filtered_docs.head(top_n)
        
        if len(filtered_docs) == 0:
            return {
                'similar_documents': pd.DataFrame(),
                'key_terms': pd.DataFrame(),
                'crime_script': pd.DataFrame(),
                'message': 'No similar documents found above threshold'
            }
        
        similar_texts = filtered_docs['text'].tolist()
        
        key_terms = self.tfidf_extractor.extract_key_terms_from_similar_scams(
            similar_texts,
            n_min=n_min,
            n_max=n_max,
            top_n=top_ngrams,
            arrange_by_sequence=True
        )
        
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
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    data_path = os.path.join(project_root, "Data Set", "scam_data_preprocessed.csv")
    output_dir = os.path.join(project_root, "Analysis Results", "case_study_example")
    
    doc2vec_model_path = os.path.join(project_root, "Trained Models", "scam_doc2vec_model.model")
    doc2vec_path = doc2vec_model_path if os.path.exists(doc2vec_model_path) else None
    
    case_study = CaseStudy(
        preprocessed_data_path=data_path,
        doc2vec_model_path=doc2vec_path,
        use_transformer=True
    )
    
    new_scam = """
    I received a scam call. It was an automated voice from the Singapore High Court, 
    stating that I have an outstanding summon. I was asked to pay it.
    """
    
    results = case_study.analyze_new_scam_report(
        new_scam,
        similarity_metric='cosine',
        quantile=0.995,
        top_n=20
    )
    
    case_study.export_case_study_results(results, output_dir)
    
    case_study.visualize_case_study(results, output_dir)
    
    print(f"\nCase study complete! Results saved to {output_dir}")
    print(f"Found {results['num_similar_docs']} similar documents")
    
    return results


if __name__ == "__main__":
    run_case_study_example()
