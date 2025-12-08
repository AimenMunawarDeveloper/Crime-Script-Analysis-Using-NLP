"""
Main script for Crime Script Analysis Using NLP
Implements complete pipeline: preprocessing, Doc2Vec, similarity measures,
TF-IDF extraction, clustering, and temporal ordering
"""

import os
import pandas as pd
import numpy as np
from preprocessing import TextPreprocessor
from doc2vec_model import Doc2VecModel
from similarity_measures import SimilarityMeasures
from tfidf_extraction import TFIDFExtractor
from clustering import ScamClustering
from temporal_ordering import TemporalOrdering
from transformer_embeddings import TransformerEmbeddings, EmbeddingComparison


def main():
    """Main function to run preprocessing and Doc2Vec training"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    DATA_DIR = os.path.join(project_root, "Data Set")
    MODELS_DIR = os.path.join(project_root, "Trained Models")
    RESULTS_DIR = os.path.join(project_root, "Analysis Results")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Crime Script Analysis Using NLP")
    print("Preprocessing + Doc2Vec Implementation")
    print("=" * 60)
    print()
    
    print("STEP 1: Dataset Loading & Preprocessing")
    print("-" * 60)
    
    preprocessor = TextPreprocessor()
    
    dataset_path = os.path.join(DATA_DIR, "scam_raw_dataset.csv")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Raw dataset not found at {dataset_path}")
        print("Please place 'scam_raw_dataset.csv' in the Data Set/ directory")
        print("This should be the raw extracted dataset before any preprocessing")
        return
    
    print(f"Found raw dataset: {dataset_path}")
    
    df = preprocessor.load_dataset(dataset_path, text_column='incident_description')
    
    text_col = getattr(preprocessor, 'detected_text_column', 'incident_description')
    
    df_processed = preprocessor.preprocess_dataset(
        df, 
        text_column=text_col,
        id_column='submission_id' if 'submission_id' in df.columns else None
    )
    
    preprocessed_path = os.path.join(DATA_DIR, "scam_data_preprocessed.csv")
    preprocessor.save_preprocessed_data(df_processed, preprocessed_path)
    
    print()
    
    print("STEP 2: Preparing Corpus for Doc2Vec")
    print("-" * 60)
    
    training_text_column = 'lemmatised' if 'lemmatised' in df_processed.columns else 'preprocessed_text'
    
    if 'submission_id' in df_processed.columns:
        doc_ids = df_processed['submission_id'].tolist()
    else:
        doc_ids = df_processed.index.tolist()
    
    texts = df_processed[training_text_column].fillna('').astype(str).tolist()
    
    print(f"Prepared {len(texts)} documents for training")
    print()
    
    print("STEP 3: Training Doc2Vec Model")
    print("-" * 60)
    
    doc2vec = Doc2VecModel(
        vector_size=50,
        min_count=2,
        epochs=100,
        dm=1,
        alpha=0.025,
        min_alpha=0.00025
    )
    
    tagged_docs = doc2vec.create_tagged_documents(texts, doc_ids)
    
    doc2vec.train(tagged_docs, verbose=True)
    
    model_path = os.path.join(MODELS_DIR, "scam_doc2vec_model.model")
    doc2vec.save_model(model_path)
    
    print()
    
    print("STEP 4: Generating Document Embeddings")
    print("-" * 60)
    
    embeddings = doc2vec.generate_embeddings()
    print(f"Generated embeddings: {embeddings.shape}")
    print(f"  - Number of documents: {embeddings.shape[0]}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")
    
    embeddings_path = os.path.join(RESULTS_DIR, "scam_document_embeddings.csv")
    doc2vec.save_embeddings(embeddings, embeddings_path)
    
    embeddings_df = pd.DataFrame(embeddings, index=doc_ids)
    embeddings_df.index.name = 'document_id'
    embeddings_df.to_csv(embeddings_path)
    print(f"Embeddings saved with document IDs to {embeddings_path}")
    
    print()
    
    print("STEP 5: Computing Cosine Similarity Matrix")
    print("-" * 60)
    
    cosine_similarity_matrix = doc2vec.compute_similarity_matrix(embeddings)
    print(f"Computed cosine similarity matrix: {cosine_similarity_matrix.shape}")
    print(f"  - Similarity scores range: [{cosine_similarity_matrix.min():.4f}, {cosine_similarity_matrix.max():.4f}]")
    print(f"  - Mean similarity: {cosine_similarity_matrix.mean():.4f}")
    
    cosine_similarity_path = os.path.join(RESULTS_DIR, "scam_cosine_similarity_matrix.csv")
    doc2vec.save_similarity_matrix(cosine_similarity_matrix, cosine_similarity_path, doc_ids)
    
    print()
    
    print("STEP 6: Computing Jaccard Similarity Matrix")
    print("-" * 60)
    
    similarity_measures = SimilarityMeasures()
    
    # Use preprocessed text for Jaccard similarity
    jaccard_texts = df_processed[training_text_column].fillna('').astype(str).tolist()
    
    jaccard_similarity_matrix = similarity_measures.compute_jaccard_similarity_matrix(
        jaccard_texts, 
        use_noun_phrases=True
    )
    print(f"Computed Jaccard similarity matrix: {jaccard_similarity_matrix.shape}")
    print(f"  - Similarity scores range: [{jaccard_similarity_matrix.min():.4f}, {jaccard_similarity_matrix.max():.4f}]")
    print(f"  - Mean similarity: {jaccard_similarity_matrix.mean():.4f}")
    
    jaccard_similarity_path = os.path.join(RESULTS_DIR, "scam_jaccard_similarity_matrix.csv")
    similarity_measures.save_jaccard_similarity_matrix(
        jaccard_similarity_matrix, 
        jaccard_similarity_path, 
        doc_ids
    )
    
    print()
    
    print("STEP 7: Generating Transformer Embeddings")
    print("-" * 60)
    
    # Initialize variables for transformer embeddings
    transformer_available = False
    transformer_similarity_matrix = None
    transformer_embeddings = None
    transformer_embeddings_path = None
    transformer_similarity_path = None
    comparison_path = None
    
    try:
        # Initialize transformer model (using MiniLM for faster processing)
        # Can be changed to 'mpnet' for better quality but slower processing
        transformer = TransformerEmbeddings(
            model_name='minilm',
            batch_size=32,
            show_progress=True
        )
        
        # Use preprocessed text for transformer embeddings
        transformer_texts = df_processed[training_text_column].fillna('').astype(str).tolist()
        
        # Generate transformer embeddings
        transformer_embeddings = transformer.generate_embeddings(transformer_texts, normalize=True)
        print(f"Generated transformer embeddings: {transformer_embeddings.shape}")
        
        # Save transformer embeddings
        transformer_embeddings_path = os.path.join(RESULTS_DIR, "scam_transformer_embeddings.csv")
        transformer.save_embeddings(transformer_embeddings, transformer_embeddings_path, doc_ids)
        
        # Compute transformer similarity matrix
        transformer_similarity_matrix = transformer.compute_similarity_matrix(transformer_embeddings)
        
        # Save transformer similarity matrix
        transformer_similarity_path = os.path.join(RESULTS_DIR, "scam_transformer_similarity_matrix.csv")
        transformer.save_similarity_matrix(
            transformer_similarity_matrix,
            transformer_similarity_path,
            doc_ids
        )
        
        print()
        
        print("STEP 7b: Comparing Embedding Methods")
        print("-" * 60)
        
        # Compare all three methods
        comparison = EmbeddingComparison()
        comparison_df = comparison.compare_similarity_matrices(
            transformer_similarity_matrix,
            cosine_similarity_matrix,
            jaccard_similarity_matrix,
            doc_ids
        )
        
        # Save comparison results
        comparison_path = os.path.join(RESULTS_DIR, "scam_embedding_comparison.csv")
        comparison.save_comparison_results(comparison_df, comparison_path)
        
        # Find agreement pairs (documents that all three methods agree are similar)
        print("\nFinding high-agreement document pairs...")
        agreement_pairs = comparison.find_agreement_pairs(
            transformer_similarity_matrix,
            cosine_similarity_matrix,
            jaccard_similarity_matrix,
            doc_ids,
            threshold=0.7,
            top_n=20
        )
        
        if len(agreement_pairs) > 0:
            agreement_path = os.path.join(RESULTS_DIR, "scam_agreement_pairs.csv")
            agreement_pairs.to_csv(agreement_path, index=False)
            print(f"Agreement pairs saved to {agreement_path}")
        
        transformer_available = True
        
    except ImportError as e:
        print(f"Warning: Transformer embeddings not available: {e}")
        print("Skipping transformer embeddings step. Install sentence-transformers to enable.")
        transformer_available = False
        transformer_similarity_matrix = None
        transformer_embeddings = None
    except Exception as e:
        print(f"Error generating transformer embeddings: {e}")
        print("Continuing with other methods...")
        transformer_available = False
        transformer_similarity_matrix = None
        transformer_embeddings = None
    
    print()
    
    print("STEP 8: Clustering Similar Scams")
    print("-" * 60)
    
    clustering = ScamClustering()
    
    # Cluster using similarity thresholds
    cluster_df = clustering.cluster_by_similarity_threshold(
        cosine_similarity_matrix,
        jaccard_similarity_matrix,
        doc_ids,
        cosine_threshold=0.7,
        jaccard_threshold=0.3,
        min_cluster_size=2
    )
    
    cluster_path = os.path.join(RESULTS_DIR, "scam_clusters.csv")
    cluster_df.to_csv(cluster_path, index=False)
    print(f"Clustering complete. Found {cluster_df['cluster_id'].nunique()} clusters")
    print(f"Cluster assignments saved to {cluster_path}")
    
    # Get cluster statistics
    cluster_stats = clustering.get_cluster_statistics(cluster_df)
    cluster_stats_path = os.path.join(RESULTS_DIR, "scam_cluster_statistics.csv")
    cluster_stats.to_csv(cluster_stats_path, index=False)
    print(f"Cluster statistics saved to {cluster_stats_path}")
    
    print()
    
    print("STEP 9: Extracting Key Terms from Similar Scams")
    print("-" * 60)
    
    tfidf_extractor = TFIDFExtractor(
        additional_stopwords=['ask', 'said', 'say', 'asked', 'claimed', 'told', 'got', 'tell', 'get']
    )
    
    # Process each cluster to extract key terms
    key_terms_results = []
    
    for cluster_id in cluster_df['cluster_id'].unique():
        cluster_members = cluster_df[cluster_df['cluster_id'] == cluster_id]['document_id'].tolist()
        
        if len(cluster_members) < 2:
            continue
        
        # Get texts for cluster members
        cluster_texts = []
        for doc_id in cluster_members:
            doc_idx = doc_ids.index(doc_id)
            cluster_texts.append(jaccard_texts[doc_idx])
        
        # Extract key terms
        key_terms = tfidf_extractor.extract_key_terms_from_similar_scams(
            cluster_texts,
            n_min=1,
            n_max=1,
            top_n=20,
            arrange_by_sequence=True
        )
        
        key_terms['cluster_id'] = cluster_id
        key_terms_results.append(key_terms)
    
    all_key_terms = None
    if key_terms_results:
        all_key_terms = pd.concat(key_terms_results, ignore_index=True)
        key_terms_path = os.path.join(RESULTS_DIR, "scam_key_terms.csv")
        all_key_terms.to_csv(key_terms_path, index=False)
        print(f"Key terms extracted for {len(key_terms_results)} clusters")
        print(f"Key terms saved to {key_terms_path}")
    else:
        print("No key terms extracted (no clusters with sufficient members)")
    
    print()
    
    print("STEP 10: Generating Crime Scripts (Temporal Ordering)")
    print("-" * 60)
    
    temporal_ordering = TemporalOrdering()
    
    crime_scripts = []
    
    if all_key_terms is not None:
        for cluster_id in cluster_df['cluster_id'].unique():
            cluster_members = cluster_df[cluster_df['cluster_id'] == cluster_id]['document_id'].tolist()
            
            if len(cluster_members) < 2:
                continue
            
            # Get texts for cluster members
            cluster_texts = []
            for doc_id in cluster_members:
                doc_idx = doc_ids.index(doc_id)
                cluster_texts.append(jaccard_texts[doc_idx])
            
            # Get key terms for this cluster
            cluster_key_terms = all_key_terms[all_key_terms['cluster_id'] == cluster_id].copy()
            
            if len(cluster_key_terms) == 0:
                continue
            
            # Generate consensus crime script
            crime_script = temporal_ordering.generate_consensus_script(
                cluster_texts,
                cluster_key_terms,
                position_column='sequence'
            )
            
            crime_script['cluster_id'] = cluster_id
            crime_scripts.append(crime_script)
    
    if crime_scripts:
        all_crime_scripts = pd.concat(crime_scripts, ignore_index=True)
        crime_scripts_path = os.path.join(RESULTS_DIR, "scam_crime_scripts.csv")
        temporal_ordering.export_crime_script(all_crime_scripts, crime_scripts_path)
        print(f"Generated crime scripts for {len(crime_scripts)} clusters")
        print(f"Crime scripts saved to {crime_scripts_path}")
        
        # Generate visualizations for top clusters
        if all_key_terms is not None:
            print("\nGenerating sequence visualizations for top clusters...")
            os.makedirs(os.path.join(RESULTS_DIR, "visualizations"), exist_ok=True)
            viz_dir = os.path.join(RESULTS_DIR, "visualizations")
            
            # Get top clusters by size
            top_clusters = cluster_stats.head(10)['cluster_id'].tolist()
            viz_count = 0
            
            for cluster_id in top_clusters:
                cluster_scripts = all_crime_scripts[all_crime_scripts['cluster_id'] == cluster_id]
                if len(cluster_scripts) > 0:
                    # Get key terms for this cluster to create sequence graph
                    cluster_key_terms = all_key_terms[all_key_terms['cluster_id'] == cluster_id].copy()
                    if len(cluster_key_terms) > 0 and 'next_term' in cluster_key_terms.columns:
                        try:
                            viz_path = os.path.join(viz_dir, f"cluster_{cluster_id}_sequence_graph.png")
                            temporal_ordering.visualize_sequence_graph(
                                cluster_key_terms,
                                term_column='term',
                                next_term_column='next_term',
                                weight_column='tfidf_score',
                                figsize=(14, 10),
                                save_path=viz_path
                            )
                            viz_count += 1
                        except Exception as e:
                            print(f"  - Warning: Could not generate visualization for cluster {cluster_id}: {e}")
            
            if viz_count > 0:
                print(f"Generated {viz_count} sequence visualizations")
                print(f"Visualizations saved to {viz_dir}")
            else:
                print("No visualizations generated (insufficient data)")
    
    print()
    
    print("STEP 11: Summary Statistics")
    print("-" * 60)
    
    np.fill_diagonal(cosine_similarity_matrix, -1)
    max_sim_idx = np.unravel_index(np.argmax(cosine_similarity_matrix), cosine_similarity_matrix.shape)
    max_similarity = cosine_similarity_matrix[max_sim_idx]
    
    print(f"Most similar document pair (Cosine):")
    print(f"  - Document 1 ID: {doc_ids[max_sim_idx[0]]}")
    print(f"  - Document 2 ID: {doc_ids[max_sim_idx[1]]}")
    print(f"  - Similarity score: {max_similarity:.4f}")
    
    np.fill_diagonal(cosine_similarity_matrix, 1.0)
    
    print(f"\nCosine similarity matrix statistics:")
    print(f"  - Mean: {cosine_similarity_matrix.mean():.4f}")
    print(f"  - Std: {cosine_similarity_matrix.std():.4f}")
    print(f"  - Min: {cosine_similarity_matrix.min():.4f}")
    print(f"  - Max: {cosine_similarity_matrix.max():.4f}")
    
    print(f"\nJaccard similarity matrix statistics:")
    print(f"  - Mean: {jaccard_similarity_matrix.mean():.4f}")
    print(f"  - Std: {jaccard_similarity_matrix.std():.4f}")
    print(f"  - Min: {jaccard_similarity_matrix.min():.4f}")
    print(f"  - Max: {jaccard_similarity_matrix.max():.4f}")
    
    if transformer_available and transformer_similarity_matrix is not None:
        print(f"\nTransformer similarity matrix statistics:")
        print(f"  - Mean: {transformer_similarity_matrix.mean():.4f}")
        print(f"  - Std: {transformer_similarity_matrix.std():.4f}")
        print(f"  - Min: {transformer_similarity_matrix.min():.4f}")
        print(f"  - Max: {transformer_similarity_matrix.max():.4f}")
    
    print(f"\nClustering statistics:")
    print(f"  - Total clusters: {cluster_df['cluster_id'].nunique()}")
    print(f"  - Average cluster size: {cluster_df['cluster_size'].mean():.2f}")
    print(f"  - Largest cluster size: {cluster_df['cluster_size'].max()}")
    
    print()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Preprocessed data: {preprocessed_path}")
    print(f"  - Trained model: {model_path}")
    print(f"  - Embeddings: {embeddings_path}")
    print(f"  - Cosine similarity matrix: {cosine_similarity_path}")
    print(f"  - Jaccard similarity matrix: {jaccard_similarity_path}")
    if transformer_available:
        print(f"  - Transformer embeddings: {transformer_embeddings_path}")
        print(f"  - Transformer similarity matrix: {transformer_similarity_path}")
        print(f"  - Embedding comparison: {comparison_path}")
    print(f"  - Cluster assignments: {cluster_path}")
    print(f"  - Cluster statistics: {cluster_stats_path}")
    if all_key_terms is not None:
        print(f"  - Key terms: {key_terms_path}")
    if crime_scripts:
        print(f"  - Crime scripts: {crime_scripts_path}")


if __name__ == "__main__":
    main()

