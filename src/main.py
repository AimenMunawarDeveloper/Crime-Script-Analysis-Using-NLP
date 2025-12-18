import os
import pandas as pd
import numpy as np
from secure_reports import (
    decrypt_and_verify_packages,
    load_private_key_json,
    load_secure_reports_jsonl,
)


def main():
    # Heavy NLP imports are deferred so the secure-report gate can run even if
    # the current Python environment lacks spaCy / transformer deps.
    from preprocessing import TextPreprocessor
    from doc2vec_model import Doc2VecModel
    from similarity_measures import SimilarityMeasures
    from tfidf_extraction import TFIDFExtractor
    from clustering import ScamClustering
    from temporal_ordering import TemporalOrdering
    from transformer_embeddings import TransformerEmbeddings, EmbeddingComparison

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
    print("Preprocessing + Transformer Embeddings Implementation")
    print("=" * 60)
    print()
    
    print("STEP 1: Dataset Loading & Preprocessing")
    print("-" * 60)
    
    preprocessor = TextPreprocessor()
    
    dataset_path = os.path.join(DATA_DIR, "scam_raw_dataset.csv")
    secure_reports_path = os.path.join(DATA_DIR, "secure_reports.jsonl")
    server_priv_path = os.path.join(DATA_DIR, "server_rsa_private.json")
    
    df = None
    if os.path.exists(secure_reports_path) and os.path.exists(server_priv_path):
        print(f"Found secure report packages: {secure_reports_path}")
        print("Decrypting + verifying signatures before NLP processing...")
        try:
            packages = load_secure_reports_jsonl(secure_reports_path)
            server_priv = load_private_key_json(server_priv_path)
            verified_rows, verified_count, total_count = decrypt_and_verify_packages(
                packages, server_priv
            )
            print(f"Verified {verified_count}/{total_count} secure reports")
            if verified_count == 0:
                print("Error: No secure reports verified. Aborting.")
                return
            df = pd.DataFrame(
                {
                    "submission_id": [r["report_id"] for r in verified_rows],
                    "incident_description": [r["plaintext"] for r in verified_rows],
                }
            )
        except Exception as e:
            print(f"Error processing secure reports: {e}")
            return
    else:
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
    
    print("STEP 2: Preparing Corpus")
    print("-" * 60)
    
    training_text_column = 'lemmatised' if 'lemmatised' in df_processed.columns else 'preprocessed_text'
    
    if 'submission_id' in df_processed.columns:
        doc_ids = df_processed['submission_id'].tolist()
    else:
        doc_ids = df_processed.index.tolist()
    
    texts = df_processed[training_text_column].fillna('').astype(str).tolist()
    
    print(f"Prepared {len(texts)} documents")
    print()
    
    print("STEP 3: Generating Transformer Embeddings (Primary Method)")
    print("-" * 60)
    
    transformer = None
    transformer_similarity_matrix = None
    transformer_embeddings = None
    transformer_model_path = None
    transformer_embeddings_path = None
    transformer_similarity_path = None
    
    try:
        transformer = TransformerEmbeddings(
            model_name='minilm',
            batch_size=32,
            show_progress=True
        )
        
        transformer_embeddings = transformer.generate_embeddings(texts, normalize=True)
        print(f"Generated transformer embeddings: {transformer_embeddings.shape}")
        
        transformer_model_path = os.path.join(MODELS_DIR, "scam_transformer_model")
        transformer.save_model(transformer_model_path)
        
        transformer_embeddings_path = os.path.join(RESULTS_DIR, "scam_transformer_embeddings.csv")
        transformer.save_embeddings(transformer_embeddings, transformer_embeddings_path, doc_ids)
        
        transformer_similarity_matrix = transformer.compute_similarity_matrix(transformer_embeddings)
        transformer_similarity_path = os.path.join(RESULTS_DIR, "scam_transformer_similarity_matrix.csv")
        transformer.save_similarity_matrix(
            transformer_similarity_matrix,
            transformer_similarity_path,
            doc_ids
        )
        
    except ImportError as e:
        print(f"ERROR: Transformer embeddings required but not available: {e}")
        print("Please install sentence-transformers: pip install sentence-transformers")
        return
    except Exception as e:
        print(f"ERROR generating transformer embeddings: {e}")
        print("Transformer embeddings are required for this pipeline.")
        return
    
    print()
    
    print("STEP 4: Computing Jaccard Similarity Matrix (for weighted combination)")
    print("-" * 60)
    
    similarity_measures = SimilarityMeasures()
    jaccard_similarity_matrix = similarity_measures.compute_jaccard_similarity_matrix(
        texts, 
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
    
    doc2vec = None
    cosine_similarity_matrix = None
    cosine_similarity_path = None
    doc2vec_available = False
    
    print()
    print("STEP 5: Optional - Generating Doc2Vec Embeddings (for comparison)")
    print("-" * 60)
    
    try:
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
        
        doc2vec_embeddings = doc2vec.generate_embeddings()
        cosine_similarity_matrix = doc2vec.compute_similarity_matrix(doc2vec_embeddings)
        cosine_similarity_path = os.path.join(RESULTS_DIR, "scam_cosine_similarity_matrix.csv")
        doc2vec.save_similarity_matrix(cosine_similarity_matrix, cosine_similarity_path, doc_ids)
        
        doc2vec_available = True
        print("Doc2Vec embeddings generated for comparison")
        
        print("\nComparing embedding methods...")
        comparison = EmbeddingComparison()
        comparison_df = comparison.compare_similarity_matrices(
            transformer_similarity_matrix,
            cosine_similarity_matrix,
            jaccard_similarity_matrix,
            doc_ids
        )
        
        comparison_path = os.path.join(RESULTS_DIR, "scam_embedding_comparison.csv")
        comparison.save_comparison_results(comparison_df, comparison_path)
        
    except Exception as e:
        print(f"Note: Doc2Vec optional step skipped: {e}")
        doc2vec_available = False
    
    print()
    
    print()
    
    print("STEP 6: Clustering Similar Scams (Using Similarity-Based Clustering)")
    print("-" * 60)
    
    clustering = ScamClustering()
    
    print("Using similarity graph-based clustering (cosine + jaccard thresholds)...")
    cluster_df = clustering.cluster_using_similarity_graph(
        similarity_matrix=transformer_similarity_matrix,
        doc_ids=doc_ids,
        threshold=0.7,
        jaccard_matrix=jaccard_similarity_matrix,
        jaccard_threshold=0.05,
        min_cluster_size=2
    )
    print(f"Similarity-based clustering complete. Found {cluster_df['cluster_id'].nunique()} clusters")
    
    cluster_path = os.path.join(RESULTS_DIR, "scam_clusters.csv")
    cluster_df.to_csv(cluster_path, index=False)
    print(f"Clustering complete. Found {cluster_df['cluster_id'].nunique()} clusters")
    print(f"Cluster assignments saved to {cluster_path}")
    
    cluster_stats = clustering.get_cluster_statistics(cluster_df)
    cluster_stats_path = os.path.join(RESULTS_DIR, "scam_cluster_statistics.csv")
    cluster_stats.to_csv(cluster_stats_path, index=False)
    print(f"Cluster statistics saved to {cluster_stats_path}")
    
    print()
    
    print("STEP 7: Extracting Key Terms from Similar Scams")
    print("-" * 60)
    
    tfidf_extractor = TFIDFExtractor(
        additional_stopwords=['ask', 'said', 'say', 'asked', 'claimed', 'told', 'got', 'tell', 'get']
    )
    
    key_terms_results = []
    
    for cluster_id in cluster_df['cluster_id'].unique():
        cluster_members = cluster_df[cluster_df['cluster_id'] == cluster_id]['document_id'].tolist()
        
        if len(cluster_members) < 2:
            continue
        
        cluster_texts = []
        for doc_id in cluster_members:
            doc_idx = doc_ids.index(doc_id)
            cluster_texts.append(texts[doc_idx])
        
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
    
    print("STEP 8: Generating Crime Scripts (Temporal Ordering)")
    print("-" * 60)
    
    temporal_ordering = TemporalOrdering()
    
    crime_scripts = []
    
    if all_key_terms is not None:
        for cluster_id in cluster_df['cluster_id'].unique():
            cluster_members = cluster_df[cluster_df['cluster_id'] == cluster_id]['document_id'].tolist()
            
            if len(cluster_members) < 2:
                continue
            
            cluster_texts = []
            for doc_id in cluster_members:
                doc_idx = doc_ids.index(doc_id)
                cluster_texts.append(texts[doc_idx])
            
            cluster_key_terms = all_key_terms[all_key_terms['cluster_id'] == cluster_id].copy()
            
            if len(cluster_key_terms) == 0:
                continue
            
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
        
        if all_key_terms is not None:
            print("\nGenerating sequence visualizations for a medium-sized cluster...")
            os.makedirs(os.path.join(RESULTS_DIR, "visualizations"), exist_ok=True)
            viz_dir = os.path.join(RESULTS_DIR, "visualizations")
            
            # Pick a "medium-sized" (near-median) cluster that is eligible for visualization.
            # Eligibility criteria:
            # - Has a generated crime script (present in all_crime_scripts)
            # - Has key terms with a 'next_term' column (required for visualize_sequence_graph)
            eligible_cluster_stats = cluster_stats.copy()
            script_cluster_ids = set(all_crime_scripts['cluster_id'].unique().tolist())
            eligible_cluster_stats = eligible_cluster_stats[
                eligible_cluster_stats['cluster_id'].isin(script_cluster_ids)
            ]

            if 'next_term' in all_key_terms.columns:
                # Prefer clusters that have at least one non-null next_term
                key_term_cluster_ids = set(
                    all_key_terms.loc[all_key_terms['next_term'].notna(), 'cluster_id']
                    .unique()
                    .tolist()
                )
                eligible_cluster_stats = eligible_cluster_stats[
                    eligible_cluster_stats['cluster_id'].isin(key_term_cluster_ids)
                ]

            if len(eligible_cluster_stats) == 0:
                # Fallback: use any cluster with scripts, closest to median size in cluster_stats
                eligible_cluster_stats = cluster_stats[
                    cluster_stats['cluster_id'].isin(script_cluster_ids)
                ].copy()

            if len(eligible_cluster_stats) == 0:
                top_clusters = []
            else:
                median_size = eligible_cluster_stats['num_documents'].median()
                chosen_idx = (eligible_cluster_stats['num_documents'] - median_size).abs().idxmin()
                chosen_cluster_id = int(eligible_cluster_stats.loc[chosen_idx, 'cluster_id'])
                top_clusters = [chosen_cluster_id]
            viz_count = 0
            
            for idx, cluster_id in enumerate(top_clusters, 1):
                print(f"  Processing cluster {cluster_id} ({idx}/{len(top_clusters)})...", end=' ')
                cluster_scripts = all_crime_scripts[all_crime_scripts['cluster_id'] == cluster_id]
                if len(cluster_scripts) > 0:
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
                                save_path=viz_path,
                                max_terms=25
                            )
                            viz_count += 1
                            print("✓")
                        except Exception as e:
                            print(f"✗ Error: {str(e)[:50]}")
                else:
                    print("- (skipped, no scripts)")
            
            if viz_count > 0:
                print(f"Generated {viz_count} sequence visualizations")
                print(f"Visualizations saved to {viz_dir}")
            else:
                print("No visualizations generated (insufficient data)")
    
    print()
    
    print("STEP 9: Summary Statistics")
    print("-" * 60)
    
    np.fill_diagonal(transformer_similarity_matrix, -1)
    max_sim_idx = np.unravel_index(np.argmax(transformer_similarity_matrix), transformer_similarity_matrix.shape)
    max_similarity = transformer_similarity_matrix[max_sim_idx]
    
    print(f"Most similar document pair (Transformer):")
    print(f"  - Document 1 ID: {doc_ids[max_sim_idx[0]]}")
    print(f"  - Document 2 ID: {doc_ids[max_sim_idx[1]]}")
    print(f"  - Similarity score: {max_similarity:.4f}")
    
    np.fill_diagonal(transformer_similarity_matrix, 1.0)
    
    print(f"\nTransformer similarity matrix statistics:")
    print(f"  - Mean: {transformer_similarity_matrix.mean():.4f}")
    print(f"  - Std: {transformer_similarity_matrix.std():.4f}")
    print(f"  - Min: {transformer_similarity_matrix.min():.4f}")
    print(f"  - Max: {transformer_similarity_matrix.max():.4f}")
    
    print(f"\nJaccard similarity matrix statistics:")
    print(f"  - Mean: {jaccard_similarity_matrix.mean():.4f}")
    print(f"  - Std: {jaccard_similarity_matrix.std():.4f}")
    print(f"  - Min: {jaccard_similarity_matrix.min():.4f}")
    print(f"  - Max: {jaccard_similarity_matrix.max():.4f}")
    
    if doc2vec_available and cosine_similarity_matrix is not None:
        print(f"\nDoc2Vec Cosine similarity matrix statistics (for comparison):")
        print(f"  - Mean: {cosine_similarity_matrix.mean():.4f}")
        print(f"  - Std: {cosine_similarity_matrix.std():.4f}")
        print(f"  - Min: {cosine_similarity_matrix.min():.4f}")
        print(f"  - Max: {cosine_similarity_matrix.max():.4f}")
    
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
    print(f"  - Transformer model: {transformer_model_path}")
    print(f"  - Transformer embeddings: {transformer_embeddings_path}")
    print(f"  - Transformer similarity matrix: {transformer_similarity_path}")
    print(f"  - Jaccard similarity matrix: {jaccard_similarity_path}")
    if doc2vec_available:
        print(f"  - Doc2Vec model: {model_path}")
        print(f"  - Doc2Vec cosine similarity matrix: {cosine_similarity_path}")
        print(f"  - Embedding comparison: {comparison_path}")
    print(f"  - Cluster assignments: {cluster_path}")
    print(f"  - Cluster statistics: {cluster_stats_path}")
    if all_key_terms is not None:
        print(f"  - Key terms: {key_terms_path}")
    if crime_scripts:
        print(f"  - Crime scripts: {crime_scripts_path}")


if __name__ == "__main__":
    main()

