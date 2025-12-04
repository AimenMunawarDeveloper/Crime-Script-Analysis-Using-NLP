"""
Main script for Crime Script Analysis Using NLP
Implements preprocessing and Doc2Vec model training
"""

import os
import pandas as pd
import numpy as np
from preprocessing import TextPreprocessor
from doc2vec_model import Doc2VecModel


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
    
    print("STEP 5: Computing Similarity Matrix")
    print("-" * 60)
    
    similarity_matrix = doc2vec.compute_similarity_matrix(embeddings)
    print(f"Computed similarity matrix: {similarity_matrix.shape}")
    print(f"  - Similarity scores range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    print(f"  - Mean similarity: {similarity_matrix.mean():.4f}")
    
    similarity_path = os.path.join(RESULTS_DIR, "scam_similarity_matrix.csv")
    doc2vec.save_similarity_matrix(similarity_matrix, similarity_path, doc_ids)
    
    print()
    
    print("STEP 6: Summary Statistics")
    print("-" * 60)
    
    np.fill_diagonal(similarity_matrix, -1)
    max_sim_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    max_similarity = similarity_matrix[max_sim_idx]
    
    print(f"Most similar document pair:")
    print(f"  - Document 1 ID: {doc_ids[max_sim_idx[0]]}")
    print(f"  - Document 2 ID: {doc_ids[max_sim_idx[1]]}")
    print(f"  - Similarity score: {max_similarity:.4f}")
    
    np.fill_diagonal(similarity_matrix, 1.0)
    
    print(f"\nSimilarity matrix statistics:")
    print(f"  - Mean: {similarity_matrix.mean():.4f}")
    print(f"  - Std: {similarity_matrix.std():.4f}")
    print(f"  - Min: {similarity_matrix.min():.4f}")
    print(f"  - Max: {similarity_matrix.max():.4f}")
    
    print()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Preprocessed data: {preprocessed_path}")
    print(f"  - Trained model: {model_path}")
    print(f"  - Embeddings: {embeddings_path}")
    print(f"  - Similarity matrix: {similarity_path}")


if __name__ == "__main__":
    main()

