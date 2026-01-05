# Crime Script Analysis Using NLP

A comprehensive NLP-based system for analyzing scam reports with secure encryption, document similarity analysis, clustering, and crime script generation. This project implements preprocessing, multiple embedding methods (Transformer and Doc2Vec), similarity measures, clustering, and security features for handling sensitive crime report data.

## Project Structure

```
Crime Script Analysis Using NLP/
├── Data Set/                    # Dataset files (input, preprocessed, and secure reports)
│   ├── scam_raw_dataset.csv     # Raw input dataset
│   ├── scam_data_preprocessed.csv  # Preprocessed dataset
│   ├── secure_reports.jsonl     # Encrypted secure reports
│   ├── server_rsa_public.json   # Server public key
│   ├── server_rsa_private.json  # Server private key
│   ├── sender_rsa_public.json   # Sender public key
│   └── sender_rsa_private.json  # Sender private key
├── Trained Models/              # Trained embedding models
│   ├── scam_doc2vec_model.model
│   └── scam_transformer_model/
├── Analysis Results/           # Output files (embeddings, similarity matrices, clusters, scripts)
│   ├── scam_transformer_embeddings.csv
│   ├── scam_transformer_similarity_matrix.csv
│   ├── scam_jaccard_similarity_matrix.csv
│   ├── scam_cosine_similarity_matrix.csv
│   ├── scam_clusters.csv
│   ├── scam_cluster_statistics.csv
│   ├── scam_key_terms.csv
│   ├── scam_crime_scripts.csv
│   ├── scam_embedding_comparison.csv
│   ├── visualizations/         # Sequence graph visualizations
│   └── case_study_example/     # Case study results
├── src/                         # Source code
│   ├── preprocessing.py        # Text preprocessing module
│   ├── doc2vec_model.py        # Doc2Vec model implementation
│   ├── transformer_embeddings.py  # Transformer embeddings (sentence-transformers)
│   ├── similarity_measures.py  # Jaccard and cosine similarity computation
│   ├── clustering.py            # Similarity-based clustering
│   ├── tfidf_extraction.py     # TF-IDF key term extraction
│   ├── temporal_ordering.py    # Crime script generation and visualization
│   ├── secure_reports.py       # Secure report encryption/decryption
│   ├── manual_crypto.py        # Cryptographic primitives (RSA, AES, HMAC)
│   ├── case_study.py           # Case study analysis for new reports
│   ├── attack_simulation.py    # Security attack demonstrations
│   └── main.py                 # Main pipeline execution script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup

### 1. Install Dependencies

**On Windows**, use `python -m pip` instead of just `pip`:

```bash
python -m pip install -r requirements.txt
```

**On Linux/Mac**, you can use either:

```bash
pip install -r requirements.txt
# or
python -m pip install -r requirements.txt
```

### 2. Download Required NLTK Data

The script will automatically download required NLTK data on first run. If you encounter issues, run:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 3. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

## Usage - Four Main Steps

### Step 1: Create Secure Reports

Generate encrypted secure report packages from your raw dataset. This step creates encrypted JSONL files with RSA-OAEP encryption, AES-256-CBC encryption, HMAC-SHA256 integrity protection, and RSA signatures.

**Command:**

```bash
python src/secure_reports.py --csv "Data Set/scam_raw_dataset.csv" --out "Data Set/secure_reports.jsonl" --limit 200 --server-bits 1024 --sender-bits 1024
```

**Parameters:**
- `--csv`: Path to raw dataset CSV file (default: `Data Set/scam_raw_dataset.csv`)
- `--out`: Output path for secure reports JSONL (default: `Data Set/secure_reports.jsonl`)
- `--text-col`: Column name containing report text (default: `incident_description`)
- `--id-col`: Column name for report IDs (default: `submission_id`)
- `--limit`: Maximum number of reports to encrypt (default: 200)
- `--server-bits`: RSA key size for server (minimum 1024, default: 1024)
- `--sender-bits`: RSA key size for sender (default: 1024)

**Output:**
- `Data Set/secure_reports.jsonl`: Encrypted report packages
- `Data Set/server_rsa_public.json` & `server_rsa_private.json`: Server key pair
- `Data Set/sender_rsa_public.json` & `sender_rsa_private.json`: Sender key pair

**Security Features:**
- **Confidentiality**: AES-256-CBC encryption with unique IVs per report
- **Integrity**: HMAC-SHA256 verification
- **Authentication**: RSA signatures with SHA-256 hashing
- **Non-repudiation**: Cryptographic signatures prove sender identity
- **Domain-aware**: Vishing-specific bindings in OAEP labels and IV generation

### Step 2: Run Main Pipeline

Execute the complete NLP analysis pipeline including preprocessing, embedding generation, similarity computation, clustering, key term extraction, and crime script generation.

**Command:**

```bash
python src/main.py
```

Or from the project root:

```bash
python src/main.py
```

**What it does:**
1. **Dataset Loading**: Loads either secure reports (if available) or raw CSV dataset
2. **Preprocessing**: Cleans and normalizes text data
3. **Transformer Embeddings**: Generates sentence embeddings using MiniLM model
4. **Jaccard Similarity**: Computes noun-phrase based Jaccard similarity
5. **Doc2Vec Embeddings** (optional): Generates Doc2Vec embeddings for comparison
6. **Clustering**: Groups similar scams using similarity graph-based clustering
7. **Key Term Extraction**: Extracts TF-IDF key terms from clusters
8. **Crime Script Generation**: Creates temporal-ordered crime scripts
9. **Visualization**: Generates sequence graphs for clusters

**Output Files:**
- `Data Set/scam_data_preprocessed.csv`: Preprocessed dataset
- `Trained Models/scam_transformer_model/`: Saved transformer model
- `Trained Models/scam_doc2vec_model.model`: Doc2Vec model (optional)
- `Analysis Results/scam_transformer_embeddings.csv`: Document embeddings
- `Analysis Results/scam_transformer_similarity_matrix.csv`: Cosine similarity matrix
- `Analysis Results/scam_jaccard_similarity_matrix.csv`: Jaccard similarity matrix
- `Analysis Results/scam_cosine_similarity_matrix.csv`: Doc2Vec cosine similarity (optional)
- `Analysis Results/scam_clusters.csv`: Cluster assignments
- `Analysis Results/scam_cluster_statistics.csv`: Cluster statistics
- `Analysis Results/scam_key_terms.csv`: Extracted key terms per cluster
- `Analysis Results/scam_crime_scripts.csv`: Generated crime scripts
- `Analysis Results/scam_embedding_comparison.csv`: Embedding method comparison (optional)
- `Analysis Results/visualizations/cluster_*.png`: Sequence graph visualizations

### Step 3: Run Case Study

Analyze a new scam report by finding similar reports, extracting key terms, and generating a crime script.

**Command:**

```bash
python src/case_study.py
```

**What it does:**
- Loads preprocessed data and trained models
- Analyzes a new scam report (example provided in code)
- Finds similar documents using transformer or Doc2Vec embeddings
- Extracts key terms from similar reports
- Generates crime script with temporal ordering
- Creates visualizations and exports results

**Output:**
- `Analysis Results/case_study_example/similar_documents.csv`: Similar reports found
- `Analysis Results/case_study_example/key_terms.csv`: Extracted key terms
- `Analysis Results/case_study_example/crime_script.csv`: Generated crime script
- `Analysis Results/case_study_example/case_study_sequence_graph.png`: Visualization

**Customization:**
You can modify the case study script to analyze your own reports by changing the `new_scam` variable in `run_case_study_example()`.

### Step 4: Run Attack Simulation

Demonstrate security vulnerabilities in old systems vs. secure new system by simulating various attacks.

**Command:**

```bash
python src/attack_simulation.py --csv "Data Set/scam_raw_dataset.csv" --secure "Data Set/secure_reports.jsonl" --server-priv "Data Set/server_rsa_private.json"
```

**Parameters:**
- `--csv`: Path to raw CSV dataset (for old system simulation)
- `--secure`: Path to secure reports JSONL (for new system)
- `--server-priv`: Path to server private key JSON

**Simulated Attacks:**
1. **Eavesdropping**: Network interception - shows plaintext vs. encrypted data
2. **Tampering**: Report modification - demonstrates HMAC and signature protection
3. **Replay Attack**: Reusing old reports - shows timestamp protection
4. **Man-in-the-Middle**: Fake report injection - demonstrates signature verification
5. **Signature Forgery**: Attempting to forge signatures - shows RSA signature security
6. **Key Swapping**: Swapping encrypted keys between reports - shows OAEP label protection
7. **Pattern Analysis**: Detecting patterns in repeated reports - shows unique IV protection

**Output:**
Detailed console output showing:
- How old system (plaintext) is vulnerable to each attack
- How new system (encrypted) prevents each attack
- Security comparison summary

## Features

### Preprocessing Module (`preprocessing.py`)

- **Dataset Loading**: Automatic detection of text columns, support for CSV files
- **Text Cleaning**:
  - URL removal using regex patterns
  - Contraction expansion (e.g., "don't" → "do not")
  - Punctuation removal
  - Digit removal
  - Lowercase conversion
- **Normalization**:
  - Acronym expansion (90+ acronyms: ICA, DBS, OTP, etc.)
  - Misspelling correction (100+ common misspellings)
- **Tokenization**: Word tokenization using NLTK Punkt tokenizer
- **Stopword Removal**: Removes common English stopwords
- **Lemmatization**: Reduces words to base forms using spaCy

### Embedding Methods

#### Transformer Embeddings (`transformer_embeddings.py`)
- **Models Supported**: MiniLM-L6-v2 (default), MPNet, MiniLM-L12-v2
- **Features**: 
  - Batch processing for efficiency
  - GPU support (automatic detection)
  - Normalized embeddings
  - Model persistence
  - Similarity matrix computation

#### Doc2Vec Model (`doc2vec_model.py`)
- **Training Modes**: PV-DM (Distributed Memory) and PV-DBOW (Distributed Bag of Words)
- **Configurable Parameters**:
  - Vector size (embedding dimensionality)
  - Training epochs
  - Minimum word count
  - Learning rate (alpha, min_alpha)
- **Features**: 
  - Tagged document creation
  - Embedding generation
  - Model persistence
  - Similarity computation

### Similarity Measures (`similarity_measures.py`)

- **Jaccard Similarity**: 
  - Token-based or noun-phrase based
  - Uses spaCy for noun phrase extraction
  - Computes intersection over union
- **Cosine Similarity**: 
  - Computed from embedding vectors
  - Supports both transformer and Doc2Vec embeddings
- **Combined Similarity**: Weighted combination of multiple metrics
- **Similar Document Finding**: Finds top-N similar documents for new reports

### Clustering (`clustering.py`)

- **Similarity Graph-Based Clustering**:
  - Uses NetworkX for graph construction
  - Threshold-based edge creation (cosine + Jaccard)
  - Connected components for cluster identification
  - Minimum cluster size filtering
- **Cluster Statistics**: Size, document count, member lists

### Key Term Extraction (`tfidf_extraction.py`)

- **TF-IDF Vectorization**: 
  - Configurable n-gram ranges (unigrams, bigrams, etc.)
  - Custom stopword lists
  - Term-document matrix generation
- **Key Term Ranking**: 
  - TF-IDF score-based ranking
  - Top-N term selection
- **Sequence Arrangement**: 
  - Orders terms by median position in documents
  - Supports temporal ordering for crime scripts

### Temporal Ordering (`temporal_ordering.py`)

- **Crime Script Generation**:
  - Creates step-by-step crime scripts from key terms
  - Orders actions by sequence position
  - Links terms with next-term relationships
- **Visualization**:
  - Network graph visualization using NetworkX and Matplotlib
  - Node size based on TF-IDF scores
  - Edge weights based on term relationships
  - Configurable figure sizes and term limits

### Security Features (`secure_reports.py`, `manual_crypto.py`)

- **Encryption**:
  - AES-256-CBC for symmetric encryption
  - RSA-OAEP for key encryption
  - Unique IVs per report (derived from report_id, crime_type, timestamp)
- **Integrity Protection**:
  - HMAC-SHA256 for ciphertext verification
  - HMAC key derived from AES key + metadata
- **Authentication**:
  - RSA signatures with SHA-256 hashing
  - Hash includes: crime_type, report_id, timestamp, text
- **Domain-Aware Security**:
  - OAEP labels include crime type and report ID
  - IV generation includes report-specific metadata
  - Prevents key swapping and pattern analysis

## Configuration

### Doc2Vec Model Parameters

You can modify Doc2Vec parameters in `src/main.py`:

```python
doc2vec = Doc2VecModel(
    vector_size=50,      # Embedding dimension
    min_count=2,         # Minimum word frequency
    epochs=100,         # Training epochs
    dm=1,               # 1 for PV-DM, 0 for PV-DBOW
    alpha=0.025,        # Initial learning rate
    min_alpha=0.00025   # Final learning rate
)
```

### Clustering Parameters

Modify clustering thresholds in `src/main.py`:

```python
cluster_df = clustering.cluster_using_similarity_graph(
    similarity_matrix=transformer_similarity_matrix,
    doc_ids=doc_ids,
    threshold=0.7,              # Cosine similarity threshold
    jaccard_matrix=jaccard_similarity_matrix,
    jaccard_threshold=0.05,     # Jaccard similarity threshold
    min_cluster_size=2          # Minimum documents per cluster
)
```

### Transformer Model Selection

Change transformer model in `src/main.py`:

```python
transformer = TransformerEmbeddings(
    model_name='minilm',        # Options: 'minilm', 'mpnet', 'minilm_l12', 'mpnet_large'
    batch_size=32,
    show_progress=True
)
```

## Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- nltk >= 3.6
- spacy >= 3.4.0
- gensim >= 4.0.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- networkx >= 2.6.0
- sentence-transformers >= 2.2.0
- torch >= 1.9.0

## Dataset Requirements

### Input Dataset Format

Your `scam_raw_dataset.csv` should contain:
- **Required column**: `incident_description` - contains the scam report text
- **Optional column**: `submission_id` - unique identifier for each report

### Example CSV Structure

```csv
submission_id,incident_description
1,"I received a call from someone claiming to be from DBS bank asking for my OTP code."
2,"A person called me saying they were from Singapore Police Force and I had an outstanding warrant."
```

## Troubleshooting

### Common Issues

1. **spaCy model not found**: Run `python -m spacy download en_core_web_sm`
2. **NLTK data missing**: The script auto-downloads, but you can manually run `nltk.download('punkt')` and `nltk.download('stopwords')`
3. **Transformer model download**: First run may take time to download the model (sentence-transformers)
4. **Memory issues**: Reduce `--limit` in secure_reports.py or batch size in transformer embeddings
5. **RSA key size errors**: Use `--server-bits 1024` or higher for OAEP with 48-byte key material

## Security Notes

- **Private Keys**: Never commit private keys to version control
- **Key Management**: In production, use proper key management systems
- **Key Sizes**: Use 2048-bit or higher RSA keys for production (1024-bit is for demo)
- **Timestamp Validation**: Add timestamp validation in production to prevent replay attacks
- **Secure Storage**: Store encrypted reports and keys securely

## License

This project is for educational and research purposes.
