# Crime Script Analysis Using NLP

Implementation of preprocessing and Doc2Vec model for crime script analysis of scam reports.

## Project Structure

```
Crime Script Analysis Using NLP/
├── Data Set/                # Dataset files (input and preprocessed)
├── Trained Models/          # Trained Doc2Vec models
├── Analysis Results/        # Output files (embeddings, similarity matrices)
├── src/                     # Source code
│   ├── preprocessing.py     # Text preprocessing module
│   ├── doc2vec_model.py    # Doc2Vec model implementation
│   └── main.py             # Main execution script
├── requirements.txt         # Python dependencies
└── README.md               # This file
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

## Usage

### Prepare Your Dataset

1. Place `scam_raw_dataset.csv` in the `Data Set/` directory
2. This should be the raw extracted dataset before any preprocessing
3. Ensure the CSV file has a column named `incident_description` containing the text data
4. Optionally include a `submission_id` column for document identification

**Note:** The script only uses `scam_raw_dataset.csv`. Intermediate files (scam_data_1.csv, scam_data_2.csv, etc.) are not needed.

### Run the Pipeline

```bash
cd src
python main.py
```

Or from the project root:

```bash
python src/main.py
```

## Features

### Preprocessing Module (`preprocessing.py`)

- **Dataset Loading**: Load CSV files with scam report data
- **Text Cleaning**:
  - URL removal
  - Contraction expansion
  - Punctuation removal
  - Digit removal
  - Lowercase conversion
- **Normalization**:
  - Acronym expansion
  - Misspelling correction
- **Tokenization**: Word tokenization using NLTK
- **Stopword Removal**: Remove common English stopwords
- **Lemmatization**: Reduce words to their base forms using spaCy

### Doc2Vec Model (`doc2vec_model.py`)

- **Model Training**: Train Doc2Vec models with configurable parameters
  - Vector size (embedding dimensionality)
  - Training epochs
  - DM mode (PV-DM or PV-DBOW)
  - Minimum word count
- **Embedding Generation**: Generate document embeddings for scam reports
- **Similarity Computation**: Compute cosine similarity matrix between all document pairs
- **Model Persistence**: Save and load trained models

## Output Files

After running the pipeline, you'll find:

1. **`Data Set/scam_data_preprocessed.csv`**: Preprocessed dataset with all text transformations
2. **`Trained Models/scam_doc2vec_model.model`**: Trained Doc2Vec model
3. **`Analysis Results/scam_document_embeddings.csv`**: Document embeddings (n_documents × vector_size)
4. **`Analysis Results/scam_similarity_matrix.csv`**: Cosine similarity matrix (n_documents × n_documents)

## Configuration

You can modify the Doc2Vec model parameters in `src/main.py`:

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

## Requirements

- Python 3.7+
- pandas
- numpy
- nltk
- spacy
- gensim
- scikit-learn
