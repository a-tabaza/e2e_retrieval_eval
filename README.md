# Multilingual Retrieval Evaluation Framework

This framework provides an end-to-end solution for evaluating retrieval systems using Wikipedia articles as a test corpus. It specializes in Arabic language content but can be adapted for other languages. It is a work in progress and I am only open sourcing a part of it for now with plans for open sourcing the entirety of my experiments and methodology in the future.

## Overview

The framework implements the following pipeline:
1. Data Collection: Fetches Wikipedia articles
2. Text Processing: Chunks articles into meaningful segments
3. Query Generation: Creates natural language queries using GPT-4
4. Embedding Generation: Generates embeddings using Cohere's multilingual model
5. Index Creation: Builds a vector index for efficient retrieval
6. Evaluation: Measures retrieval performance using standard IR metrics

## Prerequisites

```bash
pip install uv
uv venv .venv
uv pip install ir-measures openai cohere python-dotenv usearch pymediawiki numpy
```

You'll need API keys for:
- OpenAI (for query generation)
- Cohere (for embeddings)

## Configuration

Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
```

## Usage

The framework is organized into several sequential steps:

### 1. Data Collection
```python
from mediawiki import MediaWiki
wikipedia = MediaWiki(lang="ar")
results = wikipedia.search("الثورة التونسية")  # Example search term
```

### 2. Text Chunking
Uses [cluster semantic chunking](https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/cluster_semantic_chunker.py) to break documents into meaningful segments:
```python
from chunking import ClusterSemanticChunker
text_splitter = ClusterSemanticChunker()
```

### 3. Query Generation
Generates diverse queries for each document chunk using GPT-4:
```python
client = OpenAI()
# Generate queries using the Query class
```

### 4. Embedding Generation
Creates embeddings using Cohere's multilingual model:
```python
co = ClientV2()
# Generate embeddings for documents and queries
```

### 5. Indexing
Creates a vector index for efficient retrieval:
```python
chunks_index = Index(ndim=1024, metric='cos')
chunks_index.add(keys, embeddings)
```

### 6. Evaluation
Evaluates retrieval performance using standard IR metrics:
```python
ir_measures.calc_aggregate([P@1, P@3, P@5, R@1, R@3, R@5], qrels, results)
```

## Output Files

The framework generates several JSON files during processing:
- `data.json`: Raw Wikipedia articles
- `chunks.json`: Chunked documents
- `chunks_with_queries.json`: Documents with generated queries
- `query_with_ground_truth.json`: Query-document relevance pairs
- `chunks_with_queries_and_embeddings.json`: Complete document data with embeddings
- `query_with_ground_truth_and_embeddings.json`: Complete query data with embeddings
- `chunk_key_mapping.json`: Mapping between chunk IDs and index keys
- `qrels.txt`: TREC-format relevance judgments

The files aren't uploaded since I don't want to set up LFS.

## Evaluation Metrics

The framework evaluates retrieval performance using:
- Precision@k (k=1,3,5)
- Recall@k (k=1,3,5)

## Customization

To adapt the framework for different languages:
1. Modify the MediaWiki language parameter
2. Adjust the prompt for query generation
3. Use appropriate multilingual embeddings

## Contributing

Feel free to submit issues and enhancement requests. If you flag any glaring errors in my methodology please do let me know, I am new to all of this and always eager to learn.
