# Blaze-DB

Blaze-DB is a high-performance vector database written in Rust, designed for efficient storage and retrieval of vector
embeddings.

## Current State

- Reads text data from a specified file.
- Chunks the data into smaller batches (configurable batch sizes).
- Generates vector embeddings for each batch using Jina AI API.
- Stores the generated embeddings on disk in binary format for optimal performance.
- Fast parallel loading of embeddings from disk using memory-mapped files.
- **Semantic similarity search with multiple distance metrics** (Cosine, Euclidean, Dot Product).
- Top-K result retrieval with ranked scoring.
- Async/await architecture for non-blocking operations.
- Parallel processing with Rayon for compute-intensive operations.
- Performance benchmarking suite (~3.7ms per search on War and Peace dataset).

## Roadmap

- HNSW (Hierarchical Navigable Small World) indexing for improved search performance.
- Product quantization for memory optimization.
- HTTP API server for remote database access.
- Query filtering and metadata support.
- CLI client for database management.
- Incremental updates without full reindex.
- Distributed storage and sharding support.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests. 🤧🏳️