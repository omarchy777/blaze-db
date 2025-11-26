# Blaze-DB

Blaze-DB is a high-performance vector database written in Rust, designed for efficient storage and retrieval of vector
embeddings.

## Current State

- Reads text data from a specified file.
- Chunks the data into smaller batches.
- Generates vector embeddings for each batch using an external embedding provider (e.g., Ollama, LM Studio).
- Stores the generated embeddings on disk in JSON, Binary format.
- Fast loading embeddings from disk for vector operations.

## Roadmap

- Efficient indexing of vector embeddings.
- Fast vector search and similarity search.
- HSNW indexing for improved search performance.
- AN API for interacting with the database.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests. 🤧🏳️