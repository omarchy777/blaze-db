use blaze_db::prelude::{EmbeddingStore, VectorData};
use blaze_db::utils::EmbeddingData;
use tempfile::tempdir;

#[tokio::test]
async fn test_embedding_store_creation() {
    let embedding_data = vec![
        EmbeddingData {
            index: 0,
            chunk: "test chunk 1".to_string(),
            embedding: vec![1.0, 2.0, 3.0],
            dimensions: 3,
        },
        EmbeddingData {
            index: 1,
            chunk: "test chunk 2".to_string(),
            embedding: vec![4.0, 5.0, 6.0],
            dimensions: 3,
        },
    ];

    let store = EmbeddingStore::new(0, embedding_data);

    assert_eq!(store.batch_index, 0);
    assert_eq!(store.items.len(), 2);
    assert_eq!(store.items[0].chunk, "test chunk 1");
    assert_eq!(store.items[1].chunk, "test chunk 2");
}

#[tokio::test]
async fn test_write_read_binary() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test_embeddings");

    let embedding_data = vec![EmbeddingData {
        index: 0,
        chunk: "test chunk".to_string(),
        embedding: vec![1.0, 2.0, 3.0],
        dimensions: 3,
    }];

    let store = EmbeddingStore::new(0, embedding_data);

    // Write binary
    store
        .write_binary(file_path.to_str().unwrap())
        .await
        .unwrap();

    // Read binary back
    let binary_path = format!("{}.bin", file_path.to_str().unwrap());
    let loaded_store = EmbeddingStore::read_binary_file(&std::path::Path::new(&binary_path))
        .await
        .unwrap();

    assert_eq!(loaded_store.batch_index, store.batch_index);
    assert_eq!(loaded_store.items.len(), store.items.len());
    assert_eq!(loaded_store.items[0].chunk, store.items[0].chunk);
    assert_eq!(loaded_store.items[0].embedding, store.items[0].embedding);
    assert_eq!(loaded_store.items[0].dimensions, store.items[0].dimensions);
}

#[tokio::test]
async fn test_write_json() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test_embeddings");

    let embedding_data = vec![EmbeddingData {
        index: 0,
        chunk: "test chunk".to_string(),
        embedding: vec![1.0, 2.0, 3.0],
        dimensions: 3,
    }];

    let store = EmbeddingStore::new(0, embedding_data);

    // Write JSON
    store.write_json(file_path.to_str().unwrap()).await.unwrap();

    // Verify file exists and has content
    let json_path = format!("{}.json", file_path.to_str().unwrap());
    assert!(std::path::Path::new(&json_path).exists());

    let content = std::fs::read_to_string(&json_path).unwrap();
    assert!(content.contains("test chunk"));
    assert!(content.contains("\"dimensions\":3"));
}

#[tokio::test]
async fn test_read_binary_multiple_files() {
    let dir = tempdir().unwrap();
    let embeddings_dir = dir.path().join("embeddings");
    std::fs::create_dir_all(&embeddings_dir).unwrap();

    // Create multiple embedding stores
    for i in 0..3 {
        let embedding_data = vec![EmbeddingData {
            index: i,
            chunk: format!("chunk {}", i),
            embedding: vec![i as f32, (i + 1) as f32],
            dimensions: 2,
        }];

        let store = EmbeddingStore::new(i, embedding_data);
        let file_path = embeddings_dir.join(format!("batch_{}", i));
        store
            .write_binary(file_path.to_str().unwrap())
            .await
            .unwrap();
    }

    // Read all files
    let vector_data = EmbeddingStore::read_binary(embeddings_dir.to_str().unwrap())
        .await
        .unwrap();

    assert_eq!(vector_data.total_vectors, 3);
    assert_eq!(vector_data.dimensions, 2);
    assert_eq!(vector_data.chunk.len(), 3);
    assert_eq!(vector_data.embedding.len(), 3);
}

#[tokio::test]
async fn test_read_binary_empty_directory() {
    let dir = tempdir().unwrap();
    let empty_dir = dir.path().join("empty");
    std::fs::create_dir_all(&empty_dir).unwrap();

    let result = EmbeddingStore::read_binary(empty_dir.to_str().unwrap()).await;

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("No .bin files found")
    );
}

#[tokio::test]
async fn test_read_binary_nonexistent_directory() {
    let result = EmbeddingStore::read_binary("/nonexistent/directory").await;

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Failed to read directory")
    );
}

#[test]
fn test_vector_data_get_vector() {
    let vector_data = VectorData {
        chunk: vec!["chunk1".to_string(), "chunk2".to_string()],
        embedding: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        dimensions: 2,
        total_vectors: 2,
    };

    assert_eq!(vector_data.get_vector(0), Some([1.0, 2.0].as_slice()));
    assert_eq!(vector_data.get_vector(1), Some([3.0, 4.0].as_slice()));
    assert_eq!(vector_data.get_vector(2), None);
}

#[test]
fn test_vector_data_get_chunk() {
    let vector_data = VectorData {
        chunk: vec!["chunk1".to_string(), "chunk2".to_string()],
        embedding: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        dimensions: 2,
        total_vectors: 2,
    };

    assert_eq!(vector_data.get_chunk(0), Some("chunk1"));
    assert_eq!(vector_data.get_chunk(1), Some("chunk2"));
    assert_eq!(vector_data.get_chunk(2), None);
}

#[test]
fn test_vector_data_memory_usage() {
    let vector_data = VectorData {
        chunk: vec!["test".to_string()],
        embedding: vec![vec![1.0; 100]], // 100 f32 values
        dimensions: 100,
        total_vectors: 1,
    };

    let memory_mb = vector_data.memory_usage_mb();
    assert!(memory_mb > 0.0);
    // Should be approximately 400 bytes (100 * 4) + 4 bytes for "test" = ~0.0004 MB
    assert!(memory_mb < 1.0); // Should be less than 1MB
}

#[test]
fn test_vector_data_empty() {
    let vector_data = VectorData {
        chunk: vec![],
        embedding: vec![],
        dimensions: 0,
        total_vectors: 0,
    };

    assert_eq!(vector_data.get_vector(0), None);
    assert_eq!(vector_data.get_chunk(0), None);
    assert_eq!(vector_data.memory_usage_mb(), 0.0);
}

#[tokio::test]
async fn test_embedding_store_debug_print() {
    let embedding_data = vec![EmbeddingData {
        index: 0,
        chunk: "test chunk for debugging".to_string(),
        embedding: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        dimensions: 5,
    }];

    let store = EmbeddingStore::new(42, embedding_data);

    // This should not panic - just testing the debug_print method doesn't crash
    store.debug_print();

    assert_eq!(store.batch_index, 42);
    assert_eq!(store.items[0].dimensions, 5);
}
