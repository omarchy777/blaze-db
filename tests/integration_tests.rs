use blaze_db::prelude::{EmbeddingStore, Ingestor};
use blaze_db::utils::EmbeddingData;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

#[tokio::test]
async fn test_ingest_to_storage_pipeline() {
    // Setup test file
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.txt");
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "This is line 1").unwrap();
    writeln!(file, "This is line 2").unwrap();

    // Test ingestion
    let ingestor = Ingestor::new(&file_path, 8);
    let batches = ingestor.read_line().unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].len(), 2);

    // Create mock embeddings manually
    let embeddings = vec![
        EmbeddingData {
            index: 0,
            chunk: "This is line 1".to_string(),
            embedding: vec![1.0, 2.0, 3.0],
            dimensions: 3,
        },
        EmbeddingData {
            index: 1,
            chunk: "This is line 2".to_string(),
            embedding: vec![4.0, 5.0, 6.0],
            dimensions: 3,
        },
    ];

    let store = EmbeddingStore::new(0, embeddings);

    // Test storage
    let output_path = dir.path().join("embeddings");
    store
        .write_binary(output_path.to_str().unwrap())
        .await
        .unwrap();

    // Verify file was created
    let binary_path = format!("{}.bin", output_path.to_str().unwrap());
    assert!(std::path::Path::new(&binary_path).exists());

    // Load and verify
    let loaded_store = EmbeddingStore::read_binary_file(std::path::Path::new(&binary_path))
        .await
        .unwrap();

    assert_eq!(loaded_store.items.len(), 2);
    assert_eq!(loaded_store.items[0].chunk, "This is line 1");
    assert_eq!(loaded_store.items[1].chunk, "This is line 2");
}

#[tokio::test]
async fn test_multiple_batch_processing() {
    // Setup test file with many lines to force multiple batches
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("large_test.txt");
    let mut file = File::create(&file_path).unwrap();

    // Write 10 lines with batch size 8 to get 2 batches
    for i in 1..=10 {
        writeln!(file, "Line number {}", i).unwrap();
    }

    let ingestor = Ingestor::new(&file_path, 8);
    let batches = ingestor.read_line().unwrap();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].len(), 8);
    assert_eq!(batches[1].len(), 2);

    // Create embeddings for first batch
    let embeddings1: Vec<EmbeddingData> = (0..8)
        .map(|i| EmbeddingData {
            index: i,
            chunk: format!("Line number {}", i + 1),
            embedding: vec![i as f32, (i + 1) as f32],
            dimensions: 2,
        })
        .collect();

    let store1 = EmbeddingStore::new(0, embeddings1);

    // Create embeddings for second batch
    let embeddings2: Vec<EmbeddingData> = (0..2)
        .map(|i| EmbeddingData {
            index: i,
            chunk: format!("Line number {}", i + 9),
            embedding: vec![(i + 8) as f32, (i + 9) as f32],
            dimensions: 2,
        })
        .collect();

    let store2 = EmbeddingStore::new(1, embeddings2);

    assert_eq!(store1.items.len(), 8);
    assert_eq!(store2.items.len(), 2);

    // Test writing both batches
    let batch1_path = dir.path().join("batch_0");
    let batch2_path = dir.path().join("batch_1");

    store1
        .write_binary(batch1_path.to_str().unwrap())
        .await
        .unwrap();
    store2
        .write_binary(batch2_path.to_str().unwrap())
        .await
        .unwrap();

    // Verify both files exist
    assert!(std::path::Path::new(&format!("{}.bin", batch1_path.to_str().unwrap())).exists());
    assert!(std::path::Path::new(&format!("{}.bin", batch2_path.to_str().unwrap())).exists());
}

#[tokio::test]
async fn test_empty_file_processing() {
    // Setup empty test file
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("empty.txt");
    File::create(&file_path).unwrap(); // Create empty file

    let ingestor = Ingestor::new(&file_path, 8);
    let batches = ingestor.read_line().unwrap();
    assert_eq!(batches.len(), 0); // No batches for empty file
}

#[tokio::test]
async fn test_unicode_text_processing() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("unicode.txt");
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "Hello 世界! This is unicode text.").unwrap();
    writeln!(file, "Café, naïve, résumé - accented characters").unwrap();
    writeln!(file, "😭 Emoji support test 🤧").unwrap();

    let ingestor = Ingestor::new(&file_path, 8);
    let _batches = ingestor.read_line().unwrap();

    // Create embeddings
    let embeddings = vec![
        EmbeddingData {
            index: 0,
            chunk: "Hello 世界! This is unicode text.".to_string(),
            embedding: vec![1.0, 2.0],
            dimensions: 2,
        },
        EmbeddingData {
            index: 1,
            chunk: "Café, naïve, résumé - accented characters".to_string(),
            embedding: vec![3.0, 4.0],
            dimensions: 2,
        },
        EmbeddingData {
            index: 2,
            chunk: "😭 Emoji support test 🤧".to_string(),
            embedding: vec![5.0, 6.0],
            dimensions: 2,
        },
    ];

    let store = EmbeddingStore::new(0, embeddings);

    // Verify unicode text is preserved
    assert_eq!(store.items[0].chunk, "Hello 世界! This is unicode text.");
    assert_eq!(
        store.items[1].chunk,
        "Café, naïve, résumé - accented characters"
    );
    assert_eq!(store.items[2].chunk, "😭 Emoji support test 🤧");

    // Test storage and retrieval of Unicode content
    let output_path = dir.path().join("unicode_embeddings");
    store
        .write_binary(output_path.to_str().unwrap())
        .await
        .unwrap();

    let binary_path = format!("{}.bin", output_path.to_str().unwrap());
    let loaded_store = EmbeddingStore::read_binary_file(std::path::Path::new(&binary_path))
        .await
        .unwrap();

    // Verify unicode is preserved after serialization/deserialization
    assert_eq!(
        loaded_store.items[0].chunk,
        "Hello 世界! This is unicode text."
    );
    assert_eq!(
        loaded_store.items[1].chunk,
        "Café, naïve, résumé - accented characters"
    );
    assert_eq!(loaded_store.items[2].chunk, "😭 Emoji support test 🤧");
}

#[tokio::test]
async fn test_large_embedding_dimensions() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.txt");
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "Test with large embedding dimensions").unwrap();

    let ingestor = Ingestor::new(&file_path, 8);
    let _batches = ingestor.read_line().unwrap();

    // Create realistic high-dimensional embeddings (like GPT embeddings)
    let embedding_vector = (0..1536).map(|i| i as f32 * 0.01).collect::<Vec<f32>>();

    let embeddings = vec![EmbeddingData {
        index: 0,
        chunk: "Test with large embedding dimensions".to_string(),
        embedding: embedding_vector.clone(),
        dimensions: 1536,
    }];

    let store = EmbeddingStore::new(0, embeddings);

    assert_eq!(store.items[0].dimensions, 1536);
    assert_eq!(store.items[0].embedding.len(), 1536);

    // Test storage and retrieval
    let output_path = dir.path().join("large_embeddings");
    store
        .write_binary(output_path.to_str().unwrap())
        .await
        .unwrap();

    let binary_path = format!("{}.bin", output_path.to_str().unwrap());
    let loaded_store = EmbeddingStore::read_binary_file(std::path::Path::new(&binary_path))
        .await
        .unwrap();

    assert_eq!(loaded_store.items[0].dimensions, 1536);
    assert_eq!(loaded_store.items[0].embedding.len(), 1536);
    assert_eq!(loaded_store.items[0].embedding, embedding_vector);
}
