use anyhow::{Context, Result};
use rayon::iter::ParallelIterator;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::task::spawn_blocking;

use crate::utils::EmbeddingData;

///
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorData {
    pub chunk: Vec<String>,
    pub embedding: Vec<Vec<f32>>,
    pub dimensions: usize,
    pub total_vectors: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingStore {
    pub batch_index: usize,
    pub items: Vec<EmbeddingData>,
}

impl EmbeddingStore {
    pub fn new(batch_index: usize, items: Vec<EmbeddingData>) -> Self {
        Self { batch_index, items }
    }

    pub fn debug_print(&self) {
        println!("Batch Index: {}", self.batch_index);
        self.items.iter().take(3).for_each(|item| {
            println!(
                "Index: {:?}\n Chunk: {:?}\n Embeddings (first 3): {:?}\n Embedding Length : {:?}\n",
                &item.index,
                &item.chunk,
                &item.embedding[..3],
                &item.dimensions,
            );
        });
    }

    /// Load multiple binary files from a directory
    pub async fn read_binary(dir_path: &str) -> Result<VectorData> {
        let bin_files: Vec<PathBuf> = fs::read_dir(dir_path)
            .with_context(|| format!("Failed to read directory: {:?}", dir_path))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "bin")
                    .unwrap_or(false)
            })
            .collect();

        if bin_files.is_empty() {
            anyhow::bail!("No .bin files found in {:?}", dir_path);
        }

        println!("Found {} binary files to load...", bin_files.len());

        let stores: Vec<EmbeddingStore> = bin_files
            .par_iter()
            .filter_map(|path| match Self::read_binary_file(path) {
                Ok(store) => Some(store),
                Err(e) => {
                    eprintln!("Failed to load {:?}: {}", path, e);
                    None
                }
            })
            .collect();

        // Flatten all items from all stores in parallel
        let (all_chunks, all_embeddings): (Vec<String>, Vec<Vec<f32>>) = stores
            .into_par_iter()
            .flat_map(|store| store.items)
            .map(|item| (item.chunk, item.embedding))
            .unzip();

        let dimensions = all_embeddings.first().map(|v| v.len()).unwrap_or(0);
        let total_vectors = all_embeddings.len();

        Ok(VectorData {
            chunk: all_chunks,
            embedding: all_embeddings,
            dimensions,
            total_vectors,
        })
    }

    /// Load from a single binary file
    fn read_binary_file(path: &Path) -> Result<EmbeddingStore> {
        let bytes = fs::read(path).with_context(|| format!("Failed to read file: {:?}", path))?;

        let store: EmbeddingStore = bincode::deserialize(&bytes)
            .with_context(|| format!("Failed to deserialize: {:?}", path))?;

        Ok(store)
    }

    /// Get a specific vector by index
    pub fn get_vector(&self, index: usize) -> Option<&[f32]> {
        self.items.get(index).map(|item| item.embedding.as_slice())
    }

    /// Get text chunk by index
    pub fn get_chunk(&self, index: usize) -> Option<&str> {
        self.items.get(index).map(|item| item.chunk.as_str())
    }

    /// Memory usage estimate in MB
    pub fn memory_usage_mb(&self) -> f64 {
        let vector_bytes: usize = self
            .items
            .par_iter()
            .map(|item| item.embedding.len() * size_of::<f32>())
            .sum();
        let metadata_bytes: usize = self
            .items
            .par_iter()
            .map(|item| {
                item.chunk.len() * size_of::<u8>()
                    + size_of::<usize>() // for index
                    + size_of::<usize>() // for dimensions
            })
            .sum();
        (vector_bytes + metadata_bytes) as f64 / (1024.0 * 1024.0)
    }

    /// Write the embedding store to a binary file
    pub async fn write_binary(&self, file_path: &str) -> Result<()> {
        let encoded = {
            let self_clone = self.clone();
            spawn_blocking(move || bincode::serialize(&self_clone)).await??
        };

        let formatted_path = format!("{}.bin", file_path);
        let file = File::create(formatted_path).await?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&encoded).await?;
        writer.flush().await?;

        Ok(())
    }

    /// Write the embedding store to a JSON file
    pub async fn write_json(&self, file_path: &str) -> Result<()> {
        let self_clone = self.clone();
        let json_bytes = spawn_blocking(move || serde_json::to_vec(&self_clone)).await??;

        let formatted_path = format!("{}.json", file_path);
        let file = File::create(formatted_path).await?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
        writer.write_all(&json_bytes).await?;
        writer.flush().await?;

        Ok(())
    }
}
