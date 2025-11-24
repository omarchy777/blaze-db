use crate::utils::embedder::{EmbeddingData, Embeddings};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::task::spawn_blocking;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingStore {
    pub batch_index: usize,
    pub items: Vec<EmbeddingData>,
}

impl EmbeddingStore {
    pub fn new(batch_index: usize, items: Embeddings) -> Self {
        Self {
            batch_index,
            items: items.data,
        }
    }
    pub fn debug_print(&self) {
        println!("Batch Index: {}", self.batch_index);
        self.items.iter().take(3).for_each(|item| {
            println!(
                "Index: {:?}\n Chunk: {:?}\n Embeddings: {:?}\n Embedding Length: {:?}\n",
                &item.index,
                &item.chunk,
                &item.embedding[..3],
                &item.embedding.len(),
            );
        })
    }

    pub async fn write_binary(&self, file_path: &str) -> Result<()> {
        // Serialize to binary
        let encoded = {
            let self_clone = self.clone();
            spawn_blocking(move || bincode::serialize(&self_clone))
        }
        .await??;

        let file = File::create(file_path).await?;
        let mut writer = BufWriter::new(file);

        writer.write_all(&encoded).await?;
        writer.flush().await?;

        Ok(())
    }

    pub async fn write_json(&self, file_path: &str) -> Result<()> {
        // Serialize to bytes
        let self_clone = self.clone();
        let json_bytes = spawn_blocking(move || serde_json::to_vec(&self_clone)).await??;

        let file = File::create(file_path).await?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file); // 8MB buffer
        writer.write_all(&json_bytes).await?;
        writer.flush().await?;

        Ok(())
    }
}
