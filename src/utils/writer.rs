use crate::utils::embedder::{EmbeddingData, Embeddings};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::task::spawn_blocking;

#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingJson {
    pub batch_index: usize,
    pub items: Vec<EmbeddingData>,
}

impl EmbeddingJson {
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

    pub async fn json_writer(embedding_json: EmbeddingJson, file_path: &str) -> Result<()> {
        // Serialize to bytes in a blocking task for minimal overhead
        let json_bytes = spawn_blocking(move || serde_json::to_vec(&embedding_json)).await??;

        let file = File::create(file_path).await?;
        let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file); // 16MB buffer
        writer.write_all(&json_bytes).await?;
        writer.flush().await?;

        Ok(())
    }
}
