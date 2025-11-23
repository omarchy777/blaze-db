use crate::utils::embedder::EmbeddingJson;
use anyhow::Result;
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::task::spawn_blocking;

pub async fn json_writer(embedding_json: EmbeddingJson, file_path: &str) -> Result<()> {
    // Serialize in a blocking task to not block async runtime
    let json_bytes =
        spawn_blocking(move || serde_json::to_vec_pretty(&embedding_json))
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    // Write asynchronously with large buffer
    let file = File::create(file_path).await?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file); // 16MB buffer

    writer.write_all(&json_bytes).await?;
    writer.flush().await?;

    Ok(())
}
