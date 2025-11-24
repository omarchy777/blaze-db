use blaze_db::utils::{embedder, ingestor, storage};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;

#[tokio::main]
async fn main() {
    let url = "http://localhost:1234/v1/embeddings";
    let model = "text-embedding-qwen3-embedding-0.6b";
    let provider = embedder::Provider::new(url.into(), model.into());

    let batch_size: usize = 512;
    let ingestor = ingestor::Ingestor::new("./sample/War_and_peace.txt".into(), batch_size);

    match ingestor::Ingestor::read_line(&ingestor) {
        Ok(batched_data) => {
            let total_lines: usize = batched_data.par_iter().map(|b| b.len()).sum();
            println!();
            println!("Batch size: {}", batch_size.to_string().cyan());
            println!("Total batch: {}", batched_data.len().to_string().blue());
            println!("Total Lines: {}", total_lines.to_string().green());
            println!();

            let progress_bar = ProgressBar::new(batched_data.len().max(1) as u64);
            progress_bar.set_style(
                ProgressStyle::with_template("[{bar:40.cyan/blue}] {pos}/{len} batches")
                    .unwrap()
                    .progress_chars("##>-"),
            );

            for (index, chunk) in batched_data.iter().enumerate() {
                match embedder::Provider::fetch_embeddings(&provider, chunk).await {
                    Ok(embeddings) => {
                        let embedding_store = storage::EmbeddingStore::new(index, embeddings);
                        embedding_store.debug_print();
                        let filename = format!("./embeddings/embeddings_batch_{}.bin", index);
                        //embedding_store
                        //    .write_json(&filename)
                        //    .await
                        //   .expect("Failed to write embeddings to file");
                        embedding_store
                            .write_binary(&filename)
                            .await
                            .expect("Failed to write embeddings to file");
                        progress_bar.inc(1);
                    }
                    Err(e) => {
                        eprintln!("Error fetching embeddings: {}", e);
                    }
                }
            }
        }

        Err(e) => {
            eprintln!("Error reading lines: ({})", e);
        }
    }
}
