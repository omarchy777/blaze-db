use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use vector_search::utils::{embedder, ingestor, writer};

#[tokio::main]
async fn main() {
    let url = "http://localhost:1234/v1/embeddings";
    let model = "text-embedding-qwen3-embedding-0.6b";
    let provider = embedder::Provider::new(url.into(), model.into());

    let batch_size: usize = 512;
    let ingestor = ingestor::Ingestor::new("./data/WarAndPeace.txt".into(), batch_size);
    let batched_data = ingestor::Ingestor::read_line(ingestor);

    match batched_data {
        Ok(batched_data) => {
            println!();
            println!("Batch size: {}", batch_size.to_string().cyan());
            println!("Total batch: {}", batched_data.len().to_string().blue());
            println!(
                "Total Lines: {}",
                (batched_data.len() * batch_size).to_string().green()
            );
            println!();

            let progress_bar = ProgressBar::new(batched_data.len().max(1) as u64);
            progress_bar.set_style(
                ProgressStyle::with_template("[{bar:40.cyan/blue}] {pos}/{len} batches")
                    .unwrap()
                    .progress_chars("##>"),
            );

            for (i, chunk) in batched_data.iter().enumerate() {
                let input: Vec<&String> = chunk.iter().collect();
                match embedder::Provider::fetch_embeddings(provider.clone(), input).await {
                    Ok(embeddings) => {
                        let embedding_json = writer::EmbeddingJson::new(i, embeddings);
                        //embedding_json.pretty_print();
                        let filename = format!("./embeddings/embeddings_batch_{}.json", i);
                        writer::EmbeddingJson::json_writer(embedding_json, &filename)
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
