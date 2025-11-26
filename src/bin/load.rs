use blaze_db::prelude::EmbeddingStore;
use colored::Colorize;

#[tokio::main]
async fn main() {
    println!();

    match EmbeddingStore::read_binary("./embeddings").await {
        Ok(vector_data) => {
            println!("{}", "Successfully loaded embeddings".green().bold());
            println!();
            println!("{}", "Stats:".yellow().bold());
            println!(
                " Total vectors: {}",
                vector_data.total_vectors.to_string().cyan()
            );
            println!(" Dimensions: {}", vector_data.dimensions.to_string().cyan());
            println!(
                " Total chunks: {}",
                vector_data.chunk.len().to_string().cyan()
            );
            println!("  Memory Usage: {}MB", vector_data.memory_usage_mb());
            println!();

            // Display sample data
            if !vector_data.chunk.is_empty() {
                println!(" {}", "Sample Data (first 3 items):".yellow().bold());
                for (index, (chunk, embedding)) in vector_data
                    .chunk
                    .iter()
                    .zip(vector_data.embedding.iter())
                    .take(3)
                    .enumerate()
                {
                    println!();
                    println!("  {} {}", "Item".blue(), index);
                    println!(
                        "    Chunk: {}",
                        chunk.chars().take(60).collect::<String>().cyan()
                    );
                    println!(
                        "    Embedding (first 5): {:?}",
                        &embedding[..5.min(embedding.len())]
                    );
                    println!("    Embedding length: {}", embedding.len());
                }
            }
        }
        Err(e) => {
            eprintln!("{}", "Failed to load embeddings".red().bold());
            eprintln!("Error: {}", e);
        }
    }
}
