use colored::Colorize;
use rayon::iter::ParallelIterator;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator};
use vector_search::utils::{embedder, reader, writer};

#[tokio::main]
async fn main() {
    let path = "./examples/WarAndPeace.txt";

    let url = "http://localhost:1234/v1/embeddings";
    let model = "text-embedding-embeddinggemma-300m-qat";

    let content_lines = reader::read_line(path);
    let content_words = reader::read_word(path);

    let batch_size: usize = 512;

    match (content_lines, content_words) {
        (Ok(lines), Ok(words)) => {
            println!();
            println!("Lines: {}", lines.len().to_string().yellow());
            println!("Words: {}", words.len().to_string().yellow());

            println!("Batch size: {}", batch_size.to_string().cyan());
            println!(
                "Total batch: {}",
                (lines.len() as f64 / batch_size as f64)
                    .ceil()
                    .to_string()
                    .blue()
            );
            println!();

            let chunks: Vec<Vec<String>> = lines.into_par_iter().chunks(batch_size).collect();

            for (i, chunk) in chunks.iter().enumerate() {
                let input: Vec<&String> = chunk.iter().collect();
                match embedder::fetch_embeddings_from_provider(url, model, input).await {
                    Ok(embeddings) => {
                        let embedding_json = embedder::EmbeddingJson::new(i, embeddings);
                        embedding_json.pretty_print();
                        let filename = format!("./embeddings/embeddings_batch_{}.json", i);
                        writer::json_writer(embedding_json, &filename)
                            .await
                            .expect("Failed to write embeddings to file");
                    }
                    Err(e) => {
                        eprintln!("Error fetching embeddings: {}", e);
                    }
                }
            }
        }

        (Err(e), _) => {
            eprintln!("Error reading lines: {}", e);
        }

        (_, Err(e)) => {
            eprintln!("Error reading words: {}", e);
        }
    }
}
