use anyhow::{Error, Result};
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Embeddings {
    pub data: Vec<EmbeddingData>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingData {
    pub index: i32,
    #[serde(skip_deserializing)]
    pub chunk: String,
    pub embedding: Vec<f64>,
}

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
    pub fn pretty_print(&self) {
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
}

pub async fn fetch_embeddings_from_provider(
    url: &str,
    model: &str,
    chunks: Vec<&String>,
) -> Result<Embeddings, Error> {
    let body = serde_json::json!({
        "model": model,
        "input": chunks,
    });

    let response = reqwest::Client::new().post(url).json(&body).send().await?;

    if !response.status().is_success() {
        return Err(Error::msg(format!(
            "Failed to fetch embeddings, Status: {}",
            response.status()
        )));
    }

    let mut embeddings_response: Embeddings = response.json().await?;

    embeddings_response
        .data
        .iter_mut()
        .enumerate()
        .for_each(|(index, embedding)| {
            if let Some(chunk) = chunks.get(index) {
                embedding.chunk = chunk.to_string();
            } else {
                embedding.chunk = String::from("");
            }
        });

    embeddings_response.data.par_iter().for_each(|embedding| {
        if embedding.index.is_negative() {
            eprintln!(
                "Warning: Received embedding with -ve index for chunk: {}",
                embedding.chunk
            );
        }
        if embedding.embedding.is_empty() {
            eprintln!(
                "Warning: Received empty embedding for chunk index {}",
                embedding.index
            );
        }
    });

    Ok(embeddings_response)
}
