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

#[derive(Debug, Clone)]
pub struct Provider {
    pub url: String,
    pub model: String,
}

impl Provider {
    pub fn new(url: String, model: String) -> Provider {
        assert!(model.len() > 0, "Model name cannot be empty");
        assert!(url.starts_with("http"), "URL must start with http/https");
        Provider { url, model }
    }

    pub async fn fetch_embeddings(&self, chunks: &Vec<String>) -> Result<Embeddings, Error> {
        let body = serde_json::json!({
            "model": &self.model,
            "input": chunks,
        });

        let response = reqwest::Client::new()
            .post(&self.url)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::msg(format!(
                "Failed to fetch embeddings, Status: {}",
                response.status()
            )));
        }

        let mut embeddings_response: Embeddings = response.json().await?;

        // Fill in the chunk data for each embedding
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

        // Validate embeddings and log warnings if necessary
        embeddings_response.data.par_iter().for_each(|embedding| {
            assert!(
                embedding.index >= 0,
                "Invalid embedding index: {}",
                embedding.index
            );
            assert!(
                !embedding.embedding.len() > 0,
                "Embedding vector is empty, Chunk Index: {}",
                embedding.index
            );
            assert!(
                !embedding.chunk.is_empty(),
                "Data should be populated, Chunk Index: {}",
                embedding.index
            );
        });

        Ok(embeddings_response)
    }
}
