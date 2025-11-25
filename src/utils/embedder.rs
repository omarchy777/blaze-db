use anyhow::{Error, Result};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Embeddings {
    pub data: Vec<EmbeddingData>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
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
    pub fn new(url: impl Into<String>, model: impl Into<String>) -> Self {
        let url = url.into();
        let model = model.into();
        assert!(!model.is_empty(), "Model name cannot be empty");
        assert!(url.starts_with("http"), "URL must start with http/https");
        Self { url, model }
    }

    pub async fn fetch_embeddings(&self, chunks: &[String]) -> Result<Embeddings> {
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

        // Fill in the chunk for each embedding
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

        // Validate embeddings
        embeddings_response.data.par_iter().for_each(|embedding| {
            assert!(
                embedding.index >= 0,
                "Invalid embedding index: {}",
                embedding.index
            );
            assert!(
                !embedding.embedding.is_empty(),
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
