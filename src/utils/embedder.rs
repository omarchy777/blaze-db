use anyhow::{Error, Result};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
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
        if model.is_empty() {
            // Default model if none provided
            let default_model = "text-embedding-nomic-embed-text-v1.5";
            println!("Model not provided. Using default model: {}", default_model);
            return Self {
                url,
                model: default_model.to_string(),
            };
        }
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

        // Validate & filter embeddings
        embeddings_response.data = embeddings_response
            .data
            .into_par_iter()
            .filter(|embedding| embedding.index >= 0 && !embedding.embedding.is_empty())
            .collect();

        Ok(embeddings_response)
    }
}
