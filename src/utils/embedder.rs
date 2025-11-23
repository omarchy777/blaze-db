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

    pub async fn fetch_embeddings(
        self,
        chunks: Vec<&String>,
    ) -> Result<Embeddings, Error> {
        let body = serde_json::json!({
            "model": self.model,
            "input": chunks,
        });

        let response = reqwest::Client::new().post(self.url).json(&body).send().await?;

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
}
