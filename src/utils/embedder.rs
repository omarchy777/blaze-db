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
    pub embedding: Vec<f64>,
    pub index: i32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingJson {
    pub chucks: Vec<String>,
    pub embeddings: Embeddings,
}

impl EmbeddingJson {
    pub fn pretty_print(&self) {
        for (i, embedding_data) in self.embeddings.data.iter().enumerate() {
            println!(
                "Chunk: {}\nEmbedding Index: {}\nEmbedding Vector (First three index): {:?}\n",
                self.chucks[i],
                embedding_data.index,
                &embedding_data.embedding[..3] // Print first 3 values for brevity
            );
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct RequestBody {
    pub model: String,
    pub input: Vec<String>,
}

pub async fn fetch_embeddings_from_provider(
    url: &str,
    model: &str,
    chunks: Vec<&String>,
) -> Result<EmbeddingJson, Error> {
    let body = RequestBody {
        model: model.to_string(),
        input: chunks.into_iter().map(|s| s.to_string()).collect(),
    };

    let response = reqwest::Client::new().post(url).json(&body).send().await?;

    if !response.status().is_success() {
        return Err(Error::msg(format!(
            "Failed to fetch embeddings, Status: {}",
            response.status()
        )));
    }

    let embeddings_response: Embeddings = response.json().await?;

    embeddings_response.data.par_iter().for_each(|embedding| {
        if embedding.embedding.is_empty() {
            eprintln!(
                "Warning: Received empty embedding for chunk index {}",
                embedding.index
            );
        }
    });

    Ok(EmbeddingJson {
        chucks: body.input,
        embeddings: embeddings_response,
    })
}
