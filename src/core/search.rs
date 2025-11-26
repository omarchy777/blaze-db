use crate::utils::VectorData;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Serialize, Deserialize)]
pub struct SearchQuery {
    pub top_k: usize,
    pub query_vector: Vec<f32>,
    pub metric: Metrics,
}

#[derive(Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk: String,
    pub score: f32,
}

impl SearchQuery {
    pub fn new(top_k: usize, query_vector: Vec<f32>, metric: Metrics) -> Self {
        Self {
            top_k,
            query_vector,
            metric,
        }
    }

    pub fn search(&self, data: &VectorData) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = data
            .embedding
            .iter()
            .enumerate()
            .map(|(idx, vector)| {
                let score = self.metric.calculate(&self.query_vector, vector);
                SearchResult {
                    chunk: data.chunk[idx].clone(),
                    score,
                }
            })
            .collect();

        // Sort results by score in descending order
        results.sort_by(|a, b| {
            // Compare scores, treating NaN as less than any number
            match a.score.is_nan().cmp(&b.score.is_nan()) {
                Ordering::Equal => b.score.partial_cmp(&a.score).unwrap(),
                other => other,
            }
        });

        // Return top_k results
        let top_results = results.into_iter().take(self.top_k).collect();
        top_results
    }
}

#[derive(Serialize, Deserialize)]
pub enum Metrics {
    Cosine,
    Euclidean,
}

impl Metrics {
    pub fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metrics::Cosine => cosine_similarity(a, b),
            Metrics::Euclidean => euclidean_similarity(a, b),
        }
    }
}

/// Cosine similarity: dot(a,b) / (||a|| * ||b||)
/// Returns value in [-1, 1],
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let (dot, norm_a_sq, norm_b_sq) = a
        .iter()
        .zip(b.iter())
        .fold((0.0f32, 0.0f32, 0.0f32), |(dot, na, nb), (&x, &y)| {
            (dot + x * y, na + x * x, nb + y * y)
        });

    let denominator = (norm_a_sq * norm_b_sq).sqrt();
    if denominator < f32::EPSILON {
        0.0
    } else {
        dot / denominator
    }
}

/// Similarity = 1 / (1 + Euclidean distance)
/// Returns value in (0, 1]
pub fn euclidean_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let distance_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();

    1.0 / (1.0 + distance_sq.sqrt())
}
