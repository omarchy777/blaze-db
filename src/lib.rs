mod cli;
mod core;
pub mod utils;

pub mod prelude {
    pub use crate::core::{Metrics, SearchQuery, SearchResult};
    pub use crate::utils::{EmbeddingStore, Ingestor, Provider, VectorData};
}
