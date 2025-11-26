mod cli;
mod core;
pub mod utils;

pub mod prelude {
    pub use crate::utils::{EmbeddingStore, Ingestor, Provider, VectorData};
}
