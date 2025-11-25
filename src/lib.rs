mod cli;
mod core;
mod utils;

pub mod prelude {
    pub use crate::utils::{EmbeddingStore, Ingestor, Provider, VectorData};
}
