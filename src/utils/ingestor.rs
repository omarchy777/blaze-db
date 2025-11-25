use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Result;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Ingestor {
    pub source: PathBuf,
    pub batch_size: usize,
}

impl Ingestor {
    pub fn new(source: impl Into<PathBuf>, batch_size: usize) -> Self {
        let source = source.into();
        assert_eq!(batch_size % 8, 0, "Batch size must be a multiple of 8");
        assert!(
            source.exists() && source.is_file(),
            "Source file must exist and be a file"
        );
        Self { source, batch_size }
    }

    /// Read lines from the source file and batch them
    pub fn read_line(&self) -> Result<Vec<Vec<String>>> {
        let file = File::open(&self.source)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let lines: Vec<String> = mmap
            .par_split(|&b| b == b'\n')
            .filter_map(|line_bytes| {
                if line_bytes.is_empty() {
                    return None;
                }
                // Decode each line as UTF-8, ignoring invalid sequences
                let s = String::from_utf8_lossy(line_bytes).trim().to_string();
                if s.is_empty() { None } else { Some(s) }
            })
            .collect();

        Ok(lines.into_par_iter().chunks(self.batch_size).collect())
    }
}
