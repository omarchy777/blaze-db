use memmap2::Mmap;
use rayon::current_num_threads;
use rayon::prelude::*;
use std::fs::File;
use std::io::Result;
use std::path::PathBuf;

#[derive(Debug)]
pub struct Ingestor {
    pub source: PathBuf,
    pub batch_size: usize,
}

impl Ingestor {
    pub fn new(source: PathBuf, batch_size: usize) -> Self {
        assert_eq!(batch_size % 8, 0, "Batch size must be a multiple of 8");
        assert!(
            source.exists() && source.is_file(),
            "Source file must exist and be a file"
        );
        Self { source, batch_size }
    }

    pub fn read_line(self) -> Result<Vec<Vec<String>>> {
        let file = File::open(self.source)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Split into chunks and process in parallel
        let chunk_size = (mmap.len() / current_num_threads()).max(1024 * 1024);

        let lines: Vec<String> = mmap
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                let s = String::from_utf8_lossy(chunk);
                s.lines()
                    .map(|line| line.trim().to_string())
                    .filter(|line| !line.is_empty())
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(lines.into_par_iter().chunks(self.batch_size).collect())
    }
}
