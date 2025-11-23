use memmap2::Mmap;
use rayon::current_num_threads;
use rayon::prelude::*;
use std::fs::File;
use std::io::Result;

pub fn read_line(file_path: &str) -> Result<Vec<String>> {
    let file = File::open(file_path)?;
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

    Ok(lines)
}

pub fn read_word(path: &str) -> Result<Vec<String>> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // Split into chunks and process in parallel
    let chunk_size = (mmap.len() / rayon::current_num_threads()).max(1024 * 1024);

    let words: Vec<String> = mmap
        .par_chunks(chunk_size)
        .flat_map(|chunk| extract_words_from_bytes(chunk))
        .collect(); // <-- I know this is bad

    Ok(words)
}

fn extract_words_from_bytes(bytes: &[u8]) -> Vec<String> {
    let mut words = Vec::new();
    let mut start = None;

    for (i, &byte) in bytes.iter().enumerate() {
        if byte.is_ascii_alphanumeric() {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            if let Ok(word) = std::str::from_utf8(&bytes[s..i]) {
                words.push(word.to_string());
            }
            start = None;
        }
    }

    // Handle last word
    if let Some(s) = start {
        if let Ok(word) = std::str::from_utf8(&bytes[s..]) {
            words.push(word.to_string());
        }
    }

    words
}
