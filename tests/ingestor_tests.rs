use blaze_db::prelude::Ingestor;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn test_ingestor_creation_valid_file() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.txt");
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "test content").unwrap();

    let ingestor = Ingestor::new(&file_path, 8);

    assert_eq!(ingestor.source, file_path);
    assert_eq!(ingestor.batch_size, 8);
}

#[test]
#[should_panic(expected = "Batch size must be a multiple of 8")]
fn test_ingestor_invalid_batch_size() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.txt");
    File::create(&file_path).unwrap();

    Ingestor::new(&file_path, 7); // Should panic
}

#[test]
#[should_panic(expected = "Source file must exist and be a file")]
fn test_ingestor_nonexistent_file() {
    Ingestor::new("/nonexistent/file.txt", 8);
}

#[test]
#[should_panic(expected = "Source file must exist and be a file")]
fn test_ingestor_directory_instead_of_file() {
    let dir = tempdir().unwrap();
    // Try to create ingestor with directory path instead of file
    Ingestor::new(dir.path(), 8);
}

#[test]
fn test_read_lines_basic() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.txt");
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "line 1").unwrap();
    writeln!(file, "line 2").unwrap();
    writeln!(file, "line 3").unwrap();

    let ingestor = Ingestor::new(&file_path, 8);
    let result = ingestor.read_line().unwrap();

    assert_eq!(result.len(), 1); // Single batch
    assert_eq!(result[0].len(), 3); // 3 lines
    assert_eq!(result[0][0], "line 1");
    assert_eq!(result[0][1], "line 2");
    assert_eq!(result[0][2], "line 3");
}

#[test]
fn test_read_lines_multiple_batches() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.txt");
    let mut file = File::create(&file_path).unwrap();

    // Write 10 lines
    for i in 1..=10 {
        writeln!(file, "line {}", i).unwrap();
    }

    let ingestor = Ingestor::new(&file_path, 8);
    let result = ingestor.read_line().unwrap();

    assert_eq!(result.len(), 2); // Two batches: 8 + 2
    assert_eq!(result[0].len(), 8);
    assert_eq!(result[1].len(), 2);
    assert_eq!(result[0][0], "line 1");
    assert_eq!(result[1][0], "line 9");
}

#[test]
fn test_read_lines_empty_lines() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.txt");
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "line 1").unwrap();
    writeln!(file).unwrap(); // Empty line
    writeln!(file, "   ").unwrap(); // Whitespace only line
    writeln!(file, "line 4").unwrap();

    let ingestor = Ingestor::new(&file_path, 8);
    let result = ingestor.read_line().unwrap();

    assert_eq!(result[0].len(), 2); // Empty lines should be filtered
    assert_eq!(result[0][0], "line 1");
    assert_eq!(result[0][1], "line 4");
}

#[test]
fn test_read_lines_large_batch_size() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.txt");
    let mut file = File::create(&file_path).unwrap();

    // Write 5 lines with batch size larger than line count
    for i in 1..=5 {
        writeln!(file, "line {}", i).unwrap();
    }

    let ingestor = Ingestor::new(&file_path, 1024); // Batch size > line count
    let result = ingestor.read_line().unwrap();

    assert_eq!(result.len(), 1); // Single batch
    assert_eq!(result[0].len(), 5); // All 5 lines
}

#[test]
fn test_read_lines_exact_batch_boundary() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.txt");
    let mut file = File::create(&file_path).unwrap();

    // Write exactly 8 lines with batch size 8
    for i in 1..=8 {
        writeln!(file, "line {}", i).unwrap();
    }

    let ingestor = Ingestor::new(&file_path, 8);
    let result = ingestor.read_line().unwrap();

    assert_eq!(result.len(), 1); // Single batch
    assert_eq!(result[0].len(), 8); // All 8 lines
}

#[test]
fn test_read_lines_empty_file() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.txt");
    File::create(&file_path).unwrap(); // Create empty file

    let ingestor = Ingestor::new(&file_path, 8);
    let result = ingestor.read_line().unwrap();

    assert_eq!(result.len(), 0); // No batches for empty file
}

#[test]
fn test_read_lines_utf8_content() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.txt");
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "Hello 世界").unwrap();
    writeln!(file, "Café ñoño").unwrap();

    let ingestor = Ingestor::new(&file_path, 8);
    let result = ingestor.read_line().unwrap();

    assert_eq!(result[0].len(), 2);
    assert_eq!(result[0][0], "Hello 世界");
    assert_eq!(result[0][1], "Café ñoño");
}
