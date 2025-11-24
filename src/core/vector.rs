pub struct VectorData {
    pub chunk: String,
    pub vector: Vec<f32>,
    pub dimensions: usize,
}

impl VectorData {
    pub fn new(chunk: String, vector: Vec<f32>) -> VectorData {
        let dimensions = vector.len();
        VectorData {
            chunk,
            vector,
            dimensions,
        }
    }
}
