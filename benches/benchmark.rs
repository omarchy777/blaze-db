use bencher::{Bencher, benchmark_group, benchmark_main};
use blaze_db::prelude::{EmbeddingStore, Ingestor};
use tokio::runtime::Runtime;

fn read(b: &mut Bencher) {
    let ingest = Ingestor::new("./sample/War_and_peace.txt", 512);
    b.iter(|| {
        Ingestor::read_line(&ingest).expect("Bad Thing");
    });
}

fn load(b: &mut Bencher) {
    let rt = Runtime::new().unwrap();
    b.iter(|| {
        rt.block_on(async {
            let _data = EmbeddingStore::read_binary("./embeddings").await;
        })
    });
}

benchmark_group!(benches, read, load);
benchmark_main!(benches);
