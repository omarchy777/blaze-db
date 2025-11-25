use bencher::{Bencher, benchmark_group, benchmark_main};
use blaze_db::prelude::Ingestor;

fn my_bench(b: &mut Bencher) {
    let ingest = Ingestor::new("./sample/War_and_peace.txt", 512);
    b.iter(|| {
        Ingestor::read_line(&ingest).expect("Bad Thing");
    });
}

benchmark_group!(benches, my_bench);
benchmark_main!(benches);
