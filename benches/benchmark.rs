use bencher::{Bencher, benchmark_group, benchmark_main};
use blaze_db::utils::ingestor::Ingestor;

fn my_bench(b: &mut Bencher) {
    b.iter(|| {
        let ingest = Ingestor::new("./sample/War_and_peace.txt".into(), 512);
        Ingestor::read_line(&ingest).expect("Bad Thing");
        // code to test
    });
}

benchmark_group!(benches, my_bench);
benchmark_main!(benches);
