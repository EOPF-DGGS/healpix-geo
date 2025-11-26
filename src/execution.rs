#[macro_export]
macro_rules! maybe_parallelize {
    ($nthreads:ident, $first:expr, $second:expr $(,)?) => {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads($nthreads as usize)
                .build()
                .unwrap();
            pool.install(|| $first.par_for_each($second));
        }
        #[cfg(target_arch = "wasm32")]
        {
            $first.for_each($second);
        }
    };
}
