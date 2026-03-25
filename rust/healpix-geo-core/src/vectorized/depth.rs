pub enum DepthLike {
    // TODO: figure out how to avoid copying the data
    Scalar(u8),
    Array(Vec<u8>),
}
