pub enum DepthLike<'a> {
    Scalar(&'a u8),
    Array(&'a [u8]),
}
