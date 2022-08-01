# RAD - Rust Automatic Differentiation

A library for [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) written in Rust.

## Features

- [x] Forward autodiff
- [x] Simple vector library
- [x] Gradient descent
- [ ] Reverse autodiff
- [ ] Simple tensor library

## Example

Use forward autodiff and gradient descent to find the mean:

```rust
// Create a data vector and calculate the mean normally
let data: Vec<f32> = (1u8..6).map(f32::from).collect();
let data = Vector::from(data);
let mean = (&data).sum().first().unwrap() / data.len() as f32;
// Create a squared error loss function
let loss = |x: &FAD<Vector<f32>>| {
    let mut loss = (x - &data).square().sum();
    loss.grad = loss.grad.sum();
    loss
};
// Apply gradient descent for 100 iterations
let mut gd = GradientDescent::new(
    Vector::Scalar(0f32),
    loss,
    std::iter::repeat(Vector::Scalar(0.1f32)),
);
for _ in 0..100 {
    gd.next();
}
// Compare the gradient descent mean to the "proper mean"
let gd_mean = *gd.x.value.first().unwrap();
assert_eq!(mean, gd_mean);
```
