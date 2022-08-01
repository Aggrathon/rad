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
// Create a squared-error loss function
let loss = |x: &FAD<Vector<f32>>| {
    let mut loss = (x - &data).square();
    loss.grad = loss.grad.sum(); // all forward passes in one go
    loss
};
// Apply gradient descent for 100 iterations
let learning_rate = 0.1f32;
let mut gd = GradientDescent::with_lr(Vector::Scalar(0f32), loss, learning_rate);
for _ in 0..100 {
    gd.step();
}
// Compare the gradient descent mean to the "proper mean"
match gd.value() {
    Vector::Scalar(gd_mean) => assert_eq!(mean, *gd_mean),
    _ => panic!(),
};
```
