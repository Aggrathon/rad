# RAD - Rust Automatic Differentiation

A library for [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) written in Rust.

This library implements both forward and reverse autodiff. Forward autodiff is simpler to use (and implement) since you can calculate everything in just a single pass. However, the process has to be repeated separately for every variable (multiple outputs are fine).

Reverse autodiff, also called backpropagation, has two passes: forward and reverse. Reverse autodiff can calculate the gradients for multiple variables at the same time, as long as there is just one single output. This makes it especially popular for machine learning since it is common to have multiple variables but just a single loss function (at a time). Conceptually, reverse autodiff creates a directed acyclic computation graph, where the values and graph structure are filled in in the forward pass and the gradients are added in the reverse pass.

## Features

- [x] Forward autodiff
- [x] Simple vector library
- [x] Gradient descent
- [x] Reverse autodiff
- [ ] More linear algebra

## Example

Use forward autodiff and gradient descent to find the mean:

```rust
// Create a data vector and calculate the mean normally
let data: Vec<f32> = (1u8..6).map(f32::from).collect();
let data = Vector::from(data);
let mean = (&data).sum().first().unwrap() / data.len() as f32;
// Create a squared-error loss function
let loss = |x: &FAD<Vector<f32>>| {
    let mut loss = (x - &data).square().sum();
    loss.grad = loss.grad.sum(); // all forward passes in one go
    loss
};
// Apply gradient descent
let mut gd = GradientDescent::with_lr(Vector::Scalar(0f32), loss, 1.0);
let steps = gd.count(); // Steps until convergence
// Compare the "gradient descent" mean to the "proper" mean
match gd.value() {
    Vector::Scalar(gd_mean) => assert!((mean - gd_mean).abs() < 1e-4),
    _ => panic!(),
};
```
