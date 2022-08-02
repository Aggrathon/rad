use crate::forward::FAD;
use crate::ops::One;
use std::ops::{Mul, SubAssign};

pub struct GradientDescent<T1, T2, F, I>
where
    T1: Mul<T2, Output = T1> + SubAssign<T1> + One,
    F: Fn(&FAD<T1>) -> FAD<T1>,
    I: Iterator<Item = T2>,
{
    x: FAD<T1>,
    loss_fn: F,
    lr: I,
}

impl<T1, T2, F, I> GradientDescent<T1, T2, F, I>
where
    T1: Mul<T2, Output = T1> + SubAssign<T1> + One,
    F: Fn(&FAD<T1>) -> FAD<T1>,
    I: Iterator<Item = T2>,
{
    pub fn new(x0: T1, loss: F, lr: I) -> Self {
        Self {
            x: FAD::from(x0),
            loss_fn: loss,
            lr,
        }
    }

    pub fn loss(&mut self) -> FAD<T1> {
        (self.loss_fn)(&self.x)
    }

    pub fn value(&mut self) -> &T1 {
        &self.x.value
    }

    pub fn step(&mut self) -> Option<T1> {
        let lr = self.lr.next()?;
        let res = (self.loss_fn)(&self.x);
        self.x.value -= res.grad * lr;
        Some(res.value)
    }
}

impl<T1, T2, F> GradientDescent<T1, T2, F, std::iter::Repeat<T2>>
where
    T1: Mul<T2, Output = T1> + SubAssign<T1> + One,
    F: Fn(&FAD<T1>) -> FAD<T1>,
    T2: Clone,
{
    pub fn with_lr(x0: T1, loss_fn: F, lr: T2) -> Self {
        Self {
            x: FAD::from(x0),
            loss_fn,
            lr: std::iter::repeat(lr),
        }
    }
}

impl<T1, T2, F, I> Iterator for GradientDescent<T1, T2, F, I>
where
    T1: Mul<T2, Output = T1> + SubAssign<T1> + One,
    F: Fn(&FAD<T1>) -> FAD<T1>,
    I: Iterator<Item = T2>,
{
    type Item = T1;

    fn next(&mut self) -> Option<Self::Item> {
        self.step()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::*;
    use crate::vector::Vector;

    #[test]
    fn mean_opt() {
        let data: Vec<f32> = (1u8..6).map(f32::from).collect();
        let data = Vector::from(data);
        let mean = (&data).sum().first().unwrap() / data.len() as f32;
        let loss = |x: &FAD<Vector<f32>>| {
            let mut loss = (x - &data).square();
            loss.grad = loss.grad.sum(); // all forward passes in one go
            loss
        };
        let learning_rate = 0.1f32;
        let mut gd = GradientDescent::with_lr(Vector::Scalar(0f32), loss, learning_rate);
        for _ in 0..100 {
            gd.step();
        }
        match gd.value() {
            Vector::Scalar(gd_mean) => assert_eq!(mean, *gd_mean),
            _ => panic!(),
        };
    }
}
