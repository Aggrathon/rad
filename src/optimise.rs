use crate::forward::FAD;
use crate::ops::One;
use std::ops::{Mul, SubAssign};

pub struct GradientDescent<T1, T2, F, I>
where
    T1: Mul<T2, Output = T1> + SubAssign<T1> + One,
    F: Fn(&FAD<T1>) -> FAD<T1>,
    I: Iterator<Item = T2>,
{
    pub x: FAD<T1>,
    loss: F,
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
            loss,
            lr,
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
        let lr = self.lr.next()?;
        let res = (self.loss)(&self.x);
        let delta = res.grad * lr;
        self.x.value -= delta;
        Some(res.value)
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
            let mut loss = (x - &data).square().sum();
            loss.grad = loss.grad.sum();
            loss
        };
        let mut gd = GradientDescent::new(
            Vector::Scalar(0f32),
            loss,
            std::iter::repeat(Vector::Scalar(0.1f32)),
        );
        for _ in 0..100 {
            gd.next();
        }
        let gd_mean = *gd.x.value.first().unwrap();
        assert_eq!(mean, gd_mean);
        // println!(
        //     "{} : {} | {} : {}",
        //     mean,
        //     (&data - mean).square().sum().first().unwrap(),
        //     gd_mean,
        //     (&data - gd_mean).square().sum().first().unwrap()
        // );
    }
}
