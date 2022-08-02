use crate::ops::{Epsilon, One};
use crate::{forward::FAD, ops::Half};
use std::ops::{Mul, MulAssign, SubAssign};

pub struct GradientDescent<T1, T2, F>
where
    T1: Mul<T2, Output = T1> + SubAssign<T1>,
    F: Fn(&FAD<T1>) -> FAD<T1>,
{
    x: FAD<T1>,
    loss_fn: F,
    lr: T2,
    lr_alpha: T2,
    loss_last: Option<T1>,
}

impl<T1, T2, F> GradientDescent<T1, T2, F>
where
    T1: Mul<T2, Output = T1> + SubAssign<T1> + One,
    T2: Half,
    F: Fn(&FAD<T1>) -> FAD<T1>,
{
    pub fn new(x0: T1, loss: F, lr: T2) -> Self {
        Self {
            x: FAD::from(x0),
            loss_fn: loss,
            lr,
            lr_alpha: T2::half(),
            loss_last: None,
        }
    }
}

impl<T1, T2, F> GradientDescent<T1, T2, F>
where
    T1: Mul<T2, Output = T1> + SubAssign<T1>,
    F: Fn(&FAD<T1>) -> FAD<T1>,
{
    pub fn set_lr(mut self, lr: T2, alpha: T2) -> Self {
        self.lr = lr;
        self.lr_alpha = alpha;
        self.loss_last = None;
        self
    }

    pub fn change_lr(&mut self, lr: T2, alpha: T2) -> &mut Self {
        self.lr = lr;
        self.lr_alpha = alpha;
        self.loss_last = None;
        self
    }

    pub fn loss(&mut self) -> FAD<T1> {
        (self.loss_fn)(&self.x)
    }

    pub fn value(&mut self) -> &T1 {
        &self.x.value
    }
}

impl<T1, T2, F> GradientDescent<T1, T2, F>
where
    T1: Mul<T2, Output = T1> + SubAssign<T1> + PartialOrd,
    T2: Clone + MulAssign<T2>,
    F: Fn(&FAD<T1>) -> FAD<T1>,
{
    pub fn step(&mut self) -> &T1 {
        let res = (self.loss_fn)(&self.x);
        if let Some(l) = &self.loss_last {
            if res.value >= *l {
                self.lr *= self.lr_alpha.clone();
            }
        }
        self.x.value -= res.grad * self.lr.clone();
        self.loss_last = Some(res.value);
        self.loss_last.as_ref().unwrap()
    }
}

impl<'a, T1, T2, F> Iterator for &'a mut GradientDescent<T1, T2, F>
where
    T1: Mul<T2, Output = T1> + SubAssign<T1> + PartialOrd + Clone,
    T2: Clone + MulAssign<T2> + Epsilon + PartialOrd,
    F: Fn(&FAD<T1>) -> FAD<T1>,
{
    type Item = T1;

    fn next(&mut self) -> Option<Self::Item> {
        if self.lr.le(&T2::epsilon()) {
            None
        } else {
            Some(self.step().clone())
        }
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
            loss.grad = loss.grad.sum(); // all forward passes in one go
            loss
        };
        let mut gd = GradientDescent::new(Vector::Scalar(0f32), loss, 1.0);
        let steps = gd.count();
        assert_eq!(34, steps);
        match gd.value() {
            Vector::Scalar(gd_mean) => {
                assert!((mean - gd_mean).abs() < 1e-4, "{} != {}", mean, gd_mean)
            }
            _ => panic!(),
        };
    }
}
