/// Forward Automatic Differentiation
use crate::ops::*;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Wrapper for automatic differentiation in forward mode.
/// Wrap the parameter you want to differentiate and do the calculations normally.
/// The gradient vill automatically be accumulated.
///
/// # Example
/// ```
/// use rad::forward::FAD;
/// use rad::ops::*;
///
/// let x = FAD::from(2.0f32);
/// let y = ((-(x * 2.5)).exp() + 3.0).sqrt();
/// assert!((y.value - 1.7339948).abs() < 1e-6, "{:?}", y);
/// assert!((y.grad - -0.0048572426).abs() < 1e-6, "{:?}", y);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct FAD<T> {
    /// The wrapped value, usually a f32/f64 or a vector/matrix/tensor.
    pub value: T,
    /// The current gradient of the initial parameter (wrapped value).
    pub grad: T,
}

impl<T> From<T> for FAD<T>
where
    T: One,
{
    #[inline]
    fn from(value: T) -> Self {
        FAD {
            value,
            grad: T::one(),
        }
    }
}

impl<T> From<(T, T)> for FAD<T> {
    #[inline]
    fn from(value: (T, T)) -> Self {
        FAD {
            value: value.0,
            grad: value.1,
        }
    }
}

impl<T, O> NumOps<T, FAD<O>> for FAD<T>
where
    T: NumOps<T, O>
        + Mul<O, Output = O>
        + Div<O, Output = O>
        + Sub<T, Output = O>
        + Pow<O, Output = O>
        + One
        + Into<O>
        + Clone,
    O: Sub<O, Output = O> + Mul<O, Output = O> + Clone,
{
}

impl<T, O> Add<T> for FAD<T>
where
    T: Add<T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.add(rhs),
            grad: self.grad.into(),
        }
    }
}

impl<T> One for FAD<T>
where
    T: One,
{
    fn one() -> Self {
        T::one().into()
    }
}

impl<'a, T, O> Add<T> for &'a FAD<T>
where
    &'a T: Add<T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.add(rhs),
            grad: (&self.grad).into(),
        }
    }
}

impl<'a, T, O> Add<&'a T> for FAD<T>
where
    T: Add<&'a T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn add(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.add(rhs),
            grad: self.grad.into(),
        }
    }
}

impl<'a, T, O> Add<&'a T> for &'a FAD<T>
where
    &'a T: Add<&'a T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn add(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.add(rhs),
            grad: (&self.grad).into(),
        }
    }
}

impl<T, O> Sub<T> for FAD<T>
where
    T: Sub<T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.sub(rhs),
            grad: self.grad.into(),
        }
    }
}

impl<'a, T, O> Sub<T> for &'a FAD<T>
where
    &'a T: Sub<T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.sub(rhs),
            grad: (&self.grad).into(),
        }
    }
}

impl<'a, T, O> Sub<&'a T> for FAD<T>
where
    T: Sub<&'a T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sub(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.sub(rhs),
            grad: self.grad.into(),
        }
    }
}

impl<'a, T, O> Sub<&'a T> for &'a FAD<T>
where
    &'a T: Sub<&'a T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sub(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.sub(rhs),
            grad: (&self.grad).into(),
        }
    }
}

impl<T, O> Mul<T> for FAD<T>
where
    T: Mul<T, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.mul(rhs.clone()),
            grad: self.grad.mul(rhs),
        }
    }
}

impl<'a, T, O> Mul<T> for &'a FAD<T>
where
    &'a T: Mul<T, Output = O>,
    T: Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.mul(rhs.clone()),
            grad: self.grad.mul(rhs),
        }
    }
}

impl<'a, T, O> Mul<&'a T> for FAD<T>
where
    T: Mul<&'a T, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn mul(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.mul(rhs),
            grad: self.grad.mul(rhs),
        }
    }
}

impl<'a, T, O> Mul<&'a T> for &'a FAD<T>
where
    &'a T: Mul<&'a T, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn mul(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.mul(rhs),
            grad: self.grad.mul(rhs),
        }
    }
}

impl<T, O> Neg for FAD<T>
where
    T: Neg<Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn neg(self) -> Self::Output {
        FAD {
            value: self.value.neg(),
            grad: self.grad.neg(),
        }
    }
}

impl<'a, T, O> Neg for &'a FAD<T>
where
    &'a T: Neg<Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn neg(self) -> Self::Output {
        FAD {
            value: self.value.neg(),
            grad: self.grad.neg(),
        }
    }
}

impl<T, O> Abs for FAD<T>
where
    T: Abs<Output = O>,
{
    type Output = FAD<O>;

    fn abs(self) -> Self::Output {
        todo!()
    }
}

impl<T, O> Div<T> for FAD<T>
where
    T: Div<T, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.div(rhs.clone()),
            grad: self.grad.div(rhs),
        }
    }
}

impl<'a, T, O> Div<T> for &'a FAD<T>
where
    &'a T: Div<T, Output = O>,
    T: Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.div(rhs.clone()),
            grad: self.grad.div(rhs),
        }
    }
}

impl<'a, T, O> Div<&'a T> for FAD<T>
where
    T: Div<&'a T, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn div(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.div(rhs),
            grad: self.grad.div(rhs),
        }
    }
}

impl<'a, T, O> Div<&'a T> for &'a FAD<T>
where
    &'a T: Div<&'a T, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn div(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.div(rhs),
            grad: self.grad.div(rhs),
        }
    }
}

impl<T, O> Pow<T> for FAD<T>
where
    T: Pow<T, Output = O>
        + Pow<O, Output = O>
        + Mul<T, Output = O>
        + Sub<T, Output = O>
        + One
        + Clone,
    O: Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.clone().pow(rhs.clone()),
            grad: (self.grad * rhs.clone()) * self.value.pow(rhs - T::one()),
        }
    }
}

impl<'a, T, O> Pow<T> for &'a FAD<T>
where
    &'a T: Pow<T, Output = O> + Pow<O, Output = O> + Mul<T, Output = O>,
    T: Sub<T, Output = O> + One + Clone,
    O: Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, rhs: T) -> Self::Output {
        FAD {
            value: (&self.value).pow(rhs.clone()),
            grad: (&self.grad * rhs.clone()) * (&self.value).pow(rhs - T::one()),
        }
    }
}

impl<'a, T, O> Pow<&'a T> for FAD<T>
where
    T: Pow<&'a T, Output = O> + Pow<O, Output = O> + Mul<&'a T, Output = O> + One + Clone,
    &'a T: Sub<T, Output = O>,
    O: Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.clone().pow(rhs),
            grad: (self.grad * rhs) * self.value.pow(rhs - T::one()),
        }
    }
}

impl<'a, T, O> Pow<&'a T> for &'a FAD<T>
where
    &'a T:
        Pow<&'a T, Output = O> + Pow<O, Output = O> + Mul<&'a T, Output = O> + Sub<T, Output = O>,
    T: One,
    O: Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: (&self.value).pow(rhs),
            grad: (&self.grad * rhs) * (&self.value).pow(rhs - T::one()),
        }
    }
}

impl<T, O> Exp for FAD<T>
where
    T: Mul<T, Output = O> + Mul<O, Output = O> + Exp<Output = O>,
    O: Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn exp(self) -> Self::Output {
        let value = self.value.exp();
        let grad = self.grad * value.clone();
        FAD { value, grad: grad }
    }
}

impl<'a, T, O> Exp for &'a FAD<T>
where
    &'a T: Mul<O, Output = O> + Exp<Output = O>,
    O: Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn exp(self) -> Self::Output {
        let value = self.value.exp();
        let grad = &self.grad * value.clone();
        FAD { value, grad: grad }
    }
}

impl<T, O> Pow<FAD<T>> for T
where
    T: Mul<O, Output = O> + Ln<Output = O> + Pow<T, Output = O> + Clone,
    O: Mul<Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, lhs: FAD<T>) -> Self::Output {
        let value = self.clone().pow(lhs.value);
        let grad = lhs.grad * (self.ln() * value.clone());
        FAD { value, grad: grad }
    }
}

impl<'a, T, O> Pow<&'a FAD<T>> for T
where
    &'a T: Mul<O, Output = O>,
    T: Ln<Output = O> + Pow<&'a T, Output = O> + Clone,
    O: Mul<Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, lhs: &'a FAD<T>) -> Self::Output {
        let value = self.clone().pow(&lhs.value);
        let grad = &lhs.grad * (self.ln() * value.clone());
        FAD { value, grad: grad }
    }
}

impl<'a, T, O> Pow<FAD<T>> for &'a T
where
    &'a T: Pow<T, Output = O> + Ln<Output = O>,
    T: Mul<O, Output = O>,
    O: Mul<Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, lhs: FAD<T>) -> Self::Output {
        let value = self.pow(lhs.value);
        let grad = lhs.grad * (self.ln() * value.clone());
        FAD { value, grad: grad }
    }
}

impl<'a, T, O> Pow<&'a FAD<T>> for &'a T
where
    &'a T: Pow<&'a T, Output = O> + Ln<Output = O> + Mul<O, Output = O>,
    O: Mul<Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, lhs: &'a FAD<T>) -> Self::Output {
        let value = self.pow(&lhs.value);
        let grad = &lhs.grad * (self.ln() * value.clone());
        FAD { value, grad: grad }
    }
}

impl<T, O> Ln for FAD<T>
where
    T: Div<Output = O> + Ln<Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn ln(self) -> Self::Output {
        FAD {
            value: self.value.clone().ln(),
            grad: self.grad / self.value,
        }
    }
}

impl<'a, T, O> Ln for &'a FAD<T>
where
    &'a T: Div<&'a T, Output = O> + Ln<Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn ln(self) -> Self::Output {
        FAD {
            value: self.value.ln(),
            grad: &self.grad / &self.value,
        }
    }
}

impl<T, O> Log<T> for FAD<T>
where
    T: Div<O, Output = O> + Log<T, Output = O> + Ln<Output = O> + Mul<O, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn log(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.clone().log(rhs.clone()),
            grad: self.grad / (rhs * self.value.ln()),
        }
    }
}

impl<'a, T, O> Log<T> for &'a FAD<T>
where
    &'a T: Div<O, Output = O> + Log<T, Output = O> + Ln<Output = O>,
    T: Mul<O, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn log(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.log(rhs.clone()),
            grad: &self.grad / (rhs * self.value.ln()),
        }
    }
}

impl<'a, T, O> Log<&'a T> for FAD<T>
where
    T: Div<O, Output = O> + Log<&'a T, Output = O> + Ln<Output = O> + Clone,
    &'a T: Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn log(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.clone().log(rhs),
            grad: self.grad / (rhs * self.value.ln()),
        }
    }
}

impl<'a, T, O> Log<&'a T> for &'a FAD<T>
where
    &'a T: Div<O, Output = O> + Log<&'a T, Output = O> + Ln<Output = O> + Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn log(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.log(rhs),
            grad: &self.grad / (rhs * self.value.ln()),
        }
    }
}

impl<T, O> Sqrt for FAD<T>
where
    T: Sqrt<Output = O> + Half + Mul<T, Output = O>,
    O: Div<O, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn sqrt(self) -> Self::Output {
        let value = self.value.sqrt();
        FAD {
            value: value.clone(),
            grad: self.grad * T::half() / value,
        }
    }
}

impl<'a, T, O> Sqrt for &'a FAD<T>
where
    T: Half,
    &'a T: Sqrt<Output = O> + Mul<T, Output = O>,
    O: Div<O, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn sqrt(self) -> Self::Output {
        let value = self.value.sqrt();
        FAD {
            value: value.clone(),
            grad: &self.grad * T::half() / value,
        }
    }
}

impl<T, O> Trig for FAD<T>
where
    T: Mul<O, Output = O> + Div<O, Output = O> + Trig<Output = O> + Clone,
    O: Square<Output = O> + Neg<Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sin(self) -> Self::Output {
        FAD {
            value: self.value.clone().sin(),
            grad: self.grad * self.value.cos(),
        }
    }

    #[inline]
    fn cos(self) -> Self::Output {
        FAD {
            value: self.value.clone().cos(),
            grad: self.grad * (-self.value.sin()),
        }
    }

    #[inline]
    fn tan(self) -> Self::Output {
        FAD {
            value: self.value.clone().tan(),
            grad: self.grad / self.value.cos().square(),
        }
    }
}

impl<'a, T, O> Trig for &'a FAD<T>
where
    &'a T: Mul<O, Output = O> + Div<O, Output = O> + Trig<Output = O>,
    O: Square<Output = O> + Neg<Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sin(self) -> Self::Output {
        FAD {
            value: self.value.sin(),
            grad: &self.grad * self.value.cos(),
        }
    }

    #[inline]
    fn cos(self) -> Self::Output {
        FAD {
            value: self.value.cos(),
            grad: &self.grad * (-self.value.sin()),
        }
    }

    #[inline]
    fn tan(self) -> Self::Output {
        FAD {
            value: self.value.tan(),
            grad: &self.grad / self.value.cos().square(),
        }
    }
}

impl<T, E> From<FAD<Result<T, E>>> for Result<FAD<T>, E> {
    #[inline]
    fn from(val: FAD<Result<T, E>>) -> Self {
        Ok(FAD {
            value: val.value?,
            grad: val.grad?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_fad_eq {
        ($a:expr, $b:expr) => {
            let a: FAD<f32> = FAD::from($a);
            let b: FAD<f32> = FAD::from($b);
            assert!((a.value - b.value).abs() < 1e-4, "{:?} != {:?}", a, b);
            assert!((a.grad - b.grad).abs() < 1e-4, "{:?} != {:?}", a, b);
        };
    }

    #[test]
    fn scalar() {
        for a in [2.0f32, 3., 4.] {
            let b = FAD::from(a);
            for c in [2.0f32, 5., 7.] {
                assert_fad_eq!(b.clone() + c, (a + c, 1.));
                assert_fad_eq!(b.clone() * c, (a * c, c));
                assert_fad_eq!(b.clone() * 2. + c, (a * 2. + c, 2.));
                assert_fad_eq!((b.clone() + c) * -2., ((a + c) * -2., -2.));
                assert_fad_eq!(-b.clone(), (-a, -1.));
                assert_fad_eq!(b.clone() / c, (a / c, 1. / c));
                assert_fad_eq!(b.clone().pow(c), (a.powf(c), c * a.powf(c - 1.)));
                assert_fad_eq!(c.pow(b.clone()), (c.powf(a), c.ln() * c.powf(a)));
                assert_fad_eq!(b.clone().log(c), (a.log(c), 1. / c / a.ln()));
            }
            assert_fad_eq!(b.clone().exp(), (a.exp(), a.exp()));
            assert_fad_eq!(b.clone().ln(), (a.ln(), 1. / a));
            assert_fad_eq!(b.clone().sqrt(), (a.sqrt(), 0.5 / a.sqrt()));
            assert_fad_eq!(b.clone().sin(), (a.sin(), a.cos()));
            assert_fad_eq!(b.clone().cos(), (a.cos(), -a.sin()));
            assert_fad_eq!(b.clone().tan(), (a.tan(), 1. / a.cos() / a.cos()));
        }
    }
}
