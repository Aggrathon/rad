/// Forward Automatic Differentiation
use crate::ops::*;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Wrapper for automatic differentiation in forward mode.
/// Wrap the parameter you want to differentiate and do the calculations normally.
/// The gradient vill automatically be accumulated.
///
/// Note that the default implementations of gradients for numerical operations assume that the operations are implemented normally (element-wise).
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
#[derive(Debug, Clone, Copy, PartialEq)]
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

impl<T1, T2> NumOps<T2, FAD<T1>> for FAD<T1>
where
    T1: NumOps<T2, T1>
        + Signum<Output = T1>
        + Mul<T1, Output = T1>
        + One
        + Pow<T1, Output = T1>
        + Div<T1, Output = T1>
        + Clone,
    T2: Sub<T1, Output = T1> + Copy,
{
}

impl<'a, T1, T2> NumOps<T2, FAD<T1>> for &'a FAD<T1>
where
    T1: One + Clone + Mul<T2, Output = T1> + Mul<T1, Output = T1>,
    &'a T1: NumOps<T2, T1>
        + Signum<Output = T1>
        + Mul<T1, Output = T1>
        + Pow<T1, Output = T1>
        + Div<T1, Output = T1>
        + Div<&'a T1, Output = T1>,
    T2: Sub<T1, Output = T1> + Copy,
{
}

impl<T> NumOpts<FAD<T>> for FAD<T> where
    T: NumOpts<T> + Neg<Output = T> + Mul<T, Output = T> + Div<T, Output = T> + Clone + Half + Two
{
}

impl<'a, T> NumOpts<FAD<T>> for &'a FAD<T>
where
    T: Square<Output = T> + Neg<Output = T> + Div<T, Output = T> + Clone + Half + Two,
    &'a T: NumOpts<T> + Mul<T, Output = T> + Div<T, Output = T>,
{
}

impl<T> NumConsts for FAD<T> where T: NumConsts {}

impl<T> AggOps<FAD<T>> for FAD<T> where
    T: AggOps<T> + Mul<T, Output = T> + Clone + Div<T, Output = T> + Clone
{
}

impl<'a, T> AggOps<FAD<T>> for &'a FAD<T>
where
    &'a T: AggOps<T> + Mul<T, Output = T>,
    T: Div<&'a T, Output = T> + Clone,
{
}

macro_rules! impl_const_op {
    ($Trait:path, $fn:ident) => {
        impl<T> $Trait for FAD<T>
        where
            T: $Trait + Into<FAD<T>>,
        {
            #[inline]
            fn $fn() -> Self {
                T::$fn().into()
            }
        }
    };
}

impl_const_op!(crate::ops::One, one);
impl_const_op!(crate::ops::Zero, zero);
impl_const_op!(crate::ops::Half, half);
impl_const_op!(crate::ops::Two, two);

impl<T1, T2> Add<T2> for FAD<T1>
where
    T1: Add<T2, Output = T1>,
{
    type Output = FAD<T1>;

    #[inline]
    fn add(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.add(rhs),
            grad: self.grad,
        }
    }
}

impl<'a, T1, T2> Add<T2> for &'a FAD<T1>
where
    T1: Clone,
    &'a T1: Add<T2, Output = T1>,
{
    type Output = FAD<T1>;

    #[inline]
    fn add(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.add(rhs),
            grad: self.grad.clone(),
        }
    }
}

impl<T1, T2> Sub<T2> for FAD<T1>
where
    T1: Sub<T2, Output = T1>,
{
    type Output = FAD<T1>;

    #[inline]
    fn sub(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.sub(rhs),
            grad: self.grad,
        }
    }
}

impl<'a, T1, T2> Sub<T2> for &'a FAD<T1>
where
    T1: Clone,
    &'a T1: Sub<T2, Output = T1>,
{
    type Output = FAD<T1>;

    #[inline]
    fn sub(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.sub(rhs),
            grad: self.grad.clone(),
        }
    }
}

impl<T1, T2> Mul<T2> for FAD<T1>
where
    T1: Mul<T2, Output = T1>,
    T2: Copy,
{
    type Output = FAD<T1>;

    #[inline]
    fn mul(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.mul(rhs),
            grad: self.grad.mul(rhs),
        }
    }
}

impl<'a, T1, T2> Mul<T2> for &'a FAD<T1>
where
    &'a T1: Mul<T2, Output = T1>,
    T2: Copy,
{
    type Output = FAD<T1>;

    #[inline]
    fn mul(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.mul(rhs),
            grad: self.grad.mul(rhs),
        }
    }
}

impl<T> Neg for FAD<T>
where
    T: Neg<Output = T>,
{
    type Output = FAD<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        FAD {
            value: self.value.neg(),
            grad: self.grad.neg(),
        }
    }
}

impl<'a, T> Neg for &'a FAD<T>
where
    &'a T: Neg<Output = T>,
{
    type Output = FAD<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        FAD {
            value: self.value.neg(),
            grad: self.grad.neg(),
        }
    }
}

impl<T> Abs for FAD<T>
where
    T: Abs<Output = T> + Signum<Output = T> + Mul<T, Output = T> + Clone,
{
    type Output = FAD<T>;

    #[inline]
    fn abs(self) -> Self::Output {
        FAD {
            value: self.value.clone().abs(),
            grad: self.grad * self.value.signum(),
        }
    }
}

impl<'a, T> Abs for &'a FAD<T>
where
    &'a T: Abs<Output = T> + Signum<Output = T> + Mul<T, Output = T>,
{
    type Output = FAD<T>;

    #[inline]
    fn abs(self) -> Self::Output {
        FAD {
            value: self.value.abs(),
            grad: &self.grad * self.value.signum(),
        }
    }
}

impl<T1, T2> Div<T2> for FAD<T1>
where
    T1: Div<T2, Output = T1>,
    T2: Copy,
{
    type Output = FAD<T1>;

    #[inline]
    fn div(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.div(rhs),
            grad: self.grad.div(rhs),
        }
    }
}

impl<'a, T1, T2> Div<T2> for &'a FAD<T1>
where
    &'a T1: Div<T2, Output = T1>,
    T2: Copy,
{
    type Output = FAD<T1>;

    #[inline]
    fn div(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.div(rhs),
            grad: self.grad.div(rhs),
        }
    }
}

impl<T> Div2<FAD<T>> for T
where
    T: Div<T, Output = T> + Neg<Output = T> + Square<Output = T> + Clone,
{
    type Output = FAD<T>;

    #[inline]
    fn _div(self, rhs: FAD<T>) -> Self::Output {
        FAD {
            value: self.div(rhs.value.clone()),
            grad: rhs.grad.div(rhs.value.square()).neg(),
        }
    }
}

impl<'a, T> Div2<FAD<T>> for &'a T
where
    T: Clone + Neg<Output = T> + Square<Output = T> + Div<T, Output = T>,
    &'a T: Div<T, Output = T>,
{
    type Output = FAD<T>;

    #[inline]
    fn _div(self, rhs: FAD<T>) -> Self::Output {
        FAD {
            value: self.div(rhs.value.clone()),
            grad: rhs.grad.div(rhs.value.square()).neg(),
        }
    }
}

impl<'a, T> Div2<&'a FAD<T>> for T
where
    T: Div<&'a T, Output = T> + Neg<Output = T>,
    &'a T: Div<T, Output = T> + Square<Output = T> + Div<T, Output = T>,
{
    type Output = FAD<T>;

    #[inline]
    fn _div(self, rhs: &'a FAD<T>) -> Self::Output {
        FAD {
            value: self.div(&rhs.value),
            grad: (&rhs.grad).div((&rhs.value).square()).neg(),
        }
    }
}

impl<'a, T> Div2<&'a FAD<T>> for &'a T
where
    T: Clone + Neg<Output = T>,
    &'a T: Div<&'a T, Output = T> + Square<Output = T> + Div<T, Output = T>,
{
    type Output = FAD<T>;

    #[inline]
    fn _div(self, rhs: &'a FAD<T>) -> Self::Output {
        FAD {
            value: self.div(&rhs.value),
            grad: rhs.grad.div(rhs.value.square()).neg(),
        }
    }
}

impl<T1, T2> Pow<T2> for FAD<T1>
where
    T1: Pow<T2, Output = T1>
        + Mul<T2, Output = T1>
        + Mul<T1, Output = T1>
        + Pow<T1, Output = T1>
        + One
        + Clone,
    T2: Copy + Sub<T1, Output = T1>,
{
    type Output = FAD<T1>;

    #[inline]
    fn pow(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.clone().pow(rhs),
            grad: (self.grad * rhs) * self.value.pow(rhs - T1::one()),
        }
    }
}

impl<'a, T1, T2> Pow<T2> for &'a FAD<T1>
where
    &'a T1: Pow<T2, Output = T1> + Pow<T1, Output = T1> + Mul<T2, Output = T1>,
    T1: One + Mul<T1, Output = T1>,
    T2: Copy + Sub<T1, Output = T1>,
{
    type Output = FAD<T1>;

    #[inline]
    fn pow(self, rhs: T2) -> Self::Output {
        FAD {
            value: (&self.value).pow(rhs),
            grad: (&self.grad * rhs) * (&self.value).pow(rhs - T1::one()),
        }
    }
}

impl<T> Exp for FAD<T>
where
    T: Mul<T, Output = T> + Mul<T, Output = T> + Exp<Output = T> + Clone,
{
    type Output = FAD<T>;

    #[inline]
    fn exp(self) -> Self::Output {
        let value = self.value.exp();
        let grad = self.grad * value.clone();
        FAD { value, grad }
    }
}

impl<'a, T> Exp for &'a FAD<T>
where
    &'a T: Mul<T, Output = T> + Exp<Output = T>,
    T: Clone,
{
    type Output = FAD<T>;

    #[inline]
    fn exp(self) -> Self::Output {
        let value = self.value.exp();
        let grad = &self.grad * value.clone();
        FAD { value, grad }
    }
}

impl<T> Pow2<FAD<T>> for T
where
    T: Mul<T, Output = T> + Ln<Output = T> + Pow<T, Output = T> + Clone,
{
    type Output = FAD<T>;

    #[inline]
    fn _pow(self, lhs: FAD<T>) -> Self::Output {
        let value = self.clone().pow(lhs.value);
        let grad = lhs.grad * (self.ln() * value.clone());
        FAD { value, grad }
    }
}

impl<'a, T> Pow2<&'a FAD<T>> for T
where
    &'a T: Mul<T, Output = T>,
    T: Ln<Output = T> + Pow<&'a T, Output = T> + Mul<Output = T> + Clone,
{
    type Output = FAD<T>;

    #[inline]
    fn _pow(self, lhs: &'a FAD<T>) -> Self::Output {
        let value = self.clone().pow(&lhs.value);
        let grad = &lhs.grad * (self.ln() * value.clone());
        FAD { value, grad }
    }
}

impl<'a, T> Pow2<FAD<T>> for &'a T
where
    &'a T: Pow<T, Output = T> + Ln<Output = T>,
    T: Mul<T, Output = T> + Clone,
{
    type Output = FAD<T>;

    #[inline]
    fn _pow(self, lhs: FAD<T>) -> Self::Output {
        let value = self.pow(lhs.value);
        let grad = lhs.grad * (self.ln() * value.clone());
        FAD { value, grad }
    }
}

impl<'a, T> Pow2<&'a FAD<T>> for &'a T
where
    &'a T: Pow<&'a T, Output = T> + Ln<Output = T> + Mul<T, Output = T>,
    T: Mul<T, Output = T> + Clone,
{
    type Output = FAD<T>;

    #[inline]
    fn _pow(self, lhs: &'a FAD<T>) -> Self::Output {
        let value = self.pow(&lhs.value);
        let grad = &lhs.grad * (self.ln() * value.clone());
        FAD { value, grad }
    }
}

impl<T> Ln for FAD<T>
where
    T: Div<Output = T> + Ln<Output = T> + Clone,
{
    type Output = FAD<T>;

    #[inline]
    fn ln(self) -> Self::Output {
        FAD {
            value: self.value.clone().ln(),
            grad: self.grad / self.value,
        }
    }
}

impl<'a, T> Ln for &'a FAD<T>
where
    &'a T: Div<&'a T, Output = T> + Ln<Output = T>,
{
    type Output = FAD<T>;

    #[inline]
    fn ln(self) -> Self::Output {
        FAD {
            value: self.value.ln(),
            grad: &self.grad / &self.value,
        }
    }
}

impl<T1, T2> Log<T2> for FAD<T1>
where
    T1: Div<T1, Output = T1>
        + Log<T2, Output = T1>
        + Ln<Output = T1>
        + Mul<T2, Output = T1>
        + Clone,
    T2: Copy,
{
    type Output = FAD<T1>;

    #[inline]
    fn log(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.clone().log(rhs),
            grad: self.grad / (self.value.ln() * rhs),
        }
    }
}

impl<'a, T1, T2> Log<T2> for &'a FAD<T1>
where
    &'a T1: Div<T1, Output = T1> + Log<T2, Output = T1> + Ln<Output = T1>,
    T1: Mul<T2, Output = T1>,
    T2: Copy,
{
    type Output = FAD<T1>;

    #[inline]
    fn log(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.log(rhs),
            grad: &self.grad / (self.value.ln() * rhs),
        }
    }
}

impl<T> Square for FAD<T>
where
    T: Square<Output = T> + Two + Mul<T, Output = T> + Clone + Mul<T, Output = T>,
{
    type Output = FAD<T>;

    #[inline]
    fn square(self) -> Self::Output {
        FAD {
            value: self.value.clone().square(),
            grad: self.grad * (T::two() * self.value),
        }
    }
}

impl<'a, T> Square for &'a FAD<T>
where
    T: Two,
    &'a T: Square<Output = T> + Mul<T, Output = T>,
{
    type Output = FAD<T>;

    #[inline]
    fn square(self) -> Self::Output {
        FAD {
            value: self.value.square(),
            grad: &self.grad * (&self.value * T::two()),
        }
    }
}

impl<T> Sqrt for FAD<T>
where
    T: Sqrt<Output = T> + Half + Mul<T, Output = T> + Div<T, Output = T> + Clone,
{
    type Output = FAD<T>;

    #[inline]
    fn sqrt(self) -> Self::Output {
        let value = self.value.sqrt();
        FAD {
            value: value.clone(),
            grad: self.grad * T::half() / value,
        }
    }
}

impl<'a, T> Sqrt for &'a FAD<T>
where
    T: Half + Div<T, Output = T> + Clone,
    &'a T: Sqrt<Output = T> + Mul<T, Output = T>,
{
    type Output = FAD<T>;

    #[inline]
    fn sqrt(self) -> Self::Output {
        let value = self.value.sqrt();
        FAD {
            value: value.clone(),
            grad: &self.grad * T::half() / value,
        }
    }
}

impl<T> Trig for FAD<T>
where
    T: Mul<T, Output = T>
        + Div<T, Output = T>
        + Trig<Output = T>
        + Clone
        + Square<Output = T>
        + Neg<Output = T>,
{
    type Output = FAD<T>;

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

impl<'a, T> Trig for &'a FAD<T>
where
    &'a T: Mul<T, Output = T> + Div<T, Output = T> + Trig<Output = T>,
    T: Square<Output = T> + Neg<Output = T>,
{
    type Output = FAD<T>;

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

impl<T> Sum for FAD<T>
where
    T: Sum<Output = T>,
{
    type Output = FAD<T>;

    fn sum(self) -> Self::Output {
        FAD {
            value: self.value.sum(),
            grad: self.grad,
        }
    }
}

impl<'a, T> Sum for &'a FAD<T>
where
    T: Clone,
    &'a T: Sum<Output = T>,
{
    type Output = FAD<T>;

    fn sum(self) -> Self::Output {
        FAD {
            value: self.value.sum(),
            grad: self.grad.clone(),
        }
    }
}

impl<T> Prod for FAD<T>
where
    T: Prod<Output = T> + Mul<T, Output = T> + Clone,
    T: Div<T, Output = T> + Clone,
{
    type Output = FAD<T>;

    fn prod(self) -> Self::Output {
        let value = self.value.clone().prod();
        FAD {
            value: value.clone(),
            grad: self.grad * (value / self.value),
        }
    }
}

impl<'a, T> Prod for &'a FAD<T>
where
    &'a T: Prod<Output = T> + Mul<T, Output = T>,
    T: Div<&'a T, Output = T> + Clone,
{
    type Output = FAD<T>;

    fn prod(self) -> Self::Output {
        let value = self.value.prod();
        FAD {
            value: value.clone(),
            grad: &self.grad * (value / &self.value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_fad_eq {
        ($a:expr, $b:expr) => {
            let a: FAD<f32> = $a; // FAD::from($a);
            let b: FAD<f32> = FAD::from($b);
            assert!((a.value - b.value).abs() < 1e-4, "{:?} != {:?}", a, b);
            assert!((a.grad - b.grad).abs() < 1e-4, "{:?} != {:?}", a, b);
        };
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn scalar() {
        for a in [2.0f32, 3., 4.] {
            let b = FAD::from(a);
            for c in [2.0f32, 5., 7.] {
                assert_fad_eq!(b + c, (a + c, 1.));
                assert_fad_eq!(b * c, (a * c, c));
                assert_fad_eq!(b * 2. + c, (a * 2. + c, 2.));
                assert_fad_eq!((b - c) * -2., ((a - c) * -2., -2.));
                assert_fad_eq!(-b, (-a, -1.));
                assert_fad_eq!(b / c, (a / c, 1. / c));
                assert_fad_eq!(c._div(b), (c / a, -1.0 / a / a));
                assert_fad_eq!(b.pow(c), (a.powf(c), c * a.powf(c - 1.)));
                assert_fad_eq!(c._pow(b), (c.powf(a), c.ln() * c.powf(a)));
                assert_fad_eq!((&b).log(c), (a.log(c), 1. / c / a.ln()));
                assert_fad_eq!(b + c, (a + c, 1.));
                assert_fad_eq!((&b) * c, (a * c, c));
                assert_fad_eq!((&b) * 2. + c, (a * 2. + c, 2.));
                assert_fad_eq!((b - c) * -2., ((a - c) * -2., -2.));
                assert_fad_eq!(-(&b), (-a, -1.));
                assert_fad_eq!((&b) / c, (a / c, 1. / c));
                assert_fad_eq!(c._div(&b), (c / a, -1.0 / a / a));
                assert_fad_eq!((&b).pow(c), (a.powf(c), c * a.powf(c - 1.)));
                assert_fad_eq!(c._pow(&b), (c.powf(a), c.ln() * c.powf(a)));
                assert_fad_eq!((&b).log(c), (a.log(c), 1. / c / a.ln()));
            }
            assert_fad_eq!(b.exp(), (a.exp(), a.exp()));
            assert_fad_eq!(b.ln(), (a.ln(), 1. / a));
            assert_fad_eq!(b.sqrt(), (a.sqrt(), 0.5 / a.sqrt()));
            assert_fad_eq!(b.square(), (a.square(), 2.0 * a));
            assert_fad_eq!(b.sin(), (a.sin(), a.cos()));
            assert_fad_eq!(b.cos(), (a.cos(), -a.sin()));
            assert_fad_eq!(b.tan(), (a.tan(), 1. / a.cos() / a.cos()));
            assert_fad_eq!((&b).exp(), (a.exp(), a.exp()));
            assert_fad_eq!((&b).ln(), (a.ln(), 1. / a));
            assert_fad_eq!((&b).sqrt(), (a.sqrt(), 0.5 / a.sqrt()));
            assert_fad_eq!((&b).square(), (a.square(), 2.0 * a));
            assert_fad_eq!((&b).sin(), (a.sin(), a.cos()));
            assert_fad_eq!((&b).cos(), (a.cos(), -a.sin()));
            assert_fad_eq!((&b).tan(), (a.tan(), 1. / a.cos() / a.cos()));
        }
    }

    macro_rules! assert_impl {
        ($T:ty: $Trait:path) => {
            const _: () = {
                fn f<'a, T: $Trait>() {}
                fn assert_trait() {
                    f::<$T>()
                }
            };
        };
        ($T:ty) => {
            assert_impl!(FAD<$T>: NumOps<$T, FAD<$T>>);
            assert_impl!(&FAD<$T>: NumOps<$T, FAD<$T>>);
            assert_impl!(FAD<$T>: NumOps<&'a $T, FAD<$T>>);
            assert_impl!(&FAD<$T>: NumOps<&'a $T, FAD<$T>>);

            assert_impl!(FAD<$T>: NumConsts);

            assert_impl!(FAD<$T>: NumOpts<FAD<$T>>);
            assert_impl!(&FAD<$T>: NumOpts<FAD<$T>>);
        }
    }

    assert_impl!(f32);
    assert_impl!(f64);
}
