// Forward Automatic Differentiation

use crate::ops::{Ln, One};

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct FAD<T> {
    pub value: T,
    pub delta: T,
}

impl<T> From<T> for FAD<T>
where
    T: One,
{
    #[inline]
    fn from(value: T) -> Self {
        FAD {
            value,
            delta: T::one(),
        }
    }
}

impl<T> From<(T, T)> for FAD<T> {
    #[inline]
    fn from(value: (T, T)) -> Self {
        FAD {
            value: value.0,
            delta: value.1,
        }
    }
}

impl<T, O> crate::ops::NumOps<T, FAD<O>> for FAD<T>
where
    T: crate::ops::NumOps<T, O>
        + std::ops::Mul<O, Output = O>
        + Clone
        + std::ops::Div<O, Output = O>
        + std::ops::Sub<O, Output = O>
        + std::ops::Sub<T, Output = O>
        + crate::ops::Pow<O, Output = O>
        + Into<O>,
    O: std::ops::Sub<O, Output = O> + std::ops::Mul<O, Output = O> + Clone,
{
}

impl<T, O> std::ops::Add<T> for FAD<T>
where
    T: std::ops::Add<T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.add(rhs),
            delta: self.delta.into(),
        }
    }
}

impl<T> crate::ops::One for FAD<T>
where
    T: crate::ops::One,
{
    fn one() -> Self {
        T::one().into()
    }
}

impl<'a, T, O> std::ops::Add<T> for &'a FAD<T>
where
    &'a T: std::ops::Add<T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.add(rhs),
            delta: (&self.delta).into(),
        }
    }
}

impl<'a, T, O> std::ops::Add<&'a T> for FAD<T>
where
    T: std::ops::Add<&'a T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn add(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.add(rhs),
            delta: self.delta.into(),
        }
    }
}

impl<'a, T, O> std::ops::Add<&'a T> for &'a FAD<T>
where
    &'a T: std::ops::Add<&'a T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn add(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.add(rhs),
            delta: (&self.delta).into(),
        }
    }
}

impl<T, O> std::ops::Sub<T> for FAD<T>
where
    T: std::ops::Sub<T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.sub(rhs),
            delta: self.delta.into(),
        }
    }
}

impl<'a, T, O> std::ops::Sub<T> for &'a FAD<T>
where
    &'a T: std::ops::Sub<T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.sub(rhs),
            delta: (&self.delta).into(),
        }
    }
}

impl<'a, T, O> std::ops::Sub<&'a T> for FAD<T>
where
    T: std::ops::Sub<&'a T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sub(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.sub(rhs),
            delta: self.delta.into(),
        }
    }
}

impl<'a, T, O> std::ops::Sub<&'a T> for &'a FAD<T>
where
    &'a T: std::ops::Sub<&'a T, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sub(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.sub(rhs),
            delta: (&self.delta).into(),
        }
    }
}

impl<T, O> std::ops::Mul<T> for FAD<T>
where
    T: std::ops::Mul<T, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.mul(rhs.clone()),
            delta: self.delta.mul(rhs),
        }
    }
}

impl<'a, T, O> std::ops::Mul<T> for &'a FAD<T>
where
    &'a T: std::ops::Mul<T, Output = O>,
    T: Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.mul(rhs.clone()),
            delta: self.delta.mul(rhs),
        }
    }
}

impl<'a, T, O> std::ops::Mul<&'a T> for FAD<T>
where
    T: std::ops::Mul<&'a T, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn mul(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.mul(rhs),
            delta: self.delta.mul(rhs),
        }
    }
}

impl<'a, T, O> std::ops::Mul<&'a T> for &'a FAD<T>
where
    &'a T: std::ops::Mul<&'a T, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn mul(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.mul(rhs),
            delta: self.delta.mul(rhs),
        }
    }
}

impl<T, O> std::ops::Neg for FAD<T>
where
    T: std::ops::Neg<Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn neg(self) -> Self::Output {
        FAD {
            value: self.value.neg(),
            delta: self.delta.neg(),
        }
    }
}

impl<'a, T, O> std::ops::Neg for &'a FAD<T>
where
    &'a T: std::ops::Neg<Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn neg(self) -> Self::Output {
        FAD {
            value: self.value.neg(),
            delta: self.delta.neg(),
        }
    }
}

impl<T, O> crate::ops::Abs for FAD<T>
where
    T: crate::ops::Abs<Output = O>,
{
    type Output = FAD<O>;

    fn abs(self) -> Self::Output {
        todo!()
    }
}

impl<T, O> std::ops::Div<T> for FAD<T>
where
    T: std::ops::Div<T, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.div(rhs.clone()),
            delta: self.delta.div(rhs),
        }
    }
}

impl<'a, T, O> std::ops::Div<T> for &'a FAD<T>
where
    &'a T: std::ops::Div<T, Output = O>,
    T: Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.div(rhs.clone()),
            delta: self.delta.div(rhs),
        }
    }
}

impl<'a, T, O> std::ops::Div<&'a T> for FAD<T>
where
    T: std::ops::Div<&'a T, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn div(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.div(rhs),
            delta: self.delta.div(rhs),
        }
    }
}

impl<'a, T, O> std::ops::Div<&'a T> for &'a FAD<T>
where
    &'a T: std::ops::Div<&'a T, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn div(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.div(rhs),
            delta: self.delta.div(rhs),
        }
    }
}

impl<T, O> crate::ops::Pow<T> for FAD<T>
where
    T: crate::ops::Pow<T, Output = O>
        + crate::ops::Pow<O, Output = O>
        + std::ops::Mul<T, Output = O>
        + std::ops::Sub<T, Output = O>
        + One
        + Clone,
    O: std::ops::Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.clone().pow(rhs.clone()),
            delta: (self.delta * rhs.clone()) * self.value.pow(rhs - T::one()),
        }
    }
}

impl<'a, T, O> crate::ops::Pow<T> for &'a FAD<T>
where
    &'a T: crate::ops::Pow<T, Output = O>
        + crate::ops::Pow<O, Output = O>
        + std::ops::Mul<T, Output = O>,
    T: std::ops::Sub<T, Output = O> + One + Clone,
    O: std::ops::Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, rhs: T) -> Self::Output {
        FAD {
            value: (&self.value).pow(rhs.clone()),
            delta: (&self.delta * rhs.clone()) * (&self.value).pow(rhs - T::one()),
        }
    }
}

impl<'a, T, O> crate::ops::Pow<&'a T> for FAD<T>
where
    T: crate::ops::Pow<&'a T, Output = O>
        + crate::ops::Pow<O, Output = O>
        + std::ops::Mul<&'a T, Output = O>
        + One
        + Clone,
    &'a T: std::ops::Sub<T, Output = O>,
    O: std::ops::Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.clone().pow(rhs),
            delta: (self.delta * rhs) * self.value.pow(rhs - T::one()),
        }
    }
}

impl<'a, T, O> crate::ops::Pow<&'a T> for &'a FAD<T>
where
    &'a T: crate::ops::Pow<&'a T, Output = O>
        + crate::ops::Pow<O, Output = O>
        + std::ops::Mul<&'a T, Output = O>
        + std::ops::Sub<T, Output = O>,
    T: One,
    O: std::ops::Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: (&self.value).pow(rhs),
            delta: (&self.delta * rhs) * (&self.value).pow(rhs - T::one()),
        }
    }
}

impl<T, O> crate::ops::Exp for FAD<T>
where
    T: std::ops::Mul<T, Output = O> + std::ops::Mul<O, Output = O> + crate::ops::Exp<Output = O>,
    O: Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn exp(self) -> Self::Output {
        let value = self.value.exp();
        let delta = self.delta * value.clone();
        FAD { value, delta }
    }
}

impl<'a, T, O> crate::ops::Exp for &'a FAD<T>
where
    &'a T: std::ops::Mul<O, Output = O> + crate::ops::Exp<Output = O>,
    O: Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn exp(self) -> Self::Output {
        let value = self.value.exp();
        let delta = &self.delta * value.clone();
        FAD { value, delta }
    }
}

impl<T, O> crate::ops::Pow<FAD<T>> for T
where
    T: std::ops::Mul<O, Output = O>
        + crate::ops::Ln<Output = O>
        + crate::ops::Pow<T, Output = O>
        + Clone,
    O: std::ops::Mul<Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, lhs: FAD<T>) -> Self::Output {
        let value = self.clone().pow(lhs.value);
        let delta = lhs.delta * (self.ln() * value.clone());
        FAD { value, delta }
    }
}

impl<'a, T, O> crate::ops::Pow<&'a FAD<T>> for T
where
    &'a T: std::ops::Mul<O, Output = O>,
    T: crate::ops::Ln<Output = O> + crate::ops::Pow<&'a T, Output = O> + Clone,
    O: std::ops::Mul<Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, lhs: &'a FAD<T>) -> Self::Output {
        let value = self.clone().pow(&lhs.value);
        let delta = &lhs.delta * (self.ln() * value.clone());
        FAD { value, delta }
    }
}

impl<'a, T, O> crate::ops::Pow<FAD<T>> for &'a T
where
    &'a T: crate::ops::Pow<T, Output = O> + crate::ops::Ln<Output = O>,
    T: std::ops::Mul<O, Output = O>,
    O: std::ops::Mul<Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, lhs: FAD<T>) -> Self::Output {
        let value = self.pow(lhs.value);
        let delta = lhs.delta * (self.ln() * value.clone());
        FAD { value, delta }
    }
}

impl<'a, T, O> crate::ops::Pow<&'a FAD<T>> for &'a T
where
    &'a T: crate::ops::Pow<&'a T, Output = O>
        + crate::ops::Ln<Output = O>
        + std::ops::Mul<O, Output = O>,
    O: std::ops::Mul<Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, lhs: &'a FAD<T>) -> Self::Output {
        let value = self.pow(&lhs.value);
        let delta = &lhs.delta * (self.ln() * value.clone());
        FAD { value, delta }
    }
}

impl<T, O> crate::ops::Ln for FAD<T>
where
    T: std::ops::Div<Output = O> + crate::ops::Ln<Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn ln(self) -> Self::Output {
        FAD {
            value: self.value.clone().ln(),
            delta: self.delta / self.value,
        }
    }
}

impl<'a, T, O> crate::ops::Ln for &'a FAD<T>
where
    &'a T: std::ops::Div<&'a T, Output = O> + crate::ops::Ln<Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn ln(self) -> Self::Output {
        FAD {
            value: self.value.ln(),
            delta: &self.delta / &self.value,
        }
    }
}

impl<T, O> crate::ops::Log<T> for FAD<T>
where
    T: std::ops::Div<O, Output = O>
        + crate::ops::Log<T, Output = O>
        + crate::ops::Ln<Output = O>
        + std::ops::Mul<O, Output = O>
        + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn log(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.clone().log(rhs.clone()),
            delta: self.delta / (rhs * self.value.ln()),
        }
    }
}

impl<'a, T, O> crate::ops::Log<T> for &'a FAD<T>
where
    &'a T:
        std::ops::Div<O, Output = O> + crate::ops::Log<T, Output = O> + crate::ops::Ln<Output = O>,
    T: std::ops::Mul<O, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn log(self, rhs: T) -> Self::Output {
        FAD {
            value: self.value.log(rhs.clone()),
            delta: &self.delta / (rhs * self.value.ln()),
        }
    }
}

impl<'a, T, O> crate::ops::Log<&'a T> for FAD<T>
where
    T: std::ops::Div<O, Output = O>
        + crate::ops::Log<&'a T, Output = O>
        + crate::ops::Ln<Output = O>
        + Clone,
    &'a T: std::ops::Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn log(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.clone().log(rhs),
            delta: self.delta / (rhs * self.value.ln()),
        }
    }
}

impl<'a, T, O> crate::ops::Log<&'a T> for &'a FAD<T>
where
    &'a T: std::ops::Div<O, Output = O>
        + crate::ops::Log<&'a T, Output = O>
        + crate::ops::Ln<Output = O>
        + std::ops::Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn log(self, rhs: &'a T) -> Self::Output {
        FAD {
            value: self.value.log(rhs),
            delta: &self.delta / (rhs * self.value.ln()),
        }
    }
}

impl<T, O> crate::ops::Sqrt for FAD<T>
where
    T: crate::ops::Sqrt<Output = O> + crate::ops::Half + std::ops::Mul<T, Output = O>,
    O: std::ops::Div<O, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn sqrt(self) -> Self::Output {
        let value = self.value.sqrt();
        FAD {
            value: value.clone(),
            delta: self.delta * T::half() / value,
        }
    }
}

impl<'a, T, O> crate::ops::Sqrt for &'a FAD<T>
where
    T: crate::ops::Half,
    &'a T: crate::ops::Sqrt<Output = O> + std::ops::Mul<T, Output = O>,
    O: std::ops::Div<O, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn sqrt(self) -> Self::Output {
        let value = self.value.sqrt();
        FAD {
            value: value.clone(),
            delta: &self.delta * T::half() / value,
        }
    }
}

impl<T, O> crate::ops::Trig for FAD<T>
where
    T: std::ops::Mul<O, Output = O>
        + std::ops::Div<O, Output = O>
        + crate::ops::Trig<Output = O>
        + Clone,
    O: crate::ops::Square<Output = O> + std::ops::Neg<Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sin(self) -> Self::Output {
        FAD {
            value: self.value.clone().sin(),
            delta: self.delta * self.value.cos(),
        }
    }

    #[inline]
    fn cos(self) -> Self::Output {
        FAD {
            value: self.value.clone().cos(),
            delta: self.delta * (-self.value.sin()),
        }
    }

    #[inline]
    fn tan(self) -> Self::Output {
        FAD {
            value: self.value.clone().tan(),
            delta: self.delta / self.value.cos().square(),
        }
    }
}

impl<'a, T, O> crate::ops::Trig for &'a FAD<T>
where
    &'a T:
        std::ops::Mul<O, Output = O> + std::ops::Div<O, Output = O> + crate::ops::Trig<Output = O>,
    O: crate::ops::Square<Output = O> + std::ops::Neg<Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sin(self) -> Self::Output {
        FAD {
            value: self.value.sin(),
            delta: &self.delta * self.value.cos(),
        }
    }

    #[inline]
    fn cos(self) -> Self::Output {
        FAD {
            value: self.value.cos(),
            delta: &self.delta * (-self.value.sin()),
        }
    }

    #[inline]
    fn tan(self) -> Self::Output {
        FAD {
            value: self.value.tan(),
            delta: &self.delta / self.value.cos().square(),
        }
    }
}

impl<T, E> From<FAD<Result<T, E>>> for Result<FAD<T>, E> {
    #[inline]
    fn from(val: FAD<Result<T, E>>) -> Self {
        Ok(FAD {
            value: val.value?,
            delta: val.delta?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Pow;
    use crate::ops::*;

    macro_rules! assert_fad_eq {
        ($a:expr, $b:expr) => {
            let a: FAD<f32> = FAD::from($a);
            let b: FAD<f32> = FAD::from($b);
            assert!((a.value - b.value).abs() < 1e-4, "{:?} != {:?}", a, b);
            assert!((a.delta - b.delta).abs() < 1e-4, "{:?} != {:?}", a, b);
        };
    }

    #[test]
    fn scalar() {
        for a in [2.0f32, 3., 4.] {
            let b = FAD::from(a);
            for c in [2.0f32, 5., 7.] {
                assert_fad_eq!(b + c, (a + c, 1.));
                assert_fad_eq!(b * c, (a * c, c));
                assert_fad_eq!(b * 2. + c, (a * 2. + c, 2.));
                assert_fad_eq!((b + c) * -2., ((a + c) * -2., -2.));
                assert_fad_eq!(-b, (-a, -1.));
                assert_fad_eq!(b / c, (a / c, 1. / c));
                assert_fad_eq!(b.pow(c), (a.powf(c), c * a.powf(c - 1.)));
                assert_fad_eq!(c.pow(b), (c.powf(a), c.ln() * c.powf(a)));
                assert_fad_eq!(b.log(c), (a.log(c), 1. / c / a.ln()));
            }
            assert_fad_eq!(b.exp(), (a.exp(), a.exp()));
            assert_fad_eq!(b.ln(), (a.ln(), 1. / a));
            assert_fad_eq!(b.sqrt(), (a.sqrt(), 0.5 / a.sqrt()));
            assert_fad_eq!(b.sin(), (a.sin(), a.cos()));
            assert_fad_eq!(b.cos(), (a.cos(), -a.sin()));
            assert_fad_eq!(b.tan(), (a.tan(), 1. / a.cos() / a.cos()));
        }
    }
}
