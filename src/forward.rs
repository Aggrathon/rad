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

impl<T1, T2, O> std::ops::Add<T2> for FAD<T1>
where
    T1: std::ops::Add<T2, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn add(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.add(rhs),
            delta: self.delta.into(),
        }
    }
}

impl<'a, T1: 'a, T2, O> std::ops::Add<T2> for &'a FAD<T1>
where
    &'a T1: std::ops::Add<T2, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn add(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.add(rhs),
            delta: (&self.delta).into(),
        }
    }
}

impl<T1, T2, O> std::ops::Sub<T2> for FAD<T1>
where
    T1: std::ops::Sub<T2, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sub(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.sub(rhs),
            delta: self.delta.into(),
        }
    }
}

impl<'a, T1: 'a, T2, O> std::ops::Sub<T2> for &'a FAD<T1>
where
    &'a T1: std::ops::Sub<T2, Output = O> + Into<O>,
{
    type Output = FAD<O>;

    #[inline]
    fn sub(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.sub(rhs),
            delta: (&self.delta).into(),
        }
    }
}

impl<T1, T2, O> std::ops::Mul<T2> for FAD<T1>
where
    T1: std::ops::Mul<T2, Output = O>,
    T2: Copy,
{
    type Output = FAD<O>;

    #[inline]
    fn mul(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.mul(rhs),
            delta: self.delta.mul(rhs),
        }
    }
}

impl<'a, T1: 'a, T2, O> std::ops::Mul<T2> for &'a FAD<T1>
where
    &'a T1: std::ops::Mul<T2, Output = O>,
    T2: Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn mul(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.mul(rhs.clone()),
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

impl<'a, T: 'a, O> std::ops::Neg for &'a FAD<T>
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

impl<T1, T2, O> std::ops::Div<T2> for FAD<T1>
where
    T1: std::ops::Div<T2, Output = O>,
    T2: Copy,
{
    type Output = FAD<O>;

    #[inline]
    fn div(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.div(rhs),
            delta: self.delta.div(rhs),
        }
    }
}

impl<'a, T1: 'a, T2, O> std::ops::Div<T2> for &'a FAD<T1>
where
    &'a T1: std::ops::Div<T2, Output = O>,
    T2: Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn div(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.div(rhs.clone()),
            delta: self.delta.div(rhs),
        }
    }
}

impl<T1, T2, O> crate::ops::Pow<T2> for FAD<T1>
where
    T1: crate::ops::Pow<T2, Output = O> + std::ops::Mul<T2, Output = O> + Copy,
    T2: std::ops::Sub<Output = T2> + One + Copy,
    O: std::ops::Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.pow(rhs),
            delta: (self.delta * rhs) * self.value.pow(rhs - T2::one()),
        }
    }
}

impl<'a, T1: 'a, T2, O> crate::ops::Pow<T2> for &'a FAD<T1>
where
    &'a T1: crate::ops::Pow<T2, Output = O>
        + crate::ops::Pow<T2, Output = O>
        + std::ops::Mul<T2, Output = O>,
    T2: std::ops::Sub<T2, Output = T2>,
    T2: One + Clone,
    O: std::ops::Mul<O, Output = O>,
{
    type Output = FAD<O>;

    #[inline]
    fn pow(self, rhs: T2) -> Self::Output {
        FAD {
            value: (&self.value).pow(rhs.clone()),
            delta: (&self.delta * rhs.clone()) * (&self.value).pow(rhs - T2::one()),
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

impl<'a, T: 'a, O> crate::ops::Exp for &'a FAD<T>
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

impl<T1, T2, O> crate::ops::Wop<T2> for FAD<T1>
where
    T2: crate::ops::Ln<Output = O> + Clone,
    T1: std::ops::Mul<O, Output = O> + crate::ops::Wop<T2, Output = O>,
    O: std::ops::Mul<Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn wop(self, rhs: T2) -> Self::Output {
        let value = self.value.wop(rhs.clone());
        let delta = self.delta * (rhs.ln() * value.clone());
        FAD { value, delta }
    }
}

impl<'a, T1: 'a, T2, O> crate::ops::Wop<T2> for &'a FAD<T1>
where
    &'a T1: std::ops::Mul<O, Output = O> + crate::ops::Wop<T2, Output = O>,
    T2: crate::ops::Ln<Output = O> + Clone,
    O: std::ops::Mul<O, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn wop(self, rhs: T2) -> Self::Output {
        let value = self.value.wop(rhs.clone());
        let delta = &self.delta * (rhs.ln() * value.clone());
        FAD { value, delta }
    }
}

impl<T, O> crate::ops::Ln for FAD<T>
where
    T: std::ops::Div<Output = O> + crate::ops::Ln<Output = O> + Copy,
{
    type Output = FAD<O>;

    #[inline]
    fn ln(self) -> Self::Output {
        FAD {
            value: self.value.ln(),
            delta: self.delta / self.value,
        }
    }
}

impl<'a, T: 'a, O> crate::ops::Ln for &'a FAD<T>
where
    &'a T: std::ops::Div<&'a T, Output = O> + crate::ops::Ln<Output = O> + Copy,
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

impl<T1, T2, O> crate::ops::Log<T2> for FAD<T1>
where
    T1: std::ops::Div<O, Output = O>
        + crate::ops::Log<T2, Output = O>
        + crate::ops::Ln<Output = O>
        + Clone,
    T2: std::ops::Mul<O, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn log(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.clone().log(rhs.clone()),
            delta: self.delta / (rhs * self.value.ln()),
        }
    }
}

impl<'a, T1: 'a, T2, O> crate::ops::Log<T2> for &'a FAD<T1>
where
    &'a T1:
        std::ops::Div<O, Output = O> + crate::ops::Log<T2, Output = O> + crate::ops::Ln<Output = O>,
    T2: std::ops::Mul<O, Output = O> + Clone,
{
    type Output = FAD<O>;

    #[inline]
    fn log(self, rhs: T2) -> Self::Output {
        FAD {
            value: self.value.log(rhs.clone()),
            delta: &self.delta / (rhs * self.value.ln()),
        }
    }
}

macro_rules! impl_sqrt_fad {
    ($t:ty) => {
        impl crate::ops::Sqrt for FAD<$t> {
            type Output = FAD<$t>;

            #[inline]
            fn sqrt(self) -> Self::Output {
                let value = self.value.sqrt();
                FAD {
                    value,
                    delta: self.delta * (0.5 as $t) / value,
                }
            }
        }
    };
}

impl_sqrt_fad!(f32);
impl_sqrt_fad!(f64);

impl<T, O> crate::ops::Trig for FAD<T>
where
    T: std::ops::Mul<O, Output = O>
        + std::ops::Div<O, Output = O>
        + crate::ops::Trig<Output = O>
        + Clone,
    O: std::ops::Mul<O, Output = O> + std::ops::Neg<Output = O> + Clone,
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
        let cos = self.value.clone().cos();
        FAD {
            value: self.value.tan(),
            delta: self.delta / (cos.clone() * cos),
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
            let a: FAD<_> = $a.into();
            let b: FAD<_> = $b.into();
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
                assert_fad_eq!((b * 2. + c), (a * 2. + c, 2.));
                assert_fad_eq!(((b + c) * -2.), ((a + c) * -2., -2.));
                assert_fad_eq!(-b, (-a, -1.));
                assert_fad_eq!(b / c, (a / c, 1. / c));
                assert_fad_eq!(b.pow(c), (a.powf(c), c * a.powf(c - 1.)));
                assert_fad_eq!(b.wop(c), (c.powf(a), c.ln() * c.powf(a)));
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
