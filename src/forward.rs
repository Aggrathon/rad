// Forward Automatic Differentiation

use num_traits::identities::One;

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
    fn from(value: T) -> Self {
        FAD {
            value,
            delta: T::one(),
        }
    }
}

impl<T> From<(T, T)> for FAD<T> {
    fn from(value: (T, T)) -> Self {
        FAD {
            value: value.0,
            delta: value.1,
        }
    }
}

impl<T, O> std::ops::Add<O> for FAD<T>
where
    T: std::ops::Add<O, Output = T>,
{
    type Output = FAD<T>;

    fn add(self, rhs: O) -> Self::Output {
        FAD {
            value: self.value.add(rhs),
            delta: self.delta,
        }
    }
}

impl<T, O> std::ops::Sub<O> for FAD<T>
where
    T: std::ops::Sub<O, Output = T>,
{
    type Output = FAD<T>;

    fn sub(self, rhs: O) -> Self::Output {
        FAD {
            value: self.value.sub(rhs),
            delta: self.delta,
        }
    }
}

impl<T, O> std::ops::Mul<O> for FAD<T>
where
    T: std::ops::Mul<O, Output = T>,
    O: Copy,
{
    type Output = FAD<T>;

    fn mul(self, rhs: O) -> Self::Output {
        FAD {
            value: self.value.mul(rhs),
            delta: self.delta.mul(rhs),
        }
    }
}

impl<T> std::ops::Neg for FAD<T>
where
    T: std::ops::Neg<Output = T>,
{
    type Output = FAD<T>;

    fn neg(self) -> Self::Output {
        FAD {
            value: self.value.neg(),
            delta: self.delta.neg(),
        }
    }
}

impl<T, O> std::ops::Div<O> for FAD<T>
where
    T: std::ops::Div<O, Output = T>,
    O: Copy,
{
    type Output = FAD<T>;

    fn div(self, rhs: O) -> Self::Output {
        FAD {
            value: self.value.div(rhs),
            delta: self.delta.div(rhs),
        }
    }
}

impl<T, O> num_traits::pow::Pow<O> for FAD<T>
where
    T: num_traits::pow::Pow<O, Output = T> + std::ops::Mul<Output = T> + Copy,
    O: std::ops::Sub<Output = O> + std::ops::Mul<T, Output = T> + One + Copy,
{
    type Output = FAD<T>;

    fn pow(self, rhs: O) -> Self::Output {
        FAD {
            value: self.value.pow(rhs),
            delta: self.delta * (rhs * self.value.pow(rhs - O::one())),
        }
    }
}

impl<T> crate::ops::Exp for FAD<T>
where
    T: std::ops::Mul<Output = T> + crate::ops::Exp<Output = T> + Copy,
{
    type Output = FAD<T>;

    fn exp(self) -> Self::Output {
        let value = self.value.exp();
        FAD {
            value,
            delta: self.delta * value,
        }
    }
}

impl<T, O> crate::ops::ExpBase<O> for FAD<T>
where
    O: num_traits::pow::Pow<T, Output = T> + crate::ops::Ln<Output = T> + Copy,
    T: std::ops::Mul<Output = T> + crate::ops::ExpBase<O, Output = T> + Copy,
{
    type Output = FAD<T>;

    fn exp_base(self, rhs: O) -> Self::Output {
        let value = self.value.exp_base(rhs);
        FAD {
            value,
            delta: self.delta * (rhs.ln() * value),
        }
    }
}

impl<T> crate::ops::Ln for FAD<T>
where
    T: std::ops::Div<Output = T> + crate::ops::Ln<Output = T> + Copy,
{
    type Output = FAD<T>;

    fn ln(self) -> Self::Output {
        FAD {
            value: self.value.ln(),
            delta: self.delta / self.value,
        }
    }
}

impl<T, O> crate::ops::Log<O> for FAD<T>
where
    T: std::ops::Div<Output = T>
        + crate::ops::Log<O, Output = T>
        + crate::ops::Ln<Output = T>
        + Copy,
    O: std::ops::Mul<T, Output = T> + Copy,
{
    type Output = FAD<T>;

    fn log(self, rhs: O) -> Self::Output {
        FAD {
            value: self.value.log(rhs),
            delta: self.delta / (rhs * self.value.ln()),
        }
    }
}

impl crate::ops::Sqrt for FAD<f32> {
    type Output = FAD<f32>;

    fn sqrt(self) -> Self::Output {
        let value = self.value.sqrt();
        FAD {
            value,
            delta: self.delta * 0.5f32 / value,
        }
    }
}

impl crate::ops::Sqrt for FAD<f64> {
    type Output = FAD<f64>;

    fn sqrt(self) -> Self::Output {
        let value = self.value.sqrt();
        FAD {
            value,
            delta: self.delta * 0.5f64 / value,
        }
    }
}

impl<T> crate::ops::Trig for FAD<T>
where
    T: std::ops::Mul<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + crate::ops::Trig<Output = T>
        + Copy,
{
    type Output = FAD<T>;

    fn sin(self) -> Self::Output {
        FAD {
            value: self.value.sin(),
            delta: self.delta * self.value.cos(),
        }
    }

    fn cos(self) -> Self::Output {
        FAD {
            value: self.value.cos(),
            delta: self.delta * (-self.value.sin()),
        }
    }

    fn tan(self) -> Self::Output {
        let cos = self.value.cos();
        FAD {
            value: self.value.tan(),
            delta: self.delta / (cos * cos),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::*;
    use num_traits::pow::Pow;

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
                assert_fad_eq!(b.exp_base(c), (c.powf(a), c.ln() * c.powf(a)));
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
