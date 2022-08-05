/// Additional Traits for operations on numerical values
//
use crate::ops::*;
use std::ops::Div;

/// This is a copy of std::ops::Div to get around E0210.
/// The idea is that this can be implemented for `custom_object._div(number)` where the normal `Div` is used for `number.div(custom_object)`.
pub trait Div2<RHS = Self> {
    type Output;

    fn _div(self, rhs: RHS) -> Self::Output;
}

/// This is a copy of Pow to get around E0210
/// The idea is that this can be implemented for `custom_object._pow(number)` where the normal `Pow` is used for `number.pow(custom_object)`.
pub trait Pow2<RHS = Self> {
    type Output;

    fn _pow(self, rhs: RHS) -> Self::Output;
}

/// This is a copy of Log to get around E0210
/// The idea is that this can be implemented for `custom_object._log(number)` where the normal `Log` is used for `number.log(custom_object)`.
pub trait Log2<RHS = Self> {
    type Output;

    fn _log(self, rhs: RHS) -> Self::Output;
}

macro_rules! impl_rev {
    ($Trait:tt, $Trait2:tt, $fn:ident, $fn2:ident) => {
        impl<T, O> $Trait2<T> for T
        where
            T: $Trait<T, Output = O>,
        {
            type Output = O;

            fn $fn2(self, rhs: T) -> Self::Output {
                self.$fn(rhs)
            }
        }
    };
}

impl_rev!(Div, Div2, div, _div);
impl_rev!(Pow, Pow2, pow, _pow);
impl_rev!(Log, Log2, log, _log);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traits() {
        for a in [2.0f32, -3., 5.3213] {
            assert_eq!(Exp::exp(a), a.exp());
            assert_eq!(Ln::ln(a.abs()), a.abs().ln());
            assert_eq!(Sqrt::sqrt(a.abs()), a.abs().sqrt());
            assert_eq!(Abs::abs(a), a.abs());
            assert_eq!(Trig::sin(a), a.sin());
            assert_eq!(Trig::cos(a), a.cos());
            assert_eq!(Trig::tan(a), a.tan());
            for b in [1.0f32, 1. / 3., -6.] {
                assert_eq!(Pow::pow(a.abs(), b), a.abs().powf(b));
                assert_eq!(Log::log(a.abs(), b.abs()), a.abs().log(b.abs()));
            }
        }
        for a in [1.0f64, -2.453] {
            assert_eq!(Exp::exp(a), a.exp());
            assert_eq!(Ln::ln(a.abs()), a.abs().ln());
            assert_eq!(Sqrt::sqrt(a.abs()), a.abs().sqrt());
            assert_eq!(Abs::abs(a), a.abs());
            assert_eq!(Trig::sin(a), a.sin());
            assert_eq!(Trig::cos(a), a.cos());
            assert_eq!(Trig::tan(a), a.tan());
            for b in [2.0f64, -0.1] {
                assert_eq!(Pow::pow(a.abs(), b), a.abs().powf(b));
                assert_eq!(Log::log(a.abs(), b.abs()), a.abs().log(b.abs()));
            }
        }
    }
}
