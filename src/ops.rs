use std::ops::{Add, Div, Mul, Neg, Sub};

/// A trait enforcing some basic numerical operations.
/// Designed to be compatible with floating point numbers.
/// And also vectors and wrappers thereof.
pub trait NumOps<RHS = Self, O = Self>:
    Add<RHS, Output = O>
    + Sub<RHS, Output = O>
    + Mul<RHS, Output = O>
    + Div<RHS, Output = O>
    + Neg<Output = O>
    + Ln<Output = O>
    + Exp<Output = O>
    + Abs<Output = O>
    + Pow<RHS, Output = O>
    + Log<RHS, Output = O>
// These are OPTIONAL additional traits:
// + One
// + Half
// + Square<Output = O>
// + Sqrt<Output = O>
// + Trig<Output = O>
{
}

impl NumOps<f32, f32> for f32 {}
impl NumOps<&f32, f32> for f32 {}
impl NumOps<f32, f32> for &f32 {}
impl NumOps<&f32, f32> for &f32 {}
impl NumOps<f64, f64> for f64 {}
impl NumOps<&f64, f64> for f64 {}
impl NumOps<f64, f64> for &f64 {}
impl NumOps<&f64, f64> for &f64 {}

pub trait One {
    /// Returns the multiplicative identity element of `Self`, `1`.
    fn one() -> Self;
}

pub trait Half {
    /// Returns the multiplicative halving of `Self`, `0.5`.
    fn half() -> Self;
}

/// Unary operator for calculating the natural logarithm.
pub trait Ln {
    /// The result after applying the operator.
    type Output;

    /// Returns the natural logarithm of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use rad::ops::*;
    /// assert!((1.0f32 - Ln::ln(1.0f32).exp()).abs() < 1e-4);
    /// ```
    fn ln(self) -> Self::Output;
}

/// Unary operator for calculating the natural exponent.
pub trait Exp {
    /// The result after applying the operator.
    type Output;

    /// Returns the natural exponent of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use rad::ops::*;
    /// assert!((1.0 - Exp::exp(1.0f32).ln()).abs() < 1e-4);
    /// ```
    fn exp(self) -> Self::Output;
}

/// Unary operator for calculating the square root.
pub trait Sqrt {
    /// The result after applying the operator.
    type Output;

    /// Returns the square root of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use rad::ops::*;
    /// assert!((2.0f32.sqrt() - Sqrt::sqrt(2.0f32)).abs() < 1e-4);
    /// ```
    fn sqrt(self) -> Self::Output;
}

/// Unary operator for calculating the square.
pub trait Square {
    /// The result after applying the operator.
    type Output;

    /// Returns the square of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use rad::ops::*;
    /// assert_eq!(2.0f32*2.0, Square::square(2.0f32));
    /// ```
    fn square(self) -> Self::Output;
}

/// Unary operator for calculating the absolute value.
pub trait Abs {
    /// The result after applying the operator.
    type Output;

    /// Returns the absolute value of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use rad::ops::*;
    /// assert_eq!(2.0f32, Abs::abs(-2.0f32));
    /// ```
    fn abs(self) -> Self::Output;
}

/// Binary operator for raising a value to a power.
pub trait Pow<RHS = Self> {
    /// The result after applying the operator.
    type Output;

    /// Returns `self` to the power `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use rad::ops::*;
    /// assert!((2.0f32.powf(2.0) - Pow::pow(2.0f32, 2.0)).abs() < 1e-4);
    /// ```
    fn pow(self, rhs: RHS) -> Self::Output;
}

/// Binary operator for calculating the logarithm for a specified bases.
pub trait Log<RHS = Self> {
    /// The result after applying the operator.
    type Output;

    /// Returns the logarithm of `self` with the base `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use rad::ops::*;
    /// assert_eq!(3.0f32.log(2.0f32), Log::log(3.0f32, 2.0f32));
    /// ```
    fn log(self, rhs: RHS) -> Self::Output;
}

/// Unary operators for trigonometric functions.
pub trait Trig {
    /// The result after applying the operators.
    type Output;

    /// Returns the sinus `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use rad::ops::*;
    /// assert!((2.0f32.sin() - Trig::sin(2.0f32)).abs() < 1e-4);
    /// ```
    fn sin(self) -> Self::Output;

    /// Returns the cosinus `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use rad::ops::*;
    /// assert!((2.0f32.cos() - Trig::cos(2.0f32)).abs() < 1e-4);
    /// ```
    fn cos(self) -> Self::Output;

    /// Returns the Tangens `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use rad::ops::*;
    /// assert!((2.0f32.tan() - Trig::tan(2.0f32)).abs() < 1e-4);
    /// ```
    fn tan(self) -> Self::Output;
}

macro_rules! const_impl {
    ($T:ty,$Trait:ty, $fn:ident, $v:expr) => {
        impl $Trait for $T {
            #[inline]
            fn $fn() -> $T {
                $v
            }
        }
    };
}

const_impl!(f32, One, one, 1.0);
const_impl!(f64, One, one, 1.0);
const_impl!(f32, Half, half, 0.5);
const_impl!(f64, Half, half, 0.5);

macro_rules! unary_impl {
    ($T:ty, $Trait:ty, $($f:ident),+) => {
        impl $Trait for $T {
            type Output = $T;

            $(
                #[inline]
                fn $f(self) -> Self::Output {
                    self.$f()
                }
            )+
        }

        impl $Trait for &$T {
            type Output = $T;

            $(
                #[inline]
                fn $f(self) -> Self::Output {
                    (*self).$f()
                }
            )+
        }
    };
}

unary_impl!(f32, Exp, exp);
unary_impl!(f64, Exp, exp);
unary_impl!(f32, Ln, ln);
unary_impl!(f64, Ln, ln);
unary_impl!(f32, Sqrt, sqrt);
unary_impl!(f64, Sqrt, sqrt);
unary_impl!(f32, Abs, abs);
unary_impl!(f64, Abs, abs);
unary_impl!(f32, Trig, sin, cos, tan);
unary_impl!(f64, Trig, sin, cos, tan);

macro_rules! square_impl {
    ($T:ty, $Trait:ty, $f:ident) => {
        impl $Trait for $T {
            type Output = $T;

            #[inline]
            fn $f(self) -> Self::Output {
                self * self
            }
        }

        impl $Trait for &$T {
            type Output = $T;

            #[inline]
            fn $f(self) -> Self::Output {
                (*self) * self
            }
        }
    };
}

square_impl!(f32, Square, square);
square_impl!(f64, Square, square);

macro_rules! binary_impl {
    ($T:ty, $Trait:tt, $f1:tt, $f2:tt) => {
        impl $Trait<$T> for $T {
            type Output = $T;

            #[inline]
            fn $f1(self, rhs: $T) -> Self::Output {
                self.$f2(rhs)
            }
        }

        impl $Trait<$T> for &$T {
            type Output = $T;

            #[inline]
            fn $f1(self, rhs: $T) -> Self::Output {
                (*self).$f2(rhs)
            }
        }

        impl $Trait<&$T> for $T {
            type Output = $T;

            #[inline]
            fn $f1(self, rhs: &$T) -> Self::Output {
                self.$f2(*rhs)
            }
        }

        impl $Trait<&$T> for &$T {
            type Output = $T;

            #[inline]
            fn $f1(self, rhs: &$T) -> Self::Output {
                (*self).$f2(*rhs)
            }
        }
    };
}

binary_impl!(f32, Log, log, log);
binary_impl!(f64, Log, log, log);
binary_impl!(f32, Pow, pow, powf);
binary_impl!(f64, Pow, pow, powf);

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
