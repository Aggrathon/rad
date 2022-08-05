/// Traits for operations on numerical values
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A trait enforcing some basic numerical operations.
/// Designed to be compatible with floating point numbers, but also vectors and wrappers thereof.
pub trait NumOps<RHS = Self, Output = Self>:
    Add<RHS, Output = Output>
    + Sub<RHS, Output = Output>
    + Mul<RHS, Output = Output>
    + Div<RHS, Output = Output>
    + Neg<Output = Output>
    + Ln<Output = Output>
    + Exp<Output = Output>
    + Abs<Output = Output>
    + Pow<RHS, Output = Output>
    + Log<RHS, Output = Output>
// These are OPTIONAL additional traits:
// + Signum
// + Square<Output = Output>
// + Sqrt<Output = Output>
// + Trig<Output = Output>
{
}

/// A trait for additional, optional numerical operations
pub trait NumOpts<Output>:
    Square<Output = Output> + Sqrt<Output = Output> + Trig<Output = Output>
// Since these collection of Operation traits are designed for use with automatic
// gradients the optional trait `Signum` is not included since it lacks a gradient_
// + Signum<Output = Output>
{
}

/// A trait for producing some static values for numerical objects.
pub trait NumConsts: One + Zero + Half + Two + Epsilon {}

/// A trait for aggregation operations on collections of numerical values
pub trait AggOps<Output = Self>: Sum<Output = Output> + Prod<Output = Output> {}

pub trait One {
    /// Returns the multiplicative identity element of `Self`, `1`.
    fn one() -> Self;
}

pub trait Zero {
    /// Returns the additive identity element of `Self`, `0`.
    fn zero() -> Self;
}

pub trait Half {
    /// Returns the multiplicative halving of `Self`, `0.5`.
    fn half() -> Self;
}

pub trait Two {
    /// Returns the multiplicative double of `Self`, `2.0`.
    fn two() -> Self;
}

pub trait Epsilon {
    fn epsilon() -> Self;
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

/// Unary operator for calculating the sign.
pub trait Signum {
    /// The result after applying the operator.
    type Output;

    /// Returns the sign of `self`:
    /// - '+1' for `+0.0..infty`.
    /// - '-1' for `-0.0..-infty`.
    /// - 'NaN' for `NaN`
    ///
    /// # Example
    ///
    /// ```
    /// use rad::ops::*;
    /// assert_eq!(1.0f32, Signum::signum(5.0f32));
    /// assert_eq!(-1.0f32, Signum::signum(-2.0f32));
    /// assert_eq!(-1.0f32, Signum::signum(-0.0f32));
    /// ```
    fn signum(self) -> Self::Output;
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

/// Aggregation operator for summing values.
pub trait Sum {
    /// The result after applying the operator.
    type Output;

    /// Returns the sum of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use rad::ops::*;
    /// assert!((Sum::sum(vec![1.0f32, 2.0, 3.0]) - 6.0).abs() < 1e-4);
    /// ```
    fn sum(self) -> Self::Output;
}

/// Aggregation operator for multiplying values.
pub trait Prod {
    /// The result after applying the operator.
    type Output;

    /// Returns the product of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use rad::ops::*;
    /// assert!((Prod::prod(vec![1.0f32, 2.0, 3.0]) - 6.0).abs() < 1e-4);
    /// ```
    fn prod(self) -> Self::Output;
}

// ################## Implementations ##################

impl NumOps<f32, f32> for f32 {}
impl NumOps<&f32, f32> for f32 {}
impl NumOps<f32, f32> for &f32 {}
impl NumOps<&f32, f32> for &f32 {}
impl NumOps<f64, f64> for f64 {}
impl NumOps<&f64, f64> for f64 {}
impl NumOps<f64, f64> for &f64 {}
impl NumOps<&f64, f64> for &f64 {}

impl NumOpts<f32> for f32 {}
impl NumOpts<f32> for &f32 {}
impl NumOpts<f64> for f64 {}
impl NumOpts<f64> for &f64 {}

impl NumConsts for f32 {}
impl NumConsts for f64 {}

macro_rules! impl_const {
    ($T:ty,$Trait:ty, $fn:ident, $v:expr) => {
        impl $Trait for $T {
            #[inline]
            fn $fn() -> $T {
                $v
            }
        }
    };
}

impl_const!(f32, One, one, 1.0);
impl_const!(f64, One, one, 1.0);
impl_const!(f32, Zero, zero, 0.0);
impl_const!(f64, Zero, zero, 0.0);
impl_const!(f32, Half, half, 0.5);
impl_const!(f64, Half, half, 0.5);
impl_const!(f32, Two, two, 2.0);
impl_const!(f64, Two, two, 2.0);
impl_const!(f32, Epsilon, epsilon, f32::EPSILON);
impl_const!(f64, Epsilon, epsilon, f64::EPSILON);

macro_rules! impl_unary {
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

impl_unary!(f32, Exp, exp);
impl_unary!(f64, Exp, exp);
impl_unary!(f32, Ln, ln);
impl_unary!(f64, Ln, ln);
impl_unary!(f32, Sqrt, sqrt);
impl_unary!(f64, Sqrt, sqrt);
impl_unary!(f32, Abs, abs);
impl_unary!(f64, Abs, abs);
impl_unary!(f32, Signum, signum);
impl_unary!(f64, Signum, signum);
impl_unary!(f32, Trig, sin, cos, tan);
impl_unary!(f64, Trig, sin, cos, tan);

macro_rules! impl_square {
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
                (*self) * (*self)
            }
        }
    };
}

impl_square!(f32, Square, square);
impl_square!(f64, Square, square);

macro_rules! impl_binary {
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

impl_binary!(f32, Log, log, log);
impl_binary!(f64, Log, log, log);
impl_binary!(f32, Pow, pow, powf);
impl_binary!(f64, Pow, pow, powf);

macro_rules! impl_agg {
    ($T:ty, $AT:tt, $fna:ident, $ET:tt, $fne:ident, $IT:tt, $fni:ident) => {
        #[allow(clippy::extra_unused_lifetimes)]
        impl<'a, T> $AT for $T
        where
            T: $ET<T, Output = T> + $IT + Copy,
        {
            type Output = T;

            #[inline]
            fn $fna(self) -> Self::Output {
                self.iter().fold(T::$fni(), |a, b| $ET::$fne(a, *b))
            }
        }
    };
}

impl_agg!(Vec<T>, Sum, sum, Add, add, Zero, zero);
impl_agg!(&'a Vec<T>, Sum, sum, Add, add, Zero, zero);
impl_agg!(Vec<T>, Prod, prod, Mul, mul, One, one);
impl_agg!(&'a Vec<T>, Prod, prod, Mul, mul, One, one);

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
