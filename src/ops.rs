/// Unary operator for calculating the natural logarithm.
pub trait Ln {
    /// The result after applying the operator.
    type Output;

    /// Returns the natural logarithm of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!((1.0 - Ln::ln(1.0f32).exp()).abs() < 1e-4);
    /// ```
    fn ln(self) -> Self::Output;
}

/// Unary operator for calculating the natural exponent.
pub trait Exp {
    /// The result after applying the operator.
    type Output;

    /// Returns the natural exponent of `self`.
    ///
    /// # Examples
    ///
    /// ```
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
    /// # Examples
    ///
    /// ```
    /// assert!((2.0f32.sqrt() - Sqrt::sqrt(2.0f32)).abs() < 1e-4);
    /// ```
    fn sqrt(self) -> Self::Output;
}

/// Binary operator for calculating the exponent for a specified bases.
pub trait ExpBase<RHS> {
    /// The result after applying the operator.
    type Output;

    /// Returns the `base` to the power of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(2.0f32.powf(2.0f32), ExpBase::exp_base(2.0f32, 2.0f32));
    /// ```
    fn exp_base(self, base: RHS) -> Self::Output;
}

/// Binary operator for calculating the logarithm for a specified bases.
pub trait Log<RHS> {
    /// The result after applying the operator.
    type Output;

    /// Returns the `base` to the power of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(2.0f32.log(2.0f32), Log::log(2.0f32, 2.0f32));
    /// ```
    fn log(self, base: RHS) -> Self::Output;
}

/// Unary operators for trigonometric functions.
pub trait Trig {
    /// The result after applying the operators.
    type Output;

    /// Returns the sinus `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!((2.0f32.sin() - Trig::sin(2.0f32)).abs() < 1e-4);
    /// ```
    fn sin(self) -> Self::Output;

    /// Returns the cosinus `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!((2.0f32.cos() - Trig::cos(2.0f32)).abs() < 1e-4);
    /// ```
    fn cos(self) -> Self::Output;

    /// Returns the Tangens `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!((2.0f32.tan() - Trig::tan(2.0f32)).abs() < 1e-4);
    /// ```
    fn tan(self) -> Self::Output;
}

macro_rules! unary_impl {
    ($t:ty, $Trait:ty, $($f:ident),+) => {
        impl $Trait for $t {
            type Output = $t;

            $(fn $f(self) -> Self::Output {
                self.$f()
            })+
        }

        impl $Trait for &$t {
            type Output = $t;

            $(fn $f(self) -> Self::Output {
                (*self).$f()
            })+
        }
    };
}

unary_impl!(f32, Exp, exp);
unary_impl!(f64, Exp, exp);
unary_impl!(f32, Ln, ln);
unary_impl!(f64, Ln, ln);
unary_impl!(f32, Sqrt, sqrt);
unary_impl!(f64, Sqrt, sqrt);
unary_impl!(f32, Trig, sin, cos, tan);
unary_impl!(f64, Trig, sin, cos, tan);

macro_rules! binary_impl {
    ($t:ty, $RHS:ty, $Trait:tt, $f1:tt, $f2:tt) => {
        impl $Trait<$RHS> for $t {
            type Output = $t;

            fn $f1(self, rhs: $RHS) -> Self::Output {
                self.$f2(rhs)
            }
        }

        impl $Trait<$RHS> for &$t {
            type Output = $t;

            fn $f1(self, rhs: $RHS) -> Self::Output {
                (*self).$f2(rhs)
            }
        }

        impl $Trait<&$RHS> for $t {
            type Output = $t;

            fn $f1(self, rhs: &$RHS) -> Self::Output {
                self.$f2(*rhs)
            }
        }

        impl $Trait<&$RHS> for &$t {
            type Output = $t;

            fn $f1(self, rhs: &$RHS) -> Self::Output {
                (*self).$f2(*rhs)
            }
        }
    };
}

binary_impl!(f32, f32, Log, log, log);
binary_impl!(f64, f64, Log, log, log);

macro_rules! rev_binary_impl {
    ($t:ty, $RHS:ty, $Trait:tt, $f1:tt, $f2:tt) => {
        impl $Trait<$RHS> for $t {
            type Output = $t;

            fn $f1(self, rhs: $RHS) -> Self::Output {
                rhs.$f2(self)
            }
        }

        impl $Trait<$RHS> for &$t {
            type Output = $t;

            fn $f1(self, rhs: $RHS) -> Self::Output {
                rhs.$f2(*self)
            }
        }

        impl $Trait<&$RHS> for $t {
            type Output = $t;

            fn $f1(self, rhs: &$RHS) -> Self::Output {
                (*rhs).$f2(self)
            }
        }

        impl $Trait<&$RHS> for &$t {
            type Output = $t;

            fn $f1(self, rhs: &$RHS) -> Self::Output {
                (*rhs).$f2(*self)
            }
        }
    };
}

rev_binary_impl!(f32, f32, ExpBase, exp_base, powf);
rev_binary_impl!(f64, f64, ExpBase, exp_base, powf);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traits() {
        for a in [2.0f32, 3., 5.3213] {
            assert_eq!(Exp::exp(a), a.exp());
            assert_eq!(Ln::ln(a), a.ln());
            assert_eq!(Sqrt::sqrt(a), a.sqrt());
            assert_eq!(Trig::sin(a), a.sin());
            assert_eq!(Trig::cos(a), a.cos());
            assert_eq!(Trig::tan(a), a.tan());
            for b in [1.0f32, 1. / 3., 6.] {
                assert_eq!(ExpBase::exp_base(a, b), b.powf(a));
                assert_eq!(Log::log(a, b), a.log(b));
            }
        }
        for a in [1.0f64, 2.453] {
            assert_eq!(Exp::exp(a), a.exp());
            assert_eq!(Ln::ln(a), a.ln());
            assert_eq!(Sqrt::sqrt(a), a.sqrt());
            assert_eq!(Trig::sin(a), a.sin());
            assert_eq!(Trig::cos(a), a.cos());
            assert_eq!(Trig::tan(a), a.tan());
            for b in [2.0f64] {
                assert_eq!(ExpBase::exp_base(a, b), b.powf(a));
                assert_eq!(Log::log(a, b), a.log(b));
            }
        }
    }
}
