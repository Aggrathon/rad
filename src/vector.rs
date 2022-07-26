/// Simple library for vectorised math
use crate::ops::*;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Wrapper for vector math
#[derive(PartialEq, Debug, Clone)]
enum Vector<T> {
    Vec(Vec<T>),
    Scalar(T),
    // Iter(Box<dyn Iterator<Item = T>>),
}

impl<'a, T> Vector<T> {
    #[allow(unused)]
    fn first(&'a self) -> Option<&'a T> {
        match self {
            Vector::Vec(v) => v.first(),
            Vector::Scalar(s) => Some(s),
        }
    }
}

impl<T> NumOps<T, Vector<T>> for Vector<T> where T: NumOps<T, T> + Copy {}
impl<'a, T> NumOps<T, Vector<T>> for &'a Vector<T> where T: NumOps<T, T> + Copy {}
impl<'a, T> NumOps<&'a T, Vector<T>> for Vector<T> where T: NumOps<T, T> + Copy {}
impl<'a, T> NumOps<&'a T, Vector<T>> for &'a Vector<T> where T: NumOps<T, T> + Copy {}
impl<T> NumOps<Vector<T>, Vector<T>> for Vector<T> where T: NumOps<T, T> + Copy {}
impl<'a, T> NumOps<Vector<T>, Vector<T>> for &'a Vector<T> where T: NumOps<T, T> + Copy {}
impl<'a, T> NumOps<&'a Vector<T>, Vector<T>> for Vector<T> where T: NumOps<T, T> + Copy {}
impl<'a, T> NumOps<&'a Vector<T>, Vector<T>> for &'a Vector<T> where T: NumOps<T, T> + Copy {}

impl<T> From<T> for Vector<T>
where
    T: NumOps + Copy,
{
    fn from(v: T) -> Self {
        Vector::Scalar(v)
    }
}

impl<T> From<Vec<T>> for Vector<T> {
    fn from(v: Vec<T>) -> Self {
        Vector::Vec(v)
    }
}

impl<'a, T> From<&'a Vector<T>> for Vector<T>
where
    T: Clone,
{
    fn from(from: &'a Vector<T>) -> Vector<T> {
        from.clone()
    }
}

macro_rules! impl_const_op {
    ($Trait:path, $fn:ident) => {
        impl<T> $Trait for Vector<T>
        where
            T: $Trait,
        {
            fn $fn() -> Self {
                Vector::Scalar(T::$fn())
            }
        }
    };
}

impl_const_op!(crate::ops::One, one);
impl_const_op!(crate::ops::Half, half);

macro_rules! impl_binary_op {
    ($Trait:tt, $f:ident) => {
        impl_binary_op!($Trait, $f, &'a Vector<T>, &'a T, scalar*);
        impl_binary_op!($Trait, $f, Vector<T>, &'a T, scalar*);
        impl_binary_op!($Trait, $f, &'a Vector<T>, T, scalar);
        impl_binary_op!($Trait, $f, Vector<T>, T, scalar);

        impl_binary_op!($Trait, $f, &'a Vector<T>, &'a Vector<T>, vec);
        impl_binary_op!($Trait, $f, Vector<T>, &'a Vector<T>, vec);
        impl_binary_op!($Trait, $f, &'a Vector<T>, Vector<T>, vec);
        impl_binary_op!($Trait, $f, Vector<T>, Vector<T>, vec);
    };
    ($Trait:tt, $f:ident, $LHS:ty, $RHS:ty, scalar*) => {
        #[allow(clippy::extra_unused_lifetimes)]
        impl<'a, T> $Trait<$RHS> for $LHS
        where
            T: $Trait<T, Output = T> + Copy,
        {
            type Output = Vector<T>;

            fn $f(self, rhs: $RHS) -> Self::Output {
                match self {
                    Vector::Vec(v) => Vector::Vec(v.iter().map(|v| v.$f(*rhs)).collect()),
                    Vector::Scalar(v) => Vector::Scalar(v.$f(*rhs)),
                }
            }
        }
    };
    ($Trait:tt, $f:ident, $LHS:ty, $RHS:ty, scalar) => {
        #[allow(clippy::extra_unused_lifetimes)]
        impl<'a, T> $Trait<$RHS> for $LHS
        where
            T: $Trait<T, Output = T> + Copy,
        {
            type Output = Vector<T>;

            fn $f(self, rhs: $RHS) -> Self::Output {
                match self {
                    Vector::Vec(v) => Vector::Vec(v.iter().map(|v| v.$f(rhs)).collect()),
                    Vector::Scalar(v) => Vector::Scalar(v.$f(rhs)),
                }
            }
        }
    };
    ($Trait:tt, $f:ident, $LHS:ty, $RHS:ty, vec) => {
        #[allow(clippy::extra_unused_lifetimes)]
        impl<'a, T> $Trait<$RHS> for $LHS
        where
            T: $Trait<T, Output = T> + Copy,
        {
            type Output = Vector<T>;

            fn $f(self, rhs: $RHS) -> Self::Output {
                match rhs {
                    Vector::Vec(vs) => match self {
                        Vector::Vec(vr) => {
                            assert!(
                                vs.len() == vr.len(),
                                "The vectors must be of equal length: {} != {}",
                                vs.len(),
                                vr.len()
                            );
                            Vector::Vec(vr.iter().zip(vs.iter()).map(|(a, b)| a.$f(*b)).collect())
                        }
                        Vector::Scalar(s) => Vector::Vec(vs.iter().map(|a| s.$f(*a)).collect()),
                    },
                    Vector::Scalar(s) => (self).$f(s),
                }
            }
        }
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);
impl_binary_op!(Div, div);
impl_binary_op!(Pow, pow);
impl_binary_op!(Log, log);

macro_rules! impl_unary_op {
    ($Trait:tt, $($fn:ident),+) => {
        impl_unary_op!($Trait, $($fn),+: Vector<T>);
        impl_unary_op!($Trait, $($fn),+: &'a Vector<T>);
    };
    ($Trait:tt, $($fn:ident),+: $LHS:ty) => {
        #[allow(clippy::extra_unused_lifetimes)]
        impl<'a, T> $Trait for $LHS
        where
            T: $Trait<Output = T> + Copy,
        {
            type Output = Vector<T>;

            $(
                fn $fn(self) -> Self::Output {
                    match self {
                        Vector::Vec(v) => Vector::Vec(v.iter().map(|a| a.$fn()).collect()),
                        Vector::Scalar(s) => Vector::Scalar(s.$fn()),
                    }
                }
            )+
        }
    };
}

impl_unary_op!(Neg, neg);
impl_unary_op!(Abs, abs);
impl_unary_op!(Square, square);
impl_unary_op!(Sqrt, sqrt);
impl_unary_op!(Ln, ln);
impl_unary_op!(Exp, exp);
impl_unary_op!(Trig, sin, cos, tan);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::FAD;

    macro_rules! check_ops {
        ($a:ident, $b:ident, $c:ident: $($fn:ident),+) => {
            $(
                assert_eq!((&$a).$fn(&$c), ((&$b).$fn(&$c)).value);
                assert_eq!(
                    *(&$a).$fn(&$c).first().unwrap(),
                    ($a).first().unwrap().$fn(($c).first().unwrap())
                );
            )+
        };
        ($a:ident, $b:ident: $($fn:ident),+) => {
            $(
                assert_eq!((&$a).$fn(), ((&$b).$fn()).value);
                assert_eq!(
                    *(&$a).$fn().first().unwrap(),
                    ($a).first().unwrap().$fn()
                );
            )+
        };
    }

    #[test]
    fn scalar() {
        _scalar(
            Vector::from(vec![2.0, 3., 5.678, 1. / 3., 0.0]),
            Vector::from(vec![3.0, 1., 6.783, 1. / 7., 2.0]),
        );
        _scalar(Vector::from(2.0), Vector::from(3.0));
    }

    fn _scalar(a: Vector<f32>, c: Vector<f32>) {
        let b = FAD::from(a.clone());
        check_ops!(a, b, c: mul, add, sub, div);
        // assert_eq!((&a).pow(&c), ((&b).pow(&c)).value);
        // assert_eq!(&a.log(&c), (&b.log(&c)).value);
        // assert_eq!((&a).exp_base(&c), ((&b).exp_base(&c)).value);
        // check_ops!(a, b: neg);
        // assert_eq!(-(&a), (-(&b)).value);
        // assert_eq!(&a.ln(), (&b.ln()).value);
        // assert_eq!(&a.exp(), (&b.exp()).value);
        // assert_eq!(&a.sqrt(), (&b.sqrt()).value);
        // assert_eq!(&a.sin(), (&b.sin()).value);
        // assert_eq!(&a.cos(), (&b.cos()).value);
        // assert_eq!(&a.tan(), (&b.tan()).value);
    }
}
