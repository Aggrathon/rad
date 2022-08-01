/// Simple library for vectorised math
use crate::ops::*;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Wrapper for vector math
#[derive(PartialEq, Debug, Clone)]
pub enum Vector<T> {
    Vec(Vec<T>),
    Scalar(T),
    // Iter(Box<dyn Iterator<Item = T>>),
}

impl<'a, T> Vector<T> {
    #[allow(unused)]
    #[inline]
    pub fn first(&'a self) -> Option<&'a T> {
        match self {
            Vector::Vec(v) => v.first(),
            Vector::Scalar(s) => Some(s),
        }
    }
}

impl<T> Vector<T> {
    #[allow(unused)]
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Vector::Vec(v) => v.len(),
            Vector::Scalar(_) => 1,
        }
    }

    #[allow(unused)]
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            Vector::Vec(v) => v.is_empty(),
            Vector::Scalar(_) => false,
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

impl<T> NumConsts for Vector<T> where T: NumConsts {}

impl<T> NumOpts<Vector<T>> for Vector<T> where T: NumOpts<T> + Copy {}
impl<'a, T> NumOpts<Vector<T>> for &'a Vector<T> where T: NumOpts<T> + Copy {}

impl<T> AggOps<Vector<T>> for Vector<T> where T: One + Zero + Mul<T, Output = T> + Add<T, Output = T>
{}

impl<'a, T> AggOps<Vector<T>> for &'a Vector<T> where
    T: One + Zero + Mul<T, Output = T> + Add<T, Output = T> + Copy
{
}

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
            #[inline]
            fn $fn() -> Self {
                Vector::Scalar(T::$fn())
            }
        }
    };
}

impl_const_op!(crate::ops::One, one);
impl_const_op!(crate::ops::Zero, zero);
impl_const_op!(crate::ops::Half, half);
impl_const_op!(crate::ops::Two, two);

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
    ($Trait:tt, $f:ident: rev) => {
        impl_binary_op!($Trait, $f, T, &'a Vector<T>, scalar_rev);
        impl_binary_op!($Trait, $f, T, Vector<T>, scalar_rev);
    };
    ($Trait:tt, $f:ident, $LHS:ty, $RHS:ty, scalar*) => {
        #[allow(clippy::extra_unused_lifetimes)]
        impl<'a, T> $Trait<$RHS> for $LHS
        where
            T: $Trait<T, Output = T> + Copy,
        {
            type Output = Vector<T>;

            #[inline]
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

            #[inline]
            fn $f(self, rhs: $RHS) -> Self::Output {
                match self {
                    Vector::Vec(v) => Vector::Vec(v.iter().map(|v| v.$f(rhs)).collect()),
                    Vector::Scalar(v) => Vector::Scalar(v.$f(rhs)),
                }
            }
        }
    };
    ($Trait:tt, $f:ident, $LHS:ty, $RHS:ty, scalar_rev) => {
        #[allow(clippy::extra_unused_lifetimes)]
        impl<'a, T> $Trait<$RHS> for $LHS
        where
            T: $Trait<T, Output = T> + Copy,
        {
            type Output = Vector<T>;

            #[inline]
            fn $f(self, rhs: $RHS) -> Self::Output {
                match &rhs {
                    Vector::Vec(v) => Vector::Vec(v.iter().map(|v| self.$f(*v)).collect()),
                    Vector::Scalar(v) => Vector::Scalar(self.$f(*v)),
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

            #[inline]
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
impl_binary_op!(Log2, _log: rev);
impl_binary_op!(Pow2, _pow: rev);
impl_binary_op!(Div2, _div: rev);

macro_rules! impl_assign_op {
    ($Trait:tt, $fn:ident) => {
        impl<T> $Trait for Vector<T>
        where
            T: $Trait + Clone,
        {
            fn $fn(&mut self, rhs: Self) {
                match self {
                    Vector::Vec(lhs) => match rhs {
                        Vector::Vec(rhs) => {
                            assert!(
                                lhs.len() == rhs.len(),
                                "The other vector must be of equal length: {} != {}",
                                lhs.len(),
                                rhs.len()
                            );
                            lhs.iter_mut()
                                .zip(rhs.into_iter())
                                .for_each(|(a, b)| a.$fn(b));
                        }
                        Vector::Scalar(rhs) => lhs.iter_mut().for_each(|a| a.$fn(rhs.clone())),
                    },
                    Vector::Scalar(lhs) => match rhs {
                        Vector::Vec(mut rhs) => {
                            assert!(rhs.len() == 1, "The other vector must be of length one");
                            lhs.$fn(rhs.remove(0));
                        }
                        Vector::Scalar(rhs) => lhs.$fn(rhs),
                    },
                }
            }
        }

        impl<'a, T> $Trait<&'a Vector<T>> for Vector<T>
        where
            T: $Trait<&'a T>,
        {
            fn $fn(&mut self, rhs: &'a Self) {
                match self {
                    Vector::Vec(lhs) => match rhs {
                        Vector::Vec(rhs) => {
                            assert!(
                                lhs.len() == rhs.len(),
                                "The other vector must be of equal length: {} != {}",
                                lhs.len(),
                                rhs.len()
                            );
                            lhs.iter_mut()
                                .zip(rhs.into_iter())
                                .for_each(|(a, b)| a.$fn(b));
                        }
                        Vector::Scalar(rhs) => lhs.iter_mut().for_each(|a| a.$fn(rhs)),
                    },
                    Vector::Scalar(lhs) => match rhs {
                        Vector::Vec(rhs) => {
                            assert!(rhs.len() == 1, "The other vector must be of length one");
                            lhs.$fn(&rhs[0]);
                        }
                        Vector::Scalar(rhs) => lhs.$fn(rhs),
                    },
                }
            }
        }

        impl<T> $Trait<T> for Vector<T>
        where
            T: $Trait + Clone,
        {
            fn $fn(&mut self, rhs: T) {
                match self {
                    Vector::Vec(lhs) => lhs.iter_mut().for_each(|a| a.$fn(rhs.clone())),
                    Vector::Scalar(lhs) => lhs.$fn(rhs),
                }
            }
        }

        impl<'a, T> $Trait<&'a T> for Vector<T>
        where
            T: $Trait<&'a T>,
        {
            fn $fn(&mut self, rhs: &'a T) {
                match self {
                    Vector::Vec(lhs) => lhs.iter_mut().for_each(|a| a.$fn(rhs)),
                    Vector::Scalar(lhs) => lhs.$fn(rhs),
                }
            }
        }
    };
}

impl_assign_op!(AddAssign, add_assign);
impl_assign_op!(SubAssign, sub_assign);
impl_assign_op!(MulAssign, mul_assign);
impl_assign_op!(DivAssign, div_assign);

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
                #[inline]
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
impl_unary_op!(Signum, signum);
impl_unary_op!(Square, square);
impl_unary_op!(Sqrt, sqrt);
impl_unary_op!(Ln, ln);
impl_unary_op!(Exp, exp);
impl_unary_op!(Trig, sin, cos, tan);

macro_rules! impl_agg_op {
    ($Trait:tt, $fn:ident, $Trait2:tt, $fn2:ident, $Trait3:tt, $fn3:ident) => {
        impl<T> $Trait for Vector<T>
        where
            T: $Trait3 + $Trait2<T, Output = T>,
        {
            type Output = Vector<T>;

            fn $fn(self) -> Self::Output {
                match self {
                    Vector::Vec(v) => {
                        Vector::Scalar(v.into_iter().fold(T::$fn3(), |a, b| a.$fn2(b)))
                    }
                    Vector::Scalar(v) => Vector::Scalar(v),
                }
            }
        }
        impl<'a, T> $Trait for &'a Vector<T>
        where
            T: $Trait3 + $Trait2<T, Output = T> + Copy,
        {
            type Output = Vector<T>;

            fn $fn(self) -> Self::Output {
                match self {
                    Vector::Vec(v) => Vector::Scalar(v.iter().fold(T::$fn3(), |a, b| a.$fn2(*b))),
                    Vector::Scalar(v) => Vector::Scalar(*v),
                }
            }
        }
    };
}

impl_agg_op!(Sum, sum, Add, add, Zero, zero);
impl_agg_op!(Prod, prod, Mul, mul, One, one);

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
                assert_eq!(
                    (&$a).$fn(&$c).first().unwrap(),
                    (&$a).$fn(($c).first().unwrap()).first().unwrap()
                );
            )+
        };
        ($a:ident, $b:ident, $c:ident, rev: $($fn:ident),+) => {
            $(
                assert_eq!(
                    (&$a).$fn(&$c).first().unwrap(),
                    (*$a.first().unwrap()).$fn(&$c).first().unwrap(),
                    "{:?}", (&$a).$fn(&$c)
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
        _scalar(Vector::from(vec![-6.378, 0.0]), Vector::from(2.0));
        _scalar(Vector::from(-2.0), Vector::from(vec![2.0, -3., 0.0]));
    }

    fn _scalar(a: Vector<f32>, c: Vector<f32>) {
        let b = FAD::from(a.clone());
        check_ops!(a, b, c: mul, add, sub, div, pow);
        check_ops!(a, b, c, rev: _pow, _div);
        check_ops!(a, b: neg, abs, square, exp, sin, cos, tan);
        let a = a.abs();
        let b = b.abs();
        let c = c.abs();
        check_ops!(a, b, c: log);
        check_ops!(a, b, c, rev: _log);
        check_ops!(a, b: ln, sqrt);
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
            assert_impl!(Vector<$T>: NumOps<$T, Vector<$T>>);
            assert_impl!(&Vector<$T>: NumOps<$T, Vector<$T>>);
            assert_impl!(Vector<$T>: NumOps<&'a $T, Vector<$T>>);
            assert_impl!(&Vector<$T>: NumOps<&'a $T, Vector<$T>>);
            assert_impl!(Vector<$T>: NumOps<Vector<$T>, Vector<$T>>);
            assert_impl!(&Vector<$T>: NumOps<Vector<$T>, Vector<$T>>);
            assert_impl!(Vector<$T>: NumOps<&'a Vector<$T>, Vector<$T>>);
            assert_impl!(&Vector<$T>: NumOps<&'a Vector<$T>, Vector<$T>>);

            assert_impl!(FAD<Vector<$T>>: NumOps<Vector<$T>, FAD<Vector<$T>>>);
            assert_impl!(&FAD<Vector<$T>>: NumOps<Vector<$T>, FAD<Vector<$T>>>);
            assert_impl!(FAD<Vector<$T>>: NumOps<&'a Vector<$T>, FAD<Vector<$T>>>);
            assert_impl!(&FAD<Vector<$T>>: NumOps<&'a Vector<$T>, FAD<Vector<$T>>>);

            assert_impl!(Vector<$T>: NumConsts);
            assert_impl!(FAD<Vector<$T>>: NumConsts);

            assert_impl!(Vector<$T>: NumOpts<Vector<$T>>);
            assert_impl!(&Vector<$T>: NumOpts<Vector<$T>>);
            assert_impl!(FAD<Vector<$T>>: NumOpts<FAD<Vector<$T>>>);
            assert_impl!(&FAD<Vector<$T>>: NumOpts<FAD<Vector<$T>>>);

            assert_impl!(Vector<$T>: AggOps<Vector<$T>>);
            assert_impl!(&Vector<$T>: AggOps<Vector<$T>>);
            assert_impl!(FAD<Vector<$T>>: AggOps<FAD<Vector<$T>>>);
            assert_impl!(&FAD<Vector<$T>>: AggOps<FAD<Vector<$T>>>);
        };
    }

    assert_impl!(f32);
    assert_impl!(f64);
}
