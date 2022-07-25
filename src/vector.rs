use std::ops::*;

use crate::ops::{Log, NumOps, One, Pow, Wop};

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

impl<T> One for Vector<T>
where
    T: One,
{
    fn one() -> Self {
        Vector::Scalar(T::one())
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

macro_rules! _binary_element_op_scalar {
    ($Trait:tt, $f:ident, $LHS:ty, $RHS:ty, *) => {
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
    ($Trait:tt, $f:ident, $LHS:ty, $RHS:ty, ) => {
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
}
macro_rules! _binary_element_op_vec {
    ($Trait:tt, $f:ident, $LHS:ty, $RHS:ty) => {
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

macro_rules! binary_element_op {
    ($Trait:tt, $f:ident) => {
        _binary_element_op_scalar!($Trait, $f, &'a Vector<T>, &'a T, *);
        _binary_element_op_scalar!($Trait, $f, Vector<T>, &'a T, *);
        _binary_element_op_scalar!($Trait, $f, &'a Vector<T>, T, );
        _binary_element_op_scalar!($Trait, $f, Vector<T>, T, );

        _binary_element_op_vec!($Trait, $f, &'a Vector<T>, &'a Vector<T>);
        _binary_element_op_vec!($Trait, $f, Vector<T>, &'a Vector<T>);
        _binary_element_op_vec!($Trait, $f, &'a Vector<T>, Vector<T>);
        _binary_element_op_vec!($Trait, $f, Vector<T>, Vector<T>);
    };
}

binary_element_op!(Add, add);
binary_element_op!(Sub, sub);
binary_element_op!(Mul, mul);
binary_element_op!(Div, div);
binary_element_op!(Pow, pow);
binary_element_op!(Log, log);
binary_element_op!(Wop, wop);

// impl<T> Neg for Vector<T>
// where
//     T: Neg<Output = T> + Copy,
// {
//     type Output = Vector<T>;

//     fn neg(self) -> Self::Output {
//         match self {
//             Vector::Vec(v) => Vector::Vec(v.iter().map(Neg::neg).collect()),
//             Vector::Scalar(s) => Vector::Scalar(s.neg()),
//         }
//     }
// }

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
