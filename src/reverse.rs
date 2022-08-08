/// Reverse Automatic Differentiation
use crate::ops::{Abs, Exp, Half, Ln, Log, One, Pow, Prod, Signum, Sqrt, Square, Sum, Trig, Two};
use std::{
    cell::{Ref, RefCell, RefMut},
    fmt::Debug,
    ops::{Add, Deref, DerefMut, Div, Mul, Neg, Sub},
    rc::Rc,
};

/// This is a node/subgraph in an directed acyclic computation graph.
/// the internals of the node is wrapped in a reference counted cell.
#[derive(Clone)]
pub struct RAD<T>(Rc<RefCell<RadNode<T>>>);

impl<T> Deref for RAD<T> {
    type Target = Rc<RefCell<RadNode<T>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for RAD<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Debug for RAD<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let node = self.node();
        f.debug_struct("RAD")
            .field("value", &node.value)
            .field("grad", &node.grad)
            .field("prune", &node.prune)
            .field("op", &node.op)
            .finish()
    }
}

/// This is the internals for a node in the directed acylcic computation graph.
#[derive(Debug, Clone)]
pub struct RadNode<T> {
    /// The value of the node is calculated at creation.
    pub value: T,
    /// Each node represents either a variable/constant or an operation.
    /// The variables and constants are leaf nodes while the operations creates links in the graph.
    op: Operation<RAD<T>, T>,
    /// Each node also has a placeholder for the gradient.
    pub grad: GradOption<T>,
    /// No further gradients to calculate on this branch (used for internal optimisation).
    prune: bool,
}

/// An enum defining an operation in the computation graph
#[derive(Clone, Copy)]
pub enum Operation<Ref, Val> {
    /// No operation, just store the value (used for variables and constants)
    Value,
    /// Unary operation (such as `neg` and `abs`)
    Unary(
        /// Previous node
        Ref,
        /// Forward operation
        fn(Val) -> Val,
        /// Gradient update
        fn(&Val, Val, &Val) -> Val,
        /// Debug name
        &'static str,
    ),
    Binary(
        /// Previous node A
        Ref,
        /// Previous node B
        Ref,
        /// Forward operation
        fn(Val, Val) -> Val,
        /// Gradient update A
        fn(&Val, &Val, Val, &Val) -> Val,
        /// Gradient update B
        fn(&Val, &Val, Val, &Val) -> Val,
        /// Debug name
        &'static str,
    ),
}

impl<R, V> Debug for Operation<R, V>
where
    R: Debug,
    V: Debug,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value => write!(fmt, "Value"),
            Self::Unary(a, _, _, s) => fmt.debug_tuple(s).field(a).finish(),
            Self::Binary(a, b, _, _, _, s) => fmt.debug_tuple(s).field(a).field(b).finish(),
        }
    }
}

/// An enum representing the state of the gradient.
#[derive(Debug, Clone, Copy)]
pub enum GradOption<T> {
    /// Do not store the gradient.
    None,
    /// The gradient has not yet been calculated.
    Zero,
    /// The gradient has been calculated (here it is).
    Some(T),
}

impl<T> Default for GradOption<T> {
    fn default() -> Self {
        GradOption::None
    }
}

impl<T> From<Option<T>> for GradOption<T> {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(v) => GradOption::Some(v),
            None => GradOption::None,
        }
    }
}

impl<T> GradOption<T> {
    pub fn unwrap(self) -> T {
        match self {
            GradOption::None => panic!("called `GradOption::unwrap()` on a `None` value"),
            GradOption::Zero => panic!("called `GradOption::unwrap()` on a `Zero` value"),
            GradOption::Some(v) => v,
        }
    }

    pub fn requires_gradient(&self) -> bool {
        !matches!(self, GradOption::None)
    }
}

impl<T> RAD<T> {
    /// Create a new variable (which requires gradient)
    pub fn variable(value: T) -> Self {
        RAD(Rc::new(RefCell::new(RadNode {
            value,
            op: Operation::Value,
            grad: GradOption::Zero,
            prune: false,
        })))
    }

    /// Create a new constant (which does not require gradient)
    pub fn constant(value: T) -> Self {
        RAD(Rc::new(RefCell::new(RadNode {
            value,
            op: Operation::Value,
            grad: GradOption::None,
            prune: true,
        })))
    }

    #[inline]
    fn new(value: T, op: Operation<RAD<T>, T>) -> Self {
        RAD(Rc::new(RefCell::new(RadNode {
            value,
            op,
            grad: GradOption::None,
            prune: false,
        })))
    }

    /// Has this branch been pruned (no further gradients to calculate).
    #[inline]
    pub fn pruned(&self) -> bool {
        self.node().prune
    }

    /// Get the internal node as a reference
    #[inline]
    pub fn node(&self) -> Ref<'_, RadNode<T>> {
        self.0.deref().borrow()
    }

    /// Get the internal node as a mutable reference
    #[inline]
    pub fn node_mut(&mut self) -> RefMut<'_, RadNode<T>> {
        self.0.deref().borrow_mut()
    }
}

impl<T> RAD<T>
where
    T: Clone,
{
    /// Get a clone of the value
    #[inline]
    pub fn clone_value(&mut self) -> T {
        self.node_mut().value.clone()
    }

    /// Create a new node in the computation graph from an unary operation
    #[inline]
    pub fn unary(
        // Previous node
        mut prev: RAD<T>,
        // Operation
        forward: fn(T) -> T,
        // Gradient
        reverse: fn(&T, T, &T) -> T,
        // Debug name
        name: &'static str,
    ) -> Self {
        let value = forward(prev.clone_value());
        let op = Operation::Unary(prev, forward, reverse, name);
        Self::new(value, op)
    }

    /// Create a new node in the computation graph from an binary operation
    #[inline]
    pub fn binary(
        // Previous node A
        mut prev_a: RAD<T>,
        // Previous node B
        mut prev_b: RAD<T>,
        // Operation
        forward: fn(T, T) -> T,
        // Gradient A
        reverse_a: fn(&T, &T, T, &T) -> T,
        // Gradient B
        reverse_b: fn(&T, &T, T, &T) -> T,
        // Debug name
        name: &'static str,
    ) -> Self {
        let value = forward(prev_a.clone_value(), prev_b.clone_value());
        let op = Operation::Binary(prev_a, prev_b, forward, reverse_a, reverse_b, name);
        Self::new(value, op)
    }
}

impl<T> RAD<T>
where
    T: Add<T, Output = T> + One + Clone,
{
    /// Calculate the gradients backward from this node.
    /// If `clear_grad = true` then any old gradient results are reset and the unneccessary gradient calculations are pruned.
    pub fn backward(&mut self, clear_grad: bool) {
        if clear_grad {
            self.clear_grad();
        }
        self.backpropagate(T::one())
    }

    #[inline]
    fn backpropagate(&mut self, grad: T) {
        self.node_mut().backpropagate(grad);
    }
}

impl<T> RAD<T> {
    #[inline]
    fn clear_grad(&mut self) -> bool {
        self.node_mut().clear_grad(true)
    }
}

impl<T> RadNode<T> {
    /// Removes the gradient from the node.
    /// If `recursive=true` then it also removes all gradients from the subgraph and prunes unneccessary gradient calculations.
    #[inline]
    pub fn clear_grad(&mut self, recursive: bool) -> bool {
        if recursive {
            let prune = match &mut self.op {
                Operation::Value => true,
                Operation::Unary(a, ..) => a.clear_grad(),
                Operation::Binary(a, b, ..) => a.clear_grad() & b.clear_grad(),
            };
            if self.grad.requires_gradient() {
                self.grad = GradOption::Zero;
                self.prune = false;
                false
            } else {
                self.grad = GradOption::None;
                self.prune = prune;
                prune
            }
        } else {
            if !self.grad.requires_gradient() {
                self.grad = GradOption::Zero;
            }
            self.prune
        }
    }
}

impl<T> RadNode<T>
where
    T: Add<T, Output = T> + One + Clone,
{
    /// Backpropagate gradients
    #[inline]
    pub fn backpropagate(&mut self, grad: T) {
        if self.prune {
            return;
        }
        self.grad = match std::mem::take(&mut self.grad) {
            GradOption::None => GradOption::None,
            GradOption::Zero => GradOption::Some(grad.clone()),
            GradOption::Some(g) => GradOption::Some(g + grad.clone()),
        };
        match &mut self.op {
            Operation::Value => {}
            Operation::Unary(a, _, rev, _) => {
                if !a.pruned() {
                    let grad = rev(&a.node().value, grad, &self.value);
                    a.backpropagate(grad);
                }
            }
            Operation::Binary(a, b, _, rev_a, rev_b, _) => {
                if !a.pruned() {
                    let grad_a = rev_a(&a.node().value, &b.node().value, grad.clone(), &self.value);
                    a.backpropagate(grad_a);
                }
                if !b.pruned() {
                    let grad_b = rev_b(&a.node().value, &b.node().value, grad, &self.value);
                    b.backpropagate(grad_b);
                }
            }
        }
    }
}

macro_rules! impl_binary_op {
    ($Trait:tt, $fn:ident: $grad_a:expr; $grad_b:expr; $($T:path),*) => {
        impl<'a, T> $Trait<&'a RAD<T>> for &'a RAD<T>
        where
            T: $Trait<T, Output = T> + Clone $(+ $T)*,
        {
            type Output = RAD<T>;

            fn $fn(self, rhs: &'a RAD<T>) -> Self::Output {
                RAD::binary(self.clone(), rhs.clone(), T::$fn, $grad_a, $grad_b, stringify!($fn))
            }
        }

        impl<'a, T> $Trait<T> for &'a RAD<T>
        where
            T: $Trait<T, Output = T> + Clone $(+ $T)*,
        {
            type Output = RAD<T>;

            fn $fn(self, rhs: T) -> Self::Output {
                let rhs = RAD::constant(rhs);
                RAD::binary(self.clone(), rhs, T::$fn, $grad_a, $grad_b, stringify!($fn))
            }
        }
    };
}

impl_binary_op!(Add, add: |_, _, g, _| g; |_, _, g, _| g;);
impl_binary_op!(Sub, sub: |_, _, g, _| g; |_, _, g, _| g.neg(); Neg<Output = T>);
impl_binary_op!(Mul, mul: |_, b, g, _| g * b.clone(); |a, _, g, _| g * a.clone(););
impl_binary_op!(Div, div: |_, b, g, _| g / b.clone(); |a, b, g, _| g * a.clone().neg() / b.clone().square(); Neg<Output = T>, Mul<Output = T>, Square<Output = T>);
impl_binary_op!(Pow, pow: |a, b, g, _| g * b.clone() * a.clone().pow(b.clone() - T::one()); |a, _, g, v| g * a.clone().ln() * v.clone(); Mul<Output = T>, Sub<Output = T>, One, Ln<Output = T>);
impl_binary_op!(Log, log: |a, b, g, _| g / (b.clone().ln() * a.clone()); |a, b, g, _| g * a.clone().ln().neg() / (b.clone() * b.clone().ln().square()); Mul<Output = T>, Div<Output = T>, Neg<Output = T>, Square<Output = T>, Ln<Output = T>);

macro_rules! impl_unary_op {
    ($Trait:tt, $($fn:ident: $grad:expr),+; $($T:path),*) => {
        impl<'a, T> $Trait for &'a RAD<T>
        where
            T: $Trait<Output = T> + Clone $(+ $T)*,
        {
            type Output = RAD<T>;

            $(
                fn $fn(self) -> Self::Output {
                    RAD::unary(self.clone(), T::$fn, $grad, stringify!($fn))
                }
            )+
        }
    };
}

impl_unary_op!(Neg, neg: |_, g, _| g.neg(););
impl_unary_op!(Exp, exp: |a, g, _| g * a.clone(); Mul<Output = T>);
impl_unary_op!(Ln, ln: |a, g, _| g / a.clone(); Mul<Output = T>, Div<Output = T>);
impl_unary_op!(Abs, abs: |a, g, _| g * a.clone().signum(); Mul<Output = T>, Signum<Output = T>);
impl_unary_op!(Trig, sin: |a, g, _| g * a.clone().cos(), cos: |a, g, _| g * a.clone().sin().neg(), tan: |a, g, _| g / a.clone().cos().square(); Mul<Output = T>, Neg<Output = T>, Div<Output = T>, Square<Output = T>);
impl_unary_op!(Square, square: |a, g, _| g * (T::two() * a.clone()); Mul<Output = T>, Two);
impl_unary_op!(Sqrt, sqrt: |a, g, _| g * (T::half() / a.clone()); Mul<Output = T>, Div<Output = T>, Half);

macro_rules! impl_agg_op {
    ($Trait:tt, $fn:ident, $grad:expr; $($T:tt),*) => {
        impl<'a, T> $Trait for &'a RAD<T>
        where
            T: $Trait<Output = T> + Clone $(+ $T<Output = T>)*,
        {
            type Output = RAD<T>;

            fn $fn(self) -> Self::Output {
                RAD::unary(self.clone(), T::$fn, $grad, stringify!($fn))
            }
        }
    };
}

impl_agg_op!(Sum, sum, |_, g, _| g;);
impl_agg_op!(Prod, prod, |a, g, v| g * (v.clone() / a.clone()); Mul, Div);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_binary_scalar {
        ($a:expr, $b:expr, $fn:ident, $g1:expr, $g2:expr, $g3:expr) => {
            let r1 = ($a).$fn($b);
            let r2 = ($a).$fn($a);

            let a = RAD::variable($a);
            let b = $b;
            let mut c = (&a).$fn(b);
            assert_eq!(c.node().value, r1);
            c.backward(true);
            assert!(!c.pruned(), "Pruning wrong: 1");
            assert!(!a.pruned(), "Pruning wrong: 2");
            assert_eq!(a.node().grad.unwrap(), $g1);

            let a = RAD::variable($a);
            let b = RAD::constant($b);
            let mut c = (&a).$fn(&b);
            assert_eq!(c.node().value, r1);
            c.backward(true);
            assert!(!c.pruned(), "Pruning wrong: 3");
            assert!(!a.pruned(), "Pruning wrong: 4");
            assert!(b.pruned(), "Pruning wrong: 5");
            assert_eq!(a.node().grad.unwrap(), $g1);

            let a = RAD::variable($a);
            let b = RAD::variable($b);
            let mut c = (&a).$fn(&b);
            assert_eq!(c.node().value, r1);
            c.backward(true);
            assert!(!c.pruned(), "Pruning wrong: 6");
            assert!(!a.pruned(), "Pruning wrong: 7");
            assert!(!b.pruned(), "Pruning wrong: 8");
            assert_eq!(a.node().grad.unwrap(), $g1);
            assert_eq!(b.node().grad.unwrap(), $g2);

            let a = RAD::variable($a);
            let mut c = (&a).$fn(&a);
            assert_eq!(c.node().value, r2);
            c.backward(true);
            assert!(!c.pruned(), "Pruning wrong: 9");
            assert!(!a.pruned(), "Pruning wrong: 10");
            assert_eq!(a.node().grad.unwrap(), $g3);
        };
    }

    #[test]
    fn add() {
        test_binary_scalar!(2.0f32, 3.0f32, add, 1.0, 1.0, 2.0);
    }

    #[test]
    fn sub() {
        test_binary_scalar!(2.0f32, 3.0f32, sub, 1.0, -1.0, 0.0);
    }

    #[test]
    fn mul() {
        test_binary_scalar!(2.0f32, 3.0f32, mul, 3.0, 2.0, 4.0);
    }

    #[test]
    fn div() {
        test_binary_scalar!(2.0f32, 3.0, div, 1. / 3., -2. / 9., 0.0);
    }

    #[test]
    fn log() {
        test_binary_scalar!(
            2.0f32,
            3.0,
            log,
            0.5 / 3.0f32.ln(),
            -(2.0f32.ln()) / (3.0 * 3.0f32.ln().square()),
            0.0
        );
    }

    #[test]
    fn pow() {
        test_binary_scalar!(
            2.0f32,
            3.0,
            pow,
            12.,
            2.0f32.ln() * 2.0f32.pow(3.0),
            4.0 + 2.0f32.ln() * 4.0
        );
    }

    macro_rules! test_unary_scalar {
        ($fn:ident, $($a:expr, $g:expr),+) => {
            $(
            let r = ($a).$fn();
            let a = RAD::variable($a);
            let mut b = a.$fn();
            assert_eq!(r, b.clone_value());
            b.backward(false);
            let g = a.node().grad.unwrap();
            assert!(($g - g).abs() < 1e-6, "{} != {}", $g, g);
            )+
        };
    }

    #[test]
    fn neg() {
        for v in [-2.0, 0.0, 3.0, 7.0f32] {
            test_unary_scalar!(neg, v, -1.);
        }
    }

    #[test]
    fn exp() {
        for v in [-2.0, 0.0, 3.0, 7.0f32] {
            test_unary_scalar!(exp, v, v);
        }
    }

    #[test]
    fn ln() {
        for v in [0.01, 3.0, 7.0f32] {
            test_unary_scalar!(ln, v, 1. / v);
        }
    }

    #[test]
    fn abs() {
        for v in [-2.0, 0.0, 3.0, 7.0f32] {
            test_unary_scalar!(abs, v, v.signum());
        }
    }

    #[test]
    fn square() {
        for v in [-2.0, 0.0, 3.0, 7.0f32] {
            test_unary_scalar!(square, v, 2. * v);
        }
    }

    #[test]
    fn sqrt() {
        for v in [0.01, 3.0, 7.0f32] {
            test_unary_scalar!(sqrt, v, 0.5 / v);
        }
    }

    #[test]
    fn trig() {
        for v in [-2.0, 0.0, 3.0, 7.0f32] {
            test_unary_scalar!(sin, v, v.cos());
            test_unary_scalar!(cos, v, -v.sin());
            test_unary_scalar!(tan, v, 1. / v.cos().square());
        }
    }
}
