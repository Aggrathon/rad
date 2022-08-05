/// Reverse Automatic Differentiation
use crate::ops::{Abs, Exp, Half, Ln, Log, One, Pow, Signum, Sqrt, Square, Trig, Two};
use std::{
    cell::{Ref, RefCell, RefMut},
    fmt::Debug,
    ops::{Add, Deref, DerefMut, Div, Mul, Neg, Sub},
    rc::Rc,
};

/// This is a node in the directed acyclic computation graph.
/// the internals of the node is wrapped in a reference counted cell.
#[derive(Debug, Clone)]
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

/// This is the internals for a node in the directed acylcic computation graph.
#[derive(Debug, Clone)]
pub struct RadNode<T> {
    /// The value of the node is calculated at creation.
    pub value: T,
    /// Each node represents either a variable/constant or an operation.
    /// The variables and constants are leaf nodes while the operations creates links in the graph.
    op: Operation<RAD<T>>,
    /// Each node also has a placeholder for the gradient.
    pub grad: GradOption<T>,
    /// No further gradients to calculate on this branch (for internal optimisation).
    prune: bool,
}

#[derive(Debug, Clone, Copy)]
enum Operation<T> {
    Value,
    Add(T, T),
    Sub(T, T),
    Mul(T, T),
    Div(T, T),
    Pow(T, T),
    Log(T, T),
    Neg(T),
    Exp(T),
    Ln(T),
    Abs(T),
    Sin(T),
    Cos(T),
    Tan(T),
    Square(T),
    Sqrt(T),
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
    /// Create a new variable (which requires gradients)
    pub fn variable(value: T) -> Self {
        RAD(Rc::new(RefCell::new(RadNode {
            value,
            op: Operation::Value,
            grad: GradOption::Zero,
            prune: false,
        })))
    }

    /// Create a new constant (which does not require gradients)
    pub fn constant(value: T) -> Self {
        RAD(Rc::new(RefCell::new(RadNode {
            value,
            op: Operation::Value,
            grad: GradOption::None,
            prune: true,
        })))
    }

    #[inline]
    fn new(value: T, op: Operation<RAD<T>>) -> Self {
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
}

impl<T> RAD<T>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Signum<Output = T>
        + Trig<Output = T>
        + Square<Output = T>
        + Ln<Output = T>
        + Pow<Output = T>
        + One
        + Two
        + Half
        + Debug
        + Clone,
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
                Operation::Add(a, b)
                | Operation::Mul(a, b)
                | Operation::Sub(a, b)
                | Operation::Div(a, b)
                | Operation::Pow(a, b)
                | Operation::Log(a, b) => a.clear_grad() & b.clear_grad(),
                Operation::Exp(a)
                | Operation::Ln(a)
                | Operation::Abs(a)
                | Operation::Sin(a)
                | Operation::Cos(a)
                | Operation::Tan(a)
                | Operation::Square(a)
                | Operation::Sqrt(a)
                | Operation::Neg(a) => a.clear_grad(),
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
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Signum<Output = T>
        + Trig<Output = T>
        + Square<Output = T>
        + Ln<Output = T>
        + Pow<Output = T>
        + One
        + Two
        + Half
        + Debug
        + Clone,
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
            Operation::Add(a, b) => {
                a.backpropagate(grad.clone());
                b.backpropagate(grad);
            }
            Operation::Sub(a, b) => {
                a.backpropagate(grad.clone());
                if !b.pruned() {
                    b.backpropagate(grad.neg());
                }
            }
            Operation::Mul(a, b) => {
                if !a.pruned() {
                    let grad_a = grad.clone() * b.clone_value();
                    a.backpropagate(grad_a);
                }
                if !b.pruned() {
                    let grad_b = grad * a.clone_value();
                    b.backpropagate(grad_b);
                }
            }
            Operation::Div(a, b) => {
                if !a.pruned() {
                    let grad_a = grad.clone() / b.clone_value();
                    a.backpropagate(grad_a);
                }
                if !b.pruned() {
                    let grad_b = grad * a.clone_value().neg() / (b.clone_value().square());
                    b.backpropagate(grad_b);
                }
            }
            Operation::Pow(a, b) => {
                if !a.pruned() {
                    let b_value = b.clone_value();
                    let grad_a =
                        grad.clone() * b_value.clone() * a.clone_value().pow(b_value - T::one());
                    a.backpropagate(grad_a);
                }
                if !b.pruned() {
                    let grad_b = grad * a.clone_value().ln() * self.value.clone();
                    b.backpropagate(grad_b);
                }
            }
            Operation::Log(a, b) => {
                if !a.pruned() {
                    let grad_a = grad.clone() / (b.clone_value().ln() * a.clone_value());
                    a.backpropagate(grad_a);
                }
                if !b.pruned() {
                    let b_value = b.clone_value();
                    let grad_b = grad * a.clone_value().ln().neg()
                        / (b_value.clone() * b_value.ln().square());
                    b.backpropagate(grad_b);
                }
            }
            Operation::Neg(a) => a.backpropagate(grad.neg()),
            Operation::Exp(a) => {
                let mut a = a.borrow_mut();
                if !a.prune {
                    let grad = grad * a.value.clone();
                    a.backpropagate(grad);
                }
            }
            Operation::Ln(a) => {
                let mut a = a.borrow_mut();
                if !a.prune {
                    let grad = grad / a.value.clone();
                    a.backpropagate(grad);
                }
            }
            Operation::Abs(a) => {
                let mut a = a.borrow_mut();
                if !a.prune {
                    let grad = grad * a.value.clone().signum();
                    a.backpropagate(grad);
                }
            }
            Operation::Sin(a) => {
                let mut a = a.borrow_mut();
                if !a.prune {
                    let grad = grad * a.value.clone().cos();
                    a.backpropagate(grad);
                }
            }
            Operation::Cos(a) => {
                let mut a = a.borrow_mut();
                if !a.prune {
                    let grad = grad * a.value.clone().sin().neg();
                    a.backpropagate(grad);
                }
            }
            Operation::Tan(a) => {
                let mut a = a.borrow_mut();
                if !a.prune {
                    let grad = grad / a.value.clone().cos().square();
                    a.backpropagate(grad);
                }
            }
            Operation::Square(a) => {
                let mut a = a.borrow_mut();
                if !a.prune {
                    let grad = grad * (T::two() * a.value.clone());
                    a.backpropagate(grad);
                }
            }
            Operation::Sqrt(a) => {
                let mut a = a.borrow_mut();
                if !a.prune {
                    let grad = grad * (T::half() / a.value.clone());
                    a.backpropagate(grad);
                }
            }
        };
    }
}

macro_rules! impl_binary_op {
    ($Trait:tt, $fn:ident) => {
        impl<'a, T> $Trait<&'a RAD<T>> for &'a RAD<T>
        where
            T: $Trait<T, Output = T> + Clone,
        {
            type Output = RAD<T>;

            fn $fn(self, rhs: &'a RAD<T>) -> Self::Output {
                RAD::new(
                    self.node().value.clone().$fn(rhs.node().value.clone()),
                    Operation::$Trait(self.clone(), rhs.clone()),
                )
            }
        }

        impl<'a, T> $Trait<T> for &'a RAD<T>
        where
            T: $Trait<T, Output = T> + Clone,
        {
            type Output = RAD<T>;

            fn $fn(self, rhs: T) -> Self::Output {
                RAD::new(
                    self.node().value.clone().$fn(rhs.clone()),
                    Operation::$Trait(self.clone(), RAD::constant(rhs)),
                )
            }
        }
    };
}

impl_binary_op!(Mul, mul);
impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Div, div);
impl_binary_op!(Pow, pow);
impl_binary_op!(Log, log);

macro_rules! impl_unary_op {
    ($Trait:tt, $fn:ident) => {
        impl<'a, T> $Trait for &'a RAD<T>
        where
            T: $Trait<Output = T> + Clone,
        {
            type Output = RAD<T>;

            fn $fn(self) -> Self::Output {
                RAD::new(
                    self.node().value.clone().$fn(),
                    Operation::$Trait(self.clone()),
                )
            }
        }
    };
    ($Trait:tt, $($fn:ident: $op:tt),+) => {
        impl<'a, T> $Trait for &'a RAD<T>
        where
            T: $Trait<Output = T> + Clone,
        {
            type Output = RAD<T>;

            $(
                fn $fn(self) -> Self::Output {
                    RAD::new(
                        self.node().value.clone().$fn(),
                        Operation::$op(self.clone()),
                    )
                }
            )+
        }
    };
}

impl_unary_op!(Neg, neg);
impl_unary_op!(Exp, exp);
impl_unary_op!(Ln, ln);
impl_unary_op!(Abs, abs);
impl_unary_op!(Trig, sin: Sin, cos: Cos, tan: Tan);
impl_unary_op!(Square, square);
impl_unary_op!(Sqrt, sqrt);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_binary_scalar {
        ($a:expr, $b:expr, $op:pat_param, $fn:ident, $g1:expr, $g2:expr, $g3:expr) => {
            let r1 = ($a).$fn($b);
            let r2 = ($a).$fn($a);

            let a = RAD::variable($a);
            let b = $b;
            let mut c = (&a).$fn(b);
            assert_eq!(c.node().value, r1);
            assert!(matches!(c.node().op, $op), "Wrong op: {:?}", c.node().op);
            c.backward(true);
            assert!(!c.pruned(), "Pruning wrong: 1");
            assert!(!a.pruned(), "Pruning wrong: 2");
            assert_eq!(a.node().grad.unwrap(), $g1);

            let a = RAD::variable($a);
            let b = RAD::constant($b);
            let mut c = (&a).$fn(&b);
            assert_eq!(c.node().value, r1);
            assert!(matches!(c.node().op, $op), "Wrong op: {:?}", c.node().op);
            c.backward(true);
            assert!(!c.pruned(), "Pruning wrong: 3");
            assert!(!a.pruned(), "Pruning wrong: 4");
            assert!(b.pruned(), "Pruning wrong: 5");
            assert_eq!(a.node().grad.unwrap(), $g1);

            let a = RAD::variable($a);
            let b = RAD::variable($b);
            let mut c = (&a).$fn(&b);
            assert_eq!(c.node().value, r1);
            assert!(matches!(c.node().op, $op), "Wrong op: {:?}", c.node().op);
            c.backward(true);
            assert!(!c.pruned(), "Pruning wrong: 6");
            assert!(!a.pruned(), "Pruning wrong: 7");
            assert!(!b.pruned(), "Pruning wrong: 8");
            assert_eq!(a.node().grad.unwrap(), $g1);
            assert_eq!(b.node().grad.unwrap(), $g2);

            let a = RAD::variable($a);
            let mut c = (&a).$fn(&a);
            assert_eq!(c.node().value, r2);
            assert!(matches!(c.node().op, $op), "Wrong op: {:?}", c.node().op);
            c.backward(true);
            assert!(!c.pruned(), "Pruning wrong: 9");
            assert!(!a.pruned(), "Pruning wrong: 10");
            assert_eq!(a.node().grad.unwrap(), $g3);
        };
    }

    #[test]
    fn add() {
        test_binary_scalar!(2.0f32, 3.0f32, Operation::Add(..), add, 1.0, 1.0, 2.0);
    }

    #[test]
    fn sub() {
        test_binary_scalar!(2.0f32, 3.0f32, Operation::Sub(..), sub, 1.0, -1.0, 0.0);
    }

    #[test]
    fn mul() {
        test_binary_scalar!(2.0f32, 3.0f32, Operation::Mul(..), mul, 3.0, 2.0, 4.0);
    }

    #[test]
    fn div() {
        test_binary_scalar!(2.0f32, 3.0, Operation::Div(..), div, 1. / 3., -2. / 9., 0.0);
    }

    #[test]
    fn log() {
        test_binary_scalar!(
            2.0f32,
            3.0,
            Operation::Log(..),
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
            Operation::Pow(..),
            pow,
            12.,
            2.0f32.ln() * 2.0f32.pow(3.0),
            4.0f32 + 2.0f32.ln() * 4.0
        );
    }
}
