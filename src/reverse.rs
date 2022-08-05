/// Reverse Automatic Differentiation
use crate::ops::One;
use std::{
    cell::{Ref, RefCell, RefMut},
    ops::{Add, Deref, DerefMut, Mul},
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
/// The value of the node is calculated at creation.
/// Each node represents either a variable/constant or an operation.
/// The variables and constants are leaf nodes while the operations creates links in the graph.
/// Each node also has a placeholder for the gradient, which can be in one of three states:
/// - None: "do not store the gradient" (default for non-variables).
/// - Zero: "gradient not yet calculated (or reset)" (default for variables).
/// - Some: "the value of the gradient".
#[derive(Debug, Clone)]
pub struct RadNode<T> {
    pub value: T,
    op: Operation<RAD<T>>,
    pub grad: ZeroOption<T>,
}

#[derive(Debug, Clone, Copy)]
enum Operation<T> {
    Value,
    Add(T, T),
    Mul(T, T),
}

#[derive(Debug, Clone, Copy)]
pub enum ZeroOption<T> {
    None,
    Zero,
    Some(T),
}

impl<T> Default for ZeroOption<T> {
    fn default() -> Self {
        ZeroOption::None
    }
}

impl<T> From<Option<T>> for ZeroOption<T> {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(v) => ZeroOption::Some(v),
            None => ZeroOption::None,
        }
    }
}

impl<T> ZeroOption<T> {
    pub fn is_none(&self) -> bool {
        matches!(self, ZeroOption::None)
    }

    pub fn unwrap(self) -> T {
        match self {
            ZeroOption::None => panic!("called `ZeroOption::unwrap()` on a `None` value"),
            ZeroOption::Zero => panic!("called `ZeroOption::unwrap()` on a `Zero` value"),
            ZeroOption::Some(v) => v,
        }
    }
}

impl<T> RAD<T> {
    pub fn variable(value: T) -> Self {
        RAD(Rc::new(RefCell::new(RadNode {
            value,
            op: Operation::Value,
            grad: ZeroOption::Zero,
        })))
    }

    pub fn constant(value: T) -> Self {
        RAD(Rc::new(RefCell::new(RadNode {
            value,
            op: Operation::Value,
            grad: ZeroOption::None,
        })))
    }

    #[inline]
    fn new(value: T, op: Operation<RAD<T>>) -> Self {
        RAD(Rc::new(RefCell::new(RadNode {
            value,
            op,
            grad: ZeroOption::None,
        })))
    }

    #[inline]
    pub fn node(&self) -> Ref<'_, RadNode<T>> {
        self.0.deref().borrow()
    }

    #[inline]
    pub fn node_mut(&mut self) -> RefMut<'_, RadNode<T>> {
        self.0.deref().borrow_mut()
    }
}

impl<T> RAD<T>
where
    T: One + Add<Output = T> + Mul<Output = T> + Clone,
{
    pub fn backward(&mut self, clear_grad: bool) {
        if clear_grad {
            self.clear_grad();
        }
        self.backpropagate(T::one())
    }
}

impl<T> RAD<T>
where
    T: Add<Output = T> + Mul<Output = T> + Clone,
{
    #[inline]
    fn backpropagate(&mut self, grad: T) {
        self.node_mut().backpropagate(grad);
    }
    #[inline]
    fn clear_grad(&mut self) {
        self.node_mut().clear_grad(true);
    }
}

impl<T> RadNode<T>
where
    T: Add<Output = T> + Mul<Output = T> + Clone,
{
    #[inline]
    pub fn clear_grad(&mut self, recursive: bool) {
        if !self.grad.is_none() {
            self.grad = ZeroOption::Zero;
        }
        if recursive {
            match &mut self.op {
                Operation::Value => {}
                Operation::Add(a, b) => {
                    a.clear_grad();
                    b.clear_grad();
                }
                Operation::Mul(a, b) => {
                    a.clear_grad();
                    b.clear_grad();
                }
            }
        }
    }

    #[inline]
    pub fn backpropagate(&mut self, grad: T) {
        self.grad = match std::mem::take(&mut self.grad) {
            ZeroOption::None => ZeroOption::None,
            ZeroOption::Zero => ZeroOption::Some(grad.clone()),
            ZeroOption::Some(g) => ZeroOption::Some(g + grad.clone()),
        };
        match &mut self.op {
            Operation::Value => {}
            Operation::Add(a, b) => {
                a.backpropagate(grad.clone());
                b.backpropagate(grad);
            }
            Operation::Mul(a, b) => {
                let b_node = b.node();
                let grad_a = grad.clone() * b_node.value.clone();
                drop(b_node);
                a.backpropagate(grad_a);
                let a_node = a.node();
                let grad_b = grad * a_node.value.clone();
                drop(a_node);
                b.backpropagate(grad_b);
            }
        };
    }
}

impl<'a, T> Mul<&'a RAD<T>> for &'a RAD<T>
where
    T: Mul<T, Output = T> + Clone,
{
    type Output = RAD<T>;

    fn mul(self, rhs: &'a RAD<T>) -> Self::Output {
        RAD::new(
            self.node().value.clone() * rhs.node().value.clone(),
            Operation::Mul(self.clone(), rhs.clone()),
        )
    }
}

impl<'a, T> Mul<T> for &'a RAD<T>
where
    T: Mul<T, Output = T> + Clone,
{
    type Output = RAD<T>;

    fn mul(self, rhs: T) -> Self::Output {
        RAD::new(
            self.node().value.clone() * rhs.clone(),
            Operation::Mul(self.clone(), RAD::constant(rhs)),
        )
    }
}

impl<'a, T> Add<&'a RAD<T>> for &'a RAD<T>
where
    T: Add<T, Output = T> + Clone,
{
    type Output = RAD<T>;

    fn add(self, rhs: &'a RAD<T>) -> Self::Output {
        RAD::new(
            self.node().value.clone() + rhs.node().value.clone(),
            Operation::Add(self.clone(), rhs.clone()),
        )
    }
}

impl<'a, T> Add<T> for &'a RAD<T>
where
    T: Add<T, Output = T> + Clone,
{
    type Output = RAD<T>;

    fn add(self, rhs: T) -> Self::Output {
        RAD::new(
            self.node().value.clone() + rhs.clone(),
            Operation::Add(self.clone(), RAD::constant(rhs)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul() {
        let a = RAD::variable(2.0f32);
        let mut b = &a * 2.0f32;
        assert_eq!(b.node().value, 4.0);
        b.backward(true);
        assert_eq!(a.node().grad.unwrap(), 2.0);
        let mut b = &a * &a;
        assert_eq!(b.node().value, 4.0);
        b.backward(true);
        assert_eq!(a.node().grad.unwrap(), 4.0);
    }

    #[test]
    fn add() {
        let a = RAD::variable(2.0f32);
        let mut b = &a + 2.0f32;
        assert_eq!(b.node().value, 4.0);
        b.backward(true);
        assert_eq!(a.node().grad.unwrap(), 1.0);
        let mut b = &a + &a;
        assert_eq!(b.node().value, 4.0);
        b.backward(true);
        assert_eq!(a.node().grad.unwrap(), 2.0);
    }
}
