use std::collections::{HashMap, VecDeque};

use bitflags::bitflags;
use hashable::HashableHashSet;
use indexmap::IndexSet;
use num_bigint::BigInt;
use num_complex::Complex;
use ordered_float::OrderedFloat;
use python_marshal::PyString;

use crate::error::Error;
#[cfg(feature = "sir")]
use crate::{
    sir::{AuxVar, SIRExpression, SIRStatement},
    traits::GenericSIRNode,
};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum FrozenConstant {
    None,
    StopIteration,
    Ellipsis,
    Bool(bool),
    Long(BigInt),
    Float(OrderedFloat<f64>),
    Complex(Complex<OrderedFloat<f64>>),
    Bytes(Vec<u8>),
    String(PyString),
    Tuple(Vec<FrozenConstant>),
    List(Vec<FrozenConstant>),
    FrozenSet(HashableHashSet<FrozenConstant>),
}

impl std::fmt::Display for FrozenConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrozenConstant::None => write!(f, "None"),
            FrozenConstant::StopIteration => write!(f, "StopIteration"),
            FrozenConstant::Ellipsis => write!(f, "Ellipsis"),
            FrozenConstant::Bool(b) => write!(f, "{b}"),
            FrozenConstant::Long(l) => write!(f, "{l}"),
            FrozenConstant::Float(fl) => write!(f, "{fl}"),
            FrozenConstant::Complex(c) => write!(f, "{}+{}j", c.re, c.im),
            FrozenConstant::Bytes(b) => write!(f, "b{:?}", b),
            FrozenConstant::String(s) => write!(f, "\'{}\'", s.value),
            FrozenConstant::Tuple(t) => {
                write!(f, "(")?;

                let text = t
                    .iter()
                    .map(|c| format!("{c}"))
                    .collect::<Vec<_>>()
                    .join(", ");

                write!(f, "{text}")?;

                write!(f, ")")
            }
            FrozenConstant::List(l) => {
                write!(f, "[")?;

                let text = l
                    .iter()
                    .map(|c| format!("{c}"))
                    .collect::<Vec<_>>()
                    .join(", ");

                write!(f, "{text}")?;

                write!(f, "]")
            }
            FrozenConstant::FrozenSet(fs) => {
                write!(f, "frozenset({{")?;

                let text = fs
                    .iter()
                    .map(|c| format!("{c}"))
                    .collect::<Vec<_>>()
                    .join(", ");

                write!(f, "{text}")?;

                write!(f, "}})")
            }
        }
    }
}

impl TryFrom<python_marshal::Object> for FrozenConstant {
    type Error = Error;

    fn try_from(value: python_marshal::Object) -> Result<Self, Self::Error> {
        match value {
            python_marshal::Object::None => Ok(FrozenConstant::None),
            python_marshal::Object::StopIteration => Ok(FrozenConstant::StopIteration),
            python_marshal::Object::Ellipsis => Ok(FrozenConstant::Ellipsis),
            python_marshal::Object::Bool(b) => Ok(FrozenConstant::Bool(b)),
            python_marshal::Object::Long(l) => Ok(FrozenConstant::Long(l)),
            python_marshal::Object::Float(f) => Ok(FrozenConstant::Float(f)),
            python_marshal::Object::Complex(c) => {
                Ok(FrozenConstant::Complex(Complex { re: c.re, im: c.im }))
            }
            python_marshal::Object::Bytes(b) => Ok(FrozenConstant::Bytes(b)),
            python_marshal::Object::String(s) => Ok(FrozenConstant::String(s)),
            python_marshal::Object::Tuple(t) => {
                let constants = t
                    .into_iter()
                    .map(FrozenConstant::try_from)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(FrozenConstant::Tuple(constants))
            }
            python_marshal::Object::List(l) => {
                let constants = l
                    .into_iter()
                    .map(FrozenConstant::try_from)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(FrozenConstant::List(constants))
            }
            python_marshal::Object::FrozenSet(fs) => {
                let constants = fs
                    .into_iter()
                    .map(python_marshal::Object::from)
                    .map(FrozenConstant::try_from)
                    .collect::<Result<HashableHashSet<_>, _>>()?;
                Ok(FrozenConstant::FrozenSet(constants))
            }
            _ => Err(Error::InvalidConstant(value)),
        }
    }
}

impl From<FrozenConstant> for python_marshal::Object {
    fn from(val: FrozenConstant) -> Self {
        match val {
            FrozenConstant::Bool(value) => python_marshal::ObjectHashable::Bool(value).into(),
            FrozenConstant::None => python_marshal::ObjectHashable::None.into(),
            FrozenConstant::StopIteration => python_marshal::ObjectHashable::StopIteration.into(),
            FrozenConstant::Ellipsis => python_marshal::ObjectHashable::Ellipsis.into(),
            FrozenConstant::Long(value) => python_marshal::ObjectHashable::Long(value).into(),
            FrozenConstant::Float(value) => python_marshal::ObjectHashable::Float(value).into(),
            FrozenConstant::Complex(value) => python_marshal::ObjectHashable::Complex(value).into(),
            FrozenConstant::Bytes(value) => python_marshal::ObjectHashable::Bytes(value).into(),
            FrozenConstant::String(value) => python_marshal::ObjectHashable::String(value).into(),
            FrozenConstant::Tuple(values) => python_marshal::Object::Tuple(
                values
                    .into_iter()
                    .map(Into::<python_marshal::Object>::into)
                    .collect(),
            ),
            FrozenConstant::List(values) => python_marshal::Object::List(
                values
                    .into_iter()
                    .map(Into::<python_marshal::Object>::into)
                    .collect(),
            ),
            FrozenConstant::FrozenSet(values) => {
                python_marshal::Object::FrozenSet(
                    values
                        .into_iter()
                        .cloned()
                        .map(Into::<python_marshal::Object>::into)
                        .map(TryInto::<python_marshal::ObjectHashable>::try_into)
                        .map(Result::unwrap) // The frozen set can only contain values we know for sure are hashable
                        .collect::<IndexSet<_, _>>(),
                )
            }
        }
    }
}

bitflags! {
    /// Describes which optional data for a new function is present on the stack.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MakeFunctionFlags: u32 { // Or u8 if the arg is always a byte
        /// A tuple of default values for positional args.
        const POS_DEFAULTS = 0x01;
        /// A dictionary of keyword-only default values.
        const KW_DEFAULTS  = 0x02;
        /// A tuple of parameter annotations.
        const ANNOTATIONS  = 0x04;
        /// A tuple of cells for free variables (a closure).
        const CLOSURE      = 0x08;
    }
}

impl std::fmt::Display for MakeFunctionFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();
        if self.contains(MakeFunctionFlags::POS_DEFAULTS) {
            parts.push("POS_DEFAULTS");
        }
        if self.contains(MakeFunctionFlags::KW_DEFAULTS) {
            parts.push("KW_DEFAULTS");
        }
        if self.contains(MakeFunctionFlags::ANNOTATIONS) {
            parts.push("ANNOTATIONS");
        }
        if self.contains(MakeFunctionFlags::CLOSURE) {
            parts.push("CLOSURE");
        }

        write!(f, "{}", parts.join(", "))
    }
}

/// The amount of extended_args necessary to represent the arg.
/// This is more efficient than `get_extended_args` as we only calculate the count and the actual values.
pub fn get_extended_args_count(arg: u32) -> u8 {
    if arg <= u8::MAX.into() {
        0
    } else if arg <= u16::MAX.into() {
        1
    } else if arg <= 0xffffff {
        2
    } else {
        3
    }
}

/// Used to represent opargs for opcodes that don't require arguments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UnusedArgument(pub u32);

impl From<u32> for UnusedArgument {
    fn from(value: u32) -> Self {
        UnusedArgument(value)
    }
}

/// Used to represent stack operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackEffect {
    pub pushes: u32,
    pub pops: u32,
}

/// Offsets are for instructions (not bytes)
#[derive(Debug, Clone, PartialEq)]
pub struct ExceptionTableEntry {
    /// Inclusive offset
    pub start: u32,
    /// Exclusive offset
    pub end: u32,
    pub target: u32,
    /// Stack depth at the start of the try block
    pub depth: u32,
    /// Whether to push the index of the last executed instruction
    pub lasti: bool,
}

impl StackEffect {
    /// Creates a StackEffect with equal pushes and pops.
    pub fn balanced(count: u32) -> Self {
        StackEffect {
            pushes: count,
            pops: count,
        }
    }

    /// Creates a StackEffect when only pushing
    pub fn push(count: u32) -> Self {
        StackEffect {
            pushes: count,
            pops: 0,
        }
    }

    /// Creates a StackEffect when only pushing
    pub fn pop(count: u32) -> Self {
        StackEffect {
            pushes: 0,
            pops: count,
        }
    }

    /// For when there is no stack access
    pub fn zero() -> Self {
        StackEffect { pushes: 0, pops: 0 }
    }

    /// Calculates the net total for the stackeffect
    pub fn net_total(&self) -> i32 {
        self.pushes as i32 - self.pops as i32
    }
}

#[macro_export]
macro_rules! define_default_traits {
    ($variant:ident, Instruction) => {
        impl Deref for $crate::$variant::instructions::Instructions {
            type Target = [$crate::$variant::instructions::Instruction];

            /// Allow the user to get a reference slice to the instructions
            fn deref(&self) -> &Self::Target {
                self.0.deref()
            }
        }

        impl DerefMut for $crate::$variant::instructions::Instructions {
            /// Allow the user to get a mutable reference slice for making modifications to existing instructions.
            fn deref_mut(&mut self) -> &mut [$crate::$variant::instructions::Instruction] {
                self.0.deref_mut()
            }
        }

        impl AsRef<[$crate::$variant::instructions::Instruction]>
            for $crate::$variant::instructions::Instructions
        {
            fn as_ref(&self) -> &[$crate::$variant::instructions::Instruction] {
                &self.0
            }
        }

        impl From<$crate::$variant::instructions::Instructions> for Vec<u8> {
            fn from(val: $crate::$variant::instructions::Instructions) -> Self {
                val.to_bytes()
            }
        }

        impl TryFrom<&[u8]> for $crate::$variant::instructions::Instructions {
            type Error = Error;
            fn try_from(code: &[u8]) -> Result<Self, Self::Error> {
                if code.len() % 2 != 0 {
                    return Err(Error::InvalidBytecodeLength);
                }

                let mut instructions = $crate::$variant::instructions::Instructions(
                    Vec::with_capacity(code.len() / 2),
                );

                for chunk in code.chunks(2) {
                    if chunk.len() != 2 {
                        return Err(Error::InvalidBytecodeLength);
                    }
                    let opcode = Opcode::from(chunk[0]);
                    let arg = chunk[1];

                    instructions.append_instruction((opcode, arg).into());
                }

                Ok(instructions)
            }
        }

        impl From<&[Instruction]> for Instructions {
            fn from(value: &[Instruction]) -> Self {
                $crate::$variant::instructions::Instructions::new(value.to_vec())
            }
        }

        impl InstructionsOwned<$crate::$variant::instructions::Instruction>
            for $crate::$variant::instructions::Instructions
        {
            type Instruction = $crate::$variant::instructions::Instruction;

            fn push(&mut self, item: Self::Instruction) {
                self.0.push(item);
            }
        }

        // impl<T> SimpleInstructionAccess<$crate::$variant::instructions::Instruction> for T where
        //     T: Deref<Target = [Instruction]> + AsRef<[Instruction]>
        // {
        // }
    };

    ($variant:ident, ExtInstruction) => {
        impl Deref for $crate::$variant::ext_instructions::ExtInstructions {
            type Target = [$crate::$variant::ext_instructions::ExtInstruction];

            /// Allow the user to get a reference slice to the instructions
            fn deref(&self) -> &Self::Target {
                self.0.deref()
            }
        }

        impl DerefMut for $crate::$variant::ext_instructions::ExtInstructions {
            /// Allow the user to get a mutable reference slice for making modifications to existing instructions.
            fn deref_mut(&mut self) -> &mut [$crate::$variant::ext_instructions::ExtInstruction] {
                self.0.deref_mut()
            }
        }

        impl AsRef<[$crate::$variant::ext_instructions::ExtInstruction]>
            for $crate::$variant::ext_instructions::ExtInstructions
        {
            fn as_ref(&self) -> &[$crate::$variant::ext_instructions::ExtInstruction] {
                &self.0
            }
        }

        impl From<$crate::$variant::ext_instructions::ExtInstructions> for Vec<u8> {
            fn from(val: $crate::$variant::ext_instructions::ExtInstructions) -> Self {
                val.to_bytes()
            }
        }

        impl TryFrom<&[$crate::$variant::instructions::Instruction]>
            for $crate::$variant::ext_instructions::ExtInstructions
        {
            type Error = Error;

            fn try_from(
                value: &[$crate::$variant::instructions::Instruction],
            ) -> Result<Self, Self::Error> {
                $crate::$variant::ext_instructions::ExtInstructions::from_instructions(value)
            }
        }

        impl From<&[$crate::$variant::ext_instructions::ExtInstruction]>
            for $crate::$variant::ext_instructions::ExtInstructions
        {
            fn from(value: &[$crate::$variant::ext_instructions::ExtInstruction]) -> Self {
                $crate::$variant::ext_instructions::ExtInstructions::new(value.to_vec())
            }
        }
    };
}

pub fn generate_var_name(
    stack_name: &'static str,
    names: &mut HashMap<&'static str, u32>,
) -> String {
    if names.contains_key(stack_name) {
        *names.get_mut(stack_name).unwrap() += 1;
    } else {
        names.insert(stack_name, 0);
    }

    format!("{}_{}", stack_name, names[stack_name])
}

/// A vector that allows for negative indexes and automatically fills elements with an "empty" value when inserting at an arbitrary index below 0.
#[derive(Debug, Clone)]
pub struct InfiniteVec<T>
where
    T: Clone + std::fmt::Debug,
{
    data: VecDeque<Option<T>>,
    /// This is the offset that indicates where index "0" is really at
    negative_offset: usize,
}

impl<T> Default for InfiniteVec<T>
where
    T: Clone + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> InfiniteVec<T>
where
    T: Clone + std::fmt::Debug,
{
    pub fn new() -> Self {
        InfiniteVec {
            data: vec![].into(),
            negative_offset: 0,
        }
    }

    pub fn from_vec(vec: Vec<T>) -> Self {
        InfiniteVec {
            data: VecDeque::from(vec.into_iter().map(|v| Some(v)).collect::<Vec<_>>()),
            negative_offset: 0,
        }
    }

    pub fn insert(&mut self, index: isize, value: T) {
        let real_index = index + self.negative_offset as isize;

        if real_index < 0 {
            for _ in 0..(real_index.abs() - 1) {
                self.data.push_front(None)
            }

            self.data.push_front(Some(value));

            self.negative_offset += real_index.unsigned_abs();
        } else {
            self.data.insert(real_index as usize, Some(value));
        }
    }

    pub fn push(&mut self, value: T) {
        self.data.push_back(Some(value));
    }

    pub fn get(&self, index: isize) -> Option<&Option<T>> {
        let real_index = index + self.negative_offset as isize;

        if real_index < 0 {
            None
        } else {
            self.data.get(real_index as usize)
        }
    }

    pub fn get_mut(&mut self, index: isize) -> Option<&mut Option<T>> {
        let real_index = index + self.negative_offset as isize;

        if real_index < 0 {
            None
        } else {
            self.data.get_mut(real_index as usize)
        }
    }

    pub fn remove(&mut self, index: isize) -> Option<Option<T>> {
        let real_index = index + self.negative_offset as isize;

        if index < 0 {
            self.negative_offset -= 1;
        }

        self.data.remove(real_index.try_into().unwrap())
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn positive_len(&self) -> usize {
        self.data.len() - self.negative_offset
    }

    pub fn negative_len(&self) -> usize {
        debug_assert!(self.data.len() >= self.negative_offset);

        self.negative_offset
    }

    pub fn collect_negative_indexes(&self) -> Vec<usize> {
        self.data
            .iter()
            .enumerate()
            .take(self.negative_offset)
            .filter_map(|(i, e)| e.as_ref().map(|_| i))
            .collect()
    }

    /// Collects the values with Some() value and their index
    pub fn iter_pairs(&self) -> impl DoubleEndedIterator<Item = (isize, &T)> {
        self.data
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_some())
            .map(|(i, value)| {
                (
                    i as isize - self.negative_offset as isize,
                    value.as_ref().unwrap(),
                )
            })
    }

    /// Tells us whether negative items were used
    pub fn no_negative_items(&self) -> bool {
        self.negative_offset == 0
    }

    pub fn iter(&self) -> std::collections::vec_deque::Iter<'_, Option<T>> {
        self.data.iter()
    }

    /// Only iter over negative indexed values
    pub fn iter_negative(
        &self,
    ) -> std::iter::Take<std::collections::vec_deque::Iter<'_, Option<T>>> {
        self.data.iter().take(self.negative_offset)
    }
}

#[derive(Debug, Clone)]
pub struct InfiniteStack<T>
where
    T: Clone + std::fmt::Debug,
{
    pub data: InfiniteVec<T>,
    /// This points to where we currently are in the stack
    pub carrot: isize,
}

impl<T> InfiniteStack<T>
where
    T: Clone + std::fmt::Debug,
{
    pub fn new(stack: InfiniteVec<T>) -> Self {
        InfiniteStack {
            data: stack,
            carrot: 0,
        }
    }

    /// Returns the index of the top of the stack (if there is one)
    pub fn get_tos_index(&self) -> Option<isize> {
        self.data.iter_pairs().last().map(|(i, _)| i)
    }
}

impl<T> From<InfiniteVec<T>> for InfiniteStack<T>
where
    T: Clone + std::fmt::Debug,
{
    fn from(value: InfiniteVec<T>) -> Self {
        InfiniteStack::new(value)
    }
}

impl<T> From<Vec<T>> for InfiniteStack<T>
where
    T: Clone + std::fmt::Debug,
{
    fn from(value: Vec<T>) -> Self {
        InfiniteStack::new(value.into())
    }
}

impl<T> From<Vec<T>> for InfiniteVec<T>
where
    T: Clone + std::fmt::Debug,
{
    fn from(value: Vec<T>) -> Self {
        InfiniteVec {
            data: value.into_iter().map(|e| Some(e)).collect(),
            negative_offset: 0,
        }
    }
}

#[cfg(feature = "dot")]
#[derive(Debug, Clone)]
pub enum BlockKind {
    ExceptionBlock,
    InExceptionRange,
    NormalBlock,
}

#[cfg(feature = "sir")]
pub fn replace_var_in_expression<SIRNode: GenericSIRNode>(
    node: &mut SIRExpression<SIRNode>,
    og_var: &AuxVar,
    new_var: &AuxVar,
) {
    match node {
        SIRExpression::Call(call) => {
            for var in call.stack_inputs.iter_mut() {
                if var == og_var {
                    *var = new_var.clone();
                }
            }
        }
        SIRExpression::Exception(exc) => {
            for var in exc.stack_inputs.iter_mut() {
                if var == og_var {
                    *var = new_var.clone();
                }
            }
        }
        SIRExpression::PhiNode(values) => {
            for var in values {
                if var == og_var {
                    *var = new_var.clone();
                }
            }
        }
        SIRExpression::GeneratorStart => {}
    }
}

#[cfg(feature = "sir")]
pub fn replace_var_in_statement<SIRNode: GenericSIRNode>(
    node: &mut SIRStatement<SIRNode>,
    og_var: &AuxVar,
    new_var: &AuxVar,
) {
    match node {
        SIRStatement::Assignment(var, value) => {
            if var == og_var {
                *var = new_var.clone();
            }

            replace_var_in_expression(value, og_var, new_var);
        }
        SIRStatement::DisregardCall(call) => {
            for var in call.stack_inputs.iter_mut() {
                if var == og_var {
                    *var = new_var.clone();
                }
            }
        }
        SIRStatement::TupleAssignment(vars, value) => {
            for var in vars.iter_mut() {
                if var == og_var {
                    *var = new_var.clone();
                }
            }

            replace_var_in_expression(value, og_var, new_var);
        }
        SIRStatement::UseVar(var) => {
            if var == og_var {
                *var = new_var.clone();
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::utils::InfiniteVec;

    #[test]
    fn test_infinite_vec() {
        let mut infinite_vec = InfiniteVec::new();

        infinite_vec.push(1);
        infinite_vec.insert(-5, 5);

        assert_eq!(
            infinite_vec.iter().collect::<Vec<_>>(),
            [Some(5), None, None, None, None, Some(1)]
                .iter()
                .collect::<Vec<_>>()
        );

        assert_eq!(infinite_vec.get(0).unwrap(), &Some(1));
    }
}
