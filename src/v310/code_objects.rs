use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ops::{Deref, DerefMut},
};

use bitflags::bitflags;

use hashable::HashableHashSet;
use indexmap::IndexSet;
use num_bigint::BigInt;
use num_complex::Complex;
use ordered_float::OrderedFloat;
use python_marshal::{extract_object, resolver::resolve_all_refs, CodeFlags, Object, PyString};
use store_interval_tree::{EntryMut, Interval, IntervalTree};

use crate::{error::Error, PycFile};

use super::opcodes::Opcode;

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

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    FrozenConstant(FrozenConstant),
    CodeObject(Code),
}

impl From<Constant> for python_marshal::Object {
    fn from(val: Constant) -> Self {
        match val {
            Constant::CodeObject(code) => python_marshal::Object::Code(code.into()),
            Constant::FrozenConstant(constant) => constant.into(),
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

impl TryFrom<python_marshal::Object> for Constant {
    type Error = Error;

    fn try_from(value: python_marshal::Object) -> Result<Self, Self::Error> {
        match value {
            python_marshal::Object::Code(code) => match code {
                python_marshal::Code::V310(code) => {
                    let code = Code::try_from(code)?;
                    Ok(Constant::CodeObject(code))
                }
                _ => Err(Error::WrongVersion),
            },
            _ => {
                let frozen_constant = FrozenConstant::try_from(value)?;
                Ok(Constant::FrozenConstant(frozen_constant))
            }
        }
    }
}

// Low level representation of a Python code object
#[derive(Debug, Clone, PartialEq)]
pub struct Code {
    pub argcount: u32,
    pub posonlyargcount: u32,
    pub kwonlyargcount: u32,
    pub nlocals: u32,
    pub stacksize: u32,
    pub flags: CodeFlags,
    pub code: Instructions,
    pub consts: Vec<Constant>,
    pub names: Vec<PyString>,
    pub varnames: Vec<PyString>,
    pub freevars: Vec<PyString>,
    pub cellvars: Vec<PyString>,
    pub filename: PyString,
    pub name: PyString,
    pub firstlineno: u32,
    /// NOTE: https://peps.python.org/pep-0626/
    pub lnotab: Vec<u8>,
}

impl Code {
    pub fn get_line(instruction: &Instruction) {}
}

impl TryFrom<(python_marshal::Object, Vec<Object>)> for Code {
    type Error = Error;

    fn try_from(
        (code_object, refs): (python_marshal::Object, Vec<Object>),
    ) -> Result<Self, Self::Error> {
        let (code_object, refs) = resolve_all_refs(&code_object, &refs);

        if !refs.is_empty() {
            return Err(Error::RecursiveReference(
                "This pyc file contains references that cannot be resolved. This should never happen on a valid pyc file generated by Python.",
            ));
        }

        let code_object = extract_object!(Some(code_object), python_marshal::Object::Code(code) => code, python_marshal::error::Error::UnexpectedObject)?;

        match code_object {
            python_marshal::Code::V310(code) => Ok(Code::try_from(code)?),
            _ => Err(Error::WrongVersion),
        }
    }
}

impl From<Code> for python_marshal::Code {
    fn from(val: Code) -> Self {
        python_marshal::Code::V310(python_marshal::code_objects::Code310 {
            argcount: val.argcount,
            posonlyargcount: val.posonlyargcount,
            kwonlyargcount: val.kwonlyargcount,
            nlocals: val.nlocals,
            stacksize: val.stacksize,
            flags: val.flags,
            code: python_marshal::Object::Bytes(val.code.into()).into(),
            consts: python_marshal::Object::Tuple(
                val.consts.into_iter().map(|c| c.into()).collect(),
            )
            .into(),
            names: python_marshal::Object::Tuple(
                val.names
                    .into_iter()
                    .map(python_marshal::Object::String)
                    .collect(),
            )
            .into(),
            varnames: python_marshal::Object::Tuple(
                val.varnames
                    .into_iter()
                    .map(python_marshal::Object::String)
                    .collect(),
            )
            .into(),
            freevars: python_marshal::Object::Tuple(
                val.freevars
                    .into_iter()
                    .map(python_marshal::Object::String)
                    .collect(),
            )
            .into(),
            cellvars: python_marshal::Object::Tuple(
                val.cellvars
                    .into_iter()
                    .map(python_marshal::Object::String)
                    .collect(),
            )
            .into(),
            filename: python_marshal::Object::String(val.filename).into(),
            name: python_marshal::Object::String(val.name).into(),
            firstlineno: val.firstlineno,
            lnotab: python_marshal::Object::Bytes(val.lnotab).into(),
        })
    }
}

macro_rules! extract_strings_tuple {
    ($objs:expr, $refs:expr) => {
        $objs
            .iter()
            .map(|o| match o {
                python_marshal::Object::String(string) => Ok(string.clone()),
                _ => Err(python_marshal::error::Error::UnexpectedObject),
            })
            .collect::<Result<Vec<_>, _>>()
    };
}

impl TryFrom<python_marshal::code_objects::Code310> for Code {
    type Error = crate::error::Error;

    fn try_from(code: python_marshal::code_objects::Code310) -> Result<Self, Self::Error> {
        let co_code = extract_object!(Some(*code.code), python_marshal::Object::Bytes(bytes) => bytes, python_marshal::error::Error::NullInTuple)?;
        let co_consts = extract_object!(Some(*code.consts), python_marshal::Object::Tuple(objs) => objs, python_marshal::error::Error::NullInTuple)?;
        let co_names = extract_strings_tuple!(
            extract_object!(Some(*code.names), python_marshal::Object::Tuple(objs) => objs, python_marshal::error::Error::NullInTuple)?,
            self.references
        )?;
        let co_varnames = extract_strings_tuple!(
            extract_object!(Some(*code.varnames), python_marshal::Object::Tuple(objs) => objs, python_marshal::error::Error::NullInTuple)?,
            self.references
        )?;
        let co_freevars = extract_strings_tuple!(
            extract_object!(Some(*code.freevars), python_marshal::Object::Tuple(objs) => objs, python_marshal::error::Error::NullInTuple)?,
            self.references
        )?;
        let co_cellvars = extract_strings_tuple!(
            extract_object!(Some(*code.cellvars), python_marshal::Object::Tuple(objs) => objs, python_marshal::error::Error::NullInTuple)?,
            self.references
        )?;

        let co_filename = extract_object!(Some(*code.filename), python_marshal::Object::String(string) => string, python_marshal::error::Error::NullInTuple)?;
        let co_name = extract_object!(Some(*code.name), python_marshal::Object::String(string) => string, python_marshal::error::Error::NullInTuple)?;
        let co_lnotab = extract_object!(Some(*code.lnotab), python_marshal::Object::Bytes(bytes) => bytes, python_marshal::error::Error::NullInTuple)?;

        Ok(Code {
            argcount: code.argcount,
            posonlyargcount: code.posonlyargcount,
            kwonlyargcount: code.kwonlyargcount,
            nlocals: code.nlocals,
            stacksize: code.stacksize,
            flags: code.flags,
            code: Instructions::try_from(co_code.as_slice())?,
            consts: co_consts
                .iter()
                .map(|obj| Constant::try_from(obj.clone()))
                .collect::<Result<Vec<_>, _>>()?,
            names: co_names.to_vec(),
            varnames: co_varnames.to_vec(),
            freevars: co_freevars.to_vec(),
            cellvars: co_cellvars.to_vec(),
            filename: co_filename.clone(),
            name: co_name.clone(),
            firstlineno: code.firstlineno,
            lnotab: co_lnotab.to_vec(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jump {
    Relative(RelativeJump),
    Absolute(AbsoluteJump),
}

impl From<RelativeJump> for Jump {
    fn from(value: RelativeJump) -> Self {
        Self::Relative(value)
    }
}

impl From<AbsoluteJump> for Jump {
    fn from(value: AbsoluteJump) -> Self {
        Self::Absolute(value)
    }
}

/// Represents a relative jump offset from the current instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelativeJump {
    index: u32,
}

impl RelativeJump {
    pub fn new(index: u32) -> Self {
        RelativeJump { index }
    }
}

/// Represents an absolute jump target (a byte offset from the start of the code).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbsoluteJump {
    index: u32,
}

impl AbsoluteJump {
    pub fn new(index: u32) -> Self {
        AbsoluteJump { index }
    }
}

/// Holds an index into co_names. Has helper functions to get the actual PyString of the name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NameIndex {
    index: u32,
}

impl NameIndex {
    pub fn get<'a>(&self, co_names: &'a [PyString]) -> Option<&'a PyString> {
        co_names.get(self.index as usize)
    }
}

/// Holds an index into co_varnames. Has helper functions to get the actual PyString of the name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarNameIndex {
    index: u32,
}

impl VarNameIndex {
    pub fn get<'a>(&self, co_varnames: &'a [PyString]) -> Option<&'a PyString> {
        co_varnames.get(self.index as usize)
    }
}

/// Holds an index into co_consts. Has helper functions to get the actual constant at the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstIndex {
    index: u32,
}

impl ConstIndex {
    pub fn get<'a>(&self, co_consts: &'a [Constant]) -> Option<&'a Constant> {
        co_consts.get(self.index as usize)
    }
}

/// Represents a resolved reference to a variable in the cell or free variable storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClosureRefIndex {
    index: u32,
}

/// Represents a resolved reference to a variable in the cell or free variable storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosureRef {
    /// Index into `co_cellvars`.
    /// These are variables created in the current scope that will be used by nested scopes.
    Cell {
        /// The index into the `co_cellvars` list.
        index: u32,
    },
    /// Index into `co_freevars`.
    /// These are variables used in the current scope that were created in an enclosing scope.
    Free {
        /// The index into the `co_freevars` list.
        index: u32,
    },

    Invalid(u32),
}

impl ClosureRefIndex {
    pub fn into_closure_ref(i: u32, cellvars: &[PyString], freevars: &[PyString]) -> ClosureRef {
        let cell_len = cellvars.len() as u32;
        if i < cell_len {
            ClosureRef::Cell { index: i }
        } else {
            let free_index = i - cell_len;
            if (free_index as usize) < freevars.len() {
                ClosureRef::Free { index: free_index }
            } else {
                ClosureRef::Invalid(i)
            }
        }
    }
}

/// Used to represent the different comparison operations for COMPARE_OP
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOperation {
    Smaller,
    SmallerOrEqual,
    Equal,
    NotEqual,
    Bigger,
    BiggerOrEqual,
    /// We try to support invalid bytecode, so we have to represent it somehow.
    Invalid(u32),
}

impl From<u32> for CompareOperation {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Smaller,
            1 => Self::SmallerOrEqual,
            2 => Self::Equal,
            3 => Self::NotEqual,
            4 => Self::Bigger,
            5 => Self::BiggerOrEqual,
            _ => Self::Invalid(value),
        }
    }
}

impl From<&CompareOperation> for u32 {
    fn from(val: &CompareOperation) -> Self {
        match val {
            CompareOperation::Smaller => 0,
            CompareOperation::SmallerOrEqual => 1,
            CompareOperation::Equal => 2,
            CompareOperation::NotEqual => 3,
            CompareOperation::Bigger => 4,
            CompareOperation::BiggerOrEqual => 5,
            CompareOperation::Invalid(v) => *v,
        }
    }
}

/// Whether *_OP is inverted or not
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpInversion {
    NoInvert,
    Invert,
    Invalid(u32),
}

impl From<u32> for OpInversion {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::NoInvert,
            1 => Self::Invert,
            _ => Self::Invalid(value),
        }
    }
}

impl From<&OpInversion> for u32 {
    fn from(val: &OpInversion) -> Self {
        match val {
            OpInversion::NoInvert => 0,
            OpInversion::Invert => 1,
            OpInversion::Invalid(v) => *v,
        }
    }
}

/// The different types of raising forms. See https://docs.python.org/3.10/library/dis.html#opcode-RAISE_VARARGS
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaiseForms {
    ReraisePrev,
    RaiseTOS,
    RaiseTOS1FromTOS,
    Invalid(u32),
}

impl From<u32> for RaiseForms {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::ReraisePrev,
            1 => Self::RaiseTOS,
            2 => Self::RaiseTOS1FromTOS,
            _ => Self::Invalid(value),
        }
    }
}

impl From<&RaiseForms> for u32 {
    fn from(val: &RaiseForms) -> Self {
        match val {
            RaiseForms::ReraisePrev => 0,
            RaiseForms::RaiseTOS => 1,
            RaiseForms::RaiseTOS1FromTOS => 2,
            RaiseForms::Invalid(v) => *v,
        }
    }
}

/// Describes the configuration for a CALL_FUNCTION_EX instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallExFlags {
    /// The call has positional arguments only.
    /// Stack layout (top to bottom):
    /// - Positional args (an iterable)
    /// - Callable
    PositionalOnly,

    /// The call has both positional and keyword arguments.
    /// Stack layout (top to bottom):
    /// - Keyword args (a mapping)
    /// - Positional args (an iterable)
    /// - Callable
    WithKeywords,
    Invalid(u32),
}

impl From<u32> for CallExFlags {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::PositionalOnly,
            1 => Self::WithKeywords,
            _ => Self::Invalid(value),
        }
    }
}

impl From<&CallExFlags> for u32 {
    fn from(val: &CallExFlags) -> Self {
        match val {
            CallExFlags::PositionalOnly => 0,
            CallExFlags::WithKeywords => 1,
            CallExFlags::Invalid(v) => *v,
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

/// BUILD_SLICE gets an argc but it must be 2 or 3. See https://docs.python.org/3.10/library/dis.html#opcode-BUILD_SLICE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceCount {
    Two,
    Three,
    Invalid(u32),
}

impl From<u32> for SliceCount {
    fn from(value: u32) -> Self {
        match value {
            2 => Self::Two,
            3 => Self::Three,
            _ => Self::Invalid(value),
        }
    }
}

bitflags! {
    /// Represents the conversion to apply to a value before f-string formatting.
    /// From https://github.com/python/cpython/blob/3.10/Python/compile.c#L4349C5-L4361C7
    ///  Our oparg encodes 2 pieces of information: the conversion
    ///    character, and whether or not a format_spec was provided.

    ///    Convert the conversion char to 3 bits:
    ///        : 000  0x0  FVC_NONE   The default if nothing specified.
    ///    !s  : 001  0x1  FVC_STR
    ///    !r  : 010  0x2  FVC_REPR
    ///    !a  : 011  0x3  FVC_ASCII

    ///    next bit is whether or not we have a format spec:
    ///    yes : 100  0x4
    ///    no  : 000  0x0
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct FormatFlag: u8 {
        /// No conversion (default)
        const FVC_NONE = 0x0;
        /// !s conversion
        const FVC_STR = 0x1;
        /// !r conversion
        const FVC_REPR = 0x2;
        /// !a conversion
        const FVC_ASCII = 0x3;
        /// Format spec is present
        const FVS_MASK = 0x4;
    }
}

impl From<u32> for FormatFlag {
    fn from(value: u32) -> Self {
        FormatFlag::from_bits_retain(value as u8)
    }
}

impl From<&FormatFlag> for u32 {
    fn from(val: &FormatFlag) -> Self {
        val.bits() as u32
    }
}

/// Generator kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenKind {
    Generator,
    Coroutine,
    AsyncGenerator,
    Invalid(u32),
}

impl From<u32> for GenKind {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Generator,
            1 => Self::Coroutine,
            2 => Self::AsyncGenerator,
            _ => Self::Invalid(value),
        }
    }
}

impl From<&GenKind> for u32 {
    fn from(val: &GenKind) -> Self {
        match val {
            GenKind::Generator => 0,
            GenKind::Coroutine => 1,
            GenKind::AsyncGenerator => 2,
            GenKind::Invalid(v) => *v,
        }
    }
}

/// Low level representation of a Python bytecode instruction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    PopTop(u32),
    /// Python leaves the ROTN argument after optimizing. See https://github.com/python/cpython/blob/3.10/Python/compile.c#L7522
    RotTwo(u32),
    RotThree(u32),
    DupTop,
    DupTopTwo,
    RotFour(u32),
    Nop(u32),
    /// Version 3.10 has an unique bug where some NOPs are left with an arg. See https://github.com/python/cpython/issues/89918#issuecomment-1093937041
    UnaryPositive,
    UnaryNegative,
    UnaryNot,
    UnaryInvert,
    BinaryMatrixMultiply,
    InplaceMatrixMultiply,
    BinaryPower,
    BinaryMultiply,
    BinaryModulo,
    BinaryAdd,
    BinarySubtract,
    BinarySubscr,
    BinaryFloorDivide,
    BinaryTrueDivide,
    InplaceFloorDivide,
    InplaceTrueDivide,
    GetLen,
    MatchMapping,
    MatchSequence,
    MatchKeys,
    CopyDictWithoutKeys,
    WithExceptStart,
    GetAiter,
    GetAnext,
    BeforeAsyncWith,
    EndAsyncFor,
    InplaceAdd,
    InplaceSubtract,
    InplaceMultiply,
    InplaceModulo,
    StoreSubscr,
    DeleteSubscr,
    BinaryLshift,
    BinaryRshift,
    BinaryAnd,
    BinaryXor,
    BinaryOr,
    InplacePower,
    GetIter,
    GetYieldFromIter,
    PrintExpr,
    LoadBuildClass,
    YieldFrom,
    GetAwaitable,
    LoadAssertionError,
    InplaceLshift,
    InplaceRshift,
    InplaceAnd,
    InplaceXor,
    InplaceOr,
    ListToTuple,
    ReturnValue,
    ImportStar,
    SetupAnnotations,
    YieldValue,
    PopBlock,
    PopExcept,
    StoreName(NameIndex),
    DeleteName(NameIndex),
    UnpackSequence(u32),
    ForIter(RelativeJump),
    UnpackEx(u32),
    StoreAttr(NameIndex),
    DeleteAttr(NameIndex),
    StoreGlobal(NameIndex),
    DeleteGlobal(NameIndex),
    RotN(u32),
    LoadConst(ConstIndex),
    LoadName(NameIndex),
    BuildTuple(u32),
    BuildList(u32),
    BuildSet(u32),
    BuildMap(u32),
    LoadAttr(NameIndex),
    CompareOp(CompareOperation),
    ImportName(NameIndex),
    ImportFrom(NameIndex),
    JumpForward(RelativeJump),
    JumpIfFalseOrPop(AbsoluteJump),
    JumpIfTrueOrPop(AbsoluteJump),
    JumpAbsolute(AbsoluteJump),
    PopJumpIfFalse(AbsoluteJump),
    PopJumpIfTrue(AbsoluteJump),
    LoadGlobal(NameIndex),
    IsOp(OpInversion),
    ContainsOp(OpInversion),
    Reraise(RaiseForms),
    JumpIfNotExcMatch(AbsoluteJump),
    SetupFinally(RelativeJump),
    LoadFast(VarNameIndex),
    StoreFast(VarNameIndex),
    DeleteFast(VarNameIndex),
    GenStart(GenKind),
    RaiseVarargs(RaiseForms),
    CallFunction(u32),
    MakeFunction(MakeFunctionFlags),
    BuildSlice(u32),
    LoadClosure(ClosureRefIndex),
    LoadDeref(ClosureRefIndex),
    StoreDeref(ClosureRefIndex),
    DeleteDeref(ClosureRefIndex),
    CallFunctionKW(u32),
    CallFunctionEx(CallExFlags),
    SetupWith(RelativeJump),
    // ExtendedArg is skipped as it's integrated into the next instruction
    ListAppend(u32),
    SetAdd(u32),
    MapAdd(u32),
    LoadClassderef(ClosureRefIndex),
    MatchClass(u32),
    SetupAsyncWith(RelativeJump),
    FormatValue(FormatFlag),
    BuildConstKeyMap(u32),
    BuildString(u32),
    LoadMethod(NameIndex),
    CallMethod(u32),
    ListExtend(u32),
    SetUpdate(u32),
    DictMerge(u32),
    DictUpdate(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Instructions(Vec<Instruction>);

impl Instructions {
    pub fn append_instructions(&mut self, instructions: &[Instruction]) {
        for instruction in instructions {
            self.0.push(*instruction);
        }
    }

    /// Append an instruction at the end
    pub fn append_instruction(&mut self, instruction: Instruction) {
        self.0.push(instruction);
    }

    /// Delete instructions in range (ex. 1..10)
    pub fn delete_instructions(&mut self, range: std::ops::Range<usize>) {
        range
            .into_iter()
            .for_each(|index| self.delete_instruction(index));
    }

    /// Delete instruction at index
    pub fn delete_instruction(&mut self, index: usize) {
        self.0.iter_mut().enumerate().for_each(|(idx, inst)| {
            match inst {
                Instruction::JumpAbsolute(jump)
                | Instruction::PopJumpIfTrue(jump)
                | Instruction::PopJumpIfFalse(jump)
                | Instruction::JumpIfNotExcMatch(jump)
                | Instruction::JumpIfTrueOrPop(jump)
                | Instruction::JumpIfFalseOrPop(jump) => {
                    if jump.index as usize >= index {
                        // Update jump indexes that jump to this index or above it
                        jump.index -= 1
                    }
                }
                Instruction::ForIter(jump)
                | Instruction::JumpForward(jump)
                | Instruction::SetupFinally(jump)
                | Instruction::SetupWith(jump)
                | Instruction::SetupAsyncWith(jump) => {
                    // Relative jumps only need to update if the index falls within it's jump range
                    if idx <= index && index + idx <= jump.index as usize {
                        jump.index -= 1
                    }
                }
                _ => {}
            }
        });

        self.0.remove(index);
    }

    /// Insert a slice of instructions at an index
    pub fn insert_instructions(&mut self, index: usize, instructions: &[Instruction]) {
        for (idx, instruction) in instructions.iter().enumerate() {
            self.insert_instruction(index + idx, *instruction);
        }
    }

    /// Insert instruction at a specific index. It automatically fixes jump offsets in other instructions.
    pub fn insert_instruction(&mut self, index: usize, instruction: Instruction) {
        self.0.iter_mut().enumerate().for_each(|(idx, inst)| {
            match inst {
                Instruction::JumpAbsolute(jump)
                | Instruction::PopJumpIfTrue(jump)
                | Instruction::PopJumpIfFalse(jump)
                | Instruction::JumpIfNotExcMatch(jump)
                | Instruction::JumpIfTrueOrPop(jump)
                | Instruction::JumpIfFalseOrPop(jump) => {
                    if jump.index as usize >= index {
                        // Update jump indexes that jump to this index or above it
                        jump.index += 1
                    }
                }
                Instruction::ForIter(jump)
                | Instruction::JumpForward(jump)
                | Instruction::SetupFinally(jump)
                | Instruction::SetupWith(jump)
                | Instruction::SetupAsyncWith(jump) => {
                    // Relative jumps only need to update if the index falls within it's jump range
                    if idx <= index && index + idx <= jump.index as usize {
                        jump.index += 1
                    }
                }
                _ => {}
            }
        });
        self.0.insert(index, instruction);
    }

    pub fn get_instructions(&self) -> &[Instruction] {
        self.deref()
    }

    pub fn get_instructions_mut(&mut self) -> &mut [Instruction] {
        self.deref_mut()
    }

    pub fn get_jump_target(&self, jump: Jump) -> Option<Instruction> {
        match jump {
            Jump::Absolute(AbsoluteJump { index }) | Jump::Relative(RelativeJump { index }) => {
                self.0.get(index as usize).cloned()
            }
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytearray = Vec::with_capacity(self.0.len() * 2); // This will not be enough this as we dynamically generate EXTENDED_ARGS, but it's better than not reserving any length.

        macro_rules! push_inst {
            ($instruction:expr, $arg:expr) => {{
                let mut arg: u32 = $arg;
                // Emit EXTENDED_ARGs for arguments > 0xFF
                if arg > u8::MAX.into() {
                    // Python bytecode uses EXTENDED_ARG for each additional byte above the lowest.
                    // We need to emit them from most significant to least significant.
                    let mut ext_args = Vec::new();
                    while arg > u8::MAX.into() {
                        ext_args.push(((arg >> 8) & 0xFF) as u8);
                        arg >>= 8;
                    }
                    // Emit EXTENDED_ARGs in reverse order (most significant first)
                    for &ext in ext_args.iter().rev() {
                        bytearray.push(Opcode::EXTENDED_ARG as u8);
                        bytearray.push(ext);
                    }
                }

                bytearray.push($instruction.get_opcode() as u8);
                bytearray.push($arg as u8)
            }};
        }

        // mapping of original to updated index
        let mut absolute_jump_indexes: BTreeMap<u32, u32> = BTreeMap::new();
        let mut relative_jump_indexes = IntervalTree::<u32, u32>::new(); // (u32, u32) is the from and to index for relative jumps

        self.iter().enumerate().for_each(|(idx, inst)| match inst {
            Instruction::JumpAbsolute(jump)
            | Instruction::PopJumpIfTrue(jump)
            | Instruction::PopJumpIfFalse(jump)
            | Instruction::JumpIfNotExcMatch(jump)
            | Instruction::JumpIfTrueOrPop(jump)
            | Instruction::JumpIfFalseOrPop(jump) => {
                absolute_jump_indexes.insert(jump.index, jump.index);
            }
            Instruction::ForIter(jump)
            | Instruction::JumpForward(jump)
            | Instruction::SetupFinally(jump)
            | Instruction::SetupWith(jump)
            | Instruction::SetupAsyncWith(jump) => {
                relative_jump_indexes.insert(
                    Interval::new(
                        std::ops::Bound::Included(idx as u32),
                        std::ops::Bound::Included(idx as u32 + jump.index + 1),
                    ),
                    jump.index,
                );
            }
            _ => {}
        });

        for (index, instruction) in self.iter().enumerate() {
            let arg = instruction.get_raw_value();

            if arg > u8::MAX.into() {
                // Calculate how many extended args an instruction will need
                let extended_arg_count = ((32 - arg.leading_zeros()) + 7) / 8;
                let extended_arg_count = extended_arg_count.saturating_sub(1); // Don't count the instruction itself

                absolute_jump_indexes
                    .range_mut((
                        std::ops::Bound::Excluded(index as u32),
                        std::ops::Bound::Unbounded,
                    ))
                    .for_each(|(_, new)| *new += extended_arg_count);

                for mut entry in relative_jump_indexes.query_mut(&Interval::new(
                    std::ops::Bound::Included(index as u32),
                    std::ops::Bound::Excluded((index + 1) as u32),
                )) {
                    *entry.value() += extended_arg_count
                }
            }
        }

        for (idx, instruction) in self.0.iter().enumerate() {
            match instruction {
                Instruction::JumpAbsolute(jump)
                | Instruction::PopJumpIfTrue(jump)
                | Instruction::PopJumpIfFalse(jump)
                | Instruction::JumpIfNotExcMatch(jump)
                | Instruction::JumpIfTrueOrPop(jump)
                | Instruction::JumpIfFalseOrPop(jump) => {
                    push_inst!(instruction, absolute_jump_indexes[&jump.index]);
                }
                Instruction::ForIter(jump)
                | Instruction::JumpForward(jump)
                | Instruction::SetupFinally(jump)
                | Instruction::SetupWith(jump)
                | Instruction::SetupAsyncWith(jump) => {
                    let interval = Interval::new(
                        std::ops::Bound::Included(idx as u32),
                        std::ops::Bound::Included(idx as u32 + jump.index + 1),
                    );

                    // dbg!(
                    //     relative_jump_indexes.query(&interval).collect::<Vec<_>>(),
                    //     &interval
                    // );

                    if cfg!(debug_assertions) {
                        let indexes = relative_jump_indexes
                            .query(&interval)
                            .filter(|entry| *entry.interval() == interval)
                            .collect::<Vec<_>>();

                        assert!(indexes.len() == 1);

                        push_inst!(
                            instruction,
                            *indexes
                                .first()
                                .expect("This interval should always exist")
                                .value()
                        );
                    } else {
                        // This is faster, so use for release builds
                        push_inst!(
                            instruction,
                            *relative_jump_indexes
                                .query(&interval)
                                .filter(|entry| *entry.interval() == interval)
                                .next()
                                .expect("This interval should always exist")
                                .value()
                        );
                    }
                }
                _ => push_inst!(instruction, instruction.get_raw_value()),
            }
        }

        bytearray
    }
}

impl Deref for Instructions {
    type Target = [Instruction];

    /// Allow the user to get a reference slice to the instructions
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl DerefMut for Instructions {
    /// Allow the user to get a mutable reference slice for making modifications to existing instructions.
    fn deref_mut(&mut self) -> &mut [Instruction] {
        self.0.deref_mut()
    }
}

impl From<Instructions> for Vec<u8> {
    fn from(val: Instructions) -> Self {
        val.to_bytes()
    }
}

impl TryFrom<&[u8]> for Instructions {
    type Error = Error;
    fn try_from(code: &[u8]) -> Result<Self, Self::Error> {
        if code.len() % 2 != 0 {
            return Err(Error::InvalidBytecodeLength);
        }

        let mut instructions = Instructions(Vec::with_capacity(code.len() / 2));
        let mut extended_arg = 0; // Used to keep track of extended arguments between instructions
        let mut removed_extended_args = vec![]; // Used to offset jump indexes
        let mut removed_count = 0;

        for (index, chunk) in code.chunks(2).enumerate() {
            if chunk.len() != 2 {
                return Err(Error::InvalidBytecodeLength);
            }
            let opcode = Opcode::try_from(chunk[0])?;
            let arg = chunk[1];

            match opcode {
                Opcode::EXTENDED_ARG => {
                    removed_extended_args.push(index - removed_count);
                    removed_count += 1;
                    extended_arg = (extended_arg << 8) | arg as u32;
                    continue;
                }
                _ => {
                    let arg = (extended_arg << 8) | arg as u32;

                    instructions.append_instruction((opcode, arg).into());
                }
            }

            extended_arg = 0;
        }

        // mapping of original to updated index
        let mut absolute_jump_indexes: BTreeMap<u32, u32> = BTreeMap::new();
        let mut relative_jump_indexes = IntervalTree::<u32, u32>::new(); // (u32, u32) is the from and to index for relative jumps

        instructions
            .iter()
            .enumerate()
            .for_each(|(idx, inst)| match inst {
                Instruction::JumpAbsolute(jump)
                | Instruction::PopJumpIfTrue(jump)
                | Instruction::PopJumpIfFalse(jump)
                | Instruction::JumpIfNotExcMatch(jump)
                | Instruction::JumpIfTrueOrPop(jump)
                | Instruction::JumpIfFalseOrPop(jump) => {
                    absolute_jump_indexes.insert(jump.index, jump.index);
                }
                Instruction::ForIter(jump)
                | Instruction::JumpForward(jump)
                | Instruction::SetupFinally(jump)
                | Instruction::SetupWith(jump)
                | Instruction::SetupAsyncWith(jump) => {
                    relative_jump_indexes.insert(
                        Interval::new(
                            std::ops::Bound::Included(idx as u32),
                            std::ops::Bound::Included(idx as u32 + jump.index + 1),
                        ),
                        jump.index,
                    );
                }
                _ => {}
            });

        // Update jump offsets to exclude the extended args that were removed
        for index in removed_extended_args {
            absolute_jump_indexes
                .range_mut((
                    std::ops::Bound::Excluded(index as u32),
                    std::ops::Bound::Unbounded,
                ))
                .for_each(|(_, new)| *new -= 1);

            for mut entry in relative_jump_indexes.query_mut(&Interval::new(
                std::ops::Bound::Included(index as u32),
                std::ops::Bound::Excluded((index + 1) as u32),
            )) {
                *entry.value() -= 1
            }
        }

        for (idx, instruction) in instructions.iter_mut().enumerate() {
            match instruction {
                Instruction::JumpAbsolute(jump)
                | Instruction::PopJumpIfTrue(jump)
                | Instruction::PopJumpIfFalse(jump)
                | Instruction::JumpIfNotExcMatch(jump)
                | Instruction::JumpIfTrueOrPop(jump)
                | Instruction::JumpIfFalseOrPop(jump) => {
                    jump.index = absolute_jump_indexes[&jump.index];
                }
                Instruction::ForIter(jump)
                | Instruction::JumpForward(jump)
                | Instruction::SetupFinally(jump)
                | Instruction::SetupWith(jump)
                | Instruction::SetupAsyncWith(jump) => {
                    let interval = Interval::new(
                        std::ops::Bound::Included(idx as u32),
                        std::ops::Bound::Included(idx as u32 + jump.index + 1),
                    );

                    // dbg!(
                    //     relative_jump_indexes.query(&interval).collect::<Vec<_>>(),
                    //     &interval
                    // );

                    if cfg!(debug_assertions) {
                        let indexes = relative_jump_indexes
                            .query(&interval)
                            .filter(|entry| *entry.interval() == interval)
                            .collect::<Vec<_>>();

                        assert!(indexes.len() == 1);

                        jump.index = *indexes
                            .first()
                            .expect("This interval should always exist")
                            .value();
                    } else {
                        // This is faster, so use for release builds
                        jump.index = *relative_jump_indexes
                            .query(&interval)
                            .filter(|entry| *entry.interval() == interval)
                            .next()
                            .expect("This interval should always exist")
                            .value();
                    }
                }
                _ => {}
            }
        }

        Ok(instructions)
    }
}

impl From<&[Instruction]> for Instructions {
    fn from(value: &[Instruction]) -> Self {
        let mut instructions = Instructions(Vec::with_capacity(value.len()));

        instructions.append_instructions(value);

        instructions
    }
}

impl From<(Opcode, u32)> for Instruction {
    fn from(value: (Opcode, u32)) -> Self {
        match value.0 {
            Opcode::NOP => Instruction::Nop(value.1),
            Opcode::POP_TOP => Instruction::PopTop(value.1),
            Opcode::ROT_TWO => Instruction::RotTwo(value.1),
            Opcode::ROT_THREE => Instruction::RotThree(value.1),
            Opcode::ROT_FOUR => Instruction::RotFour(value.1),
            Opcode::DUP_TOP => Instruction::DupTop,
            Opcode::DUP_TOP_TWO => Instruction::DupTopTwo,
            Opcode::UNARY_POSITIVE => Instruction::UnaryPositive,
            Opcode::UNARY_NEGATIVE => Instruction::UnaryNegative,
            Opcode::UNARY_NOT => Instruction::UnaryNot,
            Opcode::UNARY_INVERT => Instruction::UnaryInvert,
            Opcode::GET_ITER => Instruction::GetIter,
            Opcode::GET_YIELD_FROM_ITER => Instruction::GetYieldFromIter,
            Opcode::BINARY_POWER => Instruction::BinaryPower,
            Opcode::BINARY_MULTIPLY => Instruction::BinaryMultiply,
            Opcode::BINARY_MATRIX_MULTIPLY => Instruction::BinaryMatrixMultiply,
            Opcode::BINARY_FLOOR_DIVIDE => Instruction::BinaryFloorDivide,
            Opcode::BINARY_TRUE_DIVIDE => Instruction::BinaryTrueDivide,
            Opcode::BINARY_MODULO => Instruction::BinaryModulo,
            Opcode::BINARY_ADD => Instruction::BinaryAdd,
            Opcode::BINARY_SUBTRACT => Instruction::BinarySubtract,
            Opcode::BINARY_SUBSCR => Instruction::BinarySubscr,
            Opcode::BINARY_LSHIFT => Instruction::BinaryLshift,
            Opcode::BINARY_RSHIFT => Instruction::BinaryRshift,
            Opcode::BINARY_AND => Instruction::BinaryAnd,
            Opcode::BINARY_XOR => Instruction::BinaryXor,
            Opcode::BINARY_OR => Instruction::BinaryOr,
            Opcode::INPLACE_POWER => Instruction::InplacePower,
            Opcode::INPLACE_MULTIPLY => Instruction::InplaceMultiply,
            Opcode::INPLACE_MATRIX_MULTIPLY => Instruction::InplaceMatrixMultiply,
            Opcode::INPLACE_FLOOR_DIVIDE => Instruction::InplaceFloorDivide,
            Opcode::INPLACE_TRUE_DIVIDE => Instruction::InplaceTrueDivide,
            Opcode::INPLACE_MODULO => Instruction::InplaceModulo,
            Opcode::INPLACE_ADD => Instruction::InplaceAdd,
            Opcode::INPLACE_SUBTRACT => Instruction::InplaceSubtract,
            Opcode::INPLACE_LSHIFT => Instruction::InplaceLshift,
            Opcode::INPLACE_RSHIFT => Instruction::InplaceRshift,
            Opcode::INPLACE_AND => Instruction::InplaceAnd,
            Opcode::INPLACE_XOR => Instruction::InplaceXor,
            Opcode::INPLACE_OR => Instruction::InplaceOr,
            Opcode::STORE_SUBSCR => Instruction::StoreSubscr,
            Opcode::DELETE_SUBSCR => Instruction::DeleteSubscr,
            Opcode::GET_AWAITABLE => Instruction::GetAwaitable,
            Opcode::GET_AITER => Instruction::GetAiter,
            Opcode::GET_ANEXT => Instruction::GetAnext,
            Opcode::END_ASYNC_FOR => Instruction::EndAsyncFor,
            Opcode::BEFORE_ASYNC_WITH => Instruction::BeforeAsyncWith,
            Opcode::SETUP_ASYNC_WITH => {
                Instruction::SetupAsyncWith(RelativeJump { index: value.1 })
            }
            Opcode::PRINT_EXPR => Instruction::PrintExpr,
            Opcode::SET_ADD => Instruction::SetAdd(value.1),
            Opcode::LIST_APPEND => Instruction::ListAppend(value.1),
            Opcode::MAP_ADD => Instruction::MapAdd(value.1),
            Opcode::RETURN_VALUE => Instruction::ReturnValue,
            Opcode::YIELD_VALUE => Instruction::YieldValue,
            Opcode::YIELD_FROM => Instruction::YieldFrom,
            Opcode::SETUP_ANNOTATIONS => Instruction::SetupAnnotations,
            Opcode::IMPORT_STAR => Instruction::ImportStar,
            Opcode::POP_BLOCK => Instruction::PopBlock,
            Opcode::POP_EXCEPT => Instruction::PopExcept,
            Opcode::RERAISE => Instruction::Reraise(value.1.into()),
            Opcode::WITH_EXCEPT_START => Instruction::WithExceptStart,
            Opcode::LOAD_ASSERTION_ERROR => Instruction::LoadAssertionError,
            Opcode::LOAD_BUILD_CLASS => Instruction::LoadBuildClass,
            Opcode::SETUP_WITH => Instruction::SetupWith(RelativeJump { index: value.1 }),
            Opcode::COPY_DICT_WITHOUT_KEYS => Instruction::CopyDictWithoutKeys,
            Opcode::GET_LEN => Instruction::GetLen,
            Opcode::MATCH_MAPPING => Instruction::MatchMapping,
            Opcode::MATCH_SEQUENCE => Instruction::MatchSequence,
            Opcode::MATCH_KEYS => Instruction::MatchKeys,
            Opcode::STORE_NAME => Instruction::StoreName(NameIndex { index: value.1 }),
            Opcode::DELETE_NAME => Instruction::DeleteName(NameIndex { index: value.1 }),
            Opcode::UNPACK_SEQUENCE => Instruction::UnpackSequence(value.1),
            Opcode::UNPACK_EX => Instruction::UnpackEx(value.1),
            Opcode::STORE_ATTR => Instruction::StoreAttr(NameIndex { index: value.1 }),
            Opcode::DELETE_ATTR => Instruction::DeleteAttr(NameIndex { index: value.1 }),
            Opcode::STORE_GLOBAL => Instruction::StoreGlobal(NameIndex { index: value.1 }),
            Opcode::DELETE_GLOBAL => Instruction::DeleteGlobal(NameIndex { index: value.1 }),
            Opcode::LOAD_CONST => Instruction::LoadConst(ConstIndex { index: value.1 }),
            Opcode::LOAD_NAME => Instruction::LoadName(NameIndex { index: value.1 }),
            Opcode::BUILD_TUPLE => Instruction::BuildTuple(value.1),
            Opcode::BUILD_LIST => Instruction::BuildList(value.1),
            Opcode::BUILD_SET => Instruction::BuildSet(value.1),
            Opcode::BUILD_MAP => Instruction::BuildMap(value.1),
            Opcode::BUILD_CONST_KEY_MAP => Instruction::BuildConstKeyMap(value.1),
            Opcode::BUILD_STRING => Instruction::BuildString(value.1),
            Opcode::LIST_TO_TUPLE => Instruction::ListToTuple,
            Opcode::LIST_EXTEND => Instruction::ListExtend(value.1),
            Opcode::SET_UPDATE => Instruction::SetUpdate(value.1),
            Opcode::DICT_UPDATE => Instruction::DictUpdate(value.1),
            Opcode::DICT_MERGE => Instruction::DictMerge(value.1),
            Opcode::LOAD_ATTR => Instruction::LoadAttr(NameIndex { index: value.1 }),
            Opcode::COMPARE_OP => Instruction::CompareOp(value.1.into()),
            Opcode::IMPORT_NAME => Instruction::ImportName(NameIndex { index: value.1 }),
            Opcode::IMPORT_FROM => Instruction::ImportFrom(NameIndex { index: value.1 }),
            Opcode::JUMP_FORWARD => Instruction::JumpForward(RelativeJump { index: value.1 }),
            Opcode::POP_JUMP_IF_TRUE => Instruction::PopJumpIfTrue(AbsoluteJump { index: value.1 }),
            Opcode::POP_JUMP_IF_FALSE => {
                Instruction::PopJumpIfFalse(AbsoluteJump { index: value.1 })
            }
            Opcode::JUMP_IF_NOT_EXC_MATCH => {
                Instruction::JumpIfNotExcMatch(AbsoluteJump { index: value.1 })
            }
            Opcode::JUMP_IF_TRUE_OR_POP => {
                Instruction::JumpIfTrueOrPop(AbsoluteJump { index: value.1 })
            }
            Opcode::JUMP_IF_FALSE_OR_POP => {
                Instruction::JumpIfFalseOrPop(AbsoluteJump { index: value.1 })
            }
            Opcode::JUMP_ABSOLUTE => Instruction::JumpAbsolute(AbsoluteJump { index: value.1 }),
            Opcode::FOR_ITER => Instruction::ForIter(RelativeJump { index: value.1 }),
            Opcode::LOAD_GLOBAL => Instruction::LoadGlobal(NameIndex { index: value.1 }),
            Opcode::IS_OP => Instruction::IsOp(value.1.into()),
            Opcode::CONTAINS_OP => Instruction::ContainsOp(value.1.into()),
            Opcode::SETUP_FINALLY => Instruction::SetupFinally(RelativeJump { index: value.1 }),
            Opcode::LOAD_FAST => Instruction::LoadFast(VarNameIndex { index: value.1 }),
            Opcode::STORE_FAST => Instruction::StoreFast(VarNameIndex { index: value.1 }),
            Opcode::DELETE_FAST => Instruction::DeleteFast(VarNameIndex { index: value.1 }),
            Opcode::LOAD_CLOSURE => Instruction::LoadClosure(ClosureRefIndex { index: value.1 }),
            Opcode::LOAD_DEREF => Instruction::LoadDeref(ClosureRefIndex { index: value.1 }),
            Opcode::LOAD_CLASSDEREF => {
                Instruction::LoadClassderef(ClosureRefIndex { index: value.1 })
            }
            Opcode::STORE_DEREF => Instruction::StoreDeref(ClosureRefIndex { index: value.1 }),
            Opcode::DELETE_DEREF => Instruction::DeleteDeref(ClosureRefIndex { index: value.1 }),
            Opcode::RAISE_VARARGS => Instruction::RaiseVarargs(value.1.into()),
            Opcode::CALL_FUNCTION => Instruction::CallFunction(value.1),
            Opcode::CALL_FUNCTION_KW => Instruction::CallFunctionKW(value.1),
            Opcode::CALL_FUNCTION_EX => Instruction::CallFunctionEx(value.1.into()),
            Opcode::LOAD_METHOD => Instruction::LoadMethod(NameIndex { index: value.1 }),
            Opcode::CALL_METHOD => Instruction::CallMethod(value.1),
            Opcode::MAKE_FUNCTION => {
                Instruction::MakeFunction(MakeFunctionFlags::from_bits_retain(value.1))
            }
            Opcode::BUILD_SLICE => Instruction::BuildSlice(value.1),
            Opcode::FORMAT_VALUE => Instruction::FormatValue(value.1.into()),
            Opcode::MATCH_CLASS => Instruction::MatchClass(value.1),
            Opcode::GEN_START => Instruction::GenStart(value.1.into()),
            Opcode::ROT_N => Instruction::RotN(value.1),
            Opcode::EXTENDED_ARG => panic!(
                "Extended arg can never be turned into an instruction. This should never happen."
            ), // ExtendedArg is handled separately
        }
    }
}

impl Instruction {
    pub fn get_opcode(&self) -> Opcode {
        match self {
            Instruction::Nop(_) => Opcode::NOP,
            Instruction::PopTop(_) => Opcode::POP_TOP,
            Instruction::RotTwo(_) => Opcode::ROT_TWO,
            Instruction::RotThree(_) => Opcode::ROT_THREE,
            Instruction::RotFour(_) => Opcode::ROT_FOUR,
            Instruction::DupTop => Opcode::DUP_TOP,
            Instruction::DupTopTwo => Opcode::DUP_TOP_TWO,
            Instruction::UnaryPositive => Opcode::UNARY_POSITIVE,
            Instruction::UnaryNegative => Opcode::UNARY_NEGATIVE,
            Instruction::UnaryNot => Opcode::UNARY_NOT,
            Instruction::UnaryInvert => Opcode::UNARY_INVERT,
            Instruction::GetIter => Opcode::GET_ITER,
            Instruction::GetYieldFromIter => Opcode::GET_YIELD_FROM_ITER,
            Instruction::BinaryPower => Opcode::BINARY_POWER,
            Instruction::BinaryMultiply => Opcode::BINARY_MULTIPLY,
            Instruction::BinaryMatrixMultiply => Opcode::BINARY_MATRIX_MULTIPLY,
            Instruction::BinaryFloorDivide => Opcode::BINARY_FLOOR_DIVIDE,
            Instruction::BinaryTrueDivide => Opcode::BINARY_TRUE_DIVIDE,
            Instruction::BinaryModulo => Opcode::BINARY_MODULO,
            Instruction::BinaryAdd => Opcode::BINARY_ADD,
            Instruction::BinarySubtract => Opcode::BINARY_SUBTRACT,
            Instruction::BinarySubscr => Opcode::BINARY_SUBSCR,
            Instruction::BinaryLshift => Opcode::BINARY_LSHIFT,
            Instruction::BinaryRshift => Opcode::BINARY_RSHIFT,
            Instruction::BinaryAnd => Opcode::BINARY_AND,
            Instruction::BinaryXor => Opcode::BINARY_XOR,
            Instruction::BinaryOr => Opcode::BINARY_OR,
            Instruction::InplacePower => Opcode::INPLACE_POWER,
            Instruction::InplaceMultiply => Opcode::INPLACE_MULTIPLY,
            Instruction::InplaceMatrixMultiply => Opcode::INPLACE_MATRIX_MULTIPLY,
            Instruction::InplaceFloorDivide => Opcode::INPLACE_FLOOR_DIVIDE,
            Instruction::InplaceTrueDivide => Opcode::INPLACE_TRUE_DIVIDE,
            Instruction::InplaceModulo => Opcode::INPLACE_MODULO,
            Instruction::InplaceAdd => Opcode::INPLACE_ADD,
            Instruction::InplaceSubtract => Opcode::INPLACE_SUBTRACT,
            Instruction::InplaceLshift => Opcode::INPLACE_LSHIFT,
            Instruction::InplaceRshift => Opcode::INPLACE_RSHIFT,
            Instruction::InplaceAnd => Opcode::INPLACE_AND,
            Instruction::InplaceXor => Opcode::INPLACE_XOR,
            Instruction::InplaceOr => Opcode::INPLACE_OR,
            Instruction::StoreSubscr => Opcode::STORE_SUBSCR,
            Instruction::DeleteSubscr => Opcode::DELETE_SUBSCR,
            Instruction::GetAwaitable => Opcode::GET_AWAITABLE,
            Instruction::GetAiter => Opcode::GET_AITER,
            Instruction::GetAnext => Opcode::GET_ANEXT,
            Instruction::EndAsyncFor => Opcode::END_ASYNC_FOR,
            Instruction::BeforeAsyncWith => Opcode::BEFORE_ASYNC_WITH,
            Instruction::SetupAsyncWith(_) => Opcode::SETUP_ASYNC_WITH,
            Instruction::PrintExpr => Opcode::PRINT_EXPR,
            Instruction::SetAdd(_) => Opcode::SET_ADD,
            Instruction::ListAppend(_) => Opcode::LIST_APPEND,
            Instruction::MapAdd(_) => Opcode::MAP_ADD,
            Instruction::ReturnValue => Opcode::RETURN_VALUE,
            Instruction::YieldValue => Opcode::YIELD_VALUE,
            Instruction::YieldFrom => Opcode::YIELD_FROM,
            Instruction::SetupAnnotations => Opcode::SETUP_ANNOTATIONS,
            Instruction::ImportStar => Opcode::IMPORT_STAR,
            Instruction::PopBlock => Opcode::POP_BLOCK,
            Instruction::PopExcept => Opcode::POP_EXCEPT,
            Instruction::Reraise(_) => Opcode::RERAISE,
            Instruction::WithExceptStart => Opcode::WITH_EXCEPT_START,
            Instruction::LoadAssertionError => Opcode::LOAD_ASSERTION_ERROR,
            Instruction::LoadBuildClass => Opcode::LOAD_BUILD_CLASS,
            Instruction::SetupWith(_) => Opcode::SETUP_WITH,
            Instruction::CopyDictWithoutKeys => Opcode::COPY_DICT_WITHOUT_KEYS,
            Instruction::GetLen => Opcode::GET_LEN,
            Instruction::MatchMapping => Opcode::MATCH_MAPPING,
            Instruction::MatchSequence => Opcode::MATCH_SEQUENCE,
            Instruction::MatchKeys => Opcode::MATCH_KEYS,
            Instruction::StoreName(_) => Opcode::STORE_NAME,
            Instruction::DeleteName(_) => Opcode::DELETE_NAME,
            Instruction::UnpackSequence(_) => Opcode::UNPACK_SEQUENCE,
            Instruction::UnpackEx(_) => Opcode::UNPACK_EX,
            Instruction::StoreAttr(_) => Opcode::STORE_ATTR,
            Instruction::DeleteAttr(_) => Opcode::DELETE_ATTR,
            Instruction::StoreGlobal(_) => Opcode::STORE_GLOBAL,
            Instruction::DeleteGlobal(_) => Opcode::DELETE_GLOBAL,
            Instruction::LoadConst(_) => Opcode::LOAD_CONST,
            Instruction::LoadName(_) => Opcode::LOAD_NAME,
            Instruction::BuildTuple(_) => Opcode::BUILD_TUPLE,
            Instruction::BuildList(_) => Opcode::BUILD_LIST,
            Instruction::BuildSet(_) => Opcode::BUILD_SET,
            Instruction::BuildMap(_) => Opcode::BUILD_MAP,
            Instruction::BuildConstKeyMap(_) => Opcode::BUILD_CONST_KEY_MAP,
            Instruction::BuildString(_) => Opcode::BUILD_STRING,
            Instruction::ListToTuple => Opcode::LIST_TO_TUPLE,
            Instruction::ListExtend(_) => Opcode::LIST_EXTEND,
            Instruction::SetUpdate(_) => Opcode::SET_UPDATE,
            Instruction::DictUpdate(_) => Opcode::DICT_UPDATE,
            Instruction::DictMerge(_) => Opcode::DICT_MERGE,
            Instruction::LoadAttr(_) => Opcode::LOAD_ATTR,
            Instruction::CompareOp(_) => Opcode::COMPARE_OP,
            Instruction::ImportName(_) => Opcode::IMPORT_NAME,
            Instruction::ImportFrom(_) => Opcode::IMPORT_FROM,
            Instruction::JumpForward(_) => Opcode::JUMP_FORWARD,
            Instruction::PopJumpIfTrue(_) => Opcode::POP_JUMP_IF_TRUE,
            Instruction::PopJumpIfFalse(_) => Opcode::POP_JUMP_IF_FALSE,
            Instruction::JumpIfNotExcMatch(_) => Opcode::JUMP_IF_NOT_EXC_MATCH,
            Instruction::JumpIfTrueOrPop(_) => Opcode::JUMP_IF_TRUE_OR_POP,
            Instruction::JumpIfFalseOrPop(_) => Opcode::JUMP_IF_FALSE_OR_POP,
            Instruction::JumpAbsolute(_) => Opcode::JUMP_ABSOLUTE,
            Instruction::ForIter(_) => Opcode::FOR_ITER,
            Instruction::LoadGlobal(_) => Opcode::LOAD_GLOBAL,
            Instruction::IsOp(_) => Opcode::IS_OP,
            Instruction::ContainsOp(_) => Opcode::CONTAINS_OP,
            Instruction::SetupFinally(_) => Opcode::SETUP_FINALLY,
            Instruction::LoadFast(_) => Opcode::LOAD_FAST,
            Instruction::StoreFast(_) => Opcode::STORE_FAST,
            Instruction::DeleteFast(_) => Opcode::DELETE_FAST,
            Instruction::LoadClosure(_) => Opcode::LOAD_CLOSURE,
            Instruction::LoadDeref(_) => Opcode::LOAD_DEREF,
            Instruction::LoadClassderef(_) => Opcode::LOAD_CLASSDEREF,
            Instruction::StoreDeref(_) => Opcode::STORE_DEREF,
            Instruction::DeleteDeref(_) => Opcode::DELETE_DEREF,
            Instruction::RaiseVarargs(_) => Opcode::RAISE_VARARGS,
            Instruction::CallFunction(_) => Opcode::CALL_FUNCTION,
            Instruction::CallFunctionKW(_) => Opcode::CALL_FUNCTION_KW,
            Instruction::CallFunctionEx(_) => Opcode::CALL_FUNCTION_EX,
            Instruction::LoadMethod(_) => Opcode::LOAD_METHOD,
            Instruction::CallMethod(_) => Opcode::CALL_METHOD,
            Instruction::MakeFunction(_) => Opcode::MAKE_FUNCTION,
            Instruction::BuildSlice(_) => Opcode::BUILD_SLICE,
            Instruction::FormatValue(_) => Opcode::FORMAT_VALUE,
            Instruction::MatchClass(_) => Opcode::MATCH_CLASS,
            Instruction::GenStart(_) => Opcode::GEN_START,
            Instruction::RotN(_) => Opcode::ROT_N,
        }
    }

    pub fn is_jump(&self) -> bool {
        self.get_opcode().is_jump()
    }

    pub fn is_absolute_jump(&self) -> bool {
        self.get_opcode().is_absolute_jump()
    }

    pub fn is_relative_jump(&self) -> bool {
        self.get_opcode().is_relative_jump()
    }

    fn get_raw_value(&self) -> u32 {
        match &self {
            Instruction::DupTop
            | Instruction::DupTopTwo
            | Instruction::UnaryPositive
            | Instruction::UnaryNegative
            | Instruction::UnaryNot
            | Instruction::UnaryInvert
            | Instruction::BinaryMatrixMultiply
            | Instruction::InplaceMatrixMultiply
            | Instruction::BinaryPower
            | Instruction::BinaryMultiply
            | Instruction::BinaryModulo
            | Instruction::BinaryAdd
            | Instruction::BinarySubtract
            | Instruction::BinarySubscr
            | Instruction::BinaryFloorDivide
            | Instruction::BinaryTrueDivide
            | Instruction::InplaceFloorDivide
            | Instruction::InplaceTrueDivide
            | Instruction::GetLen
            | Instruction::MatchMapping
            | Instruction::MatchSequence
            | Instruction::MatchKeys
            | Instruction::CopyDictWithoutKeys
            | Instruction::WithExceptStart
            | Instruction::GetAiter
            | Instruction::GetAnext
            | Instruction::BeforeAsyncWith
            | Instruction::EndAsyncFor
            | Instruction::InplaceAdd
            | Instruction::InplaceSubtract
            | Instruction::InplaceMultiply
            | Instruction::InplaceModulo
            | Instruction::StoreSubscr
            | Instruction::DeleteSubscr
            | Instruction::BinaryLshift
            | Instruction::BinaryRshift
            | Instruction::BinaryAnd
            | Instruction::BinaryXor
            | Instruction::BinaryOr
            | Instruction::InplacePower
            | Instruction::GetIter
            | Instruction::GetYieldFromIter
            | Instruction::PrintExpr
            | Instruction::LoadBuildClass
            | Instruction::YieldFrom
            | Instruction::GetAwaitable
            | Instruction::LoadAssertionError
            | Instruction::InplaceLshift
            | Instruction::InplaceRshift
            | Instruction::InplaceAnd
            | Instruction::InplaceXor
            | Instruction::InplaceOr
            | Instruction::ListToTuple
            | Instruction::ReturnValue
            | Instruction::ImportStar
            | Instruction::SetupAnnotations
            | Instruction::YieldValue
            | Instruction::PopBlock
            | Instruction::PopExcept => 0,
            Instruction::StoreName(name_index)
            | Instruction::DeleteName(name_index)
            | Instruction::StoreAttr(name_index)
            | Instruction::DeleteAttr(name_index)
            | Instruction::StoreGlobal(name_index)
            | Instruction::DeleteGlobal(name_index)
            | Instruction::LoadName(name_index)
            | Instruction::LoadAttr(name_index)
            | Instruction::ImportName(name_index)
            | Instruction::ImportFrom(name_index)
            | Instruction::LoadGlobal(name_index)
            | Instruction::LoadMethod(name_index) => name_index.index,
            Instruction::PopTop(n)
            | Instruction::RotTwo(n)
            | Instruction::RotThree(n)
            | Instruction::RotFour(n)
            | Instruction::Nop(n)
            | Instruction::UnpackSequence(n)
            | Instruction::UnpackEx(n)
            | Instruction::RotN(n)
            | Instruction::BuildTuple(n)
            | Instruction::BuildList(n)
            | Instruction::BuildSet(n)
            | Instruction::BuildMap(n)
            | Instruction::CallFunction(n)
            | Instruction::BuildSlice(n)
            | Instruction::CallFunctionKW(n)
            | Instruction::ListAppend(n)
            | Instruction::SetAdd(n)
            | Instruction::MapAdd(n)
            | Instruction::MatchClass(n)
            | Instruction::BuildConstKeyMap(n)
            | Instruction::BuildString(n)
            | Instruction::CallMethod(n)
            | Instruction::ListExtend(n)
            | Instruction::SetUpdate(n)
            | Instruction::DictUpdate(n)
            | Instruction::DictMerge(n) => *n,
            Instruction::ForIter(jump)
            | Instruction::JumpForward(jump)
            | Instruction::SetupFinally(jump)
            | Instruction::SetupWith(jump)
            | Instruction::SetupAsyncWith(jump) => jump.index,
            Instruction::LoadConst(const_index) => const_index.index,
            Instruction::CompareOp(comp_op) => Into::<u32>::into(comp_op),
            Instruction::JumpIfFalseOrPop(jump)
            | Instruction::JumpIfTrueOrPop(jump)
            | Instruction::JumpAbsolute(jump)
            | Instruction::PopJumpIfFalse(jump)
            | Instruction::PopJumpIfTrue(jump)
            | Instruction::JumpIfNotExcMatch(jump) => jump.index,
            Instruction::Reraise(forms) => Into::<u32>::into(forms),
            Instruction::IsOp(op_inv) | Instruction::ContainsOp(op_inv) => {
                Into::<u32>::into(op_inv)
            }
            Instruction::LoadFast(varname_index)
            | Instruction::StoreFast(varname_index)
            | Instruction::DeleteFast(varname_index) => varname_index.index,
            Instruction::GenStart(kind) => Into::<u32>::into(kind),
            Instruction::RaiseVarargs(form) => Into::<u32>::into(form),
            Instruction::MakeFunction(flags) => flags.bits(),
            Instruction::LoadClosure(closure_ref_index)
            | Instruction::LoadDeref(closure_ref_index)
            | Instruction::StoreDeref(closure_ref_index)
            | Instruction::DeleteDeref(closure_ref_index)
            | Instruction::LoadClassderef(closure_ref_index) => closure_ref_index.index,
            Instruction::CallFunctionEx(flags) => Into::<u32>::into(flags),
            Instruction::FormatValue(format_flag) => Into::<u32>::into(format_flag),
        }
    }
}

impl From<Instruction> for u8 {
    fn from(val: Instruction) -> Self {
        val.get_opcode() as u8
    }
}

#[derive(Debug, Clone)]
pub struct Pyc {
    pub python_version: python_marshal::magic::PyVersion,
    pub timestamp: u32,
    pub hash: u64,
    pub code_object: Code,
}

impl TryFrom<python_marshal::PycFile> for Pyc {
    type Error = Error;

    fn try_from(pyc: python_marshal::PycFile) -> Result<Self, Self::Error> {
        Ok(Pyc {
            python_version: pyc.python_version,
            timestamp: pyc.timestamp.ok_or(Error::WrongVersion)?,
            hash: pyc.hash,
            code_object: (pyc.object, pyc.references).try_into()?,
        })
    }
}

impl From<PycFile> for python_marshal::PycFile {
    fn from(val: PycFile) -> Self {
        match val.clone() {
            PycFile::V310(pyc) => {
                python_marshal::PycFile {
                    python_version: pyc.python_version,
                    timestamp: Some(pyc.timestamp),
                    hash: pyc.hash,
                    object: python_marshal::Object::Code(pyc.code_object.into()),
                    references: Vec::new(), // All references are resolved in this editor.
                }
            }
        }
    }
}
