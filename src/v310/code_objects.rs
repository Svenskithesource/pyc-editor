use bitflags::bitflags;

use hashable::HashableHashSet;
use num_bigint::BigInt;
use num_complex::Complex;
use ordered_float::OrderedFloat;
use python_marshal::{extract_object, resolver::resolve_all_refs, CodeFlags, PyString};

use crate::error::Error;

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

#[derive(Debug, Clone)]
pub enum Constant {
    FrozenConstant(FrozenConstant),
    CodeObject(Code),
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
            python_marshal::Object::Float(f) => {
                Ok(FrozenConstant::Float(ordered_float::OrderedFloat(f)))
            }
            python_marshal::Object::Complex(c) => Ok(FrozenConstant::Complex(Complex {
                re: OrderedFloat(c.re),
                im: OrderedFloat(c.im),
            })),
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

impl TryFrom<python_marshal::Object> for Constant {
    type Error = Error;

    fn try_from(value: python_marshal::Object) -> Result<Self, Self::Error> {
        match value {
            python_marshal::Object::Code(code) => match *code {
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
#[derive(Debug, Clone)]
pub struct Code {
    pub argcount: u32,
    pub posonlyargcount: u32,
    pub kwonlyargcount: u32,
    pub nlocals: u32,
    pub stacksize: u32,
    pub flags: CodeFlags,
    pub code: Vec<Instruction>,
    pub consts: Vec<Constant>,
    pub names: Vec<PyString>,
    pub varnames: Vec<PyString>,
    pub freevars: Vec<PyString>,
    pub cellvars: Vec<PyString>,
    pub filename: PyString,
    pub name: PyString,
    pub firstlineno: u32,
    pub lnotab: Vec<u8>,
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

fn bytecode_to_instructions(code: &[u8]) -> Result<Vec<Instruction>, Error> {
    if code.len() % 2 != 0 {
        return Err(Error::InvalidBytecodeLength);
    }

    let mut instructions = Vec::with_capacity(code.len() / 2);
    let mut extended_arg = 0; // Used to keep track of extended arguments

    for chunk in code.chunks(2) {
        if chunk.len() != 2 {
            return Err(Error::InvalidBytecodeLength);
        }
        let opcode = Opcode::try_from(chunk[0])?;
        let arg = chunk[1];

        match opcode {
            Opcode::EXTENDED_ARG => {
                extended_arg = (extended_arg << 8) | arg as u32;
                continue;
            }
            _ => {
                let arg = (extended_arg << 8) | arg as u32;

                instructions.push((opcode, arg).into());
            }
        }

        extended_arg = 0;
    }

    Ok(instructions)
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
            code: bytecode_to_instructions(&co_code)?,
            consts: co_consts
                .iter()
                .map(|obj| Constant::try_from(obj.clone()))
                .collect::<Result<Vec<_>, _>>()?,
            names: co_names.iter().map(|obj| obj.clone()).collect(),
            varnames: co_varnames.iter().map(|obj| obj.clone()).collect(),
            freevars: co_freevars.iter().map(|obj| obj.clone()).collect(),
            cellvars: co_cellvars.iter().map(|obj| obj.clone()).collect(),
            filename: co_filename.clone(),
            name: co_name.clone(),
            firstlineno: code.firstlineno,
            lnotab: co_lnotab.to_vec(),
        })
    }
}

/// Represents a relative jump offset from the current instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelativeJump {
    index: u32,
}

/// Represents an absolute jump target (a byte offset from the start of the code).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbsoluteJump {
    index: u32,
}

/// Holds an index into co_names. Has helper functions to get the actual PyString of the name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameIndex {
    index: u32,
}

impl NameIndex {
    pub fn get<'a>(&self, co_names: &'a [PyString]) -> Option<&'a PyString> {
        co_names.get(self.index as usize)
    }
}

/// Holds an index into co_varnames. Has helper functions to get the actual PyString of the name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarNameIndex {
    index: u32,
}

impl VarNameIndex {
    pub fn get<'a>(&self, co_varnames: &'a [PyString]) -> Option<&'a PyString> {
        co_varnames.get(self.index as usize)
    }
}

/// Holds an index into co_consts. Has helper functions to get the actual constant at the index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstIndex {
    index: u32,
}

impl ConstIndex {
    pub fn get<'a>(&self, co_consts: &'a [Constant]) -> Option<&'a Constant> {
        co_consts.get(self.index as usize)
    }
}

/// Represents a resolved reference to a variable in the cell or free variable storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureRefIndex {
    index: u32,
}

/// Represents a resolved reference to a variable in the cell or free variable storage.
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Whether *_OP is inverted or not
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// THe different types of raising forms. See https://docs.python.org/3.10/library/dis.html#opcode-RAISE_VARARGS
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Describes the configuration for a CALL_FUNCTION_EX instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Represents the conversion to apply to a value before f-string formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatFlag {
    /// No conversion, format the value as-is.
    NoConversion,
    /// Convert the value using `str()`.
    Str,
    /// Convert the value using `repr()`.
    Repr,
    /// Convert the value using `ascii()`.
    Ascii,
    /// Use special fmt_spec (see https://docs.python.org/3.10/library/dis.html#opcode-FORMAT_VALUE)
    FmtSpec,
    Invalid(u32),
}

impl From<u32> for FormatFlag {
    fn from(value: u32) -> Self {
        match value & 0x03 {
            0x00 => Self::NoConversion,
            0x01 => Self::Str,
            0x02 => Self::Repr,
            0x03 => Self::Ascii,
            0x04 => Self::FmtSpec,
            _ => Self::Invalid(value),
        }
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

/// Low level representation of a Python bytecode instruction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    Nop,
    PopTop,
    RotTwo,
    RotThree,
    RotFour,
    DupTop,
    DupTopTwo,
    UnaryPositive,
    UnaryNegative,
    UnaryNot,
    UnaryInvert,
    GetIter,
    GetYieldFromIter,
    BinaryPower,
    BinaryMultiply,
    BinaryMatrixMultiply,
    BinaryFloorDivide,
    BinaryTrueDivide,
    BinaryModulo,
    BinaryAdd,
    BinarySubtract,
    BinarySubscr,
    BinaryLshift,
    BinaryRshift,
    BinaryAnd,
    BinaryXor,
    BinaryOr,
    InplacePower,
    InplaceMultiply,
    InplaceMatrixMultiply,
    InplaceFloorDivide,
    InplaceTrueDivide,
    InplaceModulo,
    InplaceAdd,
    InplaceSubtract,
    InplaceLshift,
    InplaceRshift,
    InplaceAnd,
    InplaceXor,
    InplaceOr,
    StoreSubscr,
    DeleteSubscr,
    GetAwaitable,
    GetAiter,
    GetAnext,
    EndAsyncFor,
    BeforeAsyncWith,
    SetupAsyncWith,
    PrintExpr,
    SetAdd(u32),
    ListAppend(u32),
    MapAdd(u32),
    ReturnValue,
    YieldValue,
    YieldFrom,
    SetupAnnotations,
    ImportStar,
    PopBlock,
    PopExcept,
    Reraise,
    WithExceptStart,
    LoadAssertionError,
    LoadBuildClass,
    SetupWith(RelativeJump),
    CopyDictWithoutKeys,
    GetLen,
    MatchMapping,
    MatchSequence,
    MatchKeys,
    StoreName(NameIndex),
    DeleteName(NameIndex),
    UnpackSequence(u32),
    UnpackEx(u32),
    StoreAttr(NameIndex),
    DeleteAttr(NameIndex),
    StoreGlobal(NameIndex),
    DeleteGlobal(NameIndex),
    LoadConst(ConstIndex),
    LoadName(NameIndex),
    BuildTuple(u32),
    BuildList(u32),
    BuildSet(u32),
    BuildMap(u32),
    BuildConstKeyMap(u32),
    BuildString(u32),
    ListToTuple,
    ListExtend(u32),
    SetUpdate(u32),
    DictUpdate(u32),
    DictMerge,
    LoadAttr(NameIndex),
    CompareOp(CompareOperation),
    ImportName(NameIndex),
    ImportFrom(NameIndex),
    JumpForward(RelativeJump),
    PopJumpIfTrue(AbsoluteJump),
    PopJumpIfFalse(AbsoluteJump),
    JumpIfNotExcMatch(AbsoluteJump),
    JumpIfTrueOrPop(AbsoluteJump),
    JumpIfFalseOrPop(AbsoluteJump),
    JumpAbsolute(AbsoluteJump),
    ForIter(RelativeJump),
    LoadGlobal(NameIndex),
    IsOp(OpInversion),
    ContainsOp(OpInversion),
    SetupFinally(RelativeJump),
    LoadFast(VarNameIndex),
    StoreFast(VarNameIndex),
    DeleteFast(VarNameIndex),
    LoadClosure(ClosureRefIndex),
    LoadDeref(ClosureRefIndex),
    LoadClassderef(ClosureRefIndex),
    StoreDeref(ClosureRefIndex),
    DeleteDeref(ClosureRefIndex),
    RaiseVarargs(RaiseForms),
    CallFunction(u32),
    CallFunctionKW(u32),
    CallFunctionEx(CallExFlags),
    LoadMethod(NameIndex),
    CallMethod(u32),
    MakeFunction(MakeFunctionFlags),
    BuildSlice(u32),
    // ExtendedArg is skipped as it's integrated into the next instruction
    FormatValue(FormatFlag),
    MatchClass(u32),
    GenStart(GenKind),
    RotN(u32),
}

impl From<(Opcode, u32)> for Instruction {
    fn from(value: (Opcode, u32)) -> Self {
        match value.0 {
            Opcode::NOP => Instruction::Nop,
            Opcode::POP_TOP => Instruction::PopTop,
            Opcode::ROT_TWO => Instruction::RotTwo,
            Opcode::ROT_THREE => Instruction::RotThree,
            Opcode::ROT_FOUR => Instruction::RotFour,
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
            Opcode::SETUP_ASYNC_WITH => Instruction::SetupAsyncWith,
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
            Opcode::RERAISE => Instruction::Reraise,
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
            Opcode::DICT_MERGE => Instruction::DictMerge,
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

#[derive(Debug, Clone)]
pub struct Pyc {
    pub python_version: python_marshal::magic::PyVersion,
    pub timestamp: u32,
    pub hash: u64,
    pub code: Code,
}

impl TryFrom<python_marshal::PycFile> for Pyc {
    type Error = Error;

    fn try_from(pyc: python_marshal::PycFile) -> Result<Self, Self::Error> {
        if pyc.python_version.major != 3 || pyc.python_version.minor != 10 {
            Err(Error::WrongVersion)
        } else {
            let (code_object, refs) = resolve_all_refs(pyc.object, pyc.references)?;

            if !refs.is_empty() {
                return Err(Error::RecursiveReference(
                    "This pyc file contains references that cannot be resolved. This should never happen on a valid pyc file generated by Python.",
                ));
            }

            let code_object = extract_object!(Some(code_object), python_marshal::Object::Code(code) => code, python_marshal::error::Error::UnexpectedObject)?;

            match *code_object {
                python_marshal::Code::V310(code) => {
                    let code = Code::try_from(code)?;
                    Ok(Pyc {
                        python_version: pyc.python_version,
                        timestamp: pyc.timestamp.ok_or(Error::WrongVersion)?,
                        hash: pyc.hash,
                        code,
                    })
                }
                _ => Err(Error::WrongVersion),
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Argument {
    /// Positional-only argument (defined before `/`)
    PositionalOnly(String),

    /// Positional or keyword argument (defined before `*`)
    PositionalOrKeyword(String),

    /// Keyword-only argument (defined after `*`)
    KeywordOnly(String),
}
