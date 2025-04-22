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
    pub code: Vec<Instruction>, // Needs to contain Vec<u8> as a value or a reference
    pub consts: Vec<Constant>,  // Needs to contain Vec<Object> as a value or a reference
    pub names: Vec<PyString>,   // Needs to contain Vec<PyString> as a value or a reference
    pub varnames: Vec<PyString>, // Needs to contain Vec<PyString> as a value or a reference
    pub freevars: Vec<PyString>, // Needs to contain Vec<PyString> as a value or a reference
    pub cellvars: Vec<PyString>, // Needs to contain Vec<PyString> as a value or a reference
    pub filename: PyString,     // Needs to contain PyString as a value or a reference
    pub name: PyString,         // Needs to contain PyString as a value or a reference
    pub firstlineno: u32,
    pub lnotab: Vec<u8>, // Needs to contain Vec<u8>, as a value or a reference
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

// Low level representation of a Python bytecode instruction
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
    SetupWith(u32),
    CopyDictWithoutKeys,
    GetLen,
    MatchMapping,
    MatchSequence,
    MatchKeys,
    StoreName(u32),
    DeleteName(u32),
    UnpackSequence(u32),
    UnpackEx(u32),
    StoreAttr(u32),
    DeleteAttr(u32),
    StoreGlobal(u32),
    DeleteGlobal(u32),
    LoadConst(u32),
    LoadName(u32),
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
    LoadAttr(u32),
    CompareOp(u32),
    ImportName(u32),
    ImportFrom(u32),
    JumpForward(u32),
    PopJumpIfTrue(u32),
    PopJumpIfFalse(u32),
    JumpIfNotExcMatch(u32),
    JumpIfTrueOrPop(u32),
    JumpIfFalseOrPop(u32),
    JumpAbsolute(u32),
    ForIter(u32),
    LoadGlobal(u32),
    IsOp(u32),
    ContainsOp(u32),
    SetupFinally(u32),
    LoadFast(u32),
    StoreFast(u32),
    DeleteFast(u32),
    LoadClosure(u32),
    LoadDeref(u32),
    LoadClassderef(u32),
    StoreDeref(u32),
    DeleteDeref(u32),
    RaiseVarargs(u32),
    CallFunction(u32),
    CallFunctionKW(u32),
    CallFunctionEx(u32),
    LoadMethod(u32),
    CallMethod(u32),
    MakeFunction(u32),
    BuildSlice(u32),
    // ExtendedArg is skipped as it's integrated into the next instruction
    FormatValue(u32),
    MatchClass(u32),
    GenStart(u32),
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
            Opcode::SETUP_WITH => Instruction::SetupWith(value.1),
            Opcode::COPY_DICT_WITHOUT_KEYS => Instruction::CopyDictWithoutKeys,
            Opcode::GET_LEN => Instruction::GetLen,
            Opcode::MATCH_MAPPING => Instruction::MatchMapping,
            Opcode::MATCH_SEQUENCE => Instruction::MatchSequence,
            Opcode::MATCH_KEYS => Instruction::MatchKeys,
            Opcode::STORE_NAME => Instruction::StoreName(value.1),
            Opcode::DELETE_NAME => Instruction::DeleteName(value.1),
            Opcode::UNPACK_SEQUENCE => Instruction::UnpackSequence(value.1),
            Opcode::UNPACK_EX => Instruction::UnpackEx(value.1),
            Opcode::STORE_ATTR => Instruction::StoreAttr(value.1),
            Opcode::DELETE_ATTR => Instruction::DeleteAttr(value.1),
            Opcode::STORE_GLOBAL => Instruction::StoreGlobal(value.1),
            Opcode::DELETE_GLOBAL => Instruction::DeleteGlobal(value.1),
            Opcode::LOAD_CONST => Instruction::LoadConst(value.1),
            Opcode::LOAD_NAME => Instruction::LoadName(value.1),
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
            Opcode::LOAD_ATTR => Instruction::LoadAttr(value.1),
            Opcode::COMPARE_OP => Instruction::CompareOp(value.1),
            Opcode::IMPORT_NAME => Instruction::ImportName(value.1),
            Opcode::IMPORT_FROM => Instruction::ImportFrom(value.1),
            Opcode::JUMP_FORWARD => Instruction::JumpForward(value.1),
            Opcode::POP_JUMP_IF_TRUE => Instruction::PopJumpIfTrue(value.1),
            Opcode::POP_JUMP_IF_FALSE => Instruction::PopJumpIfFalse(value.1),
            Opcode::JUMP_IF_NOT_EXC_MATCH => Instruction::JumpIfNotExcMatch(value.1),
            Opcode::JUMP_IF_TRUE_OR_POP => Instruction::JumpIfTrueOrPop(value.1),
            Opcode::JUMP_IF_FALSE_OR_POP => Instruction::JumpIfFalseOrPop(value.1),
            Opcode::JUMP_ABSOLUTE => Instruction::JumpAbsolute(value.1),
            Opcode::FOR_ITER => Instruction::ForIter(value.1),
            Opcode::LOAD_GLOBAL => Instruction::LoadGlobal(value.1),
            Opcode::IS_OP => Instruction::IsOp(value.1),
            Opcode::CONTAINS_OP => Instruction::ContainsOp(value.1),
            Opcode::SETUP_FINALLY => Instruction::SetupFinally(value.1),
            Opcode::LOAD_FAST => Instruction::LoadFast(value.1),
            Opcode::STORE_FAST => Instruction::StoreFast(value.1),
            Opcode::DELETE_FAST => Instruction::DeleteFast(value.1),
            Opcode::LOAD_CLOSURE => Instruction::LoadClosure(value.1),
            Opcode::LOAD_DEREF => Instruction::LoadDeref(value.1),
            Opcode::LOAD_CLASSDEREF => Instruction::LoadClassderef(value.1),
            Opcode::STORE_DEREF => Instruction::StoreDeref(value.1),
            Opcode::DELETE_DEREF => Instruction::DeleteDeref(value.1),
            Opcode::RAISE_VARARGS => Instruction::RaiseVarargs(value.1),
            Opcode::CALL_FUNCTION => Instruction::CallFunction(value.1),
            Opcode::CALL_FUNCTION_KW => Instruction::CallFunctionKW(value.1),
            Opcode::CALL_FUNCTION_EX => Instruction::CallFunctionEx(value.1),
            Opcode::LOAD_METHOD => Instruction::LoadMethod(value.1),
            Opcode::CALL_METHOD => Instruction::CallMethod(value.1),
            Opcode::MAKE_FUNCTION => Instruction::MakeFunction(value.1),
            Opcode::BUILD_SLICE => Instruction::BuildSlice(value.1),
            Opcode::FORMAT_VALUE => Instruction::FormatValue(value.1),
            Opcode::MATCH_CLASS => Instruction::MatchClass(value.1),
            Opcode::GEN_START => Instruction::GenStart(value.1),
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

            assert!(
                refs.is_empty(),
                "There are still unresolved references in the code object."
            );

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
