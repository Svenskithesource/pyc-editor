use std::ops::{Deref, DerefMut};

use crate::{
    error::Error,
    v310::{ext_instructions::ExtInstructions, opcodes::Opcode},
};

/// Low level representation of a Python bytecode instruction with their original u8 argument.
/// We have arguments for every opcode, even if those aren't used. This is so we can have a full representation of the instructions, even if they're invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    PopTop(u8),
    RotTwo(u8),
    RotThree(u8),
    DupTop(u8),
    DupTopTwo(u8),
    RotFour(u8),
    /// Version 3.10 has an unique bug where some NOPs are left with an arg. See https://github.com/python/cpython/issues/89918#issuecomment-1093937041
    Nop(u8),
    UnaryPositive(u8),
    UnaryNegative(u8),
    UnaryNot(u8),
    UnaryInvert(u8),
    BinaryMatrixMultiply(u8),
    InplaceMatrixMultiply(u8),
    BinaryPower(u8),
    BinaryMultiply(u8),
    BinaryModulo(u8),
    BinaryAdd(u8),
    BinarySubtract(u8),
    BinarySubscr(u8),
    BinaryFloorDivide(u8),
    BinaryTrueDivide(u8),
    InplaceFloorDivide(u8),
    InplaceTrueDivide(u8),
    GetLen(u8),
    MatchMapping(u8),
    MatchSequence(u8),
    MatchKeys(u8),
    CopyDictWithoutKeys(u8),
    WithExceptStart(u8),
    GetAiter(u8),
    GetAnext(u8),
    BeforeAsyncWith(u8),
    EndAsyncFor(u8),
    InplaceAdd(u8),
    InplaceSubtract(u8),
    InplaceMultiply(u8),
    InplaceModulo(u8),
    StoreSubscr(u8),
    DeleteSubscr(u8),
    BinaryLshift(u8),
    BinaryRshift(u8),
    BinaryAnd(u8),
    BinaryXor(u8),
    BinaryOr(u8),
    InplacePower(u8),
    GetIter(u8),
    GetYieldFromIter(u8),
    PrintExpr(u8),
    LoadBuildClass(u8),
    YieldFrom(u8),
    GetAwaitable(u8),
    LoadAssertionError(u8),
    InplaceLshift(u8),
    InplaceRshift(u8),
    InplaceAnd(u8),
    InplaceXor(u8),
    InplaceOr(u8),
    ListToTuple(u8),
    ReturnValue(u8),
    ImportStar(u8),
    SetupAnnotations(u8),
    YieldValue(u8),
    PopBlock(u8),
    PopExcept(u8),
    StoreName(u8),
    DeleteName(u8),
    UnpackSequence(u8),
    ForIter(u8),
    UnpackEx(u8),
    StoreAttr(u8),
    DeleteAttr(u8),
    StoreGlobal(u8),
    DeleteGlobal(u8),
    RotN(u8),
    LoadConst(u8),
    LoadName(u8),
    BuildTuple(u8),
    BuildList(u8),
    BuildSet(u8),
    BuildMap(u8),
    LoadAttr(u8),
    CompareOp(u8),
    ImportName(u8),
    ImportFrom(u8),
    JumpForward(u8),
    JumpIfFalseOrPop(u8),
    JumpIfTrueOrPop(u8),
    JumpAbsolute(u8),
    PopJumpIfFalse(u8),
    PopJumpIfTrue(u8),
    LoadGlobal(u8),
    IsOp(u8),
    ContainsOp(u8),
    Reraise(u8),
    JumpIfNotExcMatch(u8),
    SetupFinally(u8),
    LoadFast(u8),
    StoreFast(u8),
    DeleteFast(u8),
    GenStart(u8),
    RaiseVarargs(u8),
    CallFunction(u8),
    MakeFunction(u8),
    BuildSlice(u8),
    LoadClosure(u8),
    LoadDeref(u8),
    StoreDeref(u8),
    DeleteDeref(u8),
    CallFunctionKW(u8),
    CallFunctionEx(u8),
    SetupWith(u8),
    ExtendedArg(u8),
    ListAppend(u8),
    SetAdd(u8),
    MapAdd(u8),
    LoadClassderef(u8),
    MatchClass(u8),
    SetupAsyncWith(u8),
    FormatValue(u8),
    BuildConstKeyMap(u8),
    BuildString(u8),
    LoadMethod(u8),
    CallMethod(u8),
    ListExtend(u8),
    SetUpdate(u8),
    DictMerge(u8),
    DictUpdate(u8),
}

/// A list of instructions
#[derive(Debug, Clone, PartialEq)]
pub struct Instructions(Vec<Instruction>);

impl Instructions {
    pub fn with_capacity(capacity: usize) -> Self {
        Instructions(Vec::with_capacity(capacity))
    }

    pub fn new(instructions: Vec<Instruction>) -> Self {
        Instructions(instructions)
    }

    /// Calculates the full argument for an index (keeping in mind extended args). None if the index is not within bounds.
    /// NOTE: If there is a jump skipping the extended arg(s) before this instruction, this will return an incorrect value.
    pub fn get_full_arg(&self, index: usize) -> Option<u32> {
        if self.len() > index {
            let mut curr_index = index;
            let mut extended_args = vec![];

            while curr_index > 0 {
                curr_index -= 1;

                match &self[curr_index] {
                    Instruction::ExtendedArg(arg) => {
                        extended_args.push(arg);
                    }
                    _ => break,
                }
            }

            let mut extended_arg = 0;

            for arg in extended_args.iter().rev() {
                // We collected them in the reverse order
                let arg = **arg as u32 | extended_arg;
                extended_arg = arg << 8;
            }

            Some(self[index].get_raw_value() as u32 | extended_arg)
        } else {
            None
        }
    }

    /// Returns the instructions but with the extended_args resolved
    pub fn to_resolved(&self) -> ExtInstructions {
        ExtInstructions::from(self.0.as_slice())
    }

    pub fn append_instructions(&mut self, instructions: &[Instruction]) {
        for instruction in instructions {
            self.0.push(*instruction);
        }
    }

    /// Append an instruction at the end
    pub fn append_instruction(&mut self, instruction: Instruction) {
        self.0.push(instruction);
    }

    pub fn get_instructions(&self) -> &[Instruction] {
        self.deref()
    }

    pub fn get_instructions_mut(&mut self) -> &mut [Instruction] {
        self.deref_mut()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytearray = Vec::with_capacity(self.0.len() * 2);

        for instruction in self.0.iter() {
            bytearray.push(instruction.get_opcode() as u8);
            bytearray.push(instruction.get_raw_value())
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

        for chunk in code.chunks(2) {
            if chunk.len() != 2 {
                return Err(Error::InvalidBytecodeLength);
            }
            let opcode = Opcode::try_from(chunk[0])?;
            let arg = chunk[1];

            instructions.append_instruction((opcode, arg).into());
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

impl From<(Opcode, u8)> for Instruction {
    fn from(value: (Opcode, u8)) -> Self {
        match value.0 {
            Opcode::NOP => Instruction::Nop(value.1),
            Opcode::POP_TOP => Instruction::PopTop(value.1),
            Opcode::ROT_TWO => Instruction::RotTwo(value.1),
            Opcode::ROT_THREE => Instruction::RotThree(value.1),
            Opcode::ROT_FOUR => Instruction::RotFour(value.1),
            Opcode::DUP_TOP => Instruction::DupTop(value.1),
            Opcode::DUP_TOP_TWO => Instruction::DupTopTwo(value.1),
            Opcode::UNARY_POSITIVE => Instruction::UnaryPositive(value.1),
            Opcode::UNARY_NEGATIVE => Instruction::UnaryNegative(value.1),
            Opcode::UNARY_NOT => Instruction::UnaryNot(value.1),
            Opcode::UNARY_INVERT => Instruction::UnaryInvert(value.1),
            Opcode::GET_ITER => Instruction::GetIter(value.1),
            Opcode::GET_YIELD_FROM_ITER => Instruction::GetYieldFromIter(value.1),
            Opcode::BINARY_POWER => Instruction::BinaryPower(value.1),
            Opcode::BINARY_MULTIPLY => Instruction::BinaryMultiply(value.1),
            Opcode::BINARY_MATRIX_MULTIPLY => Instruction::BinaryMatrixMultiply(value.1),
            Opcode::BINARY_FLOOR_DIVIDE => Instruction::BinaryFloorDivide(value.1),
            Opcode::BINARY_TRUE_DIVIDE => Instruction::BinaryTrueDivide(value.1),
            Opcode::BINARY_MODULO => Instruction::BinaryModulo(value.1),
            Opcode::BINARY_ADD => Instruction::BinaryAdd(value.1),
            Opcode::BINARY_SUBTRACT => Instruction::BinarySubtract(value.1),
            Opcode::BINARY_SUBSCR => Instruction::BinarySubscr(value.1),
            Opcode::BINARY_LSHIFT => Instruction::BinaryLshift(value.1),
            Opcode::BINARY_RSHIFT => Instruction::BinaryRshift(value.1),
            Opcode::BINARY_AND => Instruction::BinaryAnd(value.1),
            Opcode::BINARY_XOR => Instruction::BinaryXor(value.1),
            Opcode::BINARY_OR => Instruction::BinaryOr(value.1),
            Opcode::INPLACE_POWER => Instruction::InplacePower(value.1),
            Opcode::INPLACE_MULTIPLY => Instruction::InplaceMultiply(value.1),
            Opcode::INPLACE_MATRIX_MULTIPLY => Instruction::InplaceMatrixMultiply(value.1),
            Opcode::INPLACE_FLOOR_DIVIDE => Instruction::InplaceFloorDivide(value.1),
            Opcode::INPLACE_TRUE_DIVIDE => Instruction::InplaceTrueDivide(value.1),
            Opcode::INPLACE_MODULO => Instruction::InplaceModulo(value.1),
            Opcode::INPLACE_ADD => Instruction::InplaceAdd(value.1),
            Opcode::INPLACE_SUBTRACT => Instruction::InplaceSubtract(value.1),
            Opcode::INPLACE_LSHIFT => Instruction::InplaceLshift(value.1),
            Opcode::INPLACE_RSHIFT => Instruction::InplaceRshift(value.1),
            Opcode::INPLACE_AND => Instruction::InplaceAnd(value.1),
            Opcode::INPLACE_XOR => Instruction::InplaceXor(value.1),
            Opcode::INPLACE_OR => Instruction::InplaceOr(value.1),
            Opcode::STORE_SUBSCR => Instruction::StoreSubscr(value.1),
            Opcode::DELETE_SUBSCR => Instruction::DeleteSubscr(value.1),
            Opcode::GET_AWAITABLE => Instruction::GetAwaitable(value.1),
            Opcode::GET_AITER => Instruction::GetAiter(value.1),
            Opcode::GET_ANEXT => Instruction::GetAnext(value.1),
            Opcode::END_ASYNC_FOR => Instruction::EndAsyncFor(value.1),
            Opcode::BEFORE_ASYNC_WITH => Instruction::BeforeAsyncWith(value.1),
            Opcode::SETUP_ASYNC_WITH => Instruction::SetupAsyncWith(value.1),
            Opcode::PRINT_EXPR => Instruction::PrintExpr(value.1),
            Opcode::SET_ADD => Instruction::SetAdd(value.1),
            Opcode::LIST_APPEND => Instruction::ListAppend(value.1),
            Opcode::MAP_ADD => Instruction::MapAdd(value.1),
            Opcode::RETURN_VALUE => Instruction::ReturnValue(value.1),
            Opcode::YIELD_VALUE => Instruction::YieldValue(value.1),
            Opcode::YIELD_FROM => Instruction::YieldFrom(value.1),
            Opcode::SETUP_ANNOTATIONS => Instruction::SetupAnnotations(value.1),
            Opcode::IMPORT_STAR => Instruction::ImportStar(value.1),
            Opcode::POP_BLOCK => Instruction::PopBlock(value.1),
            Opcode::POP_EXCEPT => Instruction::PopExcept(value.1),
            Opcode::RERAISE => Instruction::Reraise(value.1),
            Opcode::WITH_EXCEPT_START => Instruction::WithExceptStart(value.1),
            Opcode::LOAD_ASSERTION_ERROR => Instruction::LoadAssertionError(value.1),
            Opcode::LOAD_BUILD_CLASS => Instruction::LoadBuildClass(value.1),
            Opcode::SETUP_WITH => Instruction::SetupWith(value.1),
            Opcode::COPY_DICT_WITHOUT_KEYS => Instruction::CopyDictWithoutKeys(value.1),
            Opcode::GET_LEN => Instruction::GetLen(value.1),
            Opcode::MATCH_MAPPING => Instruction::MatchMapping(value.1),
            Opcode::MATCH_SEQUENCE => Instruction::MatchSequence(value.1),
            Opcode::MATCH_KEYS => Instruction::MatchKeys(value.1),
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
            Opcode::LIST_TO_TUPLE => Instruction::ListToTuple(value.1),
            Opcode::LIST_EXTEND => Instruction::ListExtend(value.1),
            Opcode::SET_UPDATE => Instruction::SetUpdate(value.1),
            Opcode::DICT_UPDATE => Instruction::DictUpdate(value.1),
            Opcode::DICT_MERGE => Instruction::DictMerge(value.1),
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
            Opcode::EXTENDED_ARG => Instruction::ExtendedArg(value.1),
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
            Instruction::DupTop(_) => Opcode::DUP_TOP,
            Instruction::DupTopTwo(_) => Opcode::DUP_TOP_TWO,
            Instruction::UnaryPositive(_) => Opcode::UNARY_POSITIVE,
            Instruction::UnaryNegative(_) => Opcode::UNARY_NEGATIVE,
            Instruction::UnaryNot(_) => Opcode::UNARY_NOT,
            Instruction::UnaryInvert(_) => Opcode::UNARY_INVERT,
            Instruction::GetIter(_) => Opcode::GET_ITER,
            Instruction::GetYieldFromIter(_) => Opcode::GET_YIELD_FROM_ITER,
            Instruction::BinaryPower(_) => Opcode::BINARY_POWER,
            Instruction::BinaryMultiply(_) => Opcode::BINARY_MULTIPLY,
            Instruction::BinaryMatrixMultiply(_) => Opcode::BINARY_MATRIX_MULTIPLY,
            Instruction::BinaryFloorDivide(_) => Opcode::BINARY_FLOOR_DIVIDE,
            Instruction::BinaryTrueDivide(_) => Opcode::BINARY_TRUE_DIVIDE,
            Instruction::BinaryModulo(_) => Opcode::BINARY_MODULO,
            Instruction::BinaryAdd(_) => Opcode::BINARY_ADD,
            Instruction::BinarySubtract(_) => Opcode::BINARY_SUBTRACT,
            Instruction::BinarySubscr(_) => Opcode::BINARY_SUBSCR,
            Instruction::BinaryLshift(_) => Opcode::BINARY_LSHIFT,
            Instruction::BinaryRshift(_) => Opcode::BINARY_RSHIFT,
            Instruction::BinaryAnd(_) => Opcode::BINARY_AND,
            Instruction::BinaryXor(_) => Opcode::BINARY_XOR,
            Instruction::BinaryOr(_) => Opcode::BINARY_OR,
            Instruction::InplacePower(_) => Opcode::INPLACE_POWER,
            Instruction::InplaceMultiply(_) => Opcode::INPLACE_MULTIPLY,
            Instruction::InplaceMatrixMultiply(_) => Opcode::INPLACE_MATRIX_MULTIPLY,
            Instruction::InplaceFloorDivide(_) => Opcode::INPLACE_FLOOR_DIVIDE,
            Instruction::InplaceTrueDivide(_) => Opcode::INPLACE_TRUE_DIVIDE,
            Instruction::InplaceModulo(_) => Opcode::INPLACE_MODULO,
            Instruction::InplaceAdd(_) => Opcode::INPLACE_ADD,
            Instruction::InplaceSubtract(_) => Opcode::INPLACE_SUBTRACT,
            Instruction::InplaceLshift(_) => Opcode::INPLACE_LSHIFT,
            Instruction::InplaceRshift(_) => Opcode::INPLACE_RSHIFT,
            Instruction::InplaceAnd(_) => Opcode::INPLACE_AND,
            Instruction::InplaceXor(_) => Opcode::INPLACE_XOR,
            Instruction::InplaceOr(_) => Opcode::INPLACE_OR,
            Instruction::StoreSubscr(_) => Opcode::STORE_SUBSCR,
            Instruction::DeleteSubscr(_) => Opcode::DELETE_SUBSCR,
            Instruction::GetAwaitable(_) => Opcode::GET_AWAITABLE,
            Instruction::GetAiter(_) => Opcode::GET_AITER,
            Instruction::GetAnext(_) => Opcode::GET_ANEXT,
            Instruction::EndAsyncFor(_) => Opcode::END_ASYNC_FOR,
            Instruction::BeforeAsyncWith(_) => Opcode::BEFORE_ASYNC_WITH,
            Instruction::SetupAsyncWith(_) => Opcode::SETUP_ASYNC_WITH,
            Instruction::PrintExpr(_) => Opcode::PRINT_EXPR,
            Instruction::SetAdd(_) => Opcode::SET_ADD,
            Instruction::ListAppend(_) => Opcode::LIST_APPEND,
            Instruction::MapAdd(_) => Opcode::MAP_ADD,
            Instruction::ReturnValue(_) => Opcode::RETURN_VALUE,
            Instruction::YieldValue(_) => Opcode::YIELD_VALUE,
            Instruction::YieldFrom(_) => Opcode::YIELD_FROM,
            Instruction::SetupAnnotations(_) => Opcode::SETUP_ANNOTATIONS,
            Instruction::ImportStar(_) => Opcode::IMPORT_STAR,
            Instruction::PopBlock(_) => Opcode::POP_BLOCK,
            Instruction::PopExcept(_) => Opcode::POP_EXCEPT,
            Instruction::Reraise(_) => Opcode::RERAISE,
            Instruction::WithExceptStart(_) => Opcode::WITH_EXCEPT_START,
            Instruction::LoadAssertionError(_) => Opcode::LOAD_ASSERTION_ERROR,
            Instruction::LoadBuildClass(_) => Opcode::LOAD_BUILD_CLASS,
            Instruction::SetupWith(_) => Opcode::SETUP_WITH,
            Instruction::CopyDictWithoutKeys(_) => Opcode::COPY_DICT_WITHOUT_KEYS,
            Instruction::GetLen(_) => Opcode::GET_LEN,
            Instruction::MatchMapping(_) => Opcode::MATCH_MAPPING,
            Instruction::MatchSequence(_) => Opcode::MATCH_SEQUENCE,
            Instruction::MatchKeys(_) => Opcode::MATCH_KEYS,
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
            Instruction::ListToTuple(_) => Opcode::LIST_TO_TUPLE,
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
            Instruction::ExtendedArg(_) => Opcode::EXTENDED_ARG,
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

    pub fn get_raw_value(&self) -> u8 {
        match &self {
            Instruction::PopTop(arg)
            | Instruction::RotTwo(arg)
            | Instruction::RotThree(arg)
            | Instruction::DupTop(arg)
            | Instruction::DupTopTwo(arg)
            | Instruction::RotFour(arg)
            | Instruction::Nop(arg)
            | Instruction::UnaryPositive(arg)
            | Instruction::UnaryNegative(arg)
            | Instruction::UnaryNot(arg)
            | Instruction::UnaryInvert(arg)
            | Instruction::BinaryMatrixMultiply(arg)
            | Instruction::InplaceMatrixMultiply(arg)
            | Instruction::BinaryPower(arg)
            | Instruction::BinaryMultiply(arg)
            | Instruction::BinaryModulo(arg)
            | Instruction::BinaryAdd(arg)
            | Instruction::BinarySubtract(arg)
            | Instruction::BinarySubscr(arg)
            | Instruction::BinaryFloorDivide(arg)
            | Instruction::BinaryTrueDivide(arg)
            | Instruction::InplaceFloorDivide(arg)
            | Instruction::InplaceTrueDivide(arg)
            | Instruction::GetLen(arg)
            | Instruction::MatchMapping(arg)
            | Instruction::MatchSequence(arg)
            | Instruction::MatchKeys(arg)
            | Instruction::CopyDictWithoutKeys(arg)
            | Instruction::WithExceptStart(arg)
            | Instruction::GetAiter(arg)
            | Instruction::GetAnext(arg)
            | Instruction::BeforeAsyncWith(arg)
            | Instruction::EndAsyncFor(arg)
            | Instruction::InplaceAdd(arg)
            | Instruction::InplaceSubtract(arg)
            | Instruction::InplaceMultiply(arg)
            | Instruction::InplaceModulo(arg)
            | Instruction::StoreSubscr(arg)
            | Instruction::DeleteSubscr(arg)
            | Instruction::BinaryLshift(arg)
            | Instruction::BinaryRshift(arg)
            | Instruction::BinaryAnd(arg)
            | Instruction::BinaryXor(arg)
            | Instruction::BinaryOr(arg)
            | Instruction::InplacePower(arg)
            | Instruction::GetIter(arg)
            | Instruction::GetYieldFromIter(arg)
            | Instruction::PrintExpr(arg)
            | Instruction::LoadBuildClass(arg)
            | Instruction::YieldFrom(arg)
            | Instruction::GetAwaitable(arg)
            | Instruction::LoadAssertionError(arg)
            | Instruction::InplaceLshift(arg)
            | Instruction::InplaceRshift(arg)
            | Instruction::InplaceAnd(arg)
            | Instruction::InplaceXor(arg)
            | Instruction::InplaceOr(arg)
            | Instruction::ListToTuple(arg)
            | Instruction::ReturnValue(arg)
            | Instruction::ImportStar(arg)
            | Instruction::SetupAnnotations(arg)
            | Instruction::YieldValue(arg)
            | Instruction::PopBlock(arg)
            | Instruction::PopExcept(arg)
            | Instruction::StoreName(arg)
            | Instruction::DeleteName(arg)
            | Instruction::StoreAttr(arg)
            | Instruction::DeleteAttr(arg)
            | Instruction::StoreGlobal(arg)
            | Instruction::DeleteGlobal(arg)
            | Instruction::LoadName(arg)
            | Instruction::LoadAttr(arg)
            | Instruction::ImportName(arg)
            | Instruction::ImportFrom(arg)
            | Instruction::LoadGlobal(arg)
            | Instruction::LoadMethod(arg)
            | Instruction::UnpackSequence(arg)
            | Instruction::UnpackEx(arg)
            | Instruction::RotN(arg)
            | Instruction::BuildTuple(arg)
            | Instruction::BuildList(arg)
            | Instruction::BuildSet(arg)
            | Instruction::BuildMap(arg)
            | Instruction::CallFunction(arg)
            | Instruction::BuildSlice(arg)
            | Instruction::CallFunctionKW(arg)
            | Instruction::ListAppend(arg)
            | Instruction::SetAdd(arg)
            | Instruction::MapAdd(arg)
            | Instruction::MatchClass(arg)
            | Instruction::BuildConstKeyMap(arg)
            | Instruction::BuildString(arg)
            | Instruction::CallMethod(arg)
            | Instruction::ListExtend(arg)
            | Instruction::SetUpdate(arg)
            | Instruction::DictUpdate(arg)
            | Instruction::DictMerge(arg)
            | Instruction::ForIter(arg)
            | Instruction::JumpForward(arg)
            | Instruction::SetupFinally(arg)
            | Instruction::SetupWith(arg)
            | Instruction::SetupAsyncWith(arg)
            | Instruction::LoadConst(arg)
            | Instruction::CompareOp(arg)
            | Instruction::JumpIfFalseOrPop(arg)
            | Instruction::JumpIfTrueOrPop(arg)
            | Instruction::JumpAbsolute(arg)
            | Instruction::PopJumpIfFalse(arg)
            | Instruction::PopJumpIfTrue(arg)
            | Instruction::JumpIfNotExcMatch(arg)
            | Instruction::Reraise(arg)
            | Instruction::IsOp(arg)
            | Instruction::ContainsOp(arg)
            | Instruction::LoadFast(arg)
            | Instruction::StoreFast(arg)
            | Instruction::DeleteFast(arg)
            | Instruction::GenStart(arg)
            | Instruction::RaiseVarargs(arg)
            | Instruction::MakeFunction(arg)
            | Instruction::LoadClosure(arg)
            | Instruction::LoadDeref(arg)
            | Instruction::StoreDeref(arg)
            | Instruction::DeleteDeref(arg)
            | Instruction::LoadClassderef(arg)
            | Instruction::CallFunctionEx(arg)
            | Instruction::FormatValue(arg)
            | Instruction::ExtendedArg(arg) => *arg,
        }
    }
}
