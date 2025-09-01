use std::ops::{Deref, DerefMut};

use crate::{
    define_default_traits,
    error::Error,
    traits::{GenericInstruction, InstructionAccess, InstructionsOwned, SimpleInstructionAccess},
    v310::{
        code_objects::{AbsoluteJump, Jump, LinetableEntry, RelativeJump},
        ext_instructions::ExtInstructions,
        opcodes::Opcode,
    },
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
    CallFunctionKw(u8),
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
    InvalidOpcode((u8, u8)), // (opcode, arg)
}

impl GenericInstruction<u8> for Instruction {
    type Opcode = Opcode;

    fn get_opcode(&self) -> Self::Opcode {
        Opcode::from_instruction(self)
    }

    fn get_raw_value(&self) -> u8 {
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
            | Instruction::CallFunctionKw(arg)
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
            Instruction::InvalidOpcode((_, arg)) => *arg,
        }
    }
}

/// A list of instructions
#[derive(Debug, Clone, PartialEq)]
pub struct Instructions(Vec<Instruction>);

impl<T> InstructionAccess<u8, Instruction> for T
where
    T: Deref<Target = [Instruction]> + AsRef<[Instruction]>,
{
    type Instruction = Instruction;
    type Jump = Jump;

    fn get_jump_value(&self, index: u32) -> Option<Jump> {
        match self.get(index as usize)? {
            Instruction::JumpAbsolute(_)
            | Instruction::PopJumpIfTrue(_)
            | Instruction::PopJumpIfFalse(_)
            | Instruction::JumpIfNotExcMatch(_)
            | Instruction::JumpIfTrueOrPop(_)
            | Instruction::JumpIfFalseOrPop(_) => {
                let arg = self.get_full_arg(index as usize);

                Some(Jump::Absolute(AbsoluteJump { index: arg? }))
            }
            Instruction::ForIter(_)
            | Instruction::JumpForward(_)
            | Instruction::SetupFinally(_)
            | Instruction::SetupWith(_)
            | Instruction::SetupAsyncWith(_) => {
                let arg = self.get_full_arg(index as usize);

                Some(Jump::Relative(RelativeJump { index: arg? }))
            }
            _ => None,
        }
    }

    /// Returns the index and the instruction of the jump target. None if the index is invalid.
    fn get_jump_target(&self, index: u32) -> Option<(u32, Instruction)> {
        let jump = self.get_jump_value(index)?;

        match jump {
            Jump::Absolute(absolute_jump) => self
                .get(absolute_jump.index as usize)
                .cloned()
                .map(|target| (absolute_jump.index, target)),
            Jump::Relative(RelativeJump { index: jump_index }) => {
                let index = index + jump_index + 1;
                self.get(index as usize)
                    .cloned()
                    .map(|target| (index, target))
            }
        }
    }
}

impl InstructionsOwned<Instruction> for Instructions {
    type Instruction = Instruction;

    fn push(&mut self, item: Self::Instruction) {
        self.0.push(item);
    }
}

impl<T> SimpleInstructionAccess<Instruction> for T where
    T: Deref<Target = [Instruction]> + AsRef<[Instruction]>
{
}

impl Instructions {
    pub fn with_capacity(capacity: usize) -> Self {
        Instructions(Vec::with_capacity(capacity))
    }

    pub fn new(instructions: Vec<Instruction>) -> Self {
        Instructions(instructions)
    }

    /// Returns the instructions but with the extended_args resolved
    /// This can fail in a very unique case. If you're working with bytecode generated by the Python compiler you can safely unwrap.
    /// See the docs of `find_ext_arg_jumps` for more info
    pub fn to_resolved(&self) -> Result<ExtInstructions, Error> {
        ExtInstructions::try_from(self.0.as_slice())
    }

    /// Returns the index and the instruction of the jump target. None if the index is invalid.
    /// This exists so you don't have to supply the index of the jump instruction (only necessary for relative jumps)
    pub fn get_absolute_jump_target(&self, jump: AbsoluteJump) -> Option<(u32, Instruction)> {
        self.0
            .get(jump.index as usize)
            .cloned()
            .map(|target| (jump.index, target))
    }
}

/// Returns the line number of an instruction at `index` if it starts this line.
/// The index needs to be the instruction index, not the byte index.
pub fn starts_line_number(lines: &[LinetableEntry], index: u32) -> Option<u32> {
    for entry in lines {
        if entry.start == index * 2 {
            return entry.line_number;
        }
    }

    None
}

/// Returns the line number of an instruction at `index`. None if the index is out of bounds or if there is no line number.
pub fn get_line_number(lines: &[LinetableEntry], index: u32) -> Option<u32> {
    lines
        .iter()
        .find(|entry| entry.start <= index * 2 && entry.end > index * 2)
        .and_then(|entry| entry.line_number)
}

define_default_traits!(v310, Instruction);
