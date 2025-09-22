use std::ops::{Deref, DerefMut};

use crate::{
    define_default_traits,
    error::Error,
    traits::{GenericInstruction, InstructionAccess, InstructionsOwned, SimpleInstructionAccess},
    v312::{
        cache::get_cache_count,
        code_objects::{Jump, JumpDirection, LinetableEntry, RelativeJump},
        ext_instructions::ExtInstructions,
        opcodes::Opcode,
    },
};

/// Low level representation of a Python bytecode instruction with their original u8 argument.
/// We have arguments for every opcode, even if those aren't used. This is so we can have a full representation of the instructions, even if they're invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    Cache(u8),
    PopTop(u8),
    PushNull(u8),
    InterpreterExit(u8),
    EndFor(u8),
    EndSend(u8),
    Nop(u8),
    UnaryNegative(u8),
    UnaryNot(u8),
    UnaryInvert(u8),
    Reserved(u8),
    BinarySubscr(u8),
    BinarySlice(u8),
    StoreSlice(u8),
    GetLen(u8),
    MatchMapping(u8),
    MatchSequence(u8),
    MatchKeys(u8),
    PushExcInfo(u8),
    CheckExcMatch(u8),
    CheckEgMatch(u8),
    WithExceptStart(u8),
    GetAiter(u8),
    GetAnext(u8),
    BeforeAsyncWith(u8),
    BeforeWith(u8),
    EndAsyncFor(u8),
    CleanupThrow(u8),
    StoreSubscr(u8),
    DeleteSubscr(u8),
    GetIter(u8),
    GetYieldFromIter(u8),
    LoadBuildClass(u8),
    LoadAssertionError(u8),
    ReturnGenerator(u8),
    ReturnValue(u8),
    SetupAnnotations(u8),
    LoadLocals(u8),
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
    Swap(u8),
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
    PopJumpIfFalse(u8),
    PopJumpIfTrue(u8),
    LoadGlobal(u8),
    IsOp(u8),
    ContainsOp(u8),
    Reraise(u8),
    Copy(u8),
    ReturnConst(u8),
    BinaryOp(u8),
    Send(u8),
    LoadFast(u8),
    StoreFast(u8),
    DeleteFast(u8),
    LoadFastCheck(u8),
    PopJumpIfNotNone(u8),
    PopJumpIfNone(u8),
    RaiseVarargs(u8),
    GetAwaitable(u8),
    MakeFunction(u8),
    BuildSlice(u8),
    JumpBackwardNoInterrupt(u8),
    MakeCell(u8),
    LoadClosure(u8),
    LoadDeref(u8),
    StoreDeref(u8),
    DeleteDeref(u8),
    JumpBackward(u8),
    LoadSuperAttr(u8),
    CallFunctionEx(u8),
    LoadFastAndClear(u8),
    ExtendedArg(u8),
    ListAppend(u8),
    SetAdd(u8),
    MapAdd(u8),
    CopyFreeVars(u8),
    YieldValue(u8),
    Resume(u8),
    MatchClass(u8),
    FormatValue(u8),
    BuildConstKeyMap(u8),
    BuildString(u8),
    ListExtend(u8),
    SetUpdate(u8),
    DictMerge(u8),
    DictUpdate(u8),
    Call(u8),
    KwNames(u8),
    CallIntrinsic1(u8),
    CallIntrinsic2(u8),
    LoadFromDictOrGlobals(u8),
    LoadFromDictOrDeref(u8),
    InstrumentedLoadSuperAttr(u8),
    InstrumentedPopJumpIfNone(u8),
    InstrumentedPopJumpIfNotNone(u8),
    InstrumentedResume(u8),
    InstrumentedCall(u8),
    InstrumentedReturnValue(u8),
    InstrumentedYieldValue(u8),
    InstrumentedCallFunctionEx(u8),
    InstrumentedJumpForward(u8),
    InstrumentedJumpBackward(u8),
    InstrumentedReturnConst(u8),
    InstrumentedForIter(u8),
    InstrumentedPopJumpIfFalse(u8),
    InstrumentedPopJumpIfTrue(u8),
    InstrumentedEndFor(u8),
    InstrumentedEndSend(u8),
    InstrumentedInstruction(u8),
    InstrumentedLine(u8),
    // We skip psuedo opcodes as they can never appear in actual bytecode
    // MinPseudoOpcode(u8),
    // SetupFinally(u8),
    // SetupCleanup(u8),
    // SetupWith(u8),
    // PopBlock(u8),
    // Jump(u8),
    // JumpNoInterrupt(u8),
    // LoadMethod(u8),
    // LoadSuperMethod(u8),
    // LoadZeroSuperMethod(u8),
    // LoadZeroSuperAttr(u8),
    // StoreFastMaybeNull(u8),
    // MaxPseudoOpcode(u8),
    BinaryOpAddFloat(u8),
    BinaryOpAddInt(u8),
    BinaryOpAddUnicode(u8),
    BinaryOpInplaceAddUnicode(u8),
    BinaryOpMultiplyFloat(u8),
    BinaryOpMultiplyInt(u8),
    BinaryOpSubtractFloat(u8),
    BinaryOpSubtractInt(u8),
    BinarySubscrDict(u8),
    BinarySubscrGetitem(u8),
    BinarySubscrListInt(u8),
    BinarySubscrTupleInt(u8),
    CallPyExactArgs(u8),
    CallPyWithDefaults(u8),
    CallBoundMethodExactArgs(u8),
    CallBuiltinClass(u8),
    CallBuiltinFastWithKeywords(u8),
    CallMethodDescriptorFastWithKeywords(u8),
    CallNoKwBuiltinFast(u8),
    CallNoKwBuiltinO(u8),
    CallNoKwIsinstance(u8),
    CallNoKwLen(u8),
    CallNoKwListAppend(u8),
    CallNoKwMethodDescriptorFast(u8),
    CallNoKwMethodDescriptorNoargs(u8),
    CallNoKwMethodDescriptorO(u8),
    CallNoKwStr1(u8),
    CallNoKwTuple1(u8),
    CallNoKwType1(u8),
    CompareOpFloat(u8),
    CompareOpInt(u8),
    CompareOpStr(u8),
    ForIterList(u8),
    ForIterTuple(u8),
    ForIterRange(u8),
    ForIterGen(u8),
    LoadSuperAttrAttr(u8),
    LoadSuperAttrMethod(u8),
    LoadAttrClass(u8),
    LoadAttrGetattributeOverridden(u8),
    LoadAttrInstanceValue(u8),
    LoadAttrModule(u8),
    LoadAttrProperty(u8),
    LoadAttrSlot(u8),
    LoadAttrWithHint(u8),
    LoadAttrMethodLazyDict(u8),
    LoadAttrMethodNoDict(u8),
    LoadAttrMethodWithValues(u8),
    LoadConstLoadFast(u8),
    LoadFastLoadConst(u8),
    LoadFastLoadFast(u8),
    LoadGlobalBuiltin(u8),
    LoadGlobalModule(u8),
    StoreAttrInstanceValue(u8),
    StoreAttrSlot(u8),
    StoreAttrWithHint(u8),
    StoreFastLoadFast(u8),
    StoreFastStoreFast(u8),
    StoreSubscrDict(u8),
    StoreSubscrListInt(u8),
    UnpackSequenceList(u8),
    UnpackSequenceTuple(u8),
    UnpackSequenceTwoTuple(u8),
    SendGen(u8),
    InvalidOpcode((u8, u8)), // (opcode, arg)
}

impl GenericInstruction<u8> for Instruction {
    type Opcode = Opcode;

    fn get_opcode(&self) -> Self::Opcode {
        Opcode::from_instruction(self)
    }

    fn get_raw_value(&self) -> u8 {
        match &self {
            Instruction::Cache(arg)
            | Instruction::PopTop(arg)
            | Instruction::PushNull(arg)
            | Instruction::InterpreterExit(arg)
            | Instruction::EndFor(arg)
            | Instruction::EndSend(arg)
            | Instruction::Nop(arg)
            | Instruction::UnaryNegative(arg)
            | Instruction::UnaryNot(arg)
            | Instruction::UnaryInvert(arg)
            | Instruction::Reserved(arg)
            | Instruction::BinarySubscr(arg)
            | Instruction::BinarySlice(arg)
            | Instruction::StoreSlice(arg)
            | Instruction::GetLen(arg)
            | Instruction::MatchMapping(arg)
            | Instruction::MatchSequence(arg)
            | Instruction::MatchKeys(arg)
            | Instruction::PushExcInfo(arg)
            | Instruction::CheckExcMatch(arg)
            | Instruction::CheckEgMatch(arg)
            | Instruction::WithExceptStart(arg)
            | Instruction::GetAiter(arg)
            | Instruction::GetAnext(arg)
            | Instruction::BeforeAsyncWith(arg)
            | Instruction::BeforeWith(arg)
            | Instruction::EndAsyncFor(arg)
            | Instruction::CleanupThrow(arg)
            | Instruction::StoreSubscr(arg)
            | Instruction::DeleteSubscr(arg)
            | Instruction::GetIter(arg)
            | Instruction::GetYieldFromIter(arg)
            | Instruction::LoadBuildClass(arg)
            | Instruction::LoadAssertionError(arg)
            | Instruction::ReturnGenerator(arg)
            | Instruction::ReturnValue(arg)
            | Instruction::SetupAnnotations(arg)
            | Instruction::LoadLocals(arg)
            | Instruction::PopExcept(arg)
            | Instruction::StoreName(arg)
            | Instruction::DeleteName(arg)
            | Instruction::UnpackSequence(arg)
            | Instruction::ForIter(arg)
            | Instruction::UnpackEx(arg)
            | Instruction::StoreAttr(arg)
            | Instruction::DeleteAttr(arg)
            | Instruction::StoreGlobal(arg)
            | Instruction::DeleteGlobal(arg)
            | Instruction::Swap(arg)
            | Instruction::LoadConst(arg)
            | Instruction::LoadName(arg)
            | Instruction::BuildTuple(arg)
            | Instruction::BuildList(arg)
            | Instruction::BuildSet(arg)
            | Instruction::BuildMap(arg)
            | Instruction::LoadAttr(arg)
            | Instruction::CompareOp(arg)
            | Instruction::ImportName(arg)
            | Instruction::ImportFrom(arg)
            | Instruction::JumpForward(arg)
            | Instruction::PopJumpIfFalse(arg)
            | Instruction::PopJumpIfTrue(arg)
            | Instruction::LoadGlobal(arg)
            | Instruction::IsOp(arg)
            | Instruction::ContainsOp(arg)
            | Instruction::Reraise(arg)
            | Instruction::Copy(arg)
            | Instruction::ReturnConst(arg)
            | Instruction::BinaryOp(arg)
            | Instruction::Send(arg)
            | Instruction::LoadFast(arg)
            | Instruction::StoreFast(arg)
            | Instruction::DeleteFast(arg)
            | Instruction::LoadFastCheck(arg)
            | Instruction::PopJumpIfNotNone(arg)
            | Instruction::PopJumpIfNone(arg)
            | Instruction::RaiseVarargs(arg)
            | Instruction::GetAwaitable(arg)
            | Instruction::MakeFunction(arg)
            | Instruction::BuildSlice(arg)
            | Instruction::JumpBackwardNoInterrupt(arg)
            | Instruction::MakeCell(arg)
            | Instruction::LoadClosure(arg)
            | Instruction::LoadDeref(arg)
            | Instruction::StoreDeref(arg)
            | Instruction::DeleteDeref(arg)
            | Instruction::JumpBackward(arg)
            | Instruction::LoadSuperAttr(arg)
            | Instruction::CallFunctionEx(arg)
            | Instruction::LoadFastAndClear(arg)
            | Instruction::ExtendedArg(arg)
            | Instruction::ListAppend(arg)
            | Instruction::SetAdd(arg)
            | Instruction::MapAdd(arg)
            | Instruction::CopyFreeVars(arg)
            | Instruction::YieldValue(arg)
            | Instruction::Resume(arg)
            | Instruction::MatchClass(arg)
            | Instruction::FormatValue(arg)
            | Instruction::BuildConstKeyMap(arg)
            | Instruction::BuildString(arg)
            | Instruction::ListExtend(arg)
            | Instruction::SetUpdate(arg)
            | Instruction::DictMerge(arg)
            | Instruction::DictUpdate(arg)
            | Instruction::Call(arg)
            | Instruction::KwNames(arg)
            | Instruction::CallIntrinsic1(arg)
            | Instruction::CallIntrinsic2(arg)
            | Instruction::LoadFromDictOrGlobals(arg)
            | Instruction::LoadFromDictOrDeref(arg)
            | Instruction::InstrumentedLoadSuperAttr(arg)
            | Instruction::InstrumentedPopJumpIfNone(arg)
            | Instruction::InstrumentedPopJumpIfNotNone(arg)
            | Instruction::InstrumentedResume(arg)
            | Instruction::InstrumentedCall(arg)
            | Instruction::InstrumentedReturnValue(arg)
            | Instruction::InstrumentedYieldValue(arg)
            | Instruction::InstrumentedCallFunctionEx(arg)
            | Instruction::InstrumentedJumpForward(arg)
            | Instruction::InstrumentedJumpBackward(arg)
            | Instruction::InstrumentedReturnConst(arg)
            | Instruction::InstrumentedForIter(arg)
            | Instruction::InstrumentedPopJumpIfFalse(arg)
            | Instruction::InstrumentedPopJumpIfTrue(arg)
            | Instruction::InstrumentedEndFor(arg)
            | Instruction::InstrumentedEndSend(arg)
            | Instruction::InstrumentedInstruction(arg)
            | Instruction::InstrumentedLine(arg)
            // | Instruction::MinPseudoOpcode(arg)
            // | Instruction::SetupFinally(arg)
            // | Instruction::SetupCleanup(arg)
            // | Instruction::SetupWith(arg)
            // | Instruction::PopBlock(arg)
            // | Instruction::Jump(arg)
            // | Instruction::JumpNoInterrupt(arg)
            // | Instruction::LoadMethod(arg)
            // | Instruction::LoadSuperMethod(arg)
            // | Instruction::LoadZeroSuperMethod(arg)
            // | Instruction::LoadZeroSuperAttr(arg)
            // | Instruction::StoreFastMaybeNull(arg)
            // | Instruction::MaxPseudoOpcode(arg)
            | Instruction::BinaryOpAddFloat(arg)
            | Instruction::BinaryOpAddInt(arg)
            | Instruction::BinaryOpAddUnicode(arg)
            | Instruction::BinaryOpInplaceAddUnicode(arg)
            | Instruction::BinaryOpMultiplyFloat(arg)
            | Instruction::BinaryOpMultiplyInt(arg)
            | Instruction::BinaryOpSubtractFloat(arg)
            | Instruction::BinaryOpSubtractInt(arg)
            | Instruction::BinarySubscrDict(arg)
            | Instruction::BinarySubscrGetitem(arg)
            | Instruction::BinarySubscrListInt(arg)
            | Instruction::BinarySubscrTupleInt(arg)
            | Instruction::CallPyExactArgs(arg)
            | Instruction::CallPyWithDefaults(arg)
            | Instruction::CallBoundMethodExactArgs(arg)
            | Instruction::CallBuiltinClass(arg)
            | Instruction::CallBuiltinFastWithKeywords(arg)
            | Instruction::CallMethodDescriptorFastWithKeywords(arg)
            | Instruction::CallNoKwBuiltinFast(arg)
            | Instruction::CallNoKwBuiltinO(arg)
            | Instruction::CallNoKwIsinstance(arg)
            | Instruction::CallNoKwLen(arg)
            | Instruction::CallNoKwListAppend(arg)
            | Instruction::CallNoKwMethodDescriptorFast(arg)
            | Instruction::CallNoKwMethodDescriptorNoargs(arg)
            | Instruction::CallNoKwMethodDescriptorO(arg)
            | Instruction::CallNoKwStr1(arg)
            | Instruction::CallNoKwTuple1(arg)
            | Instruction::CallNoKwType1(arg)
            | Instruction::CompareOpFloat(arg)
            | Instruction::CompareOpInt(arg)
            | Instruction::CompareOpStr(arg)
            | Instruction::ForIterList(arg)
            | Instruction::ForIterTuple(arg)
            | Instruction::ForIterRange(arg)
            | Instruction::ForIterGen(arg)
            | Instruction::LoadSuperAttrAttr(arg)
            | Instruction::LoadSuperAttrMethod(arg)
            | Instruction::LoadAttrClass(arg)
            | Instruction::LoadAttrGetattributeOverridden(arg)
            | Instruction::LoadAttrInstanceValue(arg)
            | Instruction::LoadAttrModule(arg)
            | Instruction::LoadAttrProperty(arg)
            | Instruction::LoadAttrSlot(arg)
            | Instruction::LoadAttrWithHint(arg)
            | Instruction::LoadAttrMethodLazyDict(arg)
            | Instruction::LoadAttrMethodNoDict(arg)
            | Instruction::LoadAttrMethodWithValues(arg)
            | Instruction::LoadConstLoadFast(arg)
            | Instruction::LoadFastLoadConst(arg)
            | Instruction::LoadFastLoadFast(arg)
            | Instruction::LoadGlobalBuiltin(arg)
            | Instruction::LoadGlobalModule(arg)
            | Instruction::StoreAttrInstanceValue(arg)
            | Instruction::StoreAttrSlot(arg)
            | Instruction::StoreAttrWithHint(arg)
            | Instruction::StoreFastLoadFast(arg)
            | Instruction::StoreFastStoreFast(arg)
            | Instruction::StoreSubscrDict(arg)
            | Instruction::StoreSubscrListInt(arg)
            | Instruction::UnpackSequenceList(arg)
            | Instruction::UnpackSequenceTuple(arg)
            | Instruction::UnpackSequenceTwoTuple(arg)
            | Instruction::SendGen(arg) => *arg,
            Instruction::InvalidOpcode((_, arg)) => *arg,
        }
    }
}

/// Resolves the actual index of the current jump instruction.
/// In 3.12 the jump offsets are relative to the CACHE opcodes succeeding the jump instruction.
/// They're calculated from the predefined cache layout. This does not guarantee the index is actually valid.
pub fn get_real_jump_index(instructions: &[Instruction], index: usize) -> Option<usize> {
    Some(index + get_cache_count(instructions.get(index)?.get_opcode()).unwrap_or(0))
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
            Instruction::ForIter(_)
            | Instruction::JumpForward(_)
            | Instruction::PopJumpIfFalse(_)
            | Instruction::PopJumpIfTrue(_)
            | Instruction::Send(_)
            | Instruction::PopJumpIfNotNone(_)
            | Instruction::PopJumpIfNone(_)
            | Instruction::ForIterRange(_)
            | Instruction::ForIterList(_)
            | Instruction::ForIterGen(_)
            | Instruction::ForIterTuple(_)
            | Instruction::InstrumentedForIter(_)
            | Instruction::InstrumentedPopJumpIfNone(_)
            | Instruction::InstrumentedPopJumpIfNotNone(_)
            | Instruction::InstrumentedJumpForward(_)
            | Instruction::InstrumentedPopJumpIfFalse(_)
            | Instruction::InstrumentedPopJumpIfTrue(_) => {
                let arg = self.get_full_arg(index as usize);

                Some(Jump::Relative(RelativeJump {
                    index: arg?,
                    direction: JumpDirection::Forward,
                }))
            }
            Instruction::JumpBackwardNoInterrupt(_)
            | Instruction::JumpBackward(_)
            | Instruction::InstrumentedJumpBackward(_) => {
                let arg = self.get_full_arg(index as usize);

                Some(Jump::Relative(RelativeJump {
                    index: arg?,
                    direction: JumpDirection::Backward,
                }))
            }
            _ => None,
        }
    }

    /// Returns the index and the instruction of the jump target. None if the index is invalid.
    fn get_jump_target(&self, index: u32) -> Option<(u32, Instruction)> {
        let jump = self.get_jump_value(index)?;

        match jump {
            Jump::Relative(RelativeJump {
                index: jump_index,
                direction: JumpDirection::Forward,
            }) => {
                let index = get_real_jump_index(self, index as usize)? as u32 + jump_index + 1;
                self.get(index as usize)
                    .cloned()
                    .map(|target| (index, target))
            }
            Jump::Relative(RelativeJump {
                index: jump_index,
                direction: JumpDirection::Backward,
            }) => {
                let index = get_real_jump_index(self, index as usize)? as u32 - jump_index + 1;
                self.get(index as usize)
                    .cloned()
                    .map(|target| (index, target))
            }
        }
    }
}

impl Instructions {
    pub fn with_capacity(capacity: usize) -> Self {
        Instructions(Vec::with_capacity(capacity))
    }

    pub fn new(instructions: Vec<Instruction>) -> Self {
        Instructions(instructions)
    }

    /// Returns the instructions but with the extended_args resolved
    pub fn to_resolved(&self) -> Result<ExtInstructions, Error> {
        ExtInstructions::try_from(self.0.as_slice())
    }
}

/// Returns the line number of an instruction at `index` if it starts this line.
/// The index needs to be the instruction index, not the byte index.
pub fn starts_line_number(lines: &[LinetableEntry], index: u32) -> Option<u32> {
    let mut current_line = None;

    for entry in lines {
        if entry.line_number.is_some() && current_line != entry.line_number {
            if entry.start == index * 2 {
                return entry.line_number;
            }

            current_line = entry.line_number;
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

define_default_traits!(v312, Instruction);
