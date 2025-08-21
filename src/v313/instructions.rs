use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use crate::{
    error::Error,
    traits::{GenericInstruction, InstructionAccess},
    v313::{
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
    BeforeAsyncWith(u8),
    BeforeWith(u8),
    BinaryOpInplaceAddUnicode(u8),
    BinarySlice(u8),
    BinarySubscr(u8),
    CheckEgMatch(u8),
    CheckExcMatch(u8),
    CleanupThrow(u8),
    DeleteSubscr(u8),
    EndAsyncFor(u8),
    EndFor(u8),
    EndSend(u8),
    ExitInitCheck(u8),
    FormatSimple(u8),
    FormatWithSpec(u8),
    GetAiter(u8),
    Reserved(u8),
    GetAnext(u8),
    GetIter(u8),
    GetLen(u8),
    GetYieldFromIter(u8),
    InterpreterExit(u8),
    LoadAssertionError(u8),
    LoadBuildClass(u8),
    LoadLocals(u8),
    MakeFunction(u8),
    MatchKeys(u8),
    MatchMapping(u8),
    MatchSequence(u8),
    Nop(u8),
    PopExcept(u8),
    PopTop(u8),
    PushExcInfo(u8),
    PushNull(u8),
    ReturnGenerator(u8),
    ReturnValue(u8),
    SetupAnnotations(u8),
    StoreSlice(u8),
    StoreSubscr(u8),
    ToBool(u8),
    UnaryInvert(u8),
    UnaryNegative(u8),
    UnaryNot(u8),
    WithExceptStart(u8),
    BinaryOp(u8),
    BuildConstKeyMap(u8),
    BuildList(u8),
    BuildMap(u8),
    BuildSet(u8),
    BuildSlice(u8),
    BuildString(u8),
    BuildTuple(u8),
    Call(u8),
    CallFunctionEx(u8),
    CallIntrinsic1(u8),
    CallIntrinsic2(u8),
    CallKw(u8),
    CompareOp(u8),
    ContainsOp(u8),
    ConvertValue(u8),
    Copy(u8),
    CopyFreeVars(u8),
    DeleteAttr(u8),
    DeleteDeref(u8),
    DeleteFast(u8),
    DeleteGlobal(u8),
    DeleteName(u8),
    DictMerge(u8),
    DictUpdate(u8),
    EnterExecutor(u8),
    ExtendedArg(u8),
    ForIter(u8),
    GetAwaitable(u8),
    ImportFrom(u8),
    ImportName(u8),
    IsOp(u8),
    JumpBackward(u8),
    JumpBackwardNoInterrupt(u8),
    JumpForward(u8),
    ListAppend(u8),
    ListExtend(u8),
    LoadAttr(u8),
    LoadConst(u8),
    LoadDeref(u8),
    LoadFast(u8),
    LoadFastAndClear(u8),
    LoadFastCheck(u8),
    LoadFastLoadFast(u8),
    LoadFromDictOrDeref(u8),
    LoadFromDictOrGlobals(u8),
    LoadGlobal(u8),
    LoadName(u8),
    LoadSuperAttr(u8),
    MakeCell(u8),
    MapAdd(u8),
    MatchClass(u8),
    PopJumpIfFalse(u8),
    PopJumpIfNone(u8),
    PopJumpIfNotNone(u8),
    PopJumpIfTrue(u8),
    RaiseVarargs(u8),
    Reraise(u8),
    ReturnConst(u8),
    Send(u8),
    SetAdd(u8),
    SetFunctionAttribute(u8),
    SetUpdate(u8),
    StoreAttr(u8),
    StoreDeref(u8),
    StoreFast(u8),
    StoreFastLoadFast(u8),
    StoreFastStoreFast(u8),
    StoreGlobal(u8),
    StoreName(u8),
    Swap(u8),
    UnpackEx(u8),
    UnpackSequence(u8),
    YieldValue(u8),
    Resume(u8),
    BinaryOpAddFloat(u8),
    BinaryOpAddInt(u8),
    BinaryOpAddUnicode(u8),
    BinaryOpMultiplyFloat(u8),
    BinaryOpMultiplyInt(u8),
    BinaryOpSubtractFloat(u8),
    BinaryOpSubtractInt(u8),
    BinarySubscrDict(u8),
    BinarySubscrGetitem(u8),
    BinarySubscrListInt(u8),
    BinarySubscrStrInt(u8),
    BinarySubscrTupleInt(u8),
    CallAllocAndEnterInit(u8),
    CallBoundMethodExactArgs(u8),
    CallBoundMethodGeneral(u8),
    CallBuiltinClass(u8),
    CallBuiltinFast(u8),
    CallBuiltinFastWithKeywords(u8),
    CallBuiltinO(u8),
    CallIsinstance(u8),
    CallLen(u8),
    CallListAppend(u8),
    CallMethodDescriptorFast(u8),
    CallMethodDescriptorFastWithKeywords(u8),
    CallMethodDescriptorNoargs(u8),
    CallMethodDescriptorO(u8),
    CallNonPyGeneral(u8),
    CallPyExactArgs(u8),
    CallPyGeneral(u8),
    CallStr1(u8),
    CallTuple1(u8),
    CallType1(u8),
    CompareOpFloat(u8),
    CompareOpInt(u8),
    CompareOpStr(u8),
    ContainsOpDict(u8),
    ContainsOpSet(u8),
    ForIterGen(u8),
    ForIterList(u8),
    ForIterRange(u8),
    ForIterTuple(u8),
    LoadAttrClass(u8),
    LoadAttrGetattributeOverridden(u8),
    LoadAttrInstanceValue(u8),
    LoadAttrMethodLazyDict(u8),
    LoadAttrMethodNoDict(u8),
    LoadAttrMethodWithValues(u8),
    LoadAttrModule(u8),
    LoadAttrNondescriptorNoDict(u8),
    LoadAttrNondescriptorWithValues(u8),
    LoadAttrProperty(u8),
    LoadAttrSlot(u8),
    LoadAttrWithHint(u8),
    LoadGlobalBuiltin(u8),
    LoadGlobalModule(u8),
    LoadSuperAttrAttr(u8),
    LoadSuperAttrMethod(u8),
    ResumeCheck(u8),
    SendGen(u8),
    StoreAttrInstanceValue(u8),
    StoreAttrSlot(u8),
    StoreAttrWithHint(u8),
    StoreSubscrDict(u8),
    StoreSubscrListInt(u8),
    ToBoolAlwaysTrue(u8),
    ToBoolBool(u8),
    ToBoolInt(u8),
    ToBoolList(u8),
    ToBoolNone(u8),
    ToBoolStr(u8),
    UnpackSequenceList(u8),
    UnpackSequenceTuple(u8),
    UnpackSequenceTwoTuple(u8),
    InstrumentedResume(u8),
    InstrumentedEndFor(u8),
    InstrumentedEndSend(u8),
    InstrumentedReturnValue(u8),
    InstrumentedReturnConst(u8),
    InstrumentedYieldValue(u8),
    InstrumentedLoadSuperAttr(u8),
    InstrumentedForIter(u8),
    InstrumentedCall(u8),
    InstrumentedCallKw(u8),
    InstrumentedCallFunctionEx(u8),
    InstrumentedInstruction(u8),
    InstrumentedJumpForward(u8),
    InstrumentedJumpBackward(u8),
    InstrumentedPopJumpIfTrue(u8),
    InstrumentedPopJumpIfFalse(u8),
    InstrumentedPopJumpIfNone(u8),
    InstrumentedPopJumpIfNotNone(u8),
    InstrumentedLine(u8),
    // We skip psuedo opcodes as they can never appear in actual bytecode
    // Jump(u8),
    // JumpNoInterrupt(u8),
    // LoadClosure(u8),
    // LoadMethod(u8),
    // LoadSuperMethod(u8),
    // LoadZeroSuperAttr(u8),
    // LoadZeroSuperMethod(u8),
    // PopBlock(u8),
    // SetupCleanup(u8),
    // SetupFinally(u8),
    // SetupWith(u8),
    // StoreFastMaybeNull(u8),
    InvalidOpcode((u8, u8)), // (opcode, arg)
}

impl GenericInstruction for Instruction {
    type Opcode = Opcode;
    type Arg = u8;

    fn get_opcode(&self) -> Self::Opcode {
        Opcode::from_instruction(self)
    }

    fn get_raw_value(&self) -> Self::Arg {
        match &self {
            Instruction::Cache(arg)
            | Instruction::BeforeAsyncWith(arg)
            | Instruction::BeforeWith(arg)
            | Instruction::BinaryOpInplaceAddUnicode(arg)
            | Instruction::BinarySlice(arg)
            | Instruction::BinarySubscr(arg)
            | Instruction::CheckEgMatch(arg)
            | Instruction::CheckExcMatch(arg)
            | Instruction::CleanupThrow(arg)
            | Instruction::DeleteSubscr(arg)
            | Instruction::EndAsyncFor(arg)
            | Instruction::EndFor(arg)
            | Instruction::EndSend(arg)
            | Instruction::ExitInitCheck(arg)
            | Instruction::FormatSimple(arg)
            | Instruction::FormatWithSpec(arg)
            | Instruction::GetAiter(arg)
            | Instruction::Reserved(arg)
            | Instruction::GetAnext(arg)
            | Instruction::GetIter(arg)
            | Instruction::GetLen(arg)
            | Instruction::GetYieldFromIter(arg)
            | Instruction::InterpreterExit(arg)
            | Instruction::LoadAssertionError(arg)
            | Instruction::LoadBuildClass(arg)
            | Instruction::LoadLocals(arg)
            | Instruction::MakeFunction(arg)
            | Instruction::MatchKeys(arg)
            | Instruction::MatchMapping(arg)
            | Instruction::MatchSequence(arg)
            | Instruction::Nop(arg)
            | Instruction::PopExcept(arg)
            | Instruction::PopTop(arg)
            | Instruction::PushExcInfo(arg)
            | Instruction::PushNull(arg)
            | Instruction::ReturnGenerator(arg)
            | Instruction::ReturnValue(arg)
            | Instruction::SetupAnnotations(arg)
            | Instruction::StoreSlice(arg)
            | Instruction::StoreSubscr(arg)
            | Instruction::ToBool(arg)
            | Instruction::UnaryInvert(arg)
            | Instruction::UnaryNegative(arg)
            | Instruction::UnaryNot(arg)
            | Instruction::WithExceptStart(arg)
            | Instruction::BinaryOp(arg)
            | Instruction::BuildConstKeyMap(arg)
            | Instruction::BuildList(arg)
            | Instruction::BuildMap(arg)
            | Instruction::BuildSet(arg)
            | Instruction::BuildSlice(arg)
            | Instruction::BuildString(arg)
            | Instruction::BuildTuple(arg)
            | Instruction::Call(arg)
            | Instruction::CallFunctionEx(arg)
            | Instruction::CallIntrinsic1(arg)
            | Instruction::CallIntrinsic2(arg)
            | Instruction::CallKw(arg)
            | Instruction::CompareOp(arg)
            | Instruction::ContainsOp(arg)
            | Instruction::ConvertValue(arg)
            | Instruction::Copy(arg)
            | Instruction::CopyFreeVars(arg)
            | Instruction::DeleteAttr(arg)
            | Instruction::DeleteDeref(arg)
            | Instruction::DeleteFast(arg)
            | Instruction::DeleteGlobal(arg)
            | Instruction::DeleteName(arg)
            | Instruction::DictMerge(arg)
            | Instruction::DictUpdate(arg)
            | Instruction::EnterExecutor(arg)
            | Instruction::ExtendedArg(arg)
            | Instruction::ForIter(arg)
            | Instruction::GetAwaitable(arg)
            | Instruction::ImportFrom(arg)
            | Instruction::ImportName(arg)
            | Instruction::IsOp(arg)
            | Instruction::JumpBackward(arg)
            | Instruction::JumpBackwardNoInterrupt(arg)
            | Instruction::JumpForward(arg)
            | Instruction::ListAppend(arg)
            | Instruction::ListExtend(arg)
            | Instruction::LoadAttr(arg)
            | Instruction::LoadConst(arg)
            | Instruction::LoadDeref(arg)
            | Instruction::LoadFast(arg)
            | Instruction::LoadFastAndClear(arg)
            | Instruction::LoadFastCheck(arg)
            | Instruction::LoadFastLoadFast(arg)
            | Instruction::LoadFromDictOrDeref(arg)
            | Instruction::LoadFromDictOrGlobals(arg)
            | Instruction::LoadGlobal(arg)
            | Instruction::LoadName(arg)
            | Instruction::LoadSuperAttr(arg)
            | Instruction::MakeCell(arg)
            | Instruction::MapAdd(arg)
            | Instruction::MatchClass(arg)
            | Instruction::PopJumpIfFalse(arg)
            | Instruction::PopJumpIfNone(arg)
            | Instruction::PopJumpIfNotNone(arg)
            | Instruction::PopJumpIfTrue(arg)
            | Instruction::RaiseVarargs(arg)
            | Instruction::Reraise(arg)
            | Instruction::ReturnConst(arg)
            | Instruction::Send(arg)
            | Instruction::SetAdd(arg)
            | Instruction::SetFunctionAttribute(arg)
            | Instruction::SetUpdate(arg)
            | Instruction::StoreAttr(arg)
            | Instruction::StoreDeref(arg)
            | Instruction::StoreFast(arg)
            | Instruction::StoreFastLoadFast(arg)
            | Instruction::StoreFastStoreFast(arg)
            | Instruction::StoreGlobal(arg)
            | Instruction::StoreName(arg)
            | Instruction::Swap(arg)
            | Instruction::UnpackEx(arg)
            | Instruction::UnpackSequence(arg)
            | Instruction::YieldValue(arg)
            | Instruction::Resume(arg)
            | Instruction::BinaryOpAddFloat(arg)
            | Instruction::BinaryOpAddInt(arg)
            | Instruction::BinaryOpAddUnicode(arg)
            | Instruction::BinaryOpMultiplyFloat(arg)
            | Instruction::BinaryOpMultiplyInt(arg)
            | Instruction::BinaryOpSubtractFloat(arg)
            | Instruction::BinaryOpSubtractInt(arg)
            | Instruction::BinarySubscrDict(arg)
            | Instruction::BinarySubscrGetitem(arg)
            | Instruction::BinarySubscrListInt(arg)
            | Instruction::BinarySubscrStrInt(arg)
            | Instruction::BinarySubscrTupleInt(arg)
            | Instruction::CallAllocAndEnterInit(arg)
            | Instruction::CallBoundMethodExactArgs(arg)
            | Instruction::CallBoundMethodGeneral(arg)
            | Instruction::CallBuiltinClass(arg)
            | Instruction::CallBuiltinFast(arg)
            | Instruction::CallBuiltinFastWithKeywords(arg)
            | Instruction::CallBuiltinO(arg)
            | Instruction::CallIsinstance(arg)
            | Instruction::CallLen(arg)
            | Instruction::CallListAppend(arg)
            | Instruction::CallMethodDescriptorFast(arg)
            | Instruction::CallMethodDescriptorFastWithKeywords(arg)
            | Instruction::CallMethodDescriptorNoargs(arg)
            | Instruction::CallMethodDescriptorO(arg)
            | Instruction::CallNonPyGeneral(arg)
            | Instruction::CallPyExactArgs(arg)
            | Instruction::CallPyGeneral(arg)
            | Instruction::CallStr1(arg)
            | Instruction::CallTuple1(arg)
            | Instruction::CallType1(arg)
            | Instruction::CompareOpFloat(arg)
            | Instruction::CompareOpInt(arg)
            | Instruction::CompareOpStr(arg)
            | Instruction::ContainsOpDict(arg)
            | Instruction::ContainsOpSet(arg)
            | Instruction::ForIterGen(arg)
            | Instruction::ForIterList(arg)
            | Instruction::ForIterRange(arg)
            | Instruction::ForIterTuple(arg)
            | Instruction::LoadAttrClass(arg)
            | Instruction::LoadAttrGetattributeOverridden(arg)
            | Instruction::LoadAttrInstanceValue(arg)
            | Instruction::LoadAttrMethodLazyDict(arg)
            | Instruction::LoadAttrMethodNoDict(arg)
            | Instruction::LoadAttrMethodWithValues(arg)
            | Instruction::LoadAttrModule(arg)
            | Instruction::LoadAttrNondescriptorNoDict(arg)
            | Instruction::LoadAttrNondescriptorWithValues(arg)
            | Instruction::LoadAttrProperty(arg)
            | Instruction::LoadAttrSlot(arg)
            | Instruction::LoadAttrWithHint(arg)
            | Instruction::LoadGlobalBuiltin(arg)
            | Instruction::LoadGlobalModule(arg)
            | Instruction::LoadSuperAttrAttr(arg)
            | Instruction::LoadSuperAttrMethod(arg)
            | Instruction::ResumeCheck(arg)
            | Instruction::SendGen(arg)
            | Instruction::StoreAttrInstanceValue(arg)
            | Instruction::StoreAttrSlot(arg)
            | Instruction::StoreAttrWithHint(arg)
            | Instruction::StoreSubscrDict(arg)
            | Instruction::StoreSubscrListInt(arg)
            | Instruction::ToBoolAlwaysTrue(arg)
            | Instruction::ToBoolBool(arg)
            | Instruction::ToBoolInt(arg)
            | Instruction::ToBoolList(arg)
            | Instruction::ToBoolNone(arg)
            | Instruction::ToBoolStr(arg)
            | Instruction::UnpackSequenceList(arg)
            | Instruction::UnpackSequenceTuple(arg)
            | Instruction::UnpackSequenceTwoTuple(arg)
            | Instruction::InstrumentedResume(arg)
            | Instruction::InstrumentedEndFor(arg)
            | Instruction::InstrumentedEndSend(arg)
            | Instruction::InstrumentedReturnValue(arg)
            | Instruction::InstrumentedReturnConst(arg)
            | Instruction::InstrumentedYieldValue(arg)
            | Instruction::InstrumentedLoadSuperAttr(arg)
            | Instruction::InstrumentedForIter(arg)
            | Instruction::InstrumentedCall(arg)
            | Instruction::InstrumentedCallKw(arg)
            | Instruction::InstrumentedCallFunctionEx(arg)
            | Instruction::InstrumentedInstruction(arg)
            | Instruction::InstrumentedJumpForward(arg)
            | Instruction::InstrumentedJumpBackward(arg)
            | Instruction::InstrumentedPopJumpIfTrue(arg)
            | Instruction::InstrumentedPopJumpIfFalse(arg)
            | Instruction::InstrumentedPopJumpIfNone(arg)
            | Instruction::InstrumentedPopJumpIfNotNone(arg)
            | Instruction::InstrumentedLine(arg) => *arg,
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

impl InstructionAccess for Instructions {
    type Instruction = Instruction;

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytearray = Vec::with_capacity(self.0.len() * 2);

        for instruction in self.0.iter() {
            bytearray.push(instruction.get_opcode().into());
            bytearray.push(instruction.get_raw_value())
        }

        bytearray
    }
}

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

    /// Returns a hashmap of jump indexes and their jump target
    pub fn get_jump_map(&self) -> HashMap<u32, u32> {
        let mut jump_map: HashMap<u32, u32> = HashMap::new();

        for (index, instruction) in self.iter().enumerate() {
            let jump: Jump = match instruction {
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
                    let arg = self.get_full_arg(index);

                    let arg = match arg {
                        Some(arg) => arg,
                        None => continue,
                    };

                    Jump::Relative(RelativeJump {
                        index: arg,
                        direction: JumpDirection::Forward,
                    })
                }
                Instruction::JumpBackwardNoInterrupt(_)
                | Instruction::JumpBackward(_)
                | Instruction::InstrumentedJumpBackward(_) => {
                    let arg = self.get_full_arg(index);

                    let arg = match arg {
                        Some(arg) => arg,
                        None => continue,
                    };

                    Jump::Relative(RelativeJump {
                        index: arg,
                        direction: JumpDirection::Backward,
                    })
                }
                _ => continue,
            };

            let jump_target = self.get_jump_target(index as u32, jump);

            if let Some((jump_index, _)) = jump_target {
                jump_map.insert(index as u32, jump_index);
            }
        }

        jump_map
    }

    /// Returns the index and the instruction of the jump target. None if the index is invalid.
    pub fn get_jump_target(&self, index: u32, jump: Jump) -> Option<(u32, Instruction)> {
        match jump {
            Jump::Relative(RelativeJump {
                index: jump_index,
                direction: JumpDirection::Forward,
            }) => {
                let index = get_real_jump_index(self, index as usize)? as u32 + jump_index + 1;
                self.0
                    .get(index as usize)
                    .cloned()
                    .map(|target| (index, target))
            }
            Jump::Relative(RelativeJump {
                index: jump_index,
                direction: JumpDirection::Backward,
            }) => {
                let index = get_real_jump_index(self, index as usize)? as u32 - jump_index + 1;
                self.0
                    .get(index as usize)
                    .cloned()
                    .map(|target| (index, target))
            }
        }
    }

    /// Returns a list of all indexes that jump to the given index
    pub fn get_jump_xrefs(&self, index: u32) -> Vec<u32> {
        let jump_map = self.get_jump_map();

        jump_map
            .iter()
            .filter(|(_, to)| **to == index)
            .map(|(from, _)| *from)
            .collect()
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
            let opcode = Opcode::from(chunk[0]);
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
