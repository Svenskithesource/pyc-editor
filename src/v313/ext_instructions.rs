use std::ops::{Deref, DerefMut};

use store_interval_tree::{Interval, IntervalTree};

use crate::{
    error::Error,
    traits::{
        ExtInstructionsOwned, GenericInstruction, InstructionAccess, InstructionsOwned,
        SimpleInstructionAccess,
    },
    utils::get_extended_args_count,
    v313::{
        cache::get_cache_count,
        code_objects::{
            AttrNameIndex, AwaitableWhere, BinaryOperation, CallExFlags, ClosureRefIndex,
            ConstIndex, DynamicIndex, FunctionAttributeFlags, GlobalNameIndex, Intrinsic1Functions,
            Intrinsic2Functions, Jump, JumpDirection, NameIndex, OpInversion, RaiseForms,
            RelativeJump, Reraise, ResumeWhere, SliceCount, SuperAttrNameIndex, VarNameIndex,
        },
        instructions::{Instruction, Instructions},
        opcodes::Opcode,
    },
};
use crate::{
    traits::ExtInstructionAccess,
    v313::{
        code_objects::{CompareOperation, ConvertFormat},
        instructions,
    },
};

/// Used to represent opargs for opcodes that don't require arguments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UnusedArgument(u32);

impl From<u32> for UnusedArgument {
    fn from(value: u32) -> Self {
        UnusedArgument(value)
    }
}

/// Low level representation of a Python bytecode instruction with resolved arguments (extended arg is resolved)
/// We have arguments for every opcode, even if those aren't used. This is so we can have a full representation of the instructions, even if they're invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtInstruction {
    Cache(UnusedArgument),
    BeforeAsyncWith(UnusedArgument),
    BeforeWith(UnusedArgument),
    // Specialized op which we don't parse the oparg for
    BinaryOpInplaceAddUnicode(u32),
    BinarySlice(UnusedArgument),
    BinarySubscr(UnusedArgument),
    CheckEgMatch(UnusedArgument),
    CheckExcMatch(UnusedArgument),
    CleanupThrow(UnusedArgument),
    DeleteSubscr(UnusedArgument),
    EndAsyncFor(UnusedArgument),
    EndFor(UnusedArgument),
    EndSend(UnusedArgument),
    ExitInitCheck(UnusedArgument),
    FormatSimple(UnusedArgument),
    FormatWithSpec(UnusedArgument),
    GetAiter(UnusedArgument),
    Reserved(UnusedArgument),
    GetAnext(UnusedArgument),
    GetIter(UnusedArgument),
    GetLen(UnusedArgument),
    GetYieldFromIter(UnusedArgument),
    InterpreterExit(UnusedArgument),
    LoadAssertionError(UnusedArgument),
    LoadBuildClass(UnusedArgument),
    LoadLocals(UnusedArgument),
    MakeFunction(UnusedArgument),
    MatchKeys(UnusedArgument),
    MatchMapping(UnusedArgument),
    MatchSequence(UnusedArgument),
    Nop(UnusedArgument),
    PopExcept(UnusedArgument),
    PopTop(UnusedArgument),
    PushExcInfo(UnusedArgument),
    PushNull(UnusedArgument),
    ReturnGenerator(UnusedArgument),
    ReturnValue(UnusedArgument),
    SetupAnnotations(UnusedArgument),
    StoreSlice(UnusedArgument),
    StoreSubscr(UnusedArgument),
    ToBool(UnusedArgument),
    UnaryInvert(UnusedArgument),
    UnaryNegative(UnusedArgument),
    UnaryNot(UnusedArgument),
    WithExceptStart(UnusedArgument),
    BinaryOp(BinaryOperation),
    BuildConstKeyMap(u32),
    BuildList(u32),
    BuildMap(u32),
    BuildSet(u32),
    BuildSlice(SliceCount),
    BuildString(u32),
    BuildTuple(u32),
    Call(u32),
    CallFunctionEx(CallExFlags),
    CallIntrinsic1(Intrinsic1Functions),
    CallIntrinsic2(Intrinsic2Functions),
    CallKw(u32),
    CompareOp(CompareOperation),
    ContainsOp(OpInversion),
    ConvertValue(ConvertFormat),
    Copy(u32),
    CopyFreeVars(u32),
    DeleteAttr(NameIndex),
    DeleteDeref(ClosureRefIndex),
    DeleteFast(VarNameIndex),
    DeleteGlobal(NameIndex),
    DeleteName(NameIndex),
    DictMerge(u32),
    DictUpdate(u32),
    EnterExecutor(u32),
    // Extended arg is ommited in the resolved instructions
    ForIter(RelativeJump),
    GetAwaitable(AwaitableWhere),
    ImportFrom(NameIndex),
    ImportName(NameIndex),
    IsOp(OpInversion),
    JumpBackward(RelativeJump),
    JumpBackwardNoInterrupt(RelativeJump),
    JumpForward(RelativeJump),
    ListAppend(u32),
    ListExtend(u32),
    LoadAttr(AttrNameIndex),
    LoadConst(ConstIndex),
    LoadDeref(ClosureRefIndex),
    LoadFast(VarNameIndex),
    LoadFastAndClear(VarNameIndex),
    LoadFastCheck(VarNameIndex),
    LoadFastLoadFast((VarNameIndex, VarNameIndex)),
    LoadFromDictOrGlobals(DynamicIndex),
    LoadFromDictOrDeref(DynamicIndex),
    LoadGlobal(GlobalNameIndex),
    LoadName(NameIndex),
    LoadSuperAttr(SuperAttrNameIndex),
    MakeCell(ClosureRefIndex),
    MapAdd(u32),
    MatchClass(u32),
    PopJumpIfFalse(RelativeJump),
    PopJumpIfNone(RelativeJump),
    PopJumpIfNotNone(RelativeJump),
    PopJumpIfTrue(RelativeJump),
    RaiseVarargs(RaiseForms),
    Reraise(Reraise),
    ReturnConst(ConstIndex),
    Send(RelativeJump),
    SetAdd(u32),
    SetFunctionAttribute(FunctionAttributeFlags),
    SetUpdate(u32),
    StoreAttr(NameIndex),
    StoreDeref(ClosureRefIndex),
    StoreFast(VarNameIndex),
    StoreFastLoadFast((VarNameIndex, VarNameIndex)),
    StoreFastStoreFast((VarNameIndex, VarNameIndex)),
    StoreGlobal(NameIndex),
    StoreName(NameIndex),
    Swap(u32),
    UnpackEx(u32),
    UnpackSequence(u32),
    YieldValue(u32),
    Resume(ResumeWhere),
    // Specialized variations of opcodes, we don't parse the arguments for these (with some exceptions)
    BinaryOpAddFloat(u32),
    BinaryOpAddInt(u32),
    BinaryOpAddUnicode(u32),
    BinaryOpMultiplyFloat(u32),
    BinaryOpMultiplyInt(u32),
    BinaryOpSubtractFloat(u32),
    BinaryOpSubtractInt(u32),
    BinarySubscrDict(u32),
    BinarySubscrGetitem(u32),
    BinarySubscrListInt(u32),
    BinarySubscrStrInt(u32),
    BinarySubscrTupleInt(u32),
    CallAllocAndEnterInit(u32),
    CallBoundMethodExactArgs(u32),
    CallBoundMethodGeneral(u32),
    CallBuiltinClass(u32),
    CallBuiltinFast(u32),
    CallBuiltinFastWithKeywords(u32),
    CallBuiltinO(u32),
    CallIsinstance(u32),
    CallLen(u32),
    CallListAppend(u32),
    CallMethodDescriptorFast(u32),
    CallMethodDescriptorFastWithKeywords(u32),
    CallMethodDescriptorNoargs(u32),
    CallMethodDescriptorO(u32),
    CallNonPyGeneral(u32),
    CallPyExactArgs(u32),
    CallPyGeneral(u32),
    CallStr1(u32),
    CallTuple1(u32),
    CallType1(u32),
    CompareOpFloat(u32),
    CompareOpInt(u32),
    CompareOpStr(u32),
    ContainsOpDict(u32),
    ContainsOpSet(u32),
    ForIterGen(RelativeJump),
    ForIterList(RelativeJump),
    ForIterRange(RelativeJump),
    ForIterTuple(RelativeJump),
    LoadAttrClass(u32),
    LoadAttrGetattributeOverridden(u32),
    LoadAttrInstanceValue(u32),
    LoadAttrMethodLazyDict(u32),
    LoadAttrMethodNoDict(u32),
    LoadAttrMethodWithValues(u32),
    LoadAttrModule(u32),
    LoadAttrNondescriptorNoDict(u32),
    LoadAttrNondescriptorWithValues(u32),
    LoadAttrProperty(u32),
    LoadAttrSlot(u32),
    LoadAttrWithHint(u32),
    LoadGlobalBuiltin(u32),
    LoadGlobalModule(u32),
    LoadSuperAttrAttr(u32),
    LoadSuperAttrMethod(u32),
    ResumeCheck(u32),
    SendGen(u32),
    StoreAttrInstanceValue(u32),
    StoreAttrSlot(u32),
    StoreAttrWithHint(u32),
    StoreSubscrDict(u32),
    StoreSubscrListInt(u32),
    ToBoolAlwaysTrue(u32),
    ToBoolBool(u32),
    ToBoolInt(u32),
    ToBoolList(u32),
    ToBoolNone(u32),
    ToBoolStr(u32),
    UnpackSequenceList(u32),
    UnpackSequenceTuple(u32),
    UnpackSequenceTwoTuple(u32),
    InstrumentedResume(u32),
    InstrumentedEndFor(u32),
    InstrumentedEndSend(u32),
    InstrumentedReturnValue(u32),
    InstrumentedReturnConst(u32),
    InstrumentedYieldValue(u32),
    InstrumentedLoadSuperAttr(u32),
    InstrumentedForIter(RelativeJump),
    InstrumentedCall(u32),
    InstrumentedCallKw(u32),
    InstrumentedCallFunctionEx(u32),
    InstrumentedInstruction(u32),
    InstrumentedJumpForward(RelativeJump),
    InstrumentedJumpBackward(RelativeJump),
    InstrumentedPopJumpIfTrue(RelativeJump),
    InstrumentedPopJumpIfFalse(RelativeJump),
    InstrumentedPopJumpIfNone(RelativeJump),
    InstrumentedPopJumpIfNotNone(RelativeJump),
    InstrumentedLine(u32),
    InvalidOpcode((u8, u32)), // (opcode, arg)
}

/// A list of resolved instructions (extended_arg is resolved)
#[derive(Debug, Clone, PartialEq)]
pub struct ExtInstructions(Vec<ExtInstruction>);

impl<T> InstructionAccess<u32, ExtInstruction> for T
where
    T: Deref<Target = [ExtInstruction]> + AsRef<[ExtInstruction]>,
{
    type Instruction = ExtInstruction;
    type Jump = Jump;

    /// Returns the index and the instruction of the jump target. None if the index is invalid.
    fn get_jump_target(&self, index: u32) -> Option<(u32, ExtInstruction)> {
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

    fn get_jump_value(&self, index: u32) -> Option<Self::Jump> {
        match self.get(index as usize)? {
            ExtInstruction::ForIter(jump)
            | ExtInstruction::JumpForward(jump)
            | ExtInstruction::PopJumpIfFalse(jump)
            | ExtInstruction::PopJumpIfTrue(jump)
            | ExtInstruction::Send(jump)
            | ExtInstruction::PopJumpIfNotNone(jump)
            | ExtInstruction::PopJumpIfNone(jump)
            | ExtInstruction::ForIterRange(jump)
            | ExtInstruction::ForIterList(jump)
            | ExtInstruction::ForIterGen(jump)
            | ExtInstruction::ForIterTuple(jump)
            | ExtInstruction::InstrumentedForIter(jump)
            | ExtInstruction::InstrumentedPopJumpIfNone(jump)
            | ExtInstruction::InstrumentedPopJumpIfNotNone(jump)
            | ExtInstruction::InstrumentedJumpForward(jump)
            | ExtInstruction::InstrumentedPopJumpIfFalse(jump)
            | ExtInstruction::InstrumentedPopJumpIfTrue(jump)
            | ExtInstruction::JumpBackwardNoInterrupt(jump)
            | ExtInstruction::JumpBackward(jump)
            | ExtInstruction::InstrumentedJumpBackward(jump) => Some((*jump).into()),
            _ => None,
        }
    }
}

impl<T> ExtInstructionAccess<Instruction> for T
where
    T: Deref<Target = [ExtInstruction]> + AsRef<[ExtInstruction]>,
{
    type Instructions = Instructions;

    fn to_instructions(&self) -> Self::Instructions {
        // mapping of original to updated index
        let mut relative_jump_indexes = IntervalTree::<u32, u32>::new(); // (u32, u32) is the from and to index for relative jumps

        self.iter().enumerate().for_each(|(idx, inst)| match inst {
            ExtInstruction::ForIter(jump)
            | ExtInstruction::JumpForward(jump)
            | ExtInstruction::PopJumpIfFalse(jump)
            | ExtInstruction::PopJumpIfTrue(jump)
            | ExtInstruction::Send(jump)
            | ExtInstruction::PopJumpIfNotNone(jump)
            | ExtInstruction::PopJumpIfNone(jump)
            | ExtInstruction::ForIterRange(jump)
            | ExtInstruction::ForIterList(jump)
            | ExtInstruction::ForIterGen(jump)
            | ExtInstruction::ForIterTuple(jump)
            | ExtInstruction::InstrumentedForIter(jump)
            | ExtInstruction::InstrumentedPopJumpIfNone(jump)
            | ExtInstruction::InstrumentedPopJumpIfNotNone(jump)
            | ExtInstruction::InstrumentedJumpForward(jump)
            | ExtInstruction::InstrumentedPopJumpIfFalse(jump)
            | ExtInstruction::InstrumentedPopJumpIfTrue(jump)
            | ExtInstruction::JumpBackwardNoInterrupt(jump)
            | ExtInstruction::JumpBackward(jump)
            | ExtInstruction::InstrumentedJumpBackward(jump) => match jump {
                RelativeJump {
                    index,
                    direction: JumpDirection::Forward,
                } => {
                    let idx = get_real_jump_index(self, idx).expect("Index is always valid here");

                    relative_jump_indexes.insert(
                        Interval::new(
                            std::ops::Bound::Excluded(idx as u32),
                            std::ops::Bound::Excluded(idx as u32 + index + 1),
                        ),
                        *index,
                    );
                }
                RelativeJump {
                    index,
                    direction: JumpDirection::Backward,
                } => {
                    let idx = get_real_jump_index(self, idx).expect("Index is always valid here");

                    relative_jump_indexes.insert(
                        Interval::new(
                            std::ops::Bound::Included(idx as u32 - index + 1),
                            std::ops::Bound::Included(idx as u32),
                        ),
                        *index,
                    );
                }
            },
            _ => {}
        });

        // We keep a list of jump indexes that become bigger than 255 while recalculating the jump indexes.
        // We will need to account for those afterwards.
        let mut relative_jumps_to_update = vec![];

        for (index, instruction) in self.iter().enumerate() {
            let arg = instruction.get_raw_value();

            if arg > u8::MAX.into() {
                // Calculate how many extended args an instruction will need
                let extended_arg_count = get_extended_args_count(arg) as u32;

                let index = get_real_jump_index(self, index).expect("Index is always valid here");

                for mut entry in relative_jump_indexes.query_mut(&Interval::point(index as u32)) {
                    let interval_clone = (*entry.interval()).clone();
                    let entry_value = entry.value();

                    if *entry_value <= u8::MAX.into()
                        && *entry_value + extended_arg_count > u8::MAX.into()
                    {
                        relative_jumps_to_update.push(interval_clone);
                    }

                    *entry_value += extended_arg_count;
                }
            }
        }

        // Keep updating the offsets until there are no new extended args that need to be accounted for
        while !relative_jumps_to_update.is_empty() {
            let relative_clone = relative_jumps_to_update.clone();

            relative_jumps_to_update.clear();

            for (index, instruction) in self.iter().enumerate() {
                let index = get_real_jump_index(self, index).expect("Index is always valid here");

                let arg = match instruction {
                    ExtInstruction::ForIter(jump)
                    | ExtInstruction::JumpForward(jump)
                    | ExtInstruction::PopJumpIfFalse(jump)
                    | ExtInstruction::PopJumpIfTrue(jump)
                    | ExtInstruction::Send(jump)
                    | ExtInstruction::PopJumpIfNotNone(jump)
                    | ExtInstruction::PopJumpIfNone(jump)
                    | ExtInstruction::ForIterRange(jump)
                    | ExtInstruction::ForIterList(jump)
                    | ExtInstruction::ForIterGen(jump)
                    | ExtInstruction::ForIterTuple(jump)
                    | ExtInstruction::InstrumentedForIter(jump)
                    | ExtInstruction::InstrumentedPopJumpIfNone(jump)
                    | ExtInstruction::InstrumentedPopJumpIfNotNone(jump)
                    | ExtInstruction::InstrumentedJumpForward(jump)
                    | ExtInstruction::InstrumentedPopJumpIfFalse(jump)
                    | ExtInstruction::InstrumentedPopJumpIfTrue(jump)
                    | ExtInstruction::JumpBackwardNoInterrupt(jump)
                    | ExtInstruction::JumpBackward(jump)
                    | ExtInstruction::InstrumentedJumpBackward(jump) => {
                        let interval = match jump {
                            RelativeJump {
                                index: _,
                                direction: JumpDirection::Forward,
                            } => Interval::new(
                                std::ops::Bound::Excluded(index as u32),
                                std::ops::Bound::Excluded(index as u32 + jump.index + 1),
                            ),
                            RelativeJump {
                                index: _,
                                direction: JumpDirection::Backward,
                            } => Interval::new(
                                std::ops::Bound::Included(index as u32 - jump.index + 1),
                                std::ops::Bound::Included(index as u32),
                            ),
                        };

                        if relative_clone.contains(&interval) {
                            *relative_jump_indexes
                                .query(&interval)
                                .find(|e| *e.interval() == interval)
                                .expect("The jump table should always contain all jump indexes")
                                .value()
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                };

                let extended_arg_count = get_extended_args_count(arg) as u32;

                for mut entry in relative_jump_indexes.query_mut(&Interval::point(index as u32)) {
                    let interval_clone = (*entry.interval()).clone();
                    let entry_value = entry.value();

                    if *entry_value <= u8::MAX.into()
                        && *entry_value + extended_arg_count > u8::MAX.into()
                    {
                        relative_jumps_to_update.push(interval_clone);
                    }

                    *entry_value += extended_arg_count;
                }
            }
        }

        let mut instructions: Instructions = Instructions::with_capacity(self.len() * 2); // This will not be enough this as we dynamically generate EXTENDED_ARGS, but it's better than not reserving any length.

        for (index, instruction) in self.iter().enumerate() {
            let index = get_real_jump_index(self, index).expect("Index is always valid here");

            let arg = match instruction {
                ExtInstruction::ForIter(jump)
                | ExtInstruction::JumpForward(jump)
                | ExtInstruction::PopJumpIfFalse(jump)
                | ExtInstruction::PopJumpIfTrue(jump)
                | ExtInstruction::Send(jump)
                | ExtInstruction::PopJumpIfNotNone(jump)
                | ExtInstruction::PopJumpIfNone(jump)
                | ExtInstruction::ForIterRange(jump)
                | ExtInstruction::ForIterList(jump)
                | ExtInstruction::ForIterGen(jump)
                | ExtInstruction::ForIterTuple(jump)
                | ExtInstruction::InstrumentedForIter(jump)
                | ExtInstruction::InstrumentedPopJumpIfNone(jump)
                | ExtInstruction::InstrumentedPopJumpIfNotNone(jump)
                | ExtInstruction::InstrumentedJumpForward(jump)
                | ExtInstruction::InstrumentedPopJumpIfFalse(jump)
                | ExtInstruction::InstrumentedPopJumpIfTrue(jump)
                | ExtInstruction::JumpBackwardNoInterrupt(jump)
                | ExtInstruction::JumpBackward(jump)
                | ExtInstruction::InstrumentedJumpBackward(jump) => {
                    let interval = match jump {
                        RelativeJump {
                            index: _,
                            direction: JumpDirection::Forward,
                        } => Interval::new(
                            std::ops::Bound::Excluded(index as u32),
                            std::ops::Bound::Excluded(index as u32 + jump.index + 1),
                        ),
                        RelativeJump {
                            index: _,
                            direction: JumpDirection::Backward,
                        } => Interval::new(
                            std::ops::Bound::Included(index as u32 - jump.index + 1),
                            std::ops::Bound::Included(index as u32),
                        ),
                    };

                    *relative_jump_indexes
                        .query(&interval)
                        .find(|e| *e.interval() == interval)
                        .expect("The jump table should always contain all jump indexes")
                        .value()
                }
                _ => instruction.get_raw_value(),
            };

            // Emit EXTENDED_ARGs for arguments > 0xFF
            if arg > u8::MAX.into() {
                for ext in get_extended_args(arg) {
                    instructions.append_instruction(ext);
                }
            }

            instructions.append_instruction((instruction.get_opcode(), (arg & 0xff) as u8).into());
        }

        instructions
    }
}

impl InstructionsOwned<ExtInstruction> for ExtInstructions {
    type Instruction = ExtInstruction;

    fn push(&mut self, item: Self::Instruction) {
        self.0.push(item);
    }
}

impl ExtInstructionsOwned<ExtInstruction> for ExtInstructions {
    type Instruction = ExtInstruction;

    fn delete_instruction(&mut self, index: usize) {
        self.0.iter_mut().enumerate().for_each(|(idx, inst)| {
            match inst {
                ExtInstruction::ForIter(jump)
                | ExtInstruction::JumpForward(jump)
                | ExtInstruction::PopJumpIfFalse(jump)
                | ExtInstruction::PopJumpIfTrue(jump)
                | ExtInstruction::Send(jump)
                | ExtInstruction::PopJumpIfNotNone(jump)
                | ExtInstruction::PopJumpIfNone(jump)
                | ExtInstruction::ForIterRange(jump)
                | ExtInstruction::ForIterList(jump)
                | ExtInstruction::ForIterGen(jump)
                | ExtInstruction::ForIterTuple(jump)
                | ExtInstruction::InstrumentedForIter(jump)
                | ExtInstruction::InstrumentedPopJumpIfNone(jump)
                | ExtInstruction::InstrumentedPopJumpIfNotNone(jump)
                | ExtInstruction::InstrumentedJumpForward(jump)
                | ExtInstruction::InstrumentedPopJumpIfFalse(jump)
                | ExtInstruction::InstrumentedPopJumpIfTrue(jump) => {
                    // Relative jumps only need to update if the index falls within it's jump range
                    if idx <= index && index + idx <= jump.index as usize {
                        jump.index -= 1
                    }
                }
                ExtInstruction::JumpBackwardNoInterrupt(jump)
                | ExtInstruction::JumpBackward(jump)
                | ExtInstruction::InstrumentedJumpBackward(jump) => {
                    // Relative jumps only need to update if the index falls within it's jump range
                    if idx > index && index + idx >= jump.index as usize {
                        jump.index -= 1
                    }
                }
                _ => {}
            }
        });

        self.0.remove(index);
    }

    fn insert_instruction(&mut self, index: usize, instruction: Self::Instruction) {
        self.0.iter_mut().enumerate().for_each(|(idx, inst)| {
            match inst {
                ExtInstruction::ForIter(jump)
                | ExtInstruction::JumpForward(jump)
                | ExtInstruction::PopJumpIfFalse(jump)
                | ExtInstruction::PopJumpIfTrue(jump)
                | ExtInstruction::Send(jump)
                | ExtInstruction::PopJumpIfNotNone(jump)
                | ExtInstruction::PopJumpIfNone(jump)
                | ExtInstruction::ForIterRange(jump)
                | ExtInstruction::ForIterList(jump)
                | ExtInstruction::ForIterGen(jump)
                | ExtInstruction::ForIterTuple(jump)
                | ExtInstruction::InstrumentedForIter(jump)
                | ExtInstruction::InstrumentedPopJumpIfNone(jump)
                | ExtInstruction::InstrumentedPopJumpIfNotNone(jump)
                | ExtInstruction::InstrumentedJumpForward(jump)
                | ExtInstruction::InstrumentedPopJumpIfFalse(jump)
                | ExtInstruction::InstrumentedPopJumpIfTrue(jump) => {
                    // Relative jumps only need to update if the index falls within it's jump range
                    if idx <= index && index + idx <= jump.index as usize {
                        jump.index += 1
                    }
                }
                ExtInstruction::JumpBackwardNoInterrupt(jump)
                | ExtInstruction::JumpBackward(jump)
                | ExtInstruction::InstrumentedJumpBackward(jump) => {
                    // Relative jumps only need to update if the index falls within it's jump range
                    if idx > index && index + idx >= jump.index as usize {
                        jump.index += 1
                    }
                }
                _ => {}
            }
        });
        self.0.insert(index, instruction);
    }
}

/// Resolves the actual index of the current jump instruction.
/// In 3.13 the jump offsets are relative to the CACHE opcodes succeeding the jump instruction.
/// They're calculated from the predefined cache layout. This does not guarantee the index is actually valid.
pub fn get_real_jump_index(instructions: &[ExtInstruction], index: usize) -> Option<usize> {
    Some(index + get_cache_count(instructions.get(index)?.get_opcode()).unwrap_or(0))
}

impl ExtInstructions {
    pub fn with_capacity(capacity: usize) -> Self {
        ExtInstructions(Vec::with_capacity(capacity))
    }

    pub fn new(instructions: Vec<ExtInstruction>) -> Self {
        ExtInstructions(instructions)
    }

    /// Resolve instructions into extended instructions.
    pub fn from_instructions(instructions: &[Instruction]) -> Result<Self, Error> {
        if !instructions.find_ext_arg_jumps().is_empty() {
            return Err(Error::ExtendedArgJump);
        }

        let mut extended_arg = 0; // Used to keep track of extended arguments between instructions
        let mut relative_jump_indexes: IntervalTree<u32, u32> = IntervalTree::new();

        for (index, instruction) in instructions.iter().enumerate() {
            match instruction {
                Instruction::ExtendedArg(arg) => {
                    let arg = *arg as u32 | extended_arg;
                    extended_arg = arg << 8;
                    continue;
                }
                Instruction::ForIter(arg)
                | Instruction::JumpForward(arg)
                | Instruction::PopJumpIfFalse(arg)
                | Instruction::PopJumpIfTrue(arg)
                | Instruction::Send(arg)
                | Instruction::PopJumpIfNotNone(arg)
                | Instruction::PopJumpIfNone(arg)
                | Instruction::ForIterRange(arg)
                | Instruction::ForIterList(arg)
                | Instruction::ForIterGen(arg)
                | Instruction::ForIterTuple(arg)
                | Instruction::InstrumentedForIter(arg)
                | Instruction::InstrumentedPopJumpIfNone(arg)
                | Instruction::InstrumentedPopJumpIfNotNone(arg)
                | Instruction::InstrumentedJumpForward(arg)
                | Instruction::InstrumentedPopJumpIfFalse(arg)
                | Instruction::InstrumentedPopJumpIfTrue(arg) => {
                    let arg = *arg as u32 | extended_arg;
                    let index = instructions::get_real_jump_index(instructions, index)
                        .expect("Index is always valid here");

                    relative_jump_indexes.insert(
                        Interval::new(
                            std::ops::Bound::Excluded(index as u32),
                            std::ops::Bound::Excluded(index as u32 + arg + 1),
                        ),
                        arg,
                    );
                }
                Instruction::JumpBackwardNoInterrupt(arg)
                | Instruction::JumpBackward(arg)
                | Instruction::InstrumentedJumpBackward(arg) => {
                    let arg = *arg as u32 | extended_arg;
                    let index = instructions::get_real_jump_index(instructions, index)
                        .expect("Index is always valid here");

                    relative_jump_indexes.insert(
                        Interval::new(
                            std::ops::Bound::Included(index as u32 - arg + 1),
                            std::ops::Bound::Excluded(index as u32),
                        ),
                        arg,
                    );
                }
                _ => {}
            }

            extended_arg = 0;
        }

        for (index, instruction) in instructions.iter().enumerate() {
            if let Instruction::ExtendedArg(_) = instruction {
                let index = instructions::get_real_jump_index(instructions, index)
                    .expect("Index is always valid here");

                for mut entry in relative_jump_indexes.query_mut(&Interval::point(index as u32)) {
                    *entry.value() -= 1
                }
            }
        }

        let mut ext_instructions = ExtInstructions::with_capacity(instructions.len());

        for (index, instruction) in instructions.iter().enumerate() {
            let arg = match instruction {
                Instruction::ExtendedArg(arg) => {
                    let arg = *arg as u32 | extended_arg;
                    extended_arg = arg << 8;
                    continue;
                }
                Instruction::ForIter(arg)
                | Instruction::JumpForward(arg)
                | Instruction::PopJumpIfFalse(arg)
                | Instruction::PopJumpIfTrue(arg)
                | Instruction::Send(arg)
                | Instruction::PopJumpIfNotNone(arg)
                | Instruction::PopJumpIfNone(arg)
                | Instruction::ForIterRange(arg)
                | Instruction::ForIterList(arg)
                | Instruction::ForIterGen(arg)
                | Instruction::ForIterTuple(arg)
                | Instruction::InstrumentedForIter(arg)
                | Instruction::InstrumentedPopJumpIfNone(arg)
                | Instruction::InstrumentedPopJumpIfNotNone(arg)
                | Instruction::InstrumentedJumpForward(arg)
                | Instruction::InstrumentedPopJumpIfFalse(arg)
                | Instruction::InstrumentedPopJumpIfTrue(arg) => {
                    let index = instructions::get_real_jump_index(instructions, index)
                        .expect("Index is always valid here");

                    let interval = Interval::new(
                        std::ops::Bound::Excluded(index as u32),
                        std::ops::Bound::Excluded(index as u32 + (*arg as u32 | extended_arg) + 1),
                    );

                    *relative_jump_indexes
                        .query(&interval)
                        .find(|e| *e.interval() == interval)
                        .expect("The jump table should always contain all jump indexes")
                        .value()
                }
                Instruction::JumpBackwardNoInterrupt(arg)
                | Instruction::JumpBackward(arg)
                | Instruction::InstrumentedJumpBackward(arg) => {
                    let index = instructions::get_real_jump_index(instructions, index)
                        .expect("Index is always valid here");

                    let interval = Interval::new(
                        std::ops::Bound::Included(index as u32 - (*arg as u32 | extended_arg) + 1),
                        std::ops::Bound::Excluded(index as u32),
                    );

                    *relative_jump_indexes
                        .query(&interval)
                        .find(|e| *e.interval() == interval)
                        .expect("The jump table should always contain all jump indexes")
                        .value()
                }
                _ => instruction.get_raw_value() as u32 | extended_arg,
            };

            ext_instructions.append_instruction(
                (instruction.get_opcode(), arg)
                    .try_into()
                    .expect("This will never error, as we know it's not an EXTENDED_ARG"),
            );

            extended_arg = 0;
        }

        Ok(ext_instructions)
    }
}

impl Deref for ExtInstructions {
    type Target = [ExtInstruction];

    /// Allow the user to get a reference slice to the instructions
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl DerefMut for ExtInstructions {
    /// Allow the user to get a mutable reference slice for making modifications to existing instructions.
    fn deref_mut(&mut self) -> &mut [ExtInstruction] {
        self.0.deref_mut()
    }
}

impl AsRef<[ExtInstruction]> for ExtInstructions {
    fn as_ref(&self) -> &[ExtInstruction] {
        &self.0
    }
}

impl From<ExtInstructions> for Vec<u8> {
    fn from(val: ExtInstructions) -> Self {
        val.to_bytes()
    }
}

impl TryFrom<&[Instruction]> for ExtInstructions {
    type Error = Error;

    fn try_from(value: &[Instruction]) -> Result<Self, Self::Error> {
        ExtInstructions::from_instructions(value)
    }
}

impl From<&[ExtInstruction]> for ExtInstructions {
    fn from(value: &[ExtInstruction]) -> Self {
        ExtInstructions::new(value.to_vec())
    }
}

impl TryFrom<(Opcode, u32)> for ExtInstruction {
    type Error = Error;
    fn try_from(value: (Opcode, u32)) -> Result<Self, Self::Error> {
        Ok(match value.0 {
            Opcode::CACHE => ExtInstruction::Cache(value.1.into()),
            Opcode::BEFORE_ASYNC_WITH => ExtInstruction::BeforeAsyncWith(value.1.into()),
            Opcode::BEFORE_WITH => ExtInstruction::BeforeWith(value.1.into()),
            Opcode::BINARY_OP_INPLACE_ADD_UNICODE => {
                ExtInstruction::BinaryOpInplaceAddUnicode(value.1)
            }
            Opcode::BINARY_SLICE => ExtInstruction::BinarySlice(value.1.into()),
            Opcode::BINARY_SUBSCR => ExtInstruction::BinarySubscr(value.1.into()),
            Opcode::CHECK_EG_MATCH => ExtInstruction::CheckEgMatch(value.1.into()),
            Opcode::CHECK_EXC_MATCH => ExtInstruction::CheckExcMatch(value.1.into()),
            Opcode::CLEANUP_THROW => ExtInstruction::CleanupThrow(value.1.into()),
            Opcode::DELETE_SUBSCR => ExtInstruction::DeleteSubscr(value.1.into()),
            Opcode::END_ASYNC_FOR => ExtInstruction::EndAsyncFor(value.1.into()),
            Opcode::END_FOR => ExtInstruction::EndFor(value.1.into()),
            Opcode::END_SEND => ExtInstruction::EndSend(value.1.into()),
            Opcode::EXIT_INIT_CHECK => ExtInstruction::ExitInitCheck(value.1.into()),
            Opcode::FORMAT_SIMPLE => ExtInstruction::FormatSimple(value.1.into()),
            Opcode::FORMAT_WITH_SPEC => ExtInstruction::FormatWithSpec(value.1.into()),
            Opcode::GET_AITER => ExtInstruction::GetAiter(value.1.into()),
            Opcode::RESERVED => ExtInstruction::Reserved(value.1.into()),
            Opcode::GET_ANEXT => ExtInstruction::GetAnext(value.1.into()),
            Opcode::GET_ITER => ExtInstruction::GetIter(value.1.into()),
            Opcode::GET_LEN => ExtInstruction::GetLen(value.1.into()),
            Opcode::GET_YIELD_FROM_ITER => ExtInstruction::GetYieldFromIter(value.1.into()),
            Opcode::INTERPRETER_EXIT => ExtInstruction::InterpreterExit(value.1.into()),
            Opcode::LOAD_ASSERTION_ERROR => ExtInstruction::LoadAssertionError(value.1.into()),
            Opcode::LOAD_BUILD_CLASS => ExtInstruction::LoadBuildClass(value.1.into()),
            Opcode::LOAD_LOCALS => ExtInstruction::LoadLocals(value.1.into()),
            Opcode::MAKE_FUNCTION => ExtInstruction::MakeFunction(value.1.into()),
            Opcode::MATCH_KEYS => ExtInstruction::MatchKeys(value.1.into()),
            Opcode::MATCH_MAPPING => ExtInstruction::MatchMapping(value.1.into()),
            Opcode::MATCH_SEQUENCE => ExtInstruction::MatchSequence(value.1.into()),
            Opcode::NOP => ExtInstruction::Nop(value.1.into()),
            Opcode::POP_EXCEPT => ExtInstruction::PopExcept(value.1.into()),
            Opcode::POP_TOP => ExtInstruction::PopTop(value.1.into()),
            Opcode::PUSH_EXC_INFO => ExtInstruction::PushExcInfo(value.1.into()),
            Opcode::PUSH_NULL => ExtInstruction::PushNull(value.1.into()),
            Opcode::RETURN_GENERATOR => ExtInstruction::ReturnGenerator(value.1.into()),
            Opcode::RETURN_VALUE => ExtInstruction::ReturnValue(value.1.into()),
            Opcode::SETUP_ANNOTATIONS => ExtInstruction::SetupAnnotations(value.1.into()),
            Opcode::STORE_SLICE => ExtInstruction::StoreSlice(value.1.into()),
            Opcode::STORE_SUBSCR => ExtInstruction::StoreSubscr(value.1.into()),
            Opcode::TO_BOOL => ExtInstruction::ToBool(value.1.into()),
            Opcode::UNARY_INVERT => ExtInstruction::UnaryInvert(value.1.into()),
            Opcode::UNARY_NEGATIVE => ExtInstruction::UnaryNegative(value.1.into()),
            Opcode::UNARY_NOT => ExtInstruction::UnaryNot(value.1.into()),
            Opcode::WITH_EXCEPT_START => ExtInstruction::WithExceptStart(value.1.into()),
            Opcode::BINARY_OP => ExtInstruction::BinaryOp(value.1.into()),
            Opcode::BUILD_CONST_KEY_MAP => ExtInstruction::BuildConstKeyMap(value.1),
            Opcode::BUILD_LIST => ExtInstruction::BuildList(value.1),
            Opcode::BUILD_MAP => ExtInstruction::BuildMap(value.1),
            Opcode::BUILD_SET => ExtInstruction::BuildSet(value.1),
            Opcode::BUILD_SLICE => ExtInstruction::BuildSlice(value.1.into()),
            Opcode::BUILD_STRING => ExtInstruction::BuildString(value.1),
            Opcode::BUILD_TUPLE => ExtInstruction::BuildTuple(value.1),
            Opcode::CALL => ExtInstruction::Call(value.1),
            Opcode::CALL_FUNCTION_EX => ExtInstruction::CallFunctionEx(value.1.into()),
            Opcode::CALL_INTRINSIC_1 => ExtInstruction::CallIntrinsic1(value.1.into()),
            Opcode::CALL_INTRINSIC_2 => ExtInstruction::CallIntrinsic2(value.1.into()),
            Opcode::CALL_KW => ExtInstruction::CallKw(value.1),
            Opcode::COMPARE_OP => ExtInstruction::CompareOp(value.1.into()),
            Opcode::CONTAINS_OP => ExtInstruction::ContainsOp(value.1.into()),
            Opcode::CONVERT_VALUE => ExtInstruction::ConvertValue(value.1.into()),
            Opcode::COPY => ExtInstruction::Copy(value.1),
            Opcode::COPY_FREE_VARS => ExtInstruction::CopyFreeVars(value.1),
            Opcode::DELETE_ATTR => ExtInstruction::DeleteAttr(value.1.into()),
            Opcode::DELETE_DEREF => ExtInstruction::DeleteDeref(value.1.into()),
            Opcode::DELETE_FAST => ExtInstruction::DeleteFast(value.1.into()),
            Opcode::DELETE_GLOBAL => ExtInstruction::DeleteGlobal(value.1.into()),
            Opcode::DELETE_NAME => ExtInstruction::DeleteName(value.1.into()),
            Opcode::DICT_MERGE => ExtInstruction::DictMerge(value.1),
            Opcode::DICT_UPDATE => ExtInstruction::DictUpdate(value.1),
            Opcode::ENTER_EXECUTOR => ExtInstruction::EnterExecutor(value.1),
            Opcode::EXTENDED_ARG => return Err(Error::InvalidConversion),
            Opcode::FOR_ITER => ExtInstruction::ForIter(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::GET_AWAITABLE => ExtInstruction::GetAwaitable(value.1.into()),
            Opcode::IMPORT_FROM => ExtInstruction::ImportFrom(value.1.into()),
            Opcode::IMPORT_NAME => ExtInstruction::ImportName(value.1.into()),
            Opcode::IS_OP => ExtInstruction::IsOp(value.1.into()),
            Opcode::JUMP_BACKWARD => ExtInstruction::JumpBackward(RelativeJump {
                index: value.1,
                direction: JumpDirection::Backward,
            }),
            Opcode::JUMP_BACKWARD_NO_INTERRUPT => {
                ExtInstruction::JumpBackwardNoInterrupt(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Backward,
                })
            }
            Opcode::JUMP_FORWARD => ExtInstruction::JumpForward(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::LIST_APPEND => ExtInstruction::ListAppend(value.1),
            Opcode::LIST_EXTEND => ExtInstruction::ListExtend(value.1),
            Opcode::LOAD_ATTR => ExtInstruction::LoadAttr(value.1.into()),
            Opcode::LOAD_CONST => ExtInstruction::LoadConst(value.1.into()),
            Opcode::LOAD_DEREF => ExtInstruction::LoadDeref(value.1.into()),
            Opcode::LOAD_FAST => ExtInstruction::LoadFast(value.1.into()),
            Opcode::LOAD_FAST_AND_CLEAR => ExtInstruction::LoadFastAndClear(value.1.into()),
            Opcode::LOAD_FAST_CHECK => ExtInstruction::LoadFastCheck(value.1.into()),
            Opcode::LOAD_FAST_LOAD_FAST => {
                ExtInstruction::LoadFastLoadFast(((value.1 >> 4).into(), (value.1 & 15).into()))
            }
            Opcode::LOAD_FROM_DICT_OR_DEREF => ExtInstruction::LoadFromDictOrDeref(value.1.into()),
            Opcode::LOAD_FROM_DICT_OR_GLOBALS => {
                ExtInstruction::LoadFromDictOrGlobals(value.1.into())
            }
            Opcode::LOAD_GLOBAL => ExtInstruction::LoadGlobal(value.1.into()),
            Opcode::LOAD_NAME => ExtInstruction::LoadName(value.1.into()),
            Opcode::LOAD_SUPER_ATTR => ExtInstruction::LoadSuperAttr(value.1.into()),
            Opcode::MAKE_CELL => ExtInstruction::MakeCell(value.1.into()),
            Opcode::MAP_ADD => ExtInstruction::MapAdd(value.1),
            Opcode::MATCH_CLASS => ExtInstruction::MatchClass(value.1),
            Opcode::POP_JUMP_IF_FALSE => ExtInstruction::PopJumpIfFalse(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::POP_JUMP_IF_NONE => ExtInstruction::PopJumpIfNone(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::POP_JUMP_IF_NOT_NONE => ExtInstruction::PopJumpIfNotNone(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::POP_JUMP_IF_TRUE => ExtInstruction::PopJumpIfTrue(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::RAISE_VARARGS => ExtInstruction::RaiseVarargs(value.1.into()),
            Opcode::RERAISE => ExtInstruction::Reraise(value.1.into()),
            Opcode::RETURN_CONST => ExtInstruction::ReturnConst(value.1.into()),
            Opcode::SEND => ExtInstruction::Send(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::SET_ADD => ExtInstruction::SetAdd(value.1),
            Opcode::SET_FUNCTION_ATTRIBUTE => ExtInstruction::SetFunctionAttribute(
                FunctionAttributeFlags::from_bits_retain(value.1),
            ),
            Opcode::SET_UPDATE => ExtInstruction::SetUpdate(value.1),
            Opcode::STORE_ATTR => ExtInstruction::StoreAttr(value.1.into()),
            Opcode::STORE_DEREF => ExtInstruction::StoreDeref(value.1.into()),
            Opcode::STORE_FAST => ExtInstruction::StoreFast(value.1.into()),
            Opcode::STORE_FAST_LOAD_FAST => {
                ExtInstruction::StoreFastLoadFast(((value.1 >> 4).into(), (value.1 & 15).into()))
            }
            Opcode::STORE_FAST_STORE_FAST => {
                ExtInstruction::StoreFastStoreFast(((value.1 >> 4).into(), (value.1 & 15).into()))
            }
            Opcode::STORE_GLOBAL => ExtInstruction::StoreGlobal(value.1.into()),
            Opcode::STORE_NAME => ExtInstruction::StoreName(value.1.into()),
            Opcode::SWAP => ExtInstruction::Swap(value.1),
            Opcode::UNPACK_EX => ExtInstruction::UnpackEx(value.1),
            Opcode::UNPACK_SEQUENCE => ExtInstruction::UnpackSequence(value.1),
            Opcode::YIELD_VALUE => ExtInstruction::YieldValue(value.1),
            Opcode::RESUME => ExtInstruction::Resume(value.1.into()),
            Opcode::BINARY_OP_ADD_FLOAT => ExtInstruction::BinaryOpAddFloat(value.1),
            Opcode::BINARY_OP_ADD_INT => ExtInstruction::BinaryOpAddInt(value.1),
            Opcode::BINARY_OP_ADD_UNICODE => ExtInstruction::BinaryOpAddUnicode(value.1),
            Opcode::BINARY_OP_MULTIPLY_FLOAT => ExtInstruction::BinaryOpMultiplyFloat(value.1),
            Opcode::BINARY_OP_MULTIPLY_INT => ExtInstruction::BinaryOpMultiplyInt(value.1),
            Opcode::BINARY_OP_SUBTRACT_FLOAT => ExtInstruction::BinaryOpSubtractFloat(value.1),
            Opcode::BINARY_OP_SUBTRACT_INT => ExtInstruction::BinaryOpSubtractInt(value.1),
            Opcode::BINARY_SUBSCR_DICT => ExtInstruction::BinarySubscrDict(value.1),
            Opcode::BINARY_SUBSCR_GETITEM => ExtInstruction::BinarySubscrGetitem(value.1),
            Opcode::BINARY_SUBSCR_LIST_INT => ExtInstruction::BinarySubscrListInt(value.1),
            Opcode::BINARY_SUBSCR_STR_INT => ExtInstruction::BinarySubscrStrInt(value.1),
            Opcode::BINARY_SUBSCR_TUPLE_INT => ExtInstruction::BinarySubscrTupleInt(value.1),
            Opcode::CALL_ALLOC_AND_ENTER_INIT => ExtInstruction::CallAllocAndEnterInit(value.1),
            Opcode::CALL_BOUND_METHOD_EXACT_ARGS => {
                ExtInstruction::CallBoundMethodExactArgs(value.1)
            }
            Opcode::CALL_BOUND_METHOD_GENERAL => ExtInstruction::CallBoundMethodGeneral(value.1),
            Opcode::CALL_BUILTIN_CLASS => ExtInstruction::CallBuiltinClass(value.1),
            Opcode::CALL_BUILTIN_FAST => ExtInstruction::CallBuiltinFast(value.1),
            Opcode::CALL_BUILTIN_FAST_WITH_KEYWORDS => {
                ExtInstruction::CallBuiltinFastWithKeywords(value.1)
            }
            Opcode::CALL_BUILTIN_O => ExtInstruction::CallBuiltinO(value.1),
            Opcode::CALL_ISINSTANCE => ExtInstruction::CallIsinstance(value.1),
            Opcode::CALL_LEN => ExtInstruction::CallLen(value.1),
            Opcode::CALL_LIST_APPEND => ExtInstruction::CallListAppend(value.1),
            Opcode::CALL_METHOD_DESCRIPTOR_FAST => {
                ExtInstruction::CallMethodDescriptorFast(value.1)
            }
            Opcode::CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS => {
                ExtInstruction::CallMethodDescriptorFastWithKeywords(value.1)
            }
            Opcode::CALL_METHOD_DESCRIPTOR_NOARGS => {
                ExtInstruction::CallMethodDescriptorNoargs(value.1)
            }
            Opcode::CALL_METHOD_DESCRIPTOR_O => ExtInstruction::CallMethodDescriptorO(value.1),
            Opcode::CALL_NON_PY_GENERAL => ExtInstruction::CallNonPyGeneral(value.1),
            Opcode::CALL_PY_EXACT_ARGS => ExtInstruction::CallPyExactArgs(value.1),
            Opcode::CALL_PY_GENERAL => ExtInstruction::CallPyGeneral(value.1),
            Opcode::CALL_STR_1 => ExtInstruction::CallStr1(value.1),
            Opcode::CALL_TUPLE_1 => ExtInstruction::CallTuple1(value.1),
            Opcode::CALL_TYPE_1 => ExtInstruction::CallType1(value.1),
            Opcode::COMPARE_OP_FLOAT => ExtInstruction::CompareOpFloat(value.1),
            Opcode::COMPARE_OP_INT => ExtInstruction::CompareOpInt(value.1),
            Opcode::COMPARE_OP_STR => ExtInstruction::CompareOpStr(value.1),
            Opcode::CONTAINS_OP_DICT => ExtInstruction::ContainsOpDict(value.1),
            Opcode::CONTAINS_OP_SET => ExtInstruction::ContainsOpSet(value.1),
            Opcode::FOR_ITER_GEN => ExtInstruction::ForIterGen(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::FOR_ITER_LIST => ExtInstruction::ForIterList(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::FOR_ITER_RANGE => ExtInstruction::ForIterRange(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::FOR_ITER_TUPLE => ExtInstruction::ForIterTuple(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::LOAD_ATTR_CLASS => ExtInstruction::LoadAttrClass(value.1),
            Opcode::LOAD_ATTR_GETATTRIBUTE_OVERRIDDEN => {
                ExtInstruction::LoadAttrGetattributeOverridden(value.1)
            }
            Opcode::LOAD_ATTR_INSTANCE_VALUE => ExtInstruction::LoadAttrInstanceValue(value.1),
            Opcode::LOAD_ATTR_METHOD_LAZY_DICT => ExtInstruction::LoadAttrMethodLazyDict(value.1),
            Opcode::LOAD_ATTR_METHOD_NO_DICT => ExtInstruction::LoadAttrMethodNoDict(value.1),
            Opcode::LOAD_ATTR_METHOD_WITH_VALUES => {
                ExtInstruction::LoadAttrMethodWithValues(value.1)
            }
            Opcode::LOAD_ATTR_MODULE => ExtInstruction::LoadAttrModule(value.1),
            Opcode::LOAD_ATTR_NONDESCRIPTOR_NO_DICT => {
                ExtInstruction::LoadAttrNondescriptorNoDict(value.1)
            }
            Opcode::LOAD_ATTR_NONDESCRIPTOR_WITH_VALUES => {
                ExtInstruction::LoadAttrNondescriptorWithValues(value.1)
            }
            Opcode::LOAD_ATTR_PROPERTY => ExtInstruction::LoadAttrProperty(value.1),
            Opcode::LOAD_ATTR_SLOT => ExtInstruction::LoadAttrSlot(value.1),
            Opcode::LOAD_ATTR_WITH_HINT => ExtInstruction::LoadAttrWithHint(value.1),
            Opcode::LOAD_GLOBAL_BUILTIN => ExtInstruction::LoadGlobalBuiltin(value.1),
            Opcode::LOAD_GLOBAL_MODULE => ExtInstruction::LoadGlobalModule(value.1),
            Opcode::LOAD_SUPER_ATTR_ATTR => ExtInstruction::LoadSuperAttrAttr(value.1),
            Opcode::LOAD_SUPER_ATTR_METHOD => ExtInstruction::LoadSuperAttrMethod(value.1),
            Opcode::RESUME_CHECK => ExtInstruction::ResumeCheck(value.1),
            Opcode::SEND_GEN => ExtInstruction::SendGen(value.1),
            Opcode::STORE_ATTR_INSTANCE_VALUE => ExtInstruction::StoreAttrInstanceValue(value.1),
            Opcode::STORE_ATTR_SLOT => ExtInstruction::StoreAttrSlot(value.1),
            Opcode::STORE_ATTR_WITH_HINT => ExtInstruction::StoreAttrWithHint(value.1),
            Opcode::STORE_SUBSCR_DICT => ExtInstruction::StoreSubscrDict(value.1),
            Opcode::STORE_SUBSCR_LIST_INT => ExtInstruction::StoreSubscrListInt(value.1),
            Opcode::TO_BOOL_ALWAYS_TRUE => ExtInstruction::ToBoolAlwaysTrue(value.1),
            Opcode::TO_BOOL_BOOL => ExtInstruction::ToBoolBool(value.1),
            Opcode::TO_BOOL_INT => ExtInstruction::ToBoolInt(value.1),
            Opcode::TO_BOOL_LIST => ExtInstruction::ToBoolList(value.1),
            Opcode::TO_BOOL_NONE => ExtInstruction::ToBoolNone(value.1),
            Opcode::TO_BOOL_STR => ExtInstruction::ToBoolStr(value.1),
            Opcode::UNPACK_SEQUENCE_LIST => ExtInstruction::UnpackSequenceList(value.1),
            Opcode::UNPACK_SEQUENCE_TUPLE => ExtInstruction::UnpackSequenceTuple(value.1),
            Opcode::UNPACK_SEQUENCE_TWO_TUPLE => ExtInstruction::UnpackSequenceTwoTuple(value.1),
            Opcode::INSTRUMENTED_RESUME => ExtInstruction::InstrumentedResume(value.1),
            Opcode::INSTRUMENTED_END_FOR => ExtInstruction::InstrumentedEndFor(value.1),
            Opcode::INSTRUMENTED_END_SEND => ExtInstruction::InstrumentedEndSend(value.1),
            Opcode::INSTRUMENTED_RETURN_VALUE => ExtInstruction::InstrumentedReturnValue(value.1),
            Opcode::INSTRUMENTED_RETURN_CONST => ExtInstruction::InstrumentedReturnConst(value.1),
            Opcode::INSTRUMENTED_YIELD_VALUE => ExtInstruction::InstrumentedYieldValue(value.1),
            Opcode::INSTRUMENTED_LOAD_SUPER_ATTR => {
                ExtInstruction::InstrumentedLoadSuperAttr(value.1)
            }
            Opcode::INSTRUMENTED_FOR_ITER => ExtInstruction::InstrumentedForIter(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::INSTRUMENTED_CALL => ExtInstruction::InstrumentedCall(value.1),
            Opcode::INSTRUMENTED_CALL_KW => ExtInstruction::InstrumentedCallKw(value.1),
            Opcode::INSTRUMENTED_CALL_FUNCTION_EX => {
                ExtInstruction::InstrumentedCallFunctionEx(value.1)
            }
            Opcode::INSTRUMENTED_INSTRUCTION => ExtInstruction::InstrumentedInstruction(value.1),
            Opcode::INSTRUMENTED_JUMP_FORWARD => {
                ExtInstruction::InstrumentedJumpForward(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::INSTRUMENTED_JUMP_BACKWARD => {
                ExtInstruction::InstrumentedJumpBackward(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Backward,
                })
            }
            Opcode::INSTRUMENTED_POP_JUMP_IF_TRUE => {
                ExtInstruction::InstrumentedPopJumpIfTrue(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::INSTRUMENTED_POP_JUMP_IF_FALSE => {
                ExtInstruction::InstrumentedPopJumpIfFalse(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::INSTRUMENTED_POP_JUMP_IF_NONE => {
                ExtInstruction::InstrumentedPopJumpIfNone(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::INSTRUMENTED_POP_JUMP_IF_NOT_NONE => {
                ExtInstruction::InstrumentedPopJumpIfNotNone(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::INSTRUMENTED_LINE => ExtInstruction::InstrumentedLine(value.1),
            Opcode::INVALID_OPCODE(opcode) => ExtInstruction::InvalidOpcode((opcode, value.1)),
        })
    }
}

impl GenericInstruction<u32> for ExtInstruction {
    type Opcode = Opcode;

    fn get_opcode(&self) -> Self::Opcode {
        match self {
            ExtInstruction::Cache(_) => Opcode::CACHE,
            ExtInstruction::BeforeAsyncWith(_) => Opcode::BEFORE_ASYNC_WITH,
            ExtInstruction::BeforeWith(_) => Opcode::BEFORE_WITH,
            ExtInstruction::BinaryOpInplaceAddUnicode(_) => Opcode::BINARY_OP_INPLACE_ADD_UNICODE,
            ExtInstruction::BinarySlice(_) => Opcode::BINARY_SLICE,
            ExtInstruction::BinarySubscr(_) => Opcode::BINARY_SUBSCR,
            ExtInstruction::CheckEgMatch(_) => Opcode::CHECK_EG_MATCH,
            ExtInstruction::CheckExcMatch(_) => Opcode::CHECK_EXC_MATCH,
            ExtInstruction::CleanupThrow(_) => Opcode::CLEANUP_THROW,
            ExtInstruction::DeleteSubscr(_) => Opcode::DELETE_SUBSCR,
            ExtInstruction::EndAsyncFor(_) => Opcode::END_ASYNC_FOR,
            ExtInstruction::EndFor(_) => Opcode::END_FOR,
            ExtInstruction::EndSend(_) => Opcode::END_SEND,
            ExtInstruction::ExitInitCheck(_) => Opcode::EXIT_INIT_CHECK,
            ExtInstruction::FormatSimple(_) => Opcode::FORMAT_SIMPLE,
            ExtInstruction::FormatWithSpec(_) => Opcode::FORMAT_WITH_SPEC,
            ExtInstruction::GetAiter(_) => Opcode::GET_AITER,
            ExtInstruction::Reserved(_) => Opcode::RESERVED,
            ExtInstruction::GetAnext(_) => Opcode::GET_ANEXT,
            ExtInstruction::GetIter(_) => Opcode::GET_ITER,
            ExtInstruction::GetLen(_) => Opcode::GET_LEN,
            ExtInstruction::GetYieldFromIter(_) => Opcode::GET_YIELD_FROM_ITER,
            ExtInstruction::InterpreterExit(_) => Opcode::INTERPRETER_EXIT,
            ExtInstruction::LoadAssertionError(_) => Opcode::LOAD_ASSERTION_ERROR,
            ExtInstruction::LoadBuildClass(_) => Opcode::LOAD_BUILD_CLASS,
            ExtInstruction::LoadLocals(_) => Opcode::LOAD_LOCALS,
            ExtInstruction::MakeFunction(_) => Opcode::MAKE_FUNCTION,
            ExtInstruction::MatchKeys(_) => Opcode::MATCH_KEYS,
            ExtInstruction::MatchMapping(_) => Opcode::MATCH_MAPPING,
            ExtInstruction::MatchSequence(_) => Opcode::MATCH_SEQUENCE,
            ExtInstruction::Nop(_) => Opcode::NOP,
            ExtInstruction::PopExcept(_) => Opcode::POP_EXCEPT,
            ExtInstruction::PopTop(_) => Opcode::POP_TOP,
            ExtInstruction::PushExcInfo(_) => Opcode::PUSH_EXC_INFO,
            ExtInstruction::PushNull(_) => Opcode::PUSH_NULL,
            ExtInstruction::ReturnGenerator(_) => Opcode::RETURN_GENERATOR,
            ExtInstruction::ReturnValue(_) => Opcode::RETURN_VALUE,
            ExtInstruction::SetupAnnotations(_) => Opcode::SETUP_ANNOTATIONS,
            ExtInstruction::StoreSlice(_) => Opcode::STORE_SLICE,
            ExtInstruction::StoreSubscr(_) => Opcode::STORE_SUBSCR,
            ExtInstruction::ToBool(_) => Opcode::TO_BOOL,
            ExtInstruction::UnaryInvert(_) => Opcode::UNARY_INVERT,
            ExtInstruction::UnaryNegative(_) => Opcode::UNARY_NEGATIVE,
            ExtInstruction::UnaryNot(_) => Opcode::UNARY_NOT,
            ExtInstruction::WithExceptStart(_) => Opcode::WITH_EXCEPT_START,
            ExtInstruction::BinaryOp(_) => Opcode::BINARY_OP,
            ExtInstruction::BuildConstKeyMap(_) => Opcode::BUILD_CONST_KEY_MAP,
            ExtInstruction::BuildList(_) => Opcode::BUILD_LIST,
            ExtInstruction::BuildMap(_) => Opcode::BUILD_MAP,
            ExtInstruction::BuildSet(_) => Opcode::BUILD_SET,
            ExtInstruction::BuildSlice(_) => Opcode::BUILD_SLICE,
            ExtInstruction::BuildString(_) => Opcode::BUILD_STRING,
            ExtInstruction::BuildTuple(_) => Opcode::BUILD_TUPLE,
            ExtInstruction::Call(_) => Opcode::CALL,
            ExtInstruction::CallFunctionEx(_) => Opcode::CALL_FUNCTION_EX,
            ExtInstruction::CallIntrinsic1(_) => Opcode::CALL_INTRINSIC_1,
            ExtInstruction::CallIntrinsic2(_) => Opcode::CALL_INTRINSIC_2,
            ExtInstruction::CallKw(_) => Opcode::CALL_KW,
            ExtInstruction::CompareOp(_) => Opcode::COMPARE_OP,
            ExtInstruction::ContainsOp(_) => Opcode::CONTAINS_OP,
            ExtInstruction::ConvertValue(_) => Opcode::CONVERT_VALUE,
            ExtInstruction::Copy(_) => Opcode::COPY,
            ExtInstruction::CopyFreeVars(_) => Opcode::COPY_FREE_VARS,
            ExtInstruction::DeleteAttr(_) => Opcode::DELETE_ATTR,
            ExtInstruction::DeleteDeref(_) => Opcode::DELETE_DEREF,
            ExtInstruction::DeleteFast(_) => Opcode::DELETE_FAST,
            ExtInstruction::DeleteGlobal(_) => Opcode::DELETE_GLOBAL,
            ExtInstruction::DeleteName(_) => Opcode::DELETE_NAME,
            ExtInstruction::DictMerge(_) => Opcode::DICT_MERGE,
            ExtInstruction::DictUpdate(_) => Opcode::DICT_UPDATE,
            ExtInstruction::EnterExecutor(_) => Opcode::ENTER_EXECUTOR,
            // Extended arg is ommited in the resolved instructions
            ExtInstruction::ForIter(_) => Opcode::FOR_ITER,
            ExtInstruction::GetAwaitable(_) => Opcode::GET_AWAITABLE,
            ExtInstruction::ImportFrom(_) => Opcode::IMPORT_FROM,
            ExtInstruction::ImportName(_) => Opcode::IMPORT_NAME,
            ExtInstruction::IsOp(_) => Opcode::IS_OP,
            ExtInstruction::JumpBackward(_) => Opcode::JUMP_BACKWARD,
            ExtInstruction::JumpBackwardNoInterrupt(_) => Opcode::JUMP_BACKWARD_NO_INTERRUPT,
            ExtInstruction::JumpForward(_) => Opcode::JUMP_FORWARD,
            ExtInstruction::ListAppend(_) => Opcode::LIST_APPEND,
            ExtInstruction::ListExtend(_) => Opcode::LIST_EXTEND,
            ExtInstruction::LoadAttr(_) => Opcode::LOAD_ATTR,
            ExtInstruction::LoadConst(_) => Opcode::LOAD_CONST,
            ExtInstruction::LoadDeref(_) => Opcode::LOAD_DEREF,
            ExtInstruction::LoadFast(_) => Opcode::LOAD_FAST,
            ExtInstruction::LoadFastAndClear(_) => Opcode::LOAD_FAST_AND_CLEAR,
            ExtInstruction::LoadFastCheck(_) => Opcode::LOAD_FAST_CHECK,
            ExtInstruction::LoadFastLoadFast(_) => Opcode::LOAD_FAST_LOAD_FAST,
            ExtInstruction::LoadFromDictOrDeref(_) => Opcode::LOAD_FROM_DICT_OR_DEREF,
            ExtInstruction::LoadFromDictOrGlobals(_) => Opcode::LOAD_FROM_DICT_OR_GLOBALS,
            ExtInstruction::LoadGlobal(_) => Opcode::LOAD_GLOBAL,
            ExtInstruction::LoadName(_) => Opcode::LOAD_NAME,
            ExtInstruction::LoadSuperAttr(_) => Opcode::LOAD_SUPER_ATTR,
            ExtInstruction::MakeCell(_) => Opcode::MAKE_CELL,
            ExtInstruction::MapAdd(_) => Opcode::MAP_ADD,
            ExtInstruction::MatchClass(_) => Opcode::MATCH_CLASS,
            ExtInstruction::PopJumpIfFalse(_) => Opcode::POP_JUMP_IF_FALSE,
            ExtInstruction::PopJumpIfNone(_) => Opcode::POP_JUMP_IF_NONE,
            ExtInstruction::PopJumpIfNotNone(_) => Opcode::POP_JUMP_IF_NOT_NONE,
            ExtInstruction::PopJumpIfTrue(_) => Opcode::POP_JUMP_IF_TRUE,
            ExtInstruction::RaiseVarargs(_) => Opcode::RAISE_VARARGS,
            ExtInstruction::Reraise(_) => Opcode::RERAISE,
            ExtInstruction::ReturnConst(_) => Opcode::RETURN_CONST,
            ExtInstruction::Send(_) => Opcode::SEND,
            ExtInstruction::SetAdd(_) => Opcode::SET_ADD,
            ExtInstruction::SetFunctionAttribute(_) => Opcode::SET_FUNCTION_ATTRIBUTE,
            ExtInstruction::SetUpdate(_) => Opcode::SET_UPDATE,
            ExtInstruction::StoreAttr(_) => Opcode::STORE_ATTR,
            ExtInstruction::StoreDeref(_) => Opcode::STORE_DEREF,
            ExtInstruction::StoreFast(_) => Opcode::STORE_FAST,
            ExtInstruction::StoreFastLoadFast(_) => Opcode::STORE_FAST_LOAD_FAST,
            ExtInstruction::StoreFastStoreFast(_) => Opcode::STORE_FAST_STORE_FAST,
            ExtInstruction::StoreGlobal(_) => Opcode::STORE_GLOBAL,
            ExtInstruction::StoreName(_) => Opcode::STORE_NAME,
            ExtInstruction::Swap(_) => Opcode::SWAP,
            ExtInstruction::UnpackEx(_) => Opcode::UNPACK_EX,
            ExtInstruction::UnpackSequence(_) => Opcode::UNPACK_SEQUENCE,
            ExtInstruction::YieldValue(_) => Opcode::YIELD_VALUE,
            ExtInstruction::Resume(_) => Opcode::RESUME,
            ExtInstruction::BinaryOpAddFloat(_) => Opcode::BINARY_OP_ADD_FLOAT,
            ExtInstruction::BinaryOpAddInt(_) => Opcode::BINARY_OP_ADD_INT,
            ExtInstruction::BinaryOpAddUnicode(_) => Opcode::BINARY_OP_ADD_UNICODE,
            ExtInstruction::BinaryOpMultiplyFloat(_) => Opcode::BINARY_OP_MULTIPLY_FLOAT,
            ExtInstruction::BinaryOpMultiplyInt(_) => Opcode::BINARY_OP_MULTIPLY_INT,
            ExtInstruction::BinaryOpSubtractFloat(_) => Opcode::BINARY_OP_SUBTRACT_FLOAT,
            ExtInstruction::BinaryOpSubtractInt(_) => Opcode::BINARY_OP_SUBTRACT_INT,
            ExtInstruction::BinarySubscrDict(_) => Opcode::BINARY_SUBSCR_DICT,
            ExtInstruction::BinarySubscrGetitem(_) => Opcode::BINARY_SUBSCR_GETITEM,
            ExtInstruction::BinarySubscrListInt(_) => Opcode::BINARY_SUBSCR_LIST_INT,
            ExtInstruction::BinarySubscrStrInt(_) => Opcode::BINARY_SUBSCR_STR_INT,
            ExtInstruction::BinarySubscrTupleInt(_) => Opcode::BINARY_SUBSCR_TUPLE_INT,
            ExtInstruction::CallAllocAndEnterInit(_) => Opcode::CALL_ALLOC_AND_ENTER_INIT,
            ExtInstruction::CallBoundMethodExactArgs(_) => Opcode::CALL_BOUND_METHOD_EXACT_ARGS,
            ExtInstruction::CallBoundMethodGeneral(_) => Opcode::CALL_BOUND_METHOD_GENERAL,
            ExtInstruction::CallBuiltinClass(_) => Opcode::CALL_BUILTIN_CLASS,
            ExtInstruction::CallBuiltinFast(_) => Opcode::CALL_BUILTIN_FAST,
            ExtInstruction::CallBuiltinFastWithKeywords(_) => {
                Opcode::CALL_BUILTIN_FAST_WITH_KEYWORDS
            }
            ExtInstruction::CallBuiltinO(_) => Opcode::CALL_BUILTIN_O,
            ExtInstruction::CallIsinstance(_) => Opcode::CALL_ISINSTANCE,
            ExtInstruction::CallLen(_) => Opcode::CALL_LEN,
            ExtInstruction::CallListAppend(_) => Opcode::CALL_LIST_APPEND,
            ExtInstruction::CallMethodDescriptorFast(_) => Opcode::CALL_METHOD_DESCRIPTOR_FAST,
            ExtInstruction::CallMethodDescriptorFastWithKeywords(_) => {
                Opcode::CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS
            }
            ExtInstruction::CallMethodDescriptorNoargs(_) => Opcode::CALL_METHOD_DESCRIPTOR_NOARGS,
            ExtInstruction::CallMethodDescriptorO(_) => Opcode::CALL_METHOD_DESCRIPTOR_O,
            ExtInstruction::CallNonPyGeneral(_) => Opcode::CALL_NON_PY_GENERAL,
            ExtInstruction::CallPyExactArgs(_) => Opcode::CALL_PY_EXACT_ARGS,
            ExtInstruction::CallPyGeneral(_) => Opcode::CALL_PY_GENERAL,
            ExtInstruction::CallStr1(_) => Opcode::CALL_STR_1,
            ExtInstruction::CallTuple1(_) => Opcode::CALL_TUPLE_1,
            ExtInstruction::CallType1(_) => Opcode::CALL_TYPE_1,
            ExtInstruction::CompareOpFloat(_) => Opcode::COMPARE_OP_FLOAT,
            ExtInstruction::CompareOpInt(_) => Opcode::COMPARE_OP_INT,
            ExtInstruction::CompareOpStr(_) => Opcode::COMPARE_OP_STR,
            ExtInstruction::ContainsOpDict(_) => Opcode::CONTAINS_OP_DICT,
            ExtInstruction::ContainsOpSet(_) => Opcode::CONTAINS_OP_SET,
            ExtInstruction::ForIterGen(_) => Opcode::FOR_ITER_GEN,
            ExtInstruction::ForIterList(_) => Opcode::FOR_ITER_LIST,
            ExtInstruction::ForIterRange(_) => Opcode::FOR_ITER_RANGE,
            ExtInstruction::ForIterTuple(_) => Opcode::FOR_ITER_TUPLE,
            ExtInstruction::LoadAttrClass(_) => Opcode::LOAD_ATTR_CLASS,
            ExtInstruction::LoadAttrGetattributeOverridden(_) => {
                Opcode::LOAD_ATTR_GETATTRIBUTE_OVERRIDDEN
            }
            ExtInstruction::LoadAttrInstanceValue(_) => Opcode::LOAD_ATTR_INSTANCE_VALUE,
            ExtInstruction::LoadAttrMethodLazyDict(_) => Opcode::LOAD_ATTR_METHOD_LAZY_DICT,
            ExtInstruction::LoadAttrMethodNoDict(_) => Opcode::LOAD_ATTR_METHOD_NO_DICT,
            ExtInstruction::LoadAttrMethodWithValues(_) => Opcode::LOAD_ATTR_METHOD_WITH_VALUES,
            ExtInstruction::LoadAttrModule(_) => Opcode::LOAD_ATTR_MODULE,
            ExtInstruction::LoadAttrNondescriptorNoDict(_) => {
                Opcode::LOAD_ATTR_NONDESCRIPTOR_NO_DICT
            }
            ExtInstruction::LoadAttrNondescriptorWithValues(_) => {
                Opcode::LOAD_ATTR_NONDESCRIPTOR_WITH_VALUES
            }
            ExtInstruction::LoadAttrProperty(_) => Opcode::LOAD_ATTR_PROPERTY,
            ExtInstruction::LoadAttrSlot(_) => Opcode::LOAD_ATTR_SLOT,
            ExtInstruction::LoadAttrWithHint(_) => Opcode::LOAD_ATTR_WITH_HINT,
            ExtInstruction::LoadGlobalBuiltin(_) => Opcode::LOAD_GLOBAL_BUILTIN,
            ExtInstruction::LoadGlobalModule(_) => Opcode::LOAD_GLOBAL_MODULE,
            ExtInstruction::LoadSuperAttrAttr(_) => Opcode::LOAD_SUPER_ATTR_ATTR,
            ExtInstruction::LoadSuperAttrMethod(_) => Opcode::LOAD_SUPER_ATTR_METHOD,
            ExtInstruction::ResumeCheck(_) => Opcode::RESUME_CHECK,
            ExtInstruction::SendGen(_) => Opcode::SEND_GEN,
            ExtInstruction::StoreAttrInstanceValue(_) => Opcode::STORE_ATTR_INSTANCE_VALUE,
            ExtInstruction::StoreAttrSlot(_) => Opcode::STORE_ATTR_SLOT,
            ExtInstruction::StoreAttrWithHint(_) => Opcode::STORE_ATTR_WITH_HINT,
            ExtInstruction::StoreSubscrDict(_) => Opcode::STORE_SUBSCR_DICT,
            ExtInstruction::StoreSubscrListInt(_) => Opcode::STORE_SUBSCR_LIST_INT,
            ExtInstruction::ToBoolAlwaysTrue(_) => Opcode::TO_BOOL_ALWAYS_TRUE,
            ExtInstruction::ToBoolBool(_) => Opcode::TO_BOOL_BOOL,
            ExtInstruction::ToBoolInt(_) => Opcode::TO_BOOL_INT,
            ExtInstruction::ToBoolList(_) => Opcode::TO_BOOL_LIST,
            ExtInstruction::ToBoolNone(_) => Opcode::TO_BOOL_NONE,
            ExtInstruction::ToBoolStr(_) => Opcode::TO_BOOL_STR,
            ExtInstruction::UnpackSequenceList(_) => Opcode::UNPACK_SEQUENCE_LIST,
            ExtInstruction::UnpackSequenceTuple(_) => Opcode::UNPACK_SEQUENCE_TUPLE,
            ExtInstruction::UnpackSequenceTwoTuple(_) => Opcode::UNPACK_SEQUENCE_TWO_TUPLE,
            ExtInstruction::InstrumentedResume(_) => Opcode::INSTRUMENTED_RESUME,
            ExtInstruction::InstrumentedEndFor(_) => Opcode::INSTRUMENTED_END_FOR,
            ExtInstruction::InstrumentedEndSend(_) => Opcode::INSTRUMENTED_END_SEND,
            ExtInstruction::InstrumentedReturnValue(_) => Opcode::INSTRUMENTED_RETURN_VALUE,
            ExtInstruction::InstrumentedReturnConst(_) => Opcode::INSTRUMENTED_RETURN_CONST,
            ExtInstruction::InstrumentedYieldValue(_) => Opcode::INSTRUMENTED_YIELD_VALUE,
            ExtInstruction::InstrumentedLoadSuperAttr(_) => Opcode::INSTRUMENTED_LOAD_SUPER_ATTR,
            ExtInstruction::InstrumentedForIter(_) => Opcode::INSTRUMENTED_FOR_ITER,
            ExtInstruction::InstrumentedCall(_) => Opcode::INSTRUMENTED_CALL,
            ExtInstruction::InstrumentedCallKw(_) => Opcode::INSTRUMENTED_CALL_KW,
            ExtInstruction::InstrumentedCallFunctionEx(_) => Opcode::INSTRUMENTED_CALL_FUNCTION_EX,
            ExtInstruction::InstrumentedInstruction(_) => Opcode::INSTRUMENTED_INSTRUCTION,
            ExtInstruction::InstrumentedJumpForward(_) => Opcode::INSTRUMENTED_JUMP_FORWARD,
            ExtInstruction::InstrumentedJumpBackward(_) => Opcode::INSTRUMENTED_JUMP_BACKWARD,
            ExtInstruction::InstrumentedPopJumpIfTrue(_) => Opcode::INSTRUMENTED_POP_JUMP_IF_TRUE,
            ExtInstruction::InstrumentedPopJumpIfFalse(_) => Opcode::INSTRUMENTED_POP_JUMP_IF_FALSE,
            ExtInstruction::InstrumentedPopJumpIfNone(_) => Opcode::INSTRUMENTED_POP_JUMP_IF_NONE,
            ExtInstruction::InstrumentedPopJumpIfNotNone(_) => {
                Opcode::INSTRUMENTED_POP_JUMP_IF_NOT_NONE
            }
            ExtInstruction::InstrumentedLine(_) => Opcode::INSTRUMENTED_LINE,
            ExtInstruction::InvalidOpcode((opcode, _)) => Opcode::INVALID_OPCODE(*opcode),
        }
    }

    fn get_raw_value(&self) -> u32 {
        match &self {
            ExtInstruction::Cache(n)
            | ExtInstruction::BeforeAsyncWith(n)
            | ExtInstruction::BeforeWith(n)
            | ExtInstruction::BinarySlice(n)
            | ExtInstruction::BinarySubscr(n)
            | ExtInstruction::CheckEgMatch(n)
            | ExtInstruction::CheckExcMatch(n)
            | ExtInstruction::CleanupThrow(n)
            | ExtInstruction::DeleteSubscr(n)
            | ExtInstruction::EndAsyncFor(n)
            | ExtInstruction::EndFor(n)
            | ExtInstruction::EndSend(n)
            | ExtInstruction::ExitInitCheck(n)
            | ExtInstruction::FormatSimple(n)
            | ExtInstruction::FormatWithSpec(n)
            | ExtInstruction::GetAiter(n)
            | ExtInstruction::Reserved(n)
            | ExtInstruction::GetAnext(n)
            | ExtInstruction::GetIter(n)
            | ExtInstruction::GetLen(n)
            | ExtInstruction::GetYieldFromIter(n)
            | ExtInstruction::InterpreterExit(n)
            | ExtInstruction::LoadAssertionError(n)
            | ExtInstruction::LoadBuildClass(n)
            | ExtInstruction::LoadLocals(n)
            | ExtInstruction::MakeFunction(n)
            | ExtInstruction::MatchKeys(n)
            | ExtInstruction::MatchMapping(n)
            | ExtInstruction::MatchSequence(n)
            | ExtInstruction::Nop(n)
            | ExtInstruction::PopExcept(n)
            | ExtInstruction::PopTop(n)
            | ExtInstruction::PushExcInfo(n)
            | ExtInstruction::PushNull(n)
            | ExtInstruction::ReturnGenerator(n)
            | ExtInstruction::ReturnValue(n)
            | ExtInstruction::SetupAnnotations(n)
            | ExtInstruction::StoreSlice(n)
            | ExtInstruction::StoreSubscr(n)
            | ExtInstruction::ToBool(n)
            | ExtInstruction::UnaryInvert(n)
            | ExtInstruction::UnaryNegative(n)
            | ExtInstruction::UnaryNot(n)
            | ExtInstruction::WithExceptStart(n) => n.0,
            ExtInstruction::StoreName(name_index)
            | ExtInstruction::DeleteName(name_index)
            | ExtInstruction::StoreAttr(name_index)
            | ExtInstruction::DeleteAttr(name_index)
            | ExtInstruction::StoreGlobal(name_index)
            | ExtInstruction::DeleteGlobal(name_index)
            | ExtInstruction::LoadName(name_index)
            | ExtInstruction::ImportName(name_index)
            | ExtInstruction::ImportFrom(name_index) => name_index.index,
            ExtInstruction::LoadAttr(name_index) => name_index.index,
            ExtInstruction::LoadGlobal(global_name_index) => global_name_index.index,
            ExtInstruction::LoadSuperAttr(super_name_index) => super_name_index.index,
            ExtInstruction::BinaryOpInplaceAddUnicode(n)
            | ExtInstruction::UnpackSequence(n)
            | ExtInstruction::UnpackEx(n)
            | ExtInstruction::Swap(n)
            | ExtInstruction::BuildTuple(n)
            | ExtInstruction::BuildList(n)
            | ExtInstruction::BuildSet(n)
            | ExtInstruction::BuildMap(n)
            | ExtInstruction::Copy(n)
            | ExtInstruction::ListAppend(n)
            | ExtInstruction::SetAdd(n)
            | ExtInstruction::MapAdd(n)
            | ExtInstruction::CopyFreeVars(n)
            | ExtInstruction::YieldValue(n)
            | ExtInstruction::MatchClass(n)
            | ExtInstruction::BuildConstKeyMap(n)
            | ExtInstruction::BuildString(n)
            | ExtInstruction::ListExtend(n)
            | ExtInstruction::SetUpdate(n)
            | ExtInstruction::DictMerge(n)
            | ExtInstruction::DictUpdate(n)
            | ExtInstruction::EnterExecutor(n)
            | ExtInstruction::Call(n)
            | ExtInstruction::CallKw(n)
            | ExtInstruction::BinaryOpAddFloat(n)
            | ExtInstruction::BinaryOpAddInt(n)
            | ExtInstruction::BinaryOpAddUnicode(n)
            | ExtInstruction::BinaryOpMultiplyFloat(n)
            | ExtInstruction::BinaryOpMultiplyInt(n)
            | ExtInstruction::BinaryOpSubtractFloat(n)
            | ExtInstruction::BinaryOpSubtractInt(n)
            | ExtInstruction::BinarySubscrDict(n)
            | ExtInstruction::BinarySubscrGetitem(n)
            | ExtInstruction::BinarySubscrListInt(n)
            | ExtInstruction::BinarySubscrStrInt(n)
            | ExtInstruction::BinarySubscrTupleInt(n)
            | ExtInstruction::CallAllocAndEnterInit(n)
            | ExtInstruction::CallBoundMethodExactArgs(n)
            | ExtInstruction::CallBoundMethodGeneral(n)
            | ExtInstruction::CallBuiltinClass(n)
            | ExtInstruction::CallBuiltinFast(n)
            | ExtInstruction::CallBuiltinFastWithKeywords(n)
            | ExtInstruction::CallBuiltinO(n)
            | ExtInstruction::CallIsinstance(n)
            | ExtInstruction::CallLen(n)
            | ExtInstruction::CallListAppend(n)
            | ExtInstruction::CallMethodDescriptorFast(n)
            | ExtInstruction::CallMethodDescriptorFastWithKeywords(n)
            | ExtInstruction::CallMethodDescriptorNoargs(n)
            | ExtInstruction::CallMethodDescriptorO(n)
            | ExtInstruction::CallNonPyGeneral(n)
            | ExtInstruction::CallPyExactArgs(n)
            | ExtInstruction::CallPyGeneral(n)
            | ExtInstruction::CallStr1(n)
            | ExtInstruction::CallTuple1(n)
            | ExtInstruction::CallType1(n)
            | ExtInstruction::CompareOpFloat(n)
            | ExtInstruction::CompareOpInt(n)
            | ExtInstruction::CompareOpStr(n)
            | ExtInstruction::ContainsOpDict(n)
            | ExtInstruction::ContainsOpSet(n)
            | ExtInstruction::LoadAttrClass(n)
            | ExtInstruction::LoadAttrGetattributeOverridden(n)
            | ExtInstruction::LoadAttrInstanceValue(n)
            | ExtInstruction::LoadAttrMethodLazyDict(n)
            | ExtInstruction::LoadAttrMethodNoDict(n)
            | ExtInstruction::LoadAttrMethodWithValues(n)
            | ExtInstruction::LoadAttrModule(n)
            | ExtInstruction::LoadAttrNondescriptorNoDict(n)
            | ExtInstruction::LoadAttrNondescriptorWithValues(n)
            | ExtInstruction::LoadAttrProperty(n)
            | ExtInstruction::LoadAttrSlot(n)
            | ExtInstruction::LoadAttrWithHint(n)
            | ExtInstruction::LoadGlobalBuiltin(n)
            | ExtInstruction::LoadGlobalModule(n)
            | ExtInstruction::LoadSuperAttrAttr(n)
            | ExtInstruction::LoadSuperAttrMethod(n)
            | ExtInstruction::ResumeCheck(n)
            | ExtInstruction::SendGen(n)
            | ExtInstruction::StoreAttrInstanceValue(n)
            | ExtInstruction::StoreAttrSlot(n)
            | ExtInstruction::StoreAttrWithHint(n)
            | ExtInstruction::StoreSubscrDict(n)
            | ExtInstruction::StoreSubscrListInt(n)
            | ExtInstruction::ToBoolAlwaysTrue(n)
            | ExtInstruction::ToBoolBool(n)
            | ExtInstruction::ToBoolInt(n)
            | ExtInstruction::ToBoolList(n)
            | ExtInstruction::ToBoolNone(n)
            | ExtInstruction::ToBoolStr(n)
            | ExtInstruction::UnpackSequenceList(n)
            | ExtInstruction::UnpackSequenceTuple(n)
            | ExtInstruction::UnpackSequenceTwoTuple(n)
            | ExtInstruction::InstrumentedResume(n)
            | ExtInstruction::InstrumentedEndFor(n)
            | ExtInstruction::InstrumentedEndSend(n)
            | ExtInstruction::InstrumentedReturnValue(n)
            | ExtInstruction::InstrumentedReturnConst(n)
            | ExtInstruction::InstrumentedYieldValue(n)
            | ExtInstruction::InstrumentedLoadSuperAttr(n)
            | ExtInstruction::InstrumentedCall(n)
            | ExtInstruction::InstrumentedCallKw(n)
            | ExtInstruction::InstrumentedCallFunctionEx(n)
            | ExtInstruction::InstrumentedInstruction(n)
            | ExtInstruction::InstrumentedLine(n) => *n,
            ExtInstruction::CallIntrinsic1(functions) => functions.into(),
            ExtInstruction::CallIntrinsic2(functions) => functions.into(),
            ExtInstruction::LoadConst(const_index) | ExtInstruction::ReturnConst(const_index) => {
                const_index.index
            }
            ExtInstruction::CompareOp(cmp_op) => cmp_op.into(),
            ExtInstruction::ForIter(jump)
            | ExtInstruction::JumpForward(jump)
            | ExtInstruction::PopJumpIfFalse(jump)
            | ExtInstruction::PopJumpIfTrue(jump)
            | ExtInstruction::Send(jump)
            | ExtInstruction::PopJumpIfNotNone(jump)
            | ExtInstruction::PopJumpIfNone(jump)
            | ExtInstruction::ForIterRange(jump)
            | ExtInstruction::ForIterList(jump)
            | ExtInstruction::ForIterGen(jump)
            | ExtInstruction::ForIterTuple(jump)
            | ExtInstruction::InstrumentedForIter(jump)
            | ExtInstruction::InstrumentedPopJumpIfNone(jump)
            | ExtInstruction::InstrumentedPopJumpIfNotNone(jump)
            | ExtInstruction::InstrumentedJumpForward(jump)
            | ExtInstruction::InstrumentedPopJumpIfFalse(jump)
            | ExtInstruction::InstrumentedPopJumpIfTrue(jump)
            | ExtInstruction::JumpBackwardNoInterrupt(jump)
            | ExtInstruction::JumpBackward(jump)
            | ExtInstruction::InstrumentedJumpBackward(jump) => jump.index,
            ExtInstruction::IsOp(invert) | ExtInstruction::ContainsOp(invert) => invert.into(),
            ExtInstruction::Reraise(reraise) => reraise.into(),
            ExtInstruction::BinaryOp(binary_op) => binary_op.into(),
            ExtInstruction::LoadFast(varname_index)
            | ExtInstruction::StoreFast(varname_index)
            | ExtInstruction::DeleteFast(varname_index)
            | ExtInstruction::LoadFastCheck(varname_index)
            | ExtInstruction::LoadFastAndClear(varname_index) => varname_index.index,
            ExtInstruction::LoadFastLoadFast((index_1, index_2))
            | ExtInstruction::StoreFastLoadFast((index_1, index_2))
            | ExtInstruction::StoreFastStoreFast((index_1, index_2)) => {
                (index_1.index << 4) | index_2.index
            }
            ExtInstruction::LoadFromDictOrDeref(dynamic_index)
            | ExtInstruction::LoadFromDictOrGlobals(dynamic_index) => dynamic_index.index,
            ExtInstruction::RaiseVarargs(raise_var_args) => raise_var_args.into(),
            ExtInstruction::GetAwaitable(awaitable_where) => awaitable_where.into(),
            ExtInstruction::BuildSlice(slice) => slice.into(),
            ExtInstruction::MakeCell(closure_index)
            | ExtInstruction::LoadDeref(closure_index)
            | ExtInstruction::StoreDeref(closure_index)
            | ExtInstruction::DeleteDeref(closure_index) => closure_index.index,
            ExtInstruction::CallFunctionEx(flags) => flags.into(),
            ExtInstruction::SetFunctionAttribute(flags) => flags.bits(),
            ExtInstruction::ConvertValue(format) => format.into(),
            ExtInstruction::Resume(resume_where) => resume_where.into(),
            ExtInstruction::InvalidOpcode((_, arg)) => *arg,
        }
    }
}

/// Get a list of the extended args necessary to represent the arg.
/// The final arg that has to be included with the actual opcode is (arg & 0xff)
pub fn get_extended_args(arg: u32) -> Vec<Instruction> {
    if arg <= u8::MAX.into() {
        // arg is small enough that we don't need extended args
        vec![]
    } else {
        // Python bytecode uses EXTENDED_ARG for each additional byte above the lowest.
        // We need to emit them from most significant to least significant.
        let mut ext_args = Vec::new();
        let mut remaining = arg >> 8;
        while remaining > 0 {
            ext_args.push(Instruction::ExtendedArg((remaining & 0xff) as u8));
            remaining >>= 8;
        }

        ext_args.iter().rev().cloned().collect()
    }
}
