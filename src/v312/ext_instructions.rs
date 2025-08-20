use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use store_interval_tree::{Interval, IntervalTree};

use crate::v312::instructions;
use crate::{
    error::Error,
    traits::{GenericInstruction, InstructionAccess},
    utils::get_extended_args_count,
    v312::{
        cache::get_cache_count,
        code_objects::{
            AttrNameIndex, AwaitableWhere, BinaryOperation, CallExFlags, ClosureRefIndex,
            CompareOperation, ConstIndex, DynamicIndex, FormatFlag, GlobalNameIndex,
            Intrinsic1Functions, Intrinsic2Functions, Jump, JumpDirection, MakeFunctionFlags,
            NameIndex, OpInversion, RaiseForms, RelativeJump, Reraise, ResumeWhere, SliceCount,
            SuperAttrNameIndex, VarNameIndex,
        },
        instructions::{Instruction, Instructions},
        opcodes::Opcode,
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
    PopTop(UnusedArgument),
    PushNull(UnusedArgument),
    InterpreterExit(UnusedArgument),
    EndFor(UnusedArgument),
    EndSend(UnusedArgument),
    Nop(UnusedArgument),
    UnaryNegative(UnusedArgument),
    UnaryNot(UnusedArgument),
    UnaryInvert(UnusedArgument),
    Reserved(UnusedArgument),
    BinarySubscr(UnusedArgument),
    BinarySlice(UnusedArgument),
    StoreSlice(UnusedArgument),
    GetLen(UnusedArgument),
    MatchMapping(UnusedArgument),
    MatchSequence(UnusedArgument),
    MatchKeys(UnusedArgument),
    PushExcInfo(UnusedArgument),
    CheckExcMatch(UnusedArgument),
    CheckEgMatch(UnusedArgument),
    WithExceptStart(UnusedArgument),
    GetAiter(UnusedArgument),
    GetAnext(UnusedArgument),
    BeforeAsyncWith(UnusedArgument),
    BeforeWith(UnusedArgument),
    EndAsyncFor(UnusedArgument),
    CleanupThrow(UnusedArgument),
    StoreSubscr(UnusedArgument),
    DeleteSubscr(UnusedArgument),
    GetIter(UnusedArgument),
    GetYieldFromIter(UnusedArgument),
    LoadBuildClass(UnusedArgument),
    LoadAssertionError(UnusedArgument),
    ReturnGenerator(UnusedArgument),
    ReturnValue(UnusedArgument),
    SetupAnnotations(UnusedArgument),
    LoadLocals(UnusedArgument),
    PopExcept(UnusedArgument),
    StoreName(NameIndex),
    DeleteName(NameIndex),
    UnpackSequence(u32),
    ForIter(RelativeJump),
    UnpackEx(u32),
    StoreAttr(NameIndex),
    DeleteAttr(NameIndex),
    StoreGlobal(NameIndex),
    DeleteGlobal(NameIndex),
    Swap(u32),
    LoadConst(ConstIndex),
    LoadName(NameIndex),
    BuildTuple(u32),
    BuildList(u32),
    BuildSet(u32),
    BuildMap(u32),
    LoadAttr(AttrNameIndex),
    CompareOp(CompareOperation),
    ImportName(NameIndex),
    ImportFrom(NameIndex),
    JumpForward(RelativeJump),
    PopJumpIfFalse(RelativeJump),
    PopJumpIfTrue(RelativeJump),
    LoadGlobal(GlobalNameIndex),
    IsOp(OpInversion),
    ContainsOp(OpInversion),
    Reraise(Reraise),
    Copy(u32),
    ReturnConst(ConstIndex),
    BinaryOp(BinaryOperation),
    Send(RelativeJump),
    LoadFast(VarNameIndex),
    StoreFast(VarNameIndex),
    DeleteFast(VarNameIndex),
    LoadFastCheck(VarNameIndex),
    PopJumpIfNotNone(RelativeJump),
    PopJumpIfNone(RelativeJump),
    RaiseVarargs(RaiseForms),
    GetAwaitable(AwaitableWhere),
    MakeFunction(MakeFunctionFlags),
    BuildSlice(SliceCount),
    JumpBackwardNoInterrupt(RelativeJump),
    MakeCell(ClosureRefIndex),
    LoadClosure(ClosureRefIndex),
    LoadDeref(ClosureRefIndex),
    StoreDeref(ClosureRefIndex),
    DeleteDeref(ClosureRefIndex),
    JumpBackward(RelativeJump),
    LoadSuperAttr(SuperAttrNameIndex),
    CallFunctionEx(CallExFlags),
    LoadFastAndClear(VarNameIndex),
    // Extended arg is ommited in the resolved instructions
    ListAppend(u32),
    SetAdd(u32),
    MapAdd(u32),
    CopyFreeVars(u32),
    YieldValue(u32),
    Resume(ResumeWhere),
    MatchClass(u32),
    FormatValue(FormatFlag),
    BuildConstKeyMap(u32),
    BuildString(u32),
    ListExtend(u32),
    SetUpdate(u32),
    DictMerge(u32),
    DictUpdate(u32),
    Call(u32),
    KwNames(ConstIndex),
    CallIntrinsic1(Intrinsic1Functions),
    CallIntrinsic2(Intrinsic2Functions),
    LoadFromDictOrGlobals(DynamicIndex),
    LoadFromDictOrDeref(DynamicIndex),
    // Specialized variations of opcodes, we don't parse the arguments for these (with some exceptions)
    InstrumentedLoadSuperAttr(u32),
    InstrumentedPopJumpIfNone(RelativeJump),
    InstrumentedPopJumpIfNotNone(RelativeJump),
    InstrumentedResume(u32),
    InstrumentedCall(u32),
    InstrumentedReturnValue(u32),
    InstrumentedYieldValue(u32),
    InstrumentedCallFunctionEx(u32),
    InstrumentedJumpForward(RelativeJump),
    InstrumentedJumpBackward(RelativeJump),
    InstrumentedReturnConst(u32),
    InstrumentedForIter(RelativeJump),
    InstrumentedPopJumpIfFalse(RelativeJump),
    InstrumentedPopJumpIfTrue(RelativeJump),
    InstrumentedEndFor(u32),
    InstrumentedEndSend(u32),
    InstrumentedInstruction(u32),
    InstrumentedLine(u32),
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
    BinaryOpAddFloat(u32),
    BinaryOpAddInt(u32),
    BinaryOpAddUnicode(u32),
    BinaryOpInplaceAddUnicode(u32),
    BinaryOpMultiplyFloat(u32),
    BinaryOpMultiplyInt(u32),
    BinaryOpSubtractFloat(u32),
    BinaryOpSubtractInt(u32),
    BinarySubscrDict(u32),
    BinarySubscrGetitem(u32),
    BinarySubscrListInt(u32),
    BinarySubscrTupleInt(u32),
    CallPyExactArgs(u32),
    CallPyWithDefaults(u32),
    CallBoundMethodExactArgs(u32),
    CallBuiltinClass(u32),
    CallBuiltinFastWithKeywords(u32),
    CallMethodDescriptorFastWithKeywords(u32),
    CallNoKwBuiltinFast(u32),
    CallNoKwBuiltinO(u32),
    CallNoKwIsinstance(u32),
    CallNoKwLen(u32),
    CallNoKwListAppend(u32),
    CallNoKwMethodDescriptorFast(u32),
    CallNoKwMethodDescriptorNoargs(u32),
    CallNoKwMethodDescriptorO(u32),
    CallNoKwStr1(u32),
    CallNoKwTuple1(u32),
    CallNoKwType1(u32),
    CompareOpFloat(u32),
    CompareOpInt(u32),
    CompareOpStr(u32),
    ForIterList(RelativeJump),
    ForIterTuple(RelativeJump),
    ForIterRange(RelativeJump),
    ForIterGen(RelativeJump),
    LoadSuperAttrAttr(u32),
    LoadSuperAttrMethod(u32),
    LoadAttrClass(u32),
    LoadAttrGetattributeOverridden(u32),
    LoadAttrInstanceValue(u32),
    LoadAttrModule(u32),
    LoadAttrProperty(u32),
    LoadAttrSlot(u32),
    LoadAttrWithHint(u32),
    LoadAttrMethodLazyDict(u32),
    LoadAttrMethodNoDict(u32),
    LoadAttrMethodWithValues(u32),
    LoadConstLoadFast(u32),
    LoadFastLoadConst(u32),
    LoadFastLoadFast(u32),
    LoadGlobalBuiltin(u32),
    LoadGlobalModule(u32),
    StoreAttrInstanceValue(u32),
    StoreAttrSlot(u32),
    StoreAttrWithHint(u32),
    StoreFastLoadFast(u32),
    StoreFastStoreFast(u32),
    StoreSubscrDict(u32),
    StoreSubscrListInt(u32),
    UnpackSequenceList(u32),
    UnpackSequenceTuple(u32),
    UnpackSequenceTwoTuple(u32),
    SendGen(u32),
    InvalidOpcode((u8, u32)), // (opcode, arg)
}

/// A list of resolved instructions (extended_arg is resolved)
#[derive(Debug, Clone, PartialEq)]
pub struct ExtInstructions(Vec<ExtInstruction>);

impl InstructionAccess for ExtInstructions {
    type Instruction = ExtInstruction;

    /// Convert the resolved instructions into bytes. (converts to normal instructions first)
    fn to_bytes(&self) -> Vec<u8> {
        self.to_instructions().to_bytes()
    }
}

/// Resolves the actual index of the current jump instruction.
/// In 3.12 the jump offsets are relative to the CACHE opcodes succeeding the jump instruction.
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
    pub fn from_instructions(instructions: &[Instruction]) -> Self {
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
            match instruction {
                Instruction::ExtendedArg(_) => {
                    let index = instructions::get_real_jump_index(instructions, index)
                        .expect("Index is always valid here");

                    for mut entry in relative_jump_indexes.query_mut(&Interval::point(index as u32))
                    {
                        *entry.value() -= 1
                    }
                }
                _ => {}
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

        ext_instructions
    }

    pub fn append_instructions(&mut self, instructions: &[ExtInstruction]) {
        for instruction in instructions {
            self.0.push(*instruction);
        }
    }

    /// Append an instruction at the end
    pub fn append_instruction(&mut self, instruction: ExtInstruction) {
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

    /// Insert a slice of instructions at an index
    pub fn insert_instructions(&mut self, index: usize, instructions: &[ExtInstruction]) {
        for (idx, instruction) in instructions.iter().enumerate() {
            self.insert_instruction(index + idx, *instruction);
        }
    }

    /// Insert instruction at a specific index. It automatically fixes jump offsets in other instructions.
    pub fn insert_instruction(&mut self, index: usize, instruction: ExtInstruction) {
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

    /// Returns a hashmap of jump indexes and their jump target
    pub fn get_jump_map(&self) -> HashMap<u32, u32> {
        let mut jump_map: HashMap<u32, u32> = HashMap::new();

        for (index, instruction) in self.iter().enumerate() {
            let jump: Jump = match instruction {
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
                | ExtInstruction::InstrumentedJumpBackward(jump) => (*jump).into(),
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
    pub fn get_jump_target(&self, index: u32, jump: Jump) -> Option<(u32, ExtInstruction)> {
        match jump {
            Jump::Relative(RelativeJump {
                index: jump_index,
                direction: JumpDirection::Forward,
            }) => {
                let index = index + jump_index + 1;
                self.0
                    .get(index as usize)
                    .cloned()
                    .map(|target| (index, target))
            }
            Jump::Relative(RelativeJump {
                index: jump_index,
                direction: JumpDirection::Backward,
            }) => {
                let index = index - jump_index + 1;
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

    /// Convert the resolved instructions back into instructions with extended args.
    pub fn to_instructions(&self) -> Instructions {
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
                    let idx = get_real_jump_index(&self, idx).expect("Index is always valid here");

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
                    let idx = get_real_jump_index(&self, idx).expect("Index is always valid here");

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

                let index = get_real_jump_index(&self, index).expect("Index is always valid here");

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
                let index = get_real_jump_index(&self, index).expect("Index is always valid here");

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

        let mut instructions: Instructions = Instructions::with_capacity(self.0.len() * 2); // This will not be enough this as we dynamically generate EXTENDED_ARGS, but it's better than not reserving any length.

        for (index, instruction) in self.0.iter().enumerate() {
            let index = get_real_jump_index(&self, index).expect("Index is always valid here");

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

impl From<ExtInstructions> for Vec<u8> {
    fn from(val: ExtInstructions) -> Self {
        val.to_bytes()
    }
}

impl From<&[Instruction]> for ExtInstructions {
    fn from(code: &[Instruction]) -> Self {
        ExtInstructions::from_instructions(code)
    }
}

impl From<&[ExtInstruction]> for ExtInstructions {
    fn from(value: &[ExtInstruction]) -> Self {
        let mut instructions = ExtInstructions(Vec::with_capacity(value.len()));

        instructions.append_instructions(value);

        instructions
    }
}

impl TryFrom<(Opcode, u32)> for ExtInstruction {
    type Error = Error;
    fn try_from(value: (Opcode, u32)) -> Result<Self, Self::Error> {
        Ok(match value.0 {
            Opcode::CACHE => ExtInstruction::Cache(value.1.into()),
            Opcode::POP_TOP => ExtInstruction::PopTop(value.1.into()),
            Opcode::PUSH_NULL => ExtInstruction::PushNull(value.1.into()),
            Opcode::INTERPRETER_EXIT => ExtInstruction::InterpreterExit(value.1.into()),
            Opcode::END_FOR => ExtInstruction::EndFor(value.1.into()),
            Opcode::END_SEND => ExtInstruction::EndSend(value.1.into()),
            Opcode::NOP => ExtInstruction::Nop(value.1.into()),
            Opcode::UNARY_NEGATIVE => ExtInstruction::UnaryNegative(value.1.into()),
            Opcode::UNARY_NOT => ExtInstruction::UnaryNot(value.1.into()),
            Opcode::UNARY_INVERT => ExtInstruction::UnaryInvert(value.1.into()),
            Opcode::RESERVED => ExtInstruction::Reserved(value.1.into()),
            Opcode::BINARY_SUBSCR => ExtInstruction::BinarySubscr(value.1.into()),
            Opcode::BINARY_SLICE => ExtInstruction::BinarySlice(value.1.into()),
            Opcode::STORE_SLICE => ExtInstruction::StoreSlice(value.1.into()),
            Opcode::GET_LEN => ExtInstruction::GetLen(value.1.into()),
            Opcode::MATCH_MAPPING => ExtInstruction::MatchMapping(value.1.into()),
            Opcode::MATCH_SEQUENCE => ExtInstruction::MatchSequence(value.1.into()),
            Opcode::MATCH_KEYS => ExtInstruction::MatchKeys(value.1.into()),
            Opcode::PUSH_EXC_INFO => ExtInstruction::PushExcInfo(value.1.into()),
            Opcode::CHECK_EXC_MATCH => ExtInstruction::CheckExcMatch(value.1.into()),
            Opcode::CHECK_EG_MATCH => ExtInstruction::CheckEgMatch(value.1.into()),
            Opcode::WITH_EXCEPT_START => ExtInstruction::WithExceptStart(value.1.into()),
            Opcode::GET_AITER => ExtInstruction::GetAiter(value.1.into()),
            Opcode::GET_ANEXT => ExtInstruction::GetAnext(value.1.into()),
            Opcode::BEFORE_ASYNC_WITH => ExtInstruction::BeforeAsyncWith(value.1.into()),
            Opcode::BEFORE_WITH => ExtInstruction::BeforeWith(value.1.into()),
            Opcode::END_ASYNC_FOR => ExtInstruction::EndAsyncFor(value.1.into()),
            Opcode::CLEANUP_THROW => ExtInstruction::CleanupThrow(value.1.into()),
            Opcode::STORE_SUBSCR => ExtInstruction::StoreSubscr(value.1.into()),
            Opcode::DELETE_SUBSCR => ExtInstruction::DeleteSubscr(value.1.into()),
            Opcode::GET_ITER => ExtInstruction::GetIter(value.1.into()),
            Opcode::GET_YIELD_FROM_ITER => ExtInstruction::GetYieldFromIter(value.1.into()),
            Opcode::LOAD_BUILD_CLASS => ExtInstruction::LoadBuildClass(value.1.into()),
            Opcode::LOAD_ASSERTION_ERROR => ExtInstruction::LoadAssertionError(value.1.into()),
            Opcode::RETURN_GENERATOR => ExtInstruction::ReturnGenerator(value.1.into()),
            Opcode::RETURN_VALUE => ExtInstruction::ReturnValue(value.1.into()),
            Opcode::SETUP_ANNOTATIONS => ExtInstruction::SetupAnnotations(value.1.into()),
            Opcode::LOAD_LOCALS => ExtInstruction::LoadLocals(value.1.into()),
            Opcode::POP_EXCEPT => ExtInstruction::PopExcept(value.1.into()),
            Opcode::STORE_NAME => ExtInstruction::StoreName(NameIndex {
                index: value.1.into(),
            }),
            Opcode::DELETE_NAME => ExtInstruction::DeleteName(NameIndex {
                index: value.1.into(),
            }),
            Opcode::UNPACK_SEQUENCE => ExtInstruction::UnpackSequence(value.1.into()),
            Opcode::FOR_ITER => ExtInstruction::ForIter(RelativeJump {
                index: value.1.into(),
                direction: JumpDirection::Forward,
            }),
            Opcode::UNPACK_EX => ExtInstruction::UnpackEx(value.1.into()),
            Opcode::STORE_ATTR => ExtInstruction::StoreAttr(NameIndex {
                index: value.1.into(),
            }),
            Opcode::DELETE_ATTR => ExtInstruction::DeleteAttr(NameIndex {
                index: value.1.into(),
            }),
            Opcode::STORE_GLOBAL => ExtInstruction::StoreGlobal(NameIndex {
                index: value.1.into(),
            }),
            Opcode::DELETE_GLOBAL => ExtInstruction::DeleteGlobal(NameIndex {
                index: value.1.into(),
            }),
            Opcode::SWAP => ExtInstruction::Swap(value.1.into()),
            Opcode::LOAD_CONST => ExtInstruction::LoadConst(ConstIndex {
                index: value.1.into(),
            }),
            Opcode::LOAD_NAME => ExtInstruction::LoadName(NameIndex {
                index: value.1.into(),
            }),
            Opcode::BUILD_TUPLE => ExtInstruction::BuildTuple(value.1.into()),
            Opcode::BUILD_LIST => ExtInstruction::BuildList(value.1.into()),
            Opcode::BUILD_SET => ExtInstruction::BuildSet(value.1.into()),
            Opcode::BUILD_MAP => ExtInstruction::BuildMap(value.1.into()),
            Opcode::LOAD_ATTR => ExtInstruction::LoadAttr(AttrNameIndex {
                index: value.1.into(),
            }),
            Opcode::COMPARE_OP => ExtInstruction::CompareOp(value.1.into()),
            Opcode::IMPORT_NAME => ExtInstruction::ImportName(NameIndex {
                index: value.1.into(),
            }),
            Opcode::IMPORT_FROM => ExtInstruction::ImportFrom(NameIndex {
                index: value.1.into(),
            }),
            Opcode::JUMP_FORWARD => ExtInstruction::JumpForward(RelativeJump {
                index: value.1.into(),
                direction: JumpDirection::Forward,
            }),
            Opcode::POP_JUMP_IF_FALSE => ExtInstruction::PopJumpIfFalse(RelativeJump {
                index: value.1.into(),
                direction: JumpDirection::Forward,
            }),
            Opcode::POP_JUMP_IF_TRUE => ExtInstruction::PopJumpIfTrue(RelativeJump {
                index: value.1.into(),
                direction: JumpDirection::Forward,
            }),
            Opcode::LOAD_GLOBAL => ExtInstruction::LoadGlobal(GlobalNameIndex {
                index: value.1.into(),
            }),
            Opcode::IS_OP => ExtInstruction::IsOp(value.1.into()),
            Opcode::CONTAINS_OP => ExtInstruction::ContainsOp(value.1.into()),
            Opcode::RERAISE => ExtInstruction::Reraise(value.1.into()),
            Opcode::COPY => ExtInstruction::Copy(value.1.into()),
            Opcode::RETURN_CONST => ExtInstruction::ReturnConst(ConstIndex {
                index: value.1.into(),
            }),
            Opcode::BINARY_OP => ExtInstruction::BinaryOp(value.1.into()),
            Opcode::SEND => ExtInstruction::Send(RelativeJump {
                index: value.1.into(),
                direction: JumpDirection::Forward,
            }),
            Opcode::LOAD_FAST => ExtInstruction::LoadFast(VarNameIndex {
                index: value.1.into(),
            }),
            Opcode::STORE_FAST => ExtInstruction::StoreFast(VarNameIndex {
                index: value.1.into(),
            }),
            Opcode::DELETE_FAST => ExtInstruction::DeleteFast(VarNameIndex {
                index: value.1.into(),
            }),
            Opcode::LOAD_FAST_CHECK => ExtInstruction::LoadFastCheck(VarNameIndex {
                index: value.1.into(),
            }),
            Opcode::POP_JUMP_IF_NOT_NONE => ExtInstruction::PopJumpIfNotNone(RelativeJump {
                index: value.1.into(),
                direction: JumpDirection::Forward,
            }),
            Opcode::POP_JUMP_IF_NONE => ExtInstruction::PopJumpIfNone(RelativeJump {
                index: value.1.into(),
                direction: JumpDirection::Forward,
            }),
            Opcode::RAISE_VARARGS => ExtInstruction::RaiseVarargs(value.1.into()),
            Opcode::GET_AWAITABLE => ExtInstruction::GetAwaitable(value.1.into()),
            Opcode::MAKE_FUNCTION => {
                ExtInstruction::MakeFunction(MakeFunctionFlags::from_bits_retain(value.1))
            }
            Opcode::BUILD_SLICE => ExtInstruction::BuildSlice(value.1.into()),
            Opcode::JUMP_BACKWARD_NO_INTERRUPT => {
                ExtInstruction::JumpBackwardNoInterrupt(RelativeJump {
                    index: value.1.into(),
                    direction: JumpDirection::Backward,
                })
            }
            Opcode::MAKE_CELL => ExtInstruction::MakeCell(ClosureRefIndex {
                index: value.1.into(),
            }),
            Opcode::LOAD_CLOSURE => ExtInstruction::LoadClosure(ClosureRefIndex {
                index: value.1.into(),
            }),
            Opcode::LOAD_DEREF => ExtInstruction::LoadDeref(ClosureRefIndex {
                index: value.1.into(),
            }),
            Opcode::STORE_DEREF => ExtInstruction::StoreDeref(ClosureRefIndex {
                index: value.1.into(),
            }),
            Opcode::DELETE_DEREF => ExtInstruction::DeleteDeref(ClosureRefIndex {
                index: value.1.into(),
            }),
            Opcode::JUMP_BACKWARD => ExtInstruction::JumpBackward(RelativeJump {
                index: value.1.into(),
                direction: JumpDirection::Backward,
            }),
            Opcode::LOAD_SUPER_ATTR => ExtInstruction::LoadSuperAttr(SuperAttrNameIndex {
                index: value.1.into(),
            }),
            Opcode::CALL_FUNCTION_EX => ExtInstruction::CallFunctionEx(value.1.into()),
            Opcode::LOAD_FAST_AND_CLEAR => ExtInstruction::LoadFastAndClear(VarNameIndex {
                index: value.1.into(),
            }),
            Opcode::EXTENDED_ARG => return Err(Error::InvalidConversion),
            Opcode::LIST_APPEND => ExtInstruction::ListAppend(value.1.into()),
            Opcode::SET_ADD => ExtInstruction::SetAdd(value.1.into()),
            Opcode::MAP_ADD => ExtInstruction::MapAdd(value.1.into()),
            Opcode::COPY_FREE_VARS => ExtInstruction::CopyFreeVars(value.1.into()),
            Opcode::YIELD_VALUE => ExtInstruction::YieldValue(value.1.into()),
            Opcode::RESUME => ExtInstruction::Resume(value.1.into()),
            Opcode::MATCH_CLASS => ExtInstruction::MatchClass(value.1.into()),
            Opcode::FORMAT_VALUE => ExtInstruction::FormatValue(value.1.into()),
            Opcode::BUILD_CONST_KEY_MAP => ExtInstruction::BuildConstKeyMap(value.1.into()),
            Opcode::BUILD_STRING => ExtInstruction::BuildString(value.1.into()),
            Opcode::LIST_EXTEND => ExtInstruction::ListExtend(value.1.into()),
            Opcode::SET_UPDATE => ExtInstruction::SetUpdate(value.1.into()),
            Opcode::DICT_MERGE => ExtInstruction::DictMerge(value.1.into()),
            Opcode::DICT_UPDATE => ExtInstruction::DictUpdate(value.1.into()),
            Opcode::CALL => ExtInstruction::Call(value.1.into()),
            Opcode::KW_NAMES => ExtInstruction::KwNames(ConstIndex {
                index: value.1.into(),
            }),
            Opcode::CALL_INTRINSIC_1 => ExtInstruction::CallIntrinsic1(value.1.into()),
            Opcode::CALL_INTRINSIC_2 => ExtInstruction::CallIntrinsic2(value.1.into()),
            Opcode::LOAD_FROM_DICT_OR_GLOBALS => {
                ExtInstruction::LoadFromDictOrGlobals(DynamicIndex {
                    index: value.1.into(),
                })
            }
            Opcode::LOAD_FROM_DICT_OR_DEREF => ExtInstruction::LoadFromDictOrDeref(DynamicIndex {
                index: value.1.into(),
            }),
            Opcode::INSTRUMENTED_LOAD_SUPER_ATTR => {
                ExtInstruction::InstrumentedLoadSuperAttr(value.1.into())
            }
            Opcode::INSTRUMENTED_POP_JUMP_IF_NONE => {
                ExtInstruction::InstrumentedPopJumpIfNone(RelativeJump {
                    index: value.1.into(),
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::INSTRUMENTED_POP_JUMP_IF_NOT_NONE => {
                ExtInstruction::InstrumentedPopJumpIfNotNone(RelativeJump {
                    index: value.1.into(),
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::INSTRUMENTED_RESUME => ExtInstruction::InstrumentedResume(value.1.into()),
            Opcode::INSTRUMENTED_CALL => ExtInstruction::InstrumentedCall(value.1.into()),
            Opcode::INSTRUMENTED_RETURN_VALUE => {
                ExtInstruction::InstrumentedReturnValue(value.1.into())
            }
            Opcode::INSTRUMENTED_YIELD_VALUE => {
                ExtInstruction::InstrumentedYieldValue(value.1.into())
            }
            Opcode::INSTRUMENTED_CALL_FUNCTION_EX => {
                ExtInstruction::InstrumentedCallFunctionEx(value.1.into())
            }
            Opcode::INSTRUMENTED_JUMP_FORWARD => {
                ExtInstruction::InstrumentedJumpForward(RelativeJump {
                    index: value.1.into(),
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::INSTRUMENTED_JUMP_BACKWARD => {
                ExtInstruction::InstrumentedJumpBackward(RelativeJump {
                    index: value.1.into(),
                    direction: JumpDirection::Backward,
                })
            }
            Opcode::INSTRUMENTED_RETURN_CONST => {
                ExtInstruction::InstrumentedReturnConst(value.1.into())
            }
            Opcode::INSTRUMENTED_FOR_ITER => ExtInstruction::InstrumentedForIter(RelativeJump {
                index: value.1.into(),
                direction: JumpDirection::Forward,
            }),
            Opcode::INSTRUMENTED_POP_JUMP_IF_FALSE => {
                ExtInstruction::InstrumentedPopJumpIfFalse(RelativeJump {
                    index: value.1.into(),
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::INSTRUMENTED_POP_JUMP_IF_TRUE => {
                ExtInstruction::InstrumentedPopJumpIfTrue(RelativeJump {
                    index: value.1.into(),
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::INSTRUMENTED_END_FOR => ExtInstruction::InstrumentedEndFor(value.1.into()),
            Opcode::INSTRUMENTED_END_SEND => ExtInstruction::InstrumentedEndSend(value.1.into()),
            Opcode::INSTRUMENTED_INSTRUCTION => {
                ExtInstruction::InstrumentedInstruction(value.1.into())
            }
            Opcode::INSTRUMENTED_LINE => ExtInstruction::InstrumentedLine(value.1.into()),
            Opcode::BINARY_OP_ADD_FLOAT => ExtInstruction::BinaryOpAddFloat(value.1.into()),
            Opcode::BINARY_OP_ADD_INT => ExtInstruction::BinaryOpAddInt(value.1.into()),
            Opcode::BINARY_OP_ADD_UNICODE => ExtInstruction::BinaryOpAddUnicode(value.1.into()),
            Opcode::BINARY_OP_INPLACE_ADD_UNICODE => {
                ExtInstruction::BinaryOpInplaceAddUnicode(value.1.into())
            }
            Opcode::BINARY_OP_MULTIPLY_FLOAT => {
                ExtInstruction::BinaryOpMultiplyFloat(value.1.into())
            }
            Opcode::BINARY_OP_MULTIPLY_INT => ExtInstruction::BinaryOpMultiplyInt(value.1.into()),
            Opcode::BINARY_OP_SUBTRACT_FLOAT => {
                ExtInstruction::BinaryOpSubtractFloat(value.1.into())
            }
            Opcode::BINARY_OP_SUBTRACT_INT => ExtInstruction::BinaryOpSubtractInt(value.1.into()),
            Opcode::BINARY_SUBSCR_DICT => ExtInstruction::BinarySubscrDict(value.1.into()),
            Opcode::BINARY_SUBSCR_GETITEM => ExtInstruction::BinarySubscrGetitem(value.1.into()),
            Opcode::BINARY_SUBSCR_LIST_INT => ExtInstruction::BinarySubscrListInt(value.1.into()),
            Opcode::BINARY_SUBSCR_TUPLE_INT => ExtInstruction::BinarySubscrTupleInt(value.1.into()),
            Opcode::CALL_PY_EXACT_ARGS => ExtInstruction::CallPyExactArgs(value.1.into()),
            Opcode::CALL_PY_WITH_DEFAULTS => ExtInstruction::CallPyWithDefaults(value.1.into()),
            Opcode::CALL_BOUND_METHOD_EXACT_ARGS => {
                ExtInstruction::CallBoundMethodExactArgs(value.1.into())
            }
            Opcode::CALL_BUILTIN_CLASS => ExtInstruction::CallBuiltinClass(value.1.into()),
            Opcode::CALL_BUILTIN_FAST_WITH_KEYWORDS => {
                ExtInstruction::CallBuiltinFastWithKeywords(value.1.into())
            }
            Opcode::CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS => {
                ExtInstruction::CallMethodDescriptorFastWithKeywords(value.1.into())
            }
            Opcode::CALL_NO_KW_BUILTIN_FAST => ExtInstruction::CallNoKwBuiltinFast(value.1.into()),
            Opcode::CALL_NO_KW_BUILTIN_O => ExtInstruction::CallNoKwBuiltinO(value.1.into()),
            Opcode::CALL_NO_KW_ISINSTANCE => ExtInstruction::CallNoKwIsinstance(value.1.into()),
            Opcode::CALL_NO_KW_LEN => ExtInstruction::CallNoKwLen(value.1.into()),
            Opcode::CALL_NO_KW_LIST_APPEND => ExtInstruction::CallNoKwListAppend(value.1.into()),
            Opcode::CALL_NO_KW_METHOD_DESCRIPTOR_FAST => {
                ExtInstruction::CallNoKwMethodDescriptorFast(value.1.into())
            }
            Opcode::CALL_NO_KW_METHOD_DESCRIPTOR_NOARGS => {
                ExtInstruction::CallNoKwMethodDescriptorNoargs(value.1.into())
            }
            Opcode::CALL_NO_KW_METHOD_DESCRIPTOR_O => {
                ExtInstruction::CallNoKwMethodDescriptorO(value.1.into())
            }
            Opcode::CALL_NO_KW_STR_1 => ExtInstruction::CallNoKwStr1(value.1.into()),
            Opcode::CALL_NO_KW_TUPLE_1 => ExtInstruction::CallNoKwTuple1(value.1.into()),
            Opcode::CALL_NO_KW_TYPE_1 => ExtInstruction::CallNoKwType1(value.1.into()),
            Opcode::COMPARE_OP_FLOAT => ExtInstruction::CompareOpFloat(value.1.into()),
            Opcode::COMPARE_OP_INT => ExtInstruction::CompareOpInt(value.1.into()),
            Opcode::COMPARE_OP_STR => ExtInstruction::CompareOpStr(value.1.into()),
            Opcode::FOR_ITER_LIST => ExtInstruction::ForIterList(RelativeJump {
                index: value.1.into(),
                direction: JumpDirection::Forward,
            }),
            Opcode::FOR_ITER_TUPLE => ExtInstruction::ForIterTuple(RelativeJump {
                index: value.1.into(),
                direction: JumpDirection::Forward,
            }),
            Opcode::FOR_ITER_RANGE => ExtInstruction::ForIterRange(RelativeJump {
                index: value.1.into(),
                direction: JumpDirection::Forward,
            }),
            Opcode::FOR_ITER_GEN => ExtInstruction::ForIterGen(RelativeJump {
                index: value.1.into(),
                direction: JumpDirection::Forward,
            }),
            Opcode::LOAD_SUPER_ATTR_ATTR => ExtInstruction::LoadSuperAttrAttr(value.1.into()),
            Opcode::LOAD_SUPER_ATTR_METHOD => ExtInstruction::LoadSuperAttrMethod(value.1.into()),
            Opcode::LOAD_ATTR_CLASS => ExtInstruction::LoadAttrClass(value.1.into()),
            Opcode::LOAD_ATTR_GETATTRIBUTE_OVERRIDDEN => {
                ExtInstruction::LoadAttrGetattributeOverridden(value.1.into())
            }
            Opcode::LOAD_ATTR_INSTANCE_VALUE => {
                ExtInstruction::LoadAttrInstanceValue(value.1.into())
            }
            Opcode::LOAD_ATTR_MODULE => ExtInstruction::LoadAttrModule(value.1.into()),
            Opcode::LOAD_ATTR_PROPERTY => ExtInstruction::LoadAttrProperty(value.1.into()),
            Opcode::LOAD_ATTR_SLOT => ExtInstruction::LoadAttrSlot(value.1.into()),
            Opcode::LOAD_ATTR_WITH_HINT => ExtInstruction::LoadAttrWithHint(value.1.into()),
            Opcode::LOAD_ATTR_METHOD_LAZY_DICT => {
                ExtInstruction::LoadAttrMethodLazyDict(value.1.into())
            }
            Opcode::LOAD_ATTR_METHOD_NO_DICT => {
                ExtInstruction::LoadAttrMethodNoDict(value.1.into())
            }
            Opcode::LOAD_ATTR_METHOD_WITH_VALUES => {
                ExtInstruction::LoadAttrMethodWithValues(value.1.into())
            }
            Opcode::LOAD_CONST__LOAD_FAST => ExtInstruction::LoadConstLoadFast(value.1.into()),
            Opcode::LOAD_FAST__LOAD_CONST => ExtInstruction::LoadFastLoadConst(value.1.into()),
            Opcode::LOAD_FAST__LOAD_FAST => ExtInstruction::LoadFastLoadFast(value.1.into()),
            Opcode::LOAD_GLOBAL_BUILTIN => ExtInstruction::LoadGlobalBuiltin(value.1.into()),
            Opcode::LOAD_GLOBAL_MODULE => ExtInstruction::LoadGlobalModule(value.1.into()),
            Opcode::STORE_ATTR_INSTANCE_VALUE => {
                ExtInstruction::StoreAttrInstanceValue(value.1.into())
            }
            Opcode::STORE_ATTR_SLOT => ExtInstruction::StoreAttrSlot(value.1.into()),
            Opcode::STORE_ATTR_WITH_HINT => ExtInstruction::StoreAttrWithHint(value.1.into()),
            Opcode::STORE_FAST__LOAD_FAST => ExtInstruction::StoreFastLoadFast(value.1.into()),
            Opcode::STORE_FAST__STORE_FAST => ExtInstruction::StoreFastStoreFast(value.1.into()),
            Opcode::STORE_SUBSCR_DICT => ExtInstruction::StoreSubscrDict(value.1.into()),
            Opcode::STORE_SUBSCR_LIST_INT => ExtInstruction::StoreSubscrListInt(value.1.into()),
            Opcode::UNPACK_SEQUENCE_LIST => ExtInstruction::UnpackSequenceList(value.1.into()),
            Opcode::UNPACK_SEQUENCE_TUPLE => ExtInstruction::UnpackSequenceTuple(value.1.into()),
            Opcode::UNPACK_SEQUENCE_TWO_TUPLE => {
                ExtInstruction::UnpackSequenceTwoTuple(value.1.into())
            }
            Opcode::SEND_GEN => ExtInstruction::SendGen(value.1.into()),
            Opcode::INVALID_OPCODE(opcode) => ExtInstruction::InvalidOpcode((opcode, value.1)),
        })
    }
}

impl GenericInstruction for ExtInstruction {
    type Opcode = Opcode;
    type Arg = u32;

    fn get_opcode(&self) -> Self::Opcode {
        match self {
            ExtInstruction::Cache(_) => Opcode::CACHE,
            ExtInstruction::PopTop(_) => Opcode::POP_TOP,
            ExtInstruction::PushNull(_) => Opcode::PUSH_NULL,
            ExtInstruction::InterpreterExit(_) => Opcode::INTERPRETER_EXIT,
            ExtInstruction::EndFor(_) => Opcode::END_FOR,
            ExtInstruction::EndSend(_) => Opcode::END_SEND,
            ExtInstruction::Nop(_) => Opcode::NOP,
            ExtInstruction::UnaryNegative(_) => Opcode::UNARY_NEGATIVE,
            ExtInstruction::UnaryNot(_) => Opcode::UNARY_NOT,
            ExtInstruction::UnaryInvert(_) => Opcode::UNARY_INVERT,
            ExtInstruction::Reserved(_) => Opcode::RESERVED,
            ExtInstruction::BinarySubscr(_) => Opcode::BINARY_SUBSCR,
            ExtInstruction::BinarySlice(_) => Opcode::BINARY_SLICE,
            ExtInstruction::StoreSlice(_) => Opcode::STORE_SLICE,
            ExtInstruction::GetLen(_) => Opcode::GET_LEN,
            ExtInstruction::MatchMapping(_) => Opcode::MATCH_MAPPING,
            ExtInstruction::MatchSequence(_) => Opcode::MATCH_SEQUENCE,
            ExtInstruction::MatchKeys(_) => Opcode::MATCH_KEYS,
            ExtInstruction::PushExcInfo(_) => Opcode::PUSH_EXC_INFO,
            ExtInstruction::CheckExcMatch(_) => Opcode::CHECK_EXC_MATCH,
            ExtInstruction::CheckEgMatch(_) => Opcode::CHECK_EG_MATCH,
            ExtInstruction::WithExceptStart(_) => Opcode::WITH_EXCEPT_START,
            ExtInstruction::GetAiter(_) => Opcode::GET_AITER,
            ExtInstruction::GetAnext(_) => Opcode::GET_ANEXT,
            ExtInstruction::BeforeAsyncWith(_) => Opcode::BEFORE_ASYNC_WITH,
            ExtInstruction::BeforeWith(_) => Opcode::BEFORE_WITH,
            ExtInstruction::EndAsyncFor(_) => Opcode::END_ASYNC_FOR,
            ExtInstruction::CleanupThrow(_) => Opcode::CLEANUP_THROW,
            ExtInstruction::StoreSubscr(_) => Opcode::STORE_SUBSCR,
            ExtInstruction::DeleteSubscr(_) => Opcode::DELETE_SUBSCR,
            ExtInstruction::GetIter(_) => Opcode::GET_ITER,
            ExtInstruction::GetYieldFromIter(_) => Opcode::GET_YIELD_FROM_ITER,
            ExtInstruction::LoadBuildClass(_) => Opcode::LOAD_BUILD_CLASS,
            ExtInstruction::LoadAssertionError(_) => Opcode::LOAD_ASSERTION_ERROR,
            ExtInstruction::ReturnGenerator(_) => Opcode::RETURN_GENERATOR,
            ExtInstruction::ReturnValue(_) => Opcode::RETURN_VALUE,
            ExtInstruction::SetupAnnotations(_) => Opcode::SETUP_ANNOTATIONS,
            ExtInstruction::LoadLocals(_) => Opcode::LOAD_LOCALS,
            ExtInstruction::PopExcept(_) => Opcode::POP_EXCEPT,
            ExtInstruction::StoreName(_) => Opcode::STORE_NAME,
            ExtInstruction::DeleteName(_) => Opcode::DELETE_NAME,
            ExtInstruction::UnpackSequence(_) => Opcode::UNPACK_SEQUENCE,
            ExtInstruction::ForIter(_) => Opcode::FOR_ITER,
            ExtInstruction::UnpackEx(_) => Opcode::UNPACK_EX,
            ExtInstruction::StoreAttr(_) => Opcode::STORE_ATTR,
            ExtInstruction::DeleteAttr(_) => Opcode::DELETE_ATTR,
            ExtInstruction::StoreGlobal(_) => Opcode::STORE_GLOBAL,
            ExtInstruction::DeleteGlobal(_) => Opcode::DELETE_GLOBAL,
            ExtInstruction::Swap(_) => Opcode::SWAP,
            ExtInstruction::LoadConst(_) => Opcode::LOAD_CONST,
            ExtInstruction::LoadName(_) => Opcode::LOAD_NAME,
            ExtInstruction::BuildTuple(_) => Opcode::BUILD_TUPLE,
            ExtInstruction::BuildList(_) => Opcode::BUILD_LIST,
            ExtInstruction::BuildSet(_) => Opcode::BUILD_SET,
            ExtInstruction::BuildMap(_) => Opcode::BUILD_MAP,
            ExtInstruction::LoadAttr(_) => Opcode::LOAD_ATTR,
            ExtInstruction::CompareOp(_) => Opcode::COMPARE_OP,
            ExtInstruction::ImportName(_) => Opcode::IMPORT_NAME,
            ExtInstruction::ImportFrom(_) => Opcode::IMPORT_FROM,
            ExtInstruction::JumpForward(_) => Opcode::JUMP_FORWARD,
            ExtInstruction::PopJumpIfFalse(_) => Opcode::POP_JUMP_IF_FALSE,
            ExtInstruction::PopJumpIfTrue(_) => Opcode::POP_JUMP_IF_TRUE,
            ExtInstruction::LoadGlobal(_) => Opcode::LOAD_GLOBAL,
            ExtInstruction::IsOp(_) => Opcode::IS_OP,
            ExtInstruction::ContainsOp(_) => Opcode::CONTAINS_OP,
            ExtInstruction::Reraise(_) => Opcode::RERAISE,
            ExtInstruction::Copy(_) => Opcode::COPY,
            ExtInstruction::ReturnConst(_) => Opcode::RETURN_CONST,
            ExtInstruction::BinaryOp(_) => Opcode::BINARY_OP,
            ExtInstruction::Send(_) => Opcode::SEND,
            ExtInstruction::LoadFast(_) => Opcode::LOAD_FAST,
            ExtInstruction::StoreFast(_) => Opcode::STORE_FAST,
            ExtInstruction::DeleteFast(_) => Opcode::DELETE_FAST,
            ExtInstruction::LoadFastCheck(_) => Opcode::LOAD_FAST_CHECK,
            ExtInstruction::PopJumpIfNotNone(_) => Opcode::POP_JUMP_IF_NOT_NONE,
            ExtInstruction::PopJumpIfNone(_) => Opcode::POP_JUMP_IF_NONE,
            ExtInstruction::RaiseVarargs(_) => Opcode::RAISE_VARARGS,
            ExtInstruction::GetAwaitable(_) => Opcode::GET_AWAITABLE,
            ExtInstruction::MakeFunction(_) => Opcode::MAKE_FUNCTION,
            ExtInstruction::BuildSlice(_) => Opcode::BUILD_SLICE,
            ExtInstruction::JumpBackwardNoInterrupt(_) => Opcode::JUMP_BACKWARD_NO_INTERRUPT,
            ExtInstruction::MakeCell(_) => Opcode::MAKE_CELL,
            ExtInstruction::LoadClosure(_) => Opcode::LOAD_CLOSURE,
            ExtInstruction::LoadDeref(_) => Opcode::LOAD_DEREF,
            ExtInstruction::StoreDeref(_) => Opcode::STORE_DEREF,
            ExtInstruction::DeleteDeref(_) => Opcode::DELETE_DEREF,
            ExtInstruction::JumpBackward(_) => Opcode::JUMP_BACKWARD,
            ExtInstruction::LoadSuperAttr(_) => Opcode::LOAD_SUPER_ATTR,
            ExtInstruction::CallFunctionEx(_) => Opcode::CALL_FUNCTION_EX,
            ExtInstruction::LoadFastAndClear(_) => Opcode::LOAD_FAST_AND_CLEAR,
            ExtInstruction::ListAppend(_) => Opcode::LIST_APPEND,
            ExtInstruction::SetAdd(_) => Opcode::SET_ADD,
            ExtInstruction::MapAdd(_) => Opcode::MAP_ADD,
            ExtInstruction::CopyFreeVars(_) => Opcode::COPY_FREE_VARS,
            ExtInstruction::YieldValue(_) => Opcode::YIELD_VALUE,
            ExtInstruction::Resume(_) => Opcode::RESUME,
            ExtInstruction::MatchClass(_) => Opcode::MATCH_CLASS,
            ExtInstruction::FormatValue(_) => Opcode::FORMAT_VALUE,
            ExtInstruction::BuildConstKeyMap(_) => Opcode::BUILD_CONST_KEY_MAP,
            ExtInstruction::BuildString(_) => Opcode::BUILD_STRING,
            ExtInstruction::ListExtend(_) => Opcode::LIST_EXTEND,
            ExtInstruction::SetUpdate(_) => Opcode::SET_UPDATE,
            ExtInstruction::DictMerge(_) => Opcode::DICT_MERGE,
            ExtInstruction::DictUpdate(_) => Opcode::DICT_UPDATE,
            ExtInstruction::Call(_) => Opcode::CALL,
            ExtInstruction::KwNames(_) => Opcode::KW_NAMES,
            ExtInstruction::CallIntrinsic1(_) => Opcode::CALL_INTRINSIC_1,
            ExtInstruction::CallIntrinsic2(_) => Opcode::CALL_INTRINSIC_2,
            ExtInstruction::LoadFromDictOrGlobals(_) => Opcode::LOAD_FROM_DICT_OR_GLOBALS,
            ExtInstruction::LoadFromDictOrDeref(_) => Opcode::LOAD_FROM_DICT_OR_DEREF,
            ExtInstruction::InstrumentedLoadSuperAttr(_) => Opcode::INSTRUMENTED_LOAD_SUPER_ATTR,
            ExtInstruction::InstrumentedPopJumpIfNone(_) => Opcode::INSTRUMENTED_POP_JUMP_IF_NONE,
            ExtInstruction::InstrumentedPopJumpIfNotNone(_) => {
                Opcode::INSTRUMENTED_POP_JUMP_IF_NOT_NONE
            }
            ExtInstruction::InstrumentedResume(_) => Opcode::INSTRUMENTED_RESUME,
            ExtInstruction::InstrumentedCall(_) => Opcode::INSTRUMENTED_CALL,
            ExtInstruction::InstrumentedReturnValue(_) => Opcode::INSTRUMENTED_RETURN_VALUE,
            ExtInstruction::InstrumentedYieldValue(_) => Opcode::INSTRUMENTED_YIELD_VALUE,
            ExtInstruction::InstrumentedCallFunctionEx(_) => Opcode::INSTRUMENTED_CALL_FUNCTION_EX,
            ExtInstruction::InstrumentedJumpForward(_) => Opcode::INSTRUMENTED_JUMP_FORWARD,
            ExtInstruction::InstrumentedJumpBackward(_) => Opcode::INSTRUMENTED_JUMP_BACKWARD,
            ExtInstruction::InstrumentedReturnConst(_) => Opcode::INSTRUMENTED_RETURN_CONST,
            ExtInstruction::InstrumentedForIter(_) => Opcode::INSTRUMENTED_FOR_ITER,
            ExtInstruction::InstrumentedPopJumpIfFalse(_) => Opcode::INSTRUMENTED_POP_JUMP_IF_FALSE,
            ExtInstruction::InstrumentedPopJumpIfTrue(_) => Opcode::INSTRUMENTED_POP_JUMP_IF_TRUE,
            ExtInstruction::InstrumentedEndFor(_) => Opcode::INSTRUMENTED_END_FOR,
            ExtInstruction::InstrumentedEndSend(_) => Opcode::INSTRUMENTED_END_SEND,
            ExtInstruction::InstrumentedInstruction(_) => Opcode::INSTRUMENTED_INSTRUCTION,
            ExtInstruction::InstrumentedLine(_) => Opcode::INSTRUMENTED_LINE,
            ExtInstruction::BinaryOpAddFloat(_) => Opcode::BINARY_OP_ADD_FLOAT,
            ExtInstruction::BinaryOpAddInt(_) => Opcode::BINARY_OP_ADD_INT,
            ExtInstruction::BinaryOpAddUnicode(_) => Opcode::BINARY_OP_ADD_UNICODE,
            ExtInstruction::BinaryOpInplaceAddUnicode(_) => Opcode::BINARY_OP_INPLACE_ADD_UNICODE,
            ExtInstruction::BinaryOpMultiplyFloat(_) => Opcode::BINARY_OP_MULTIPLY_FLOAT,
            ExtInstruction::BinaryOpMultiplyInt(_) => Opcode::BINARY_OP_MULTIPLY_INT,
            ExtInstruction::BinaryOpSubtractFloat(_) => Opcode::BINARY_OP_SUBTRACT_FLOAT,
            ExtInstruction::BinaryOpSubtractInt(_) => Opcode::BINARY_OP_SUBTRACT_INT,
            ExtInstruction::BinarySubscrDict(_) => Opcode::BINARY_SUBSCR_DICT,
            ExtInstruction::BinarySubscrGetitem(_) => Opcode::BINARY_SUBSCR_GETITEM,
            ExtInstruction::BinarySubscrListInt(_) => Opcode::BINARY_SUBSCR_LIST_INT,
            ExtInstruction::BinarySubscrTupleInt(_) => Opcode::BINARY_SUBSCR_TUPLE_INT,
            ExtInstruction::CallPyExactArgs(_) => Opcode::CALL_PY_EXACT_ARGS,
            ExtInstruction::CallPyWithDefaults(_) => Opcode::CALL_PY_WITH_DEFAULTS,
            ExtInstruction::CallBoundMethodExactArgs(_) => Opcode::CALL_BOUND_METHOD_EXACT_ARGS,
            ExtInstruction::CallBuiltinClass(_) => Opcode::CALL_BUILTIN_CLASS,
            ExtInstruction::CallBuiltinFastWithKeywords(_) => {
                Opcode::CALL_BUILTIN_FAST_WITH_KEYWORDS
            }
            ExtInstruction::CallMethodDescriptorFastWithKeywords(_) => {
                Opcode::CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS
            }
            ExtInstruction::CallNoKwBuiltinFast(_) => Opcode::CALL_NO_KW_BUILTIN_FAST,
            ExtInstruction::CallNoKwBuiltinO(_) => Opcode::CALL_NO_KW_BUILTIN_O,
            ExtInstruction::CallNoKwIsinstance(_) => Opcode::CALL_NO_KW_ISINSTANCE,
            ExtInstruction::CallNoKwLen(_) => Opcode::CALL_NO_KW_LEN,
            ExtInstruction::CallNoKwListAppend(_) => Opcode::CALL_NO_KW_LIST_APPEND,
            ExtInstruction::CallNoKwMethodDescriptorFast(_) => {
                Opcode::CALL_NO_KW_METHOD_DESCRIPTOR_FAST
            }
            ExtInstruction::CallNoKwMethodDescriptorNoargs(_) => {
                Opcode::CALL_NO_KW_METHOD_DESCRIPTOR_NOARGS
            }
            ExtInstruction::CallNoKwMethodDescriptorO(_) => Opcode::CALL_NO_KW_METHOD_DESCRIPTOR_O,
            ExtInstruction::CallNoKwStr1(_) => Opcode::CALL_NO_KW_STR_1,
            ExtInstruction::CallNoKwTuple1(_) => Opcode::CALL_NO_KW_TUPLE_1,
            ExtInstruction::CallNoKwType1(_) => Opcode::CALL_NO_KW_TYPE_1,
            ExtInstruction::CompareOpFloat(_) => Opcode::COMPARE_OP_FLOAT,
            ExtInstruction::CompareOpInt(_) => Opcode::COMPARE_OP_INT,
            ExtInstruction::CompareOpStr(_) => Opcode::COMPARE_OP_STR,
            ExtInstruction::ForIterList(_) => Opcode::FOR_ITER_LIST,
            ExtInstruction::ForIterTuple(_) => Opcode::FOR_ITER_TUPLE,
            ExtInstruction::ForIterRange(_) => Opcode::FOR_ITER_RANGE,
            ExtInstruction::ForIterGen(_) => Opcode::FOR_ITER_GEN,
            ExtInstruction::LoadSuperAttrAttr(_) => Opcode::LOAD_SUPER_ATTR_ATTR,
            ExtInstruction::LoadSuperAttrMethod(_) => Opcode::LOAD_SUPER_ATTR_METHOD,
            ExtInstruction::LoadAttrClass(_) => Opcode::LOAD_ATTR_CLASS,
            ExtInstruction::LoadAttrGetattributeOverridden(_) => {
                Opcode::LOAD_ATTR_GETATTRIBUTE_OVERRIDDEN
            }
            ExtInstruction::LoadAttrInstanceValue(_) => Opcode::LOAD_ATTR_INSTANCE_VALUE,
            ExtInstruction::LoadAttrModule(_) => Opcode::LOAD_ATTR_MODULE,
            ExtInstruction::LoadAttrProperty(_) => Opcode::LOAD_ATTR_PROPERTY,
            ExtInstruction::LoadAttrSlot(_) => Opcode::LOAD_ATTR_SLOT,
            ExtInstruction::LoadAttrWithHint(_) => Opcode::LOAD_ATTR_WITH_HINT,
            ExtInstruction::LoadAttrMethodLazyDict(_) => Opcode::LOAD_ATTR_METHOD_LAZY_DICT,
            ExtInstruction::LoadAttrMethodNoDict(_) => Opcode::LOAD_ATTR_METHOD_NO_DICT,
            ExtInstruction::LoadAttrMethodWithValues(_) => Opcode::LOAD_ATTR_METHOD_WITH_VALUES,
            ExtInstruction::LoadConstLoadFast(_) => Opcode::LOAD_CONST__LOAD_FAST,
            ExtInstruction::LoadFastLoadConst(_) => Opcode::LOAD_FAST__LOAD_CONST,
            ExtInstruction::LoadFastLoadFast(_) => Opcode::LOAD_FAST__LOAD_FAST,
            ExtInstruction::LoadGlobalBuiltin(_) => Opcode::LOAD_GLOBAL_BUILTIN,
            ExtInstruction::LoadGlobalModule(_) => Opcode::LOAD_GLOBAL_MODULE,
            ExtInstruction::StoreAttrInstanceValue(_) => Opcode::STORE_ATTR_INSTANCE_VALUE,
            ExtInstruction::StoreAttrSlot(_) => Opcode::STORE_ATTR_SLOT,
            ExtInstruction::StoreAttrWithHint(_) => Opcode::STORE_ATTR_WITH_HINT,
            ExtInstruction::StoreFastLoadFast(_) => Opcode::STORE_FAST__LOAD_FAST,
            ExtInstruction::StoreFastStoreFast(_) => Opcode::STORE_FAST__STORE_FAST,
            ExtInstruction::StoreSubscrDict(_) => Opcode::STORE_SUBSCR_DICT,
            ExtInstruction::StoreSubscrListInt(_) => Opcode::STORE_SUBSCR_LIST_INT,
            ExtInstruction::UnpackSequenceList(_) => Opcode::UNPACK_SEQUENCE_LIST,
            ExtInstruction::UnpackSequenceTuple(_) => Opcode::UNPACK_SEQUENCE_TUPLE,
            ExtInstruction::UnpackSequenceTwoTuple(_) => Opcode::UNPACK_SEQUENCE_TWO_TUPLE,
            ExtInstruction::SendGen(_) => Opcode::SEND_GEN,
            ExtInstruction::InvalidOpcode((opcode, _)) => Opcode::INVALID_OPCODE(*opcode),
        }
    }

    fn get_raw_value(&self) -> Self::Arg {
        match &self {
            ExtInstruction::Cache(n)
            | ExtInstruction::PopTop(n)
            | ExtInstruction::PushNull(n)
            | ExtInstruction::InterpreterExit(n)
            | ExtInstruction::EndFor(n)
            | ExtInstruction::EndSend(n)
            | ExtInstruction::Nop(n)
            | ExtInstruction::UnaryNegative(n)
            | ExtInstruction::UnaryNot(n)
            | ExtInstruction::UnaryInvert(n)
            | ExtInstruction::Reserved(n)
            | ExtInstruction::BinarySubscr(n)
            | ExtInstruction::BinarySlice(n)
            | ExtInstruction::StoreSlice(n)
            | ExtInstruction::GetLen(n)
            | ExtInstruction::MatchMapping(n)
            | ExtInstruction::MatchSequence(n)
            | ExtInstruction::MatchKeys(n)
            | ExtInstruction::PushExcInfo(n)
            | ExtInstruction::CheckExcMatch(n)
            | ExtInstruction::CheckEgMatch(n)
            | ExtInstruction::WithExceptStart(n)
            | ExtInstruction::GetAiter(n)
            | ExtInstruction::GetAnext(n)
            | ExtInstruction::BeforeAsyncWith(n)
            | ExtInstruction::BeforeWith(n)
            | ExtInstruction::EndAsyncFor(n)
            | ExtInstruction::CleanupThrow(n)
            | ExtInstruction::StoreSubscr(n)
            | ExtInstruction::DeleteSubscr(n)
            | ExtInstruction::GetIter(n)
            | ExtInstruction::GetYieldFromIter(n)
            | ExtInstruction::LoadBuildClass(n)
            | ExtInstruction::LoadAssertionError(n)
            | ExtInstruction::ReturnGenerator(n)
            | ExtInstruction::ReturnValue(n)
            | ExtInstruction::SetupAnnotations(n)
            | ExtInstruction::LoadLocals(n)
            | ExtInstruction::PopExcept(n) => n.0,
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
            ExtInstruction::UnpackSequence(n)
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
            | ExtInstruction::Call(n)
            | ExtInstruction::InstrumentedLoadSuperAttr(n)
            | ExtInstruction::InstrumentedResume(n)
            | ExtInstruction::InstrumentedCall(n)
            | ExtInstruction::InstrumentedReturnValue(n)
            | ExtInstruction::InstrumentedYieldValue(n)
            | ExtInstruction::InstrumentedCallFunctionEx(n)
            | ExtInstruction::InstrumentedReturnConst(n)
            | ExtInstruction::InstrumentedEndFor(n)
            | ExtInstruction::InstrumentedEndSend(n)
            | ExtInstruction::InstrumentedInstruction(n)
            | ExtInstruction::InstrumentedLine(n)
            | ExtInstruction::BinaryOpAddFloat(n)
            | ExtInstruction::BinaryOpAddInt(n)
            | ExtInstruction::BinaryOpAddUnicode(n)
            | ExtInstruction::BinaryOpInplaceAddUnicode(n)
            | ExtInstruction::BinaryOpMultiplyFloat(n)
            | ExtInstruction::BinaryOpMultiplyInt(n)
            | ExtInstruction::BinaryOpSubtractFloat(n)
            | ExtInstruction::BinaryOpSubtractInt(n)
            | ExtInstruction::BinarySubscrDict(n)
            | ExtInstruction::BinarySubscrGetitem(n)
            | ExtInstruction::BinarySubscrListInt(n)
            | ExtInstruction::BinarySubscrTupleInt(n)
            | ExtInstruction::CallPyExactArgs(n)
            | ExtInstruction::CallPyWithDefaults(n)
            | ExtInstruction::CallBoundMethodExactArgs(n)
            | ExtInstruction::CallBuiltinClass(n)
            | ExtInstruction::CallBuiltinFastWithKeywords(n)
            | ExtInstruction::CallMethodDescriptorFastWithKeywords(n)
            | ExtInstruction::CallNoKwBuiltinFast(n)
            | ExtInstruction::CallNoKwBuiltinO(n)
            | ExtInstruction::CallNoKwIsinstance(n)
            | ExtInstruction::CallNoKwLen(n)
            | ExtInstruction::CallNoKwListAppend(n)
            | ExtInstruction::CallNoKwMethodDescriptorFast(n)
            | ExtInstruction::CallNoKwMethodDescriptorNoargs(n)
            | ExtInstruction::CallNoKwMethodDescriptorO(n)
            | ExtInstruction::CallNoKwStr1(n)
            | ExtInstruction::CallNoKwTuple1(n)
            | ExtInstruction::CallNoKwType1(n)
            | ExtInstruction::CompareOpFloat(n)
            | ExtInstruction::CompareOpInt(n)
            | ExtInstruction::CompareOpStr(n)
            | ExtInstruction::LoadSuperAttrAttr(n)
            | ExtInstruction::LoadSuperAttrMethod(n)
            | ExtInstruction::LoadAttrClass(n)
            | ExtInstruction::LoadAttrGetattributeOverridden(n)
            | ExtInstruction::LoadAttrInstanceValue(n)
            | ExtInstruction::LoadAttrModule(n)
            | ExtInstruction::LoadAttrProperty(n)
            | ExtInstruction::LoadAttrSlot(n)
            | ExtInstruction::LoadAttrWithHint(n)
            | ExtInstruction::LoadAttrMethodLazyDict(n)
            | ExtInstruction::LoadAttrMethodNoDict(n)
            | ExtInstruction::LoadAttrMethodWithValues(n)
            | ExtInstruction::LoadConstLoadFast(n)
            | ExtInstruction::LoadFastLoadConst(n)
            | ExtInstruction::LoadFastLoadFast(n)
            | ExtInstruction::LoadGlobalBuiltin(n)
            | ExtInstruction::LoadGlobalModule(n)
            | ExtInstruction::StoreAttrInstanceValue(n)
            | ExtInstruction::StoreAttrSlot(n)
            | ExtInstruction::StoreAttrWithHint(n)
            | ExtInstruction::StoreFastLoadFast(n)
            | ExtInstruction::StoreFastStoreFast(n)
            | ExtInstruction::StoreSubscrDict(n)
            | ExtInstruction::StoreSubscrListInt(n)
            | ExtInstruction::UnpackSequenceList(n)
            | ExtInstruction::UnpackSequenceTuple(n)
            | ExtInstruction::UnpackSequenceTwoTuple(n)
            | ExtInstruction::SendGen(n) => *n,
            ExtInstruction::CallIntrinsic1(functions) => functions.into(),
            ExtInstruction::CallIntrinsic2(functions) => functions.into(),
            ExtInstruction::LoadConst(const_index)
            | ExtInstruction::ReturnConst(const_index)
            | ExtInstruction::KwNames(const_index) => const_index.index,
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
            ExtInstruction::LoadFromDictOrDeref(dynamic_index)
            | ExtInstruction::LoadFromDictOrGlobals(dynamic_index) => dynamic_index.index,
            ExtInstruction::RaiseVarargs(raise_var_args) => raise_var_args.into(),
            ExtInstruction::GetAwaitable(awaitable_where) => awaitable_where.into(),
            ExtInstruction::MakeFunction(flags) => flags.bits(),
            ExtInstruction::BuildSlice(slice) => slice.into(),
            ExtInstruction::MakeCell(closure_index)
            | ExtInstruction::LoadClosure(closure_index)
            | ExtInstruction::LoadDeref(closure_index)
            | ExtInstruction::StoreDeref(closure_index)
            | ExtInstruction::DeleteDeref(closure_index) => closure_index.index,
            ExtInstruction::CallFunctionEx(flags) => flags.into(),
            ExtInstruction::Resume(resume_where) => resume_where.into(),
            ExtInstruction::FormatValue(format) => format.bits().into(),
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
