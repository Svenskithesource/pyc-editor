use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use store_interval_tree::{Interval, IntervalTree};

use crate::{
    error::Error,
    traits::{GenericInstruction, InstructionAccess},
    utils::get_extended_args_count,
    v311::{
        code_objects::{
            AwaitableWhere, BinaryOperation, CallExFlags, ClosureRefIndex, CompareOperation,
            ConstIndex, FormatFlag, GlobalNameIndex, Jump, JumpDirection, MakeFunctionFlags,
            NameIndex, OpInversion, RaiseForms, RelativeJump, Reraise, ResumeWhere, SliceCount,
            VarNameIndex,
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
    Nop(UnusedArgument),
    UnaryPositive(UnusedArgument),
    UnaryNegative(UnusedArgument),
    UnaryNot(UnusedArgument),
    UnaryInvert(UnusedArgument),
    BinarySubscr(UnusedArgument),
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
    StoreSubscr(UnusedArgument),
    DeleteSubscr(UnusedArgument),
    GetIter(UnusedArgument),
    GetYieldFromIter(UnusedArgument),
    PrintExpr(UnusedArgument),
    LoadBuildClass(UnusedArgument),
    LoadAssertionError(UnusedArgument),
    ReturnGenerator(UnusedArgument),
    ListToTuple(UnusedArgument),
    ReturnValue(UnusedArgument),
    ImportStar(UnusedArgument),
    SetupAnnotations(UnusedArgument),
    YieldValue(UnusedArgument),
    AsyncGenWrap(UnusedArgument),
    PrepReraiseStar(UnusedArgument),
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
    LoadAttr(NameIndex),
    CompareOp(CompareOperation),
    ImportName(NameIndex),
    ImportFrom(NameIndex),
    JumpForward(RelativeJump),
    JumpIfFalseOrPop(RelativeJump),
    JumpIfTrueOrPop(RelativeJump),
    PopJumpForwardIfFalse(RelativeJump),
    PopJumpForwardIfTrue(RelativeJump),
    LoadGlobal(GlobalNameIndex),
    IsOp(OpInversion),
    ContainsOp(OpInversion),
    Reraise(Reraise),
    Copy(u32),
    BinaryOp(BinaryOperation),
    Send(RelativeJump),
    LoadFast(VarNameIndex),
    StoreFast(VarNameIndex),
    DeleteFast(VarNameIndex),
    PopJumpForwardIfNotNone(RelativeJump),
    PopJumpForwardIfNone(RelativeJump),
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
    CallFunctionEx(CallExFlags),
    // Extended arg is ommited in the resolved instruction
    ListAppend(u32),
    SetAdd(u32),
    MapAdd(u32),
    LoadClassderef(ClosureRefIndex),
    CopyFreeVars(u32),
    Resume(ResumeWhere),
    MatchClass(u32),
    FormatValue(FormatFlag),
    BuildConstKeyMap(u32),
    BuildString(u32),
    LoadMethod(NameIndex),
    ListExtend(u32),
    SetUpdate(u32),
    DictMerge(u32),
    DictUpdate(u32),
    Precall(u32),
    Call(u32),
    KwNames(ConstIndex),
    PopJumpBackwardIfNotNone(RelativeJump),
    PopJumpBackwardIfNone(RelativeJump),
    PopJumpBackwardIfFalse(RelativeJump),
    PopJumpBackwardIfTrue(RelativeJump),
    // Specialized variations of opcodes, we don't parse the arguments for these (with some exceptions)
    BinaryOpAdaptive(u32),
    BinaryOpAddFloat(u32),
    BinaryOpAddInt(u32),
    BinaryOpAddUnicode(u32),
    BinaryOpInplaceAddUnicode(u32),
    BinaryOpMultiplyFloat(u32),
    BinaryOpMultiplyInt(u32),
    BinaryOpSubtractFloat(u32),
    BinaryOpSubtractInt(u32),
    BinarySubscrAdaptive(u32),
    BinarySubscrDict(u32),
    BinarySubscrGetitem(u32),
    BinarySubscrListInt(u32),
    BinarySubscrTupleInt(u32),
    CallAdaptive(u32),
    CallPyExactArgs(u32),
    CallPyWithDefaults(u32),
    CompareOpAdaptive(u32),
    CompareOpFloatJump(u32),
    CompareOpIntJump(u32),
    CompareOpStrJump(u32),
    // Extended arg quick ommited in resolved instructions
    JumpBackwardQuick(RelativeJump),
    LoadAttrAdaptive(u32),
    LoadAttrInstanceValue(u32),
    LoadAttrModule(u32),
    LoadAttrSlot(u32),
    LoadAttrWithHint(u32),
    LoadConstLoadFast(u32),
    LoadFastLoadConst(u32),
    LoadFastLoadFast(u32),
    LoadGlobalAdaptive(u32),
    LoadGlobalBuiltin(u32),
    LoadGlobalModule(u32),
    LoadMethodAdaptive(u32),
    LoadMethodClass(u32),
    LoadMethodModule(u32),
    LoadMethodNoDict(u32),
    LoadMethodWithDict(u32),
    LoadMethodWithValues(u32),
    PrecallAdaptive(u32),
    PrecallBoundMethod(u32),
    PrecallBuiltinClass(u32),
    PrecallBuiltinFastWithKeywords(u32),
    PrecallMethodDescriptorFastWithKeywords(u32),
    PrecallNoKwBuiltinFast(u32),
    PrecallNoKwBuiltinO(u32),
    PrecallNoKwIsinstance(u32),
    PrecallNoKwLen(u32),
    PrecallNoKwListAppend(u32),
    PrecallNoKwMethodDescriptorFast(u32),
    PrecallNoKwMethodDescriptorNoargs(u32),
    PrecallNoKwMethodDescriptorO(u32),
    PrecallNoKwStr1(u32),
    PrecallNoKwTuple1(u32),
    PrecallNoKwType1(u32),
    PrecallPyfunc(u32),
    ResumeQuick(ResumeWhere),
    StoreAttrAdaptive(u32),
    StoreAttrInstanceValue(u32),
    StoreAttrSlot(u32),
    StoreAttrWithHint(u32),
    StoreFastLoadFast(u32),
    StoreFastStoreFast(u32),
    StoreSubscrAdaptive(u32),
    StoreSubscrDict(u32),
    StoreSubscrListInt(u32),
    UnpackSequenceAdaptive(u32),
    UnpackSequenceList(u32),
    UnpackSequenceTuple(u32),
    UnpackSequenceTwoTuple(u32),
    DoTracing(u32),
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
                Instruction::ExtendedArg(arg) | Instruction::ExtendedArgQuick(arg) => {
                    let arg = *arg as u32 | extended_arg;
                    extended_arg = arg << 8;
                    continue;
                }
                Instruction::ForIter(arg)
                | Instruction::JumpForward(arg)
                | Instruction::JumpIfFalseOrPop(arg)
                | Instruction::JumpIfTrueOrPop(arg)
                | Instruction::PopJumpForwardIfFalse(arg)
                | Instruction::PopJumpForwardIfTrue(arg)
                | Instruction::Send(arg)
                | Instruction::PopJumpForwardIfNotNone(arg)
                | Instruction::PopJumpForwardIfNone(arg) => {
                    let arg = *arg as u32 | extended_arg;
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
                | Instruction::JumpBackwardQuick(arg)
                | Instruction::PopJumpBackwardIfNotNone(arg)
                | Instruction::PopJumpBackwardIfNone(arg)
                | Instruction::PopJumpBackwardIfFalse(arg)
                | Instruction::PopJumpBackwardIfTrue(arg) => {
                    let arg = *arg as u32 | extended_arg;
                    relative_jump_indexes.insert(
                        Interval::new(
                            std::ops::Bound::Excluded(index as u32 - arg + 1),
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
                Instruction::ExtendedArg(_) | Instruction::ExtendedArgQuick(_) => {
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
                Instruction::ExtendedArg(arg) | Instruction::ExtendedArgQuick(arg) => {
                    let arg = *arg as u32 | extended_arg;
                    extended_arg = arg << 8;
                    continue;
                }
                Instruction::ForIter(arg)
                | Instruction::JumpForward(arg)
                | Instruction::JumpIfFalseOrPop(arg)
                | Instruction::JumpIfTrueOrPop(arg)
                | Instruction::PopJumpForwardIfFalse(arg)
                | Instruction::PopJumpForwardIfTrue(arg)
                | Instruction::Send(arg)
                | Instruction::PopJumpForwardIfNotNone(arg)
                | Instruction::PopJumpForwardIfNone(arg) => {
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
                | Instruction::JumpBackwardQuick(arg)
                | Instruction::PopJumpBackwardIfNotNone(arg)
                | Instruction::PopJumpBackwardIfNone(arg)
                | Instruction::PopJumpBackwardIfFalse(arg)
                | Instruction::PopJumpBackwardIfTrue(arg) => {
                    let interval = Interval::new(
                        std::ops::Bound::Excluded(index as u32 - (*arg as u32 | extended_arg) + 1),
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
                | ExtInstruction::JumpIfFalseOrPop(jump)
                | ExtInstruction::JumpIfTrueOrPop(jump)
                | ExtInstruction::PopJumpForwardIfFalse(jump)
                | ExtInstruction::PopJumpForwardIfTrue(jump)
                | ExtInstruction::Send(jump)
                | ExtInstruction::PopJumpForwardIfNotNone(jump)
                | ExtInstruction::PopJumpForwardIfNone(jump) => {
                    // Relative jumps only need to update if the index falls within it's jump range
                    if idx <= index && index + idx <= jump.index as usize {
                        jump.index -= 1
                    }
                }
                ExtInstruction::JumpBackwardNoInterrupt(jump)
                | ExtInstruction::JumpBackward(jump)
                | ExtInstruction::JumpBackwardQuick(jump)
                | ExtInstruction::PopJumpBackwardIfNotNone(jump)
                | ExtInstruction::PopJumpBackwardIfNone(jump)
                | ExtInstruction::PopJumpBackwardIfFalse(jump)
                | ExtInstruction::PopJumpBackwardIfTrue(jump) => {
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
                | ExtInstruction::JumpIfFalseOrPop(jump)
                | ExtInstruction::JumpIfTrueOrPop(jump)
                | ExtInstruction::PopJumpForwardIfFalse(jump)
                | ExtInstruction::PopJumpForwardIfTrue(jump)
                | ExtInstruction::Send(jump)
                | ExtInstruction::PopJumpForwardIfNotNone(jump)
                | ExtInstruction::PopJumpForwardIfNone(jump) => {
                    // Relative jumps only need to update if the index falls within it's jump range
                    if idx <= index && index + idx <= jump.index as usize {
                        jump.index += 1
                    }
                }
                ExtInstruction::JumpBackwardNoInterrupt(jump)
                | ExtInstruction::JumpBackward(jump)
                | ExtInstruction::JumpBackwardQuick(jump)
                | ExtInstruction::PopJumpBackwardIfNotNone(jump)
                | ExtInstruction::PopJumpBackwardIfNone(jump)
                | ExtInstruction::PopJumpBackwardIfFalse(jump)
                | ExtInstruction::PopJumpBackwardIfTrue(jump) => {
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
                | ExtInstruction::JumpIfFalseOrPop(jump)
                | ExtInstruction::JumpIfTrueOrPop(jump)
                | ExtInstruction::PopJumpForwardIfFalse(jump)
                | ExtInstruction::PopJumpForwardIfTrue(jump)
                | ExtInstruction::Send(jump)
                | ExtInstruction::PopJumpForwardIfNotNone(jump)
                | ExtInstruction::PopJumpForwardIfNone(jump)
                | ExtInstruction::JumpBackwardNoInterrupt(jump)
                | ExtInstruction::JumpBackward(jump)
                | ExtInstruction::JumpBackwardQuick(jump)
                | ExtInstruction::PopJumpBackwardIfNotNone(jump)
                | ExtInstruction::PopJumpBackwardIfNone(jump)
                | ExtInstruction::PopJumpBackwardIfFalse(jump)
                | ExtInstruction::PopJumpBackwardIfTrue(jump) => (*jump).into(),
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
            | ExtInstruction::JumpIfFalseOrPop(jump)
            | ExtInstruction::JumpIfTrueOrPop(jump)
            | ExtInstruction::PopJumpForwardIfFalse(jump)
            | ExtInstruction::PopJumpForwardIfTrue(jump)
            | ExtInstruction::Send(jump)
            | ExtInstruction::PopJumpForwardIfNotNone(jump)
            | ExtInstruction::PopJumpForwardIfNone(jump)
            | ExtInstruction::JumpBackwardNoInterrupt(jump)
            | ExtInstruction::JumpBackward(jump)
            | ExtInstruction::JumpBackwardQuick(jump)
            | ExtInstruction::PopJumpBackwardIfNotNone(jump)
            | ExtInstruction::PopJumpBackwardIfNone(jump)
            | ExtInstruction::PopJumpBackwardIfFalse(jump)
            | ExtInstruction::PopJumpBackwardIfTrue(jump) => match jump {
                RelativeJump {
                    index,
                    direction: JumpDirection::Forward,
                } => {
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
                    relative_jump_indexes.insert(
                        Interval::new(
                            std::ops::Bound::Excluded(idx as u32 - index + 1),
                            std::ops::Bound::Excluded(idx as u32),
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
                let arg = match instruction {
                    ExtInstruction::ForIter(jump)
                    | ExtInstruction::JumpForward(jump)
                    | ExtInstruction::JumpIfFalseOrPop(jump)
                    | ExtInstruction::JumpIfTrueOrPop(jump)
                    | ExtInstruction::PopJumpForwardIfFalse(jump)
                    | ExtInstruction::PopJumpForwardIfTrue(jump)
                    | ExtInstruction::Send(jump)
                    | ExtInstruction::PopJumpForwardIfNotNone(jump)
                    | ExtInstruction::PopJumpForwardIfNone(jump)
                    | ExtInstruction::JumpBackwardNoInterrupt(jump)
                    | ExtInstruction::JumpBackward(jump)
                    | ExtInstruction::JumpBackwardQuick(jump)
                    | ExtInstruction::PopJumpBackwardIfNotNone(jump)
                    | ExtInstruction::PopJumpBackwardIfNone(jump)
                    | ExtInstruction::PopJumpBackwardIfFalse(jump)
                    | ExtInstruction::PopJumpBackwardIfTrue(jump) => {
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
                                std::ops::Bound::Excluded(index as u32 - jump.index + 1),
                                std::ops::Bound::Excluded(index as u32),
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
                    *entry.value() += extended_arg_count
                }
            }
        }

        let mut instructions: Instructions = Instructions::with_capacity(self.0.len() * 2); // This will not be enough this as we dynamically generate EXTENDED_ARGS, but it's better than not reserving any length.

        for (index, instruction) in self.0.iter().enumerate() {
            let arg = match instruction {
                ExtInstruction::ForIter(jump)
                | ExtInstruction::JumpForward(jump)
                | ExtInstruction::JumpIfFalseOrPop(jump)
                | ExtInstruction::JumpIfTrueOrPop(jump)
                | ExtInstruction::PopJumpForwardIfFalse(jump)
                | ExtInstruction::PopJumpForwardIfTrue(jump)
                | ExtInstruction::Send(jump)
                | ExtInstruction::PopJumpForwardIfNotNone(jump)
                | ExtInstruction::PopJumpForwardIfNone(jump)
                | ExtInstruction::JumpBackwardNoInterrupt(jump)
                | ExtInstruction::JumpBackward(jump)
                | ExtInstruction::JumpBackwardQuick(jump)
                | ExtInstruction::PopJumpBackwardIfNotNone(jump)
                | ExtInstruction::PopJumpBackwardIfNone(jump)
                | ExtInstruction::PopJumpBackwardIfFalse(jump)
                | ExtInstruction::PopJumpBackwardIfTrue(jump) => {
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
                            std::ops::Bound::Excluded(index as u32 - jump.index + 1),
                            std::ops::Bound::Excluded(index as u32),
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
            Opcode::NOP => ExtInstruction::Nop(value.1.into()),
            Opcode::UNARY_POSITIVE => ExtInstruction::UnaryPositive(value.1.into()),
            Opcode::UNARY_NEGATIVE => ExtInstruction::UnaryNegative(value.1.into()),
            Opcode::UNARY_NOT => ExtInstruction::UnaryNot(value.1.into()),
            Opcode::UNARY_INVERT => ExtInstruction::UnaryInvert(value.1.into()),
            Opcode::BINARY_SUBSCR => ExtInstruction::BinarySubscr(value.1.into()),
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
            Opcode::STORE_SUBSCR => ExtInstruction::StoreSubscr(value.1.into()),
            Opcode::DELETE_SUBSCR => ExtInstruction::DeleteSubscr(value.1.into()),
            Opcode::GET_ITER => ExtInstruction::GetIter(value.1.into()),
            Opcode::GET_YIELD_FROM_ITER => ExtInstruction::GetYieldFromIter(value.1.into()),
            Opcode::PRINT_EXPR => ExtInstruction::PrintExpr(value.1.into()),
            Opcode::LOAD_BUILD_CLASS => ExtInstruction::LoadBuildClass(value.1.into()),
            Opcode::LOAD_ASSERTION_ERROR => ExtInstruction::LoadAssertionError(value.1.into()),
            Opcode::RETURN_GENERATOR => ExtInstruction::ReturnGenerator(value.1.into()),
            Opcode::LIST_TO_TUPLE => ExtInstruction::ListToTuple(value.1.into()),
            Opcode::RETURN_VALUE => ExtInstruction::ReturnValue(value.1.into()),
            Opcode::IMPORT_STAR => ExtInstruction::ImportStar(value.1.into()),
            Opcode::SETUP_ANNOTATIONS => ExtInstruction::SetupAnnotations(value.1.into()),
            Opcode::YIELD_VALUE => ExtInstruction::YieldValue(value.1.into()),
            Opcode::ASYNC_GEN_WRAP => ExtInstruction::AsyncGenWrap(value.1.into()),
            Opcode::PREP_RERAISE_STAR => ExtInstruction::PrepReraiseStar(value.1.into()),
            Opcode::POP_EXCEPT => ExtInstruction::PopExcept(value.1.into()),
            Opcode::STORE_NAME => ExtInstruction::StoreName(NameIndex { index: value.1 }),
            Opcode::DELETE_NAME => ExtInstruction::DeleteName(NameIndex { index: value.1 }),
            Opcode::UNPACK_SEQUENCE => ExtInstruction::UnpackSequence(value.1),
            Opcode::FOR_ITER => ExtInstruction::ForIter(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::UNPACK_EX => ExtInstruction::UnpackEx(value.1),
            Opcode::STORE_ATTR => ExtInstruction::StoreAttr(NameIndex { index: value.1 }),
            Opcode::DELETE_ATTR => ExtInstruction::DeleteAttr(NameIndex { index: value.1 }),
            Opcode::STORE_GLOBAL => ExtInstruction::StoreGlobal(NameIndex { index: value.1 }),
            Opcode::DELETE_GLOBAL => ExtInstruction::DeleteGlobal(NameIndex { index: value.1 }),
            Opcode::SWAP => ExtInstruction::Swap(value.1),
            Opcode::LOAD_CONST => ExtInstruction::LoadConst(ConstIndex { index: value.1 }),
            Opcode::LOAD_NAME => ExtInstruction::LoadName(NameIndex { index: value.1 }),
            Opcode::BUILD_TUPLE => ExtInstruction::BuildTuple(value.1),
            Opcode::BUILD_LIST => ExtInstruction::BuildList(value.1),
            Opcode::BUILD_SET => ExtInstruction::BuildSet(value.1),
            Opcode::BUILD_MAP => ExtInstruction::BuildMap(value.1),
            Opcode::LOAD_ATTR => ExtInstruction::LoadAttr(NameIndex { index: value.1 }),
            Opcode::COMPARE_OP => ExtInstruction::CompareOp(value.1.into()),
            Opcode::IMPORT_NAME => ExtInstruction::ImportName(NameIndex { index: value.1 }),
            Opcode::IMPORT_FROM => ExtInstruction::ImportFrom(NameIndex { index: value.1 }),
            Opcode::JUMP_FORWARD => ExtInstruction::JumpForward(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::JUMP_IF_FALSE_OR_POP => ExtInstruction::JumpIfFalseOrPop(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::JUMP_IF_TRUE_OR_POP => ExtInstruction::JumpIfTrueOrPop(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::POP_JUMP_FORWARD_IF_FALSE => {
                ExtInstruction::PopJumpForwardIfFalse(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::POP_JUMP_FORWARD_IF_TRUE => {
                ExtInstruction::PopJumpForwardIfTrue(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::LOAD_GLOBAL => ExtInstruction::LoadGlobal(GlobalNameIndex { index: value.1 }),
            Opcode::IS_OP => ExtInstruction::IsOp(value.1.into()),
            Opcode::CONTAINS_OP => ExtInstruction::ContainsOp(value.1.into()),
            Opcode::RERAISE => ExtInstruction::Reraise(value.1.into()),
            Opcode::COPY => ExtInstruction::Copy(value.1),
            Opcode::BINARY_OP => ExtInstruction::BinaryOp(value.1.into()),
            Opcode::SEND => ExtInstruction::Send(RelativeJump {
                index: value.1,
                direction: JumpDirection::Forward,
            }),
            Opcode::LOAD_FAST => ExtInstruction::LoadFast(VarNameIndex { index: value.1 }),
            Opcode::STORE_FAST => ExtInstruction::StoreFast(VarNameIndex { index: value.1 }),
            Opcode::DELETE_FAST => ExtInstruction::DeleteFast(VarNameIndex { index: value.1 }),
            Opcode::POP_JUMP_FORWARD_IF_NOT_NONE => {
                ExtInstruction::PopJumpForwardIfNotNone(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::POP_JUMP_FORWARD_IF_NONE => {
                ExtInstruction::PopJumpForwardIfNone(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Forward,
                })
            }
            Opcode::RAISE_VARARGS => ExtInstruction::RaiseVarargs(value.1.into()),
            Opcode::GET_AWAITABLE => ExtInstruction::GetAwaitable(value.1.into()),
            Opcode::MAKE_FUNCTION => {
                ExtInstruction::MakeFunction(MakeFunctionFlags::from_bits_retain(value.1))
            }
            Opcode::BUILD_SLICE => ExtInstruction::BuildSlice(value.1.into()),
            Opcode::JUMP_BACKWARD_NO_INTERRUPT => {
                ExtInstruction::JumpBackwardNoInterrupt(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Backward,
                })
            }
            Opcode::MAKE_CELL => ExtInstruction::MakeCell(ClosureRefIndex { index: value.1 }),
            Opcode::LOAD_CLOSURE => ExtInstruction::LoadClosure(ClosureRefIndex { index: value.1 }),
            Opcode::LOAD_DEREF => ExtInstruction::LoadDeref(ClosureRefIndex { index: value.1 }),
            Opcode::STORE_DEREF => ExtInstruction::StoreDeref(ClosureRefIndex { index: value.1 }),
            Opcode::DELETE_DEREF => ExtInstruction::DeleteDeref(ClosureRefIndex { index: value.1 }),
            Opcode::JUMP_BACKWARD => ExtInstruction::JumpBackward(RelativeJump {
                index: value.1,
                direction: JumpDirection::Backward,
            }),
            Opcode::CALL_FUNCTION_EX => ExtInstruction::CallFunctionEx(value.1.into()),
            Opcode::EXTENDED_ARG => return Err(Error::InvalidConversion),
            Opcode::LIST_APPEND => ExtInstruction::ListAppend(value.1),
            Opcode::SET_ADD => ExtInstruction::SetAdd(value.1),
            Opcode::MAP_ADD => ExtInstruction::MapAdd(value.1),
            Opcode::LOAD_CLASSDEREF => {
                ExtInstruction::LoadClassderef(ClosureRefIndex { index: value.1 })
            }
            Opcode::COPY_FREE_VARS => ExtInstruction::CopyFreeVars(value.1),
            Opcode::RESUME => ExtInstruction::Resume(value.1.into()),
            Opcode::MATCH_CLASS => ExtInstruction::MatchClass(value.1),
            Opcode::FORMAT_VALUE => ExtInstruction::FormatValue(value.1.into()),
            Opcode::BUILD_CONST_KEY_MAP => ExtInstruction::BuildConstKeyMap(value.1),
            Opcode::BUILD_STRING => ExtInstruction::BuildString(value.1),
            Opcode::LOAD_METHOD => ExtInstruction::LoadMethod(NameIndex { index: value.1 }),
            Opcode::LIST_EXTEND => ExtInstruction::ListExtend(value.1),
            Opcode::SET_UPDATE => ExtInstruction::SetUpdate(value.1),
            Opcode::DICT_MERGE => ExtInstruction::DictMerge(value.1),
            Opcode::DICT_UPDATE => ExtInstruction::DictUpdate(value.1),
            Opcode::PRECALL => ExtInstruction::Precall(value.1),
            Opcode::CALL => ExtInstruction::Call(value.1),
            Opcode::KW_NAMES => ExtInstruction::KwNames(ConstIndex { index: value.1 }),
            Opcode::POP_JUMP_BACKWARD_IF_NOT_NONE => {
                ExtInstruction::PopJumpBackwardIfNotNone(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Backward,
                })
            }
            Opcode::POP_JUMP_BACKWARD_IF_NONE => {
                ExtInstruction::PopJumpBackwardIfNone(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Backward,
                })
            }
            Opcode::POP_JUMP_BACKWARD_IF_FALSE => {
                ExtInstruction::PopJumpBackwardIfFalse(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Backward,
                })
            }
            Opcode::POP_JUMP_BACKWARD_IF_TRUE => {
                ExtInstruction::PopJumpBackwardIfTrue(RelativeJump {
                    index: value.1,
                    direction: JumpDirection::Backward,
                })
            }
            Opcode::BINARY_OP_ADAPTIVE => ExtInstruction::BinaryOpAdaptive(value.1),
            Opcode::BINARY_OP_ADD_FLOAT => ExtInstruction::BinaryOpAddFloat(value.1),
            Opcode::BINARY_OP_ADD_INT => ExtInstruction::BinaryOpAddInt(value.1),
            Opcode::BINARY_OP_ADD_UNICODE => ExtInstruction::BinaryOpAddUnicode(value.1),
            Opcode::BINARY_OP_INPLACE_ADD_UNICODE => {
                ExtInstruction::BinaryOpInplaceAddUnicode(value.1)
            }
            Opcode::BINARY_OP_MULTIPLY_FLOAT => ExtInstruction::BinaryOpMultiplyFloat(value.1),
            Opcode::BINARY_OP_MULTIPLY_INT => ExtInstruction::BinaryOpMultiplyInt(value.1),
            Opcode::BINARY_OP_SUBTRACT_FLOAT => ExtInstruction::BinaryOpSubtractFloat(value.1),
            Opcode::BINARY_OP_SUBTRACT_INT => ExtInstruction::BinaryOpSubtractInt(value.1),
            Opcode::BINARY_SUBSCR_ADAPTIVE => ExtInstruction::BinarySubscrAdaptive(value.1),
            Opcode::BINARY_SUBSCR_DICT => ExtInstruction::BinarySubscrDict(value.1),
            Opcode::BINARY_SUBSCR_GETITEM => ExtInstruction::BinarySubscrGetitem(value.1),
            Opcode::BINARY_SUBSCR_LIST_INT => ExtInstruction::BinarySubscrListInt(value.1),
            Opcode::BINARY_SUBSCR_TUPLE_INT => ExtInstruction::BinarySubscrTupleInt(value.1),
            Opcode::CALL_ADAPTIVE => ExtInstruction::CallAdaptive(value.1),
            Opcode::CALL_PY_EXACT_ARGS => ExtInstruction::CallPyExactArgs(value.1),
            Opcode::CALL_PY_WITH_DEFAULTS => ExtInstruction::CallPyWithDefaults(value.1),
            Opcode::COMPARE_OP_ADAPTIVE => ExtInstruction::CompareOpAdaptive(value.1),
            Opcode::COMPARE_OP_FLOAT_JUMP => ExtInstruction::CompareOpFloatJump(value.1),
            Opcode::COMPARE_OP_INT_JUMP => ExtInstruction::CompareOpIntJump(value.1),
            Opcode::COMPARE_OP_STR_JUMP => ExtInstruction::CompareOpStrJump(value.1),
            Opcode::EXTENDED_ARG_QUICK => return Err(Error::InvalidConversion),
            Opcode::JUMP_BACKWARD_QUICK => ExtInstruction::JumpBackwardQuick(RelativeJump {
                index: value.1,
                direction: JumpDirection::Backward,
            }),
            Opcode::LOAD_ATTR_ADAPTIVE => ExtInstruction::LoadAttrAdaptive(value.1),
            Opcode::LOAD_ATTR_INSTANCE_VALUE => ExtInstruction::LoadAttrInstanceValue(value.1),
            Opcode::LOAD_ATTR_MODULE => ExtInstruction::LoadAttrModule(value.1),
            Opcode::LOAD_ATTR_SLOT => ExtInstruction::LoadAttrSlot(value.1),
            Opcode::LOAD_ATTR_WITH_HINT => ExtInstruction::LoadAttrWithHint(value.1),
            Opcode::LOAD_CONST__LOAD_FAST => ExtInstruction::LoadConstLoadFast(value.1),
            Opcode::LOAD_FAST__LOAD_CONST => ExtInstruction::LoadFastLoadConst(value.1),
            Opcode::LOAD_FAST__LOAD_FAST => ExtInstruction::LoadFastLoadFast(value.1),
            Opcode::LOAD_GLOBAL_ADAPTIVE => ExtInstruction::LoadGlobalAdaptive(value.1),
            Opcode::LOAD_GLOBAL_BUILTIN => ExtInstruction::LoadGlobalBuiltin(value.1),
            Opcode::LOAD_GLOBAL_MODULE => ExtInstruction::LoadGlobalModule(value.1),
            Opcode::LOAD_METHOD_ADAPTIVE => ExtInstruction::LoadMethodAdaptive(value.1),
            Opcode::LOAD_METHOD_CLASS => ExtInstruction::LoadMethodClass(value.1),
            Opcode::LOAD_METHOD_MODULE => ExtInstruction::LoadMethodModule(value.1),
            Opcode::LOAD_METHOD_NO_DICT => ExtInstruction::LoadMethodNoDict(value.1),
            Opcode::LOAD_METHOD_WITH_DICT => ExtInstruction::LoadMethodWithDict(value.1),
            Opcode::LOAD_METHOD_WITH_VALUES => ExtInstruction::LoadMethodWithValues(value.1),
            Opcode::PRECALL_ADAPTIVE => ExtInstruction::PrecallAdaptive(value.1),
            Opcode::PRECALL_BOUND_METHOD => ExtInstruction::PrecallBoundMethod(value.1),
            Opcode::PRECALL_BUILTIN_CLASS => ExtInstruction::PrecallBuiltinClass(value.1),
            Opcode::PRECALL_BUILTIN_FAST_WITH_KEYWORDS => {
                ExtInstruction::PrecallBuiltinFastWithKeywords(value.1)
            }
            Opcode::PRECALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS => {
                ExtInstruction::PrecallMethodDescriptorFastWithKeywords(value.1)
            }
            Opcode::PRECALL_NO_KW_BUILTIN_FAST => ExtInstruction::PrecallNoKwBuiltinFast(value.1),
            Opcode::PRECALL_NO_KW_BUILTIN_O => ExtInstruction::PrecallNoKwBuiltinO(value.1),
            Opcode::PRECALL_NO_KW_ISINSTANCE => ExtInstruction::PrecallNoKwIsinstance(value.1),
            Opcode::PRECALL_NO_KW_LEN => ExtInstruction::PrecallNoKwLen(value.1),
            Opcode::PRECALL_NO_KW_LIST_APPEND => ExtInstruction::PrecallNoKwListAppend(value.1),
            Opcode::PRECALL_NO_KW_METHOD_DESCRIPTOR_FAST => {
                ExtInstruction::PrecallNoKwMethodDescriptorFast(value.1)
            }
            Opcode::PRECALL_NO_KW_METHOD_DESCRIPTOR_NOARGS => {
                ExtInstruction::PrecallNoKwMethodDescriptorNoargs(value.1)
            }
            Opcode::PRECALL_NO_KW_METHOD_DESCRIPTOR_O => {
                ExtInstruction::PrecallNoKwMethodDescriptorO(value.1)
            }
            Opcode::PRECALL_NO_KW_STR_1 => ExtInstruction::PrecallNoKwStr1(value.1),
            Opcode::PRECALL_NO_KW_TUPLE_1 => ExtInstruction::PrecallNoKwTuple1(value.1),
            Opcode::PRECALL_NO_KW_TYPE_1 => ExtInstruction::PrecallNoKwType1(value.1),
            Opcode::PRECALL_PYFUNC => ExtInstruction::PrecallPyfunc(value.1),
            Opcode::RESUME_QUICK => ExtInstruction::ResumeQuick(value.1.into()),
            Opcode::STORE_ATTR_ADAPTIVE => ExtInstruction::StoreAttrAdaptive(value.1),
            Opcode::STORE_ATTR_INSTANCE_VALUE => ExtInstruction::StoreAttrInstanceValue(value.1),
            Opcode::STORE_ATTR_SLOT => ExtInstruction::StoreAttrSlot(value.1),
            Opcode::STORE_ATTR_WITH_HINT => ExtInstruction::StoreAttrWithHint(value.1),
            Opcode::STORE_FAST__LOAD_FAST => ExtInstruction::StoreFastLoadFast(value.1),
            Opcode::STORE_FAST__STORE_FAST => ExtInstruction::StoreFastStoreFast(value.1),
            Opcode::STORE_SUBSCR_ADAPTIVE => ExtInstruction::StoreSubscrAdaptive(value.1),
            Opcode::STORE_SUBSCR_DICT => ExtInstruction::StoreSubscrDict(value.1),
            Opcode::STORE_SUBSCR_LIST_INT => ExtInstruction::StoreSubscrListInt(value.1),
            Opcode::UNPACK_SEQUENCE_ADAPTIVE => ExtInstruction::UnpackSequenceAdaptive(value.1),
            Opcode::UNPACK_SEQUENCE_LIST => ExtInstruction::UnpackSequenceList(value.1),
            Opcode::UNPACK_SEQUENCE_TUPLE => ExtInstruction::UnpackSequenceTuple(value.1),
            Opcode::UNPACK_SEQUENCE_TWO_TUPLE => ExtInstruction::UnpackSequenceTwoTuple(value.1),
            Opcode::DO_TRACING => ExtInstruction::DoTracing(value.1),
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
            ExtInstruction::Nop(_) => Opcode::NOP,
            ExtInstruction::UnaryPositive(_) => Opcode::UNARY_POSITIVE,
            ExtInstruction::UnaryNegative(_) => Opcode::UNARY_NEGATIVE,
            ExtInstruction::UnaryNot(_) => Opcode::UNARY_NOT,
            ExtInstruction::UnaryInvert(_) => Opcode::UNARY_INVERT,
            ExtInstruction::BinarySubscr(_) => Opcode::BINARY_SUBSCR,
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
            ExtInstruction::StoreSubscr(_) => Opcode::STORE_SUBSCR,
            ExtInstruction::DeleteSubscr(_) => Opcode::DELETE_SUBSCR,
            ExtInstruction::GetIter(_) => Opcode::GET_ITER,
            ExtInstruction::GetYieldFromIter(_) => Opcode::GET_YIELD_FROM_ITER,
            ExtInstruction::PrintExpr(_) => Opcode::PRINT_EXPR,
            ExtInstruction::LoadBuildClass(_) => Opcode::LOAD_BUILD_CLASS,
            ExtInstruction::LoadAssertionError(_) => Opcode::LOAD_ASSERTION_ERROR,
            ExtInstruction::ReturnGenerator(_) => Opcode::RETURN_GENERATOR,
            ExtInstruction::ListToTuple(_) => Opcode::LIST_TO_TUPLE,
            ExtInstruction::ReturnValue(_) => Opcode::RETURN_VALUE,
            ExtInstruction::ImportStar(_) => Opcode::IMPORT_STAR,
            ExtInstruction::SetupAnnotations(_) => Opcode::SETUP_ANNOTATIONS,
            ExtInstruction::YieldValue(_) => Opcode::YIELD_VALUE,
            ExtInstruction::AsyncGenWrap(_) => Opcode::ASYNC_GEN_WRAP,
            ExtInstruction::PrepReraiseStar(_) => Opcode::PREP_RERAISE_STAR,
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
            ExtInstruction::JumpIfFalseOrPop(_) => Opcode::JUMP_IF_FALSE_OR_POP,
            ExtInstruction::JumpIfTrueOrPop(_) => Opcode::JUMP_IF_TRUE_OR_POP,
            ExtInstruction::PopJumpForwardIfFalse(_) => Opcode::POP_JUMP_FORWARD_IF_FALSE,
            ExtInstruction::PopJumpForwardIfTrue(_) => Opcode::POP_JUMP_FORWARD_IF_TRUE,
            ExtInstruction::LoadGlobal(_) => Opcode::LOAD_GLOBAL,
            ExtInstruction::IsOp(_) => Opcode::IS_OP,
            ExtInstruction::ContainsOp(_) => Opcode::CONTAINS_OP,
            ExtInstruction::Reraise(_) => Opcode::RERAISE,
            ExtInstruction::Copy(_) => Opcode::COPY,
            ExtInstruction::BinaryOp(_) => Opcode::BINARY_OP,
            ExtInstruction::Send(_) => Opcode::SEND,
            ExtInstruction::LoadFast(_) => Opcode::LOAD_FAST,
            ExtInstruction::StoreFast(_) => Opcode::STORE_FAST,
            ExtInstruction::DeleteFast(_) => Opcode::DELETE_FAST,
            ExtInstruction::PopJumpForwardIfNotNone(_) => Opcode::POP_JUMP_FORWARD_IF_NOT_NONE,
            ExtInstruction::PopJumpForwardIfNone(_) => Opcode::POP_JUMP_FORWARD_IF_NONE,
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
            ExtInstruction::CallFunctionEx(_) => Opcode::CALL_FUNCTION_EX,
            ExtInstruction::ListAppend(_) => Opcode::LIST_APPEND,
            ExtInstruction::SetAdd(_) => Opcode::SET_ADD,
            ExtInstruction::MapAdd(_) => Opcode::MAP_ADD,
            ExtInstruction::LoadClassderef(_) => Opcode::LOAD_CLASSDEREF,
            ExtInstruction::CopyFreeVars(_) => Opcode::COPY_FREE_VARS,
            ExtInstruction::Resume(_) => Opcode::RESUME,
            ExtInstruction::MatchClass(_) => Opcode::MATCH_CLASS,
            ExtInstruction::FormatValue(_) => Opcode::FORMAT_VALUE,
            ExtInstruction::BuildConstKeyMap(_) => Opcode::BUILD_CONST_KEY_MAP,
            ExtInstruction::BuildString(_) => Opcode::BUILD_STRING,
            ExtInstruction::LoadMethod(_) => Opcode::LOAD_METHOD,
            ExtInstruction::ListExtend(_) => Opcode::LIST_EXTEND,
            ExtInstruction::SetUpdate(_) => Opcode::SET_UPDATE,
            ExtInstruction::DictMerge(_) => Opcode::DICT_MERGE,
            ExtInstruction::DictUpdate(_) => Opcode::DICT_UPDATE,
            ExtInstruction::Precall(_) => Opcode::PRECALL,
            ExtInstruction::Call(_) => Opcode::CALL,
            ExtInstruction::KwNames(_) => Opcode::KW_NAMES,
            ExtInstruction::PopJumpBackwardIfNotNone(_) => Opcode::POP_JUMP_BACKWARD_IF_NOT_NONE,
            ExtInstruction::PopJumpBackwardIfNone(_) => Opcode::POP_JUMP_BACKWARD_IF_NONE,
            ExtInstruction::PopJumpBackwardIfFalse(_) => Opcode::POP_JUMP_BACKWARD_IF_FALSE,
            ExtInstruction::PopJumpBackwardIfTrue(_) => Opcode::POP_JUMP_BACKWARD_IF_TRUE,
            ExtInstruction::BinaryOpAdaptive(_) => Opcode::BINARY_OP_ADAPTIVE,
            ExtInstruction::BinaryOpAddFloat(_) => Opcode::BINARY_OP_ADD_FLOAT,
            ExtInstruction::BinaryOpAddInt(_) => Opcode::BINARY_OP_ADD_INT,
            ExtInstruction::BinaryOpAddUnicode(_) => Opcode::BINARY_OP_ADD_UNICODE,
            ExtInstruction::BinaryOpInplaceAddUnicode(_) => Opcode::BINARY_OP_INPLACE_ADD_UNICODE,
            ExtInstruction::BinaryOpMultiplyFloat(_) => Opcode::BINARY_OP_MULTIPLY_FLOAT,
            ExtInstruction::BinaryOpMultiplyInt(_) => Opcode::BINARY_OP_MULTIPLY_INT,
            ExtInstruction::BinaryOpSubtractFloat(_) => Opcode::BINARY_OP_SUBTRACT_FLOAT,
            ExtInstruction::BinaryOpSubtractInt(_) => Opcode::BINARY_OP_SUBTRACT_INT,
            ExtInstruction::BinarySubscrAdaptive(_) => Opcode::BINARY_SUBSCR_ADAPTIVE,
            ExtInstruction::BinarySubscrDict(_) => Opcode::BINARY_SUBSCR_DICT,
            ExtInstruction::BinarySubscrGetitem(_) => Opcode::BINARY_SUBSCR_GETITEM,
            ExtInstruction::BinarySubscrListInt(_) => Opcode::BINARY_SUBSCR_LIST_INT,
            ExtInstruction::BinarySubscrTupleInt(_) => Opcode::BINARY_SUBSCR_TUPLE_INT,
            ExtInstruction::CallAdaptive(_) => Opcode::CALL_ADAPTIVE,
            ExtInstruction::CallPyExactArgs(_) => Opcode::CALL_PY_EXACT_ARGS,
            ExtInstruction::CallPyWithDefaults(_) => Opcode::CALL_PY_WITH_DEFAULTS,
            ExtInstruction::CompareOpAdaptive(_) => Opcode::COMPARE_OP_ADAPTIVE,
            ExtInstruction::CompareOpFloatJump(_) => Opcode::COMPARE_OP_FLOAT_JUMP,
            ExtInstruction::CompareOpIntJump(_) => Opcode::COMPARE_OP_INT_JUMP,
            ExtInstruction::CompareOpStrJump(_) => Opcode::COMPARE_OP_STR_JUMP,
            ExtInstruction::JumpBackwardQuick(_) => Opcode::JUMP_BACKWARD_QUICK,
            ExtInstruction::LoadAttrAdaptive(_) => Opcode::LOAD_ATTR_ADAPTIVE,
            ExtInstruction::LoadAttrInstanceValue(_) => Opcode::LOAD_ATTR_INSTANCE_VALUE,
            ExtInstruction::LoadAttrModule(_) => Opcode::LOAD_ATTR_MODULE,
            ExtInstruction::LoadAttrSlot(_) => Opcode::LOAD_ATTR_SLOT,
            ExtInstruction::LoadAttrWithHint(_) => Opcode::LOAD_ATTR_WITH_HINT,
            ExtInstruction::LoadConstLoadFast(_) => Opcode::LOAD_CONST__LOAD_FAST,
            ExtInstruction::LoadFastLoadConst(_) => Opcode::LOAD_FAST__LOAD_CONST,
            ExtInstruction::LoadFastLoadFast(_) => Opcode::LOAD_FAST__LOAD_FAST,
            ExtInstruction::LoadGlobalAdaptive(_) => Opcode::LOAD_GLOBAL_ADAPTIVE,
            ExtInstruction::LoadGlobalBuiltin(_) => Opcode::LOAD_GLOBAL_BUILTIN,
            ExtInstruction::LoadGlobalModule(_) => Opcode::LOAD_GLOBAL_MODULE,
            ExtInstruction::LoadMethodAdaptive(_) => Opcode::LOAD_METHOD_ADAPTIVE,
            ExtInstruction::LoadMethodClass(_) => Opcode::LOAD_METHOD_CLASS,
            ExtInstruction::LoadMethodModule(_) => Opcode::LOAD_METHOD_MODULE,
            ExtInstruction::LoadMethodNoDict(_) => Opcode::LOAD_METHOD_NO_DICT,
            ExtInstruction::LoadMethodWithDict(_) => Opcode::LOAD_METHOD_WITH_DICT,
            ExtInstruction::LoadMethodWithValues(_) => Opcode::LOAD_METHOD_WITH_VALUES,
            ExtInstruction::PrecallAdaptive(_) => Opcode::PRECALL_ADAPTIVE,
            ExtInstruction::PrecallBoundMethod(_) => Opcode::PRECALL_BOUND_METHOD,
            ExtInstruction::PrecallBuiltinClass(_) => Opcode::PRECALL_BUILTIN_CLASS,
            ExtInstruction::PrecallBuiltinFastWithKeywords(_) => {
                Opcode::PRECALL_BUILTIN_FAST_WITH_KEYWORDS
            }
            ExtInstruction::PrecallMethodDescriptorFastWithKeywords(_) => {
                Opcode::PRECALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS
            }
            ExtInstruction::PrecallNoKwBuiltinFast(_) => Opcode::PRECALL_NO_KW_BUILTIN_FAST,
            ExtInstruction::PrecallNoKwBuiltinO(_) => Opcode::PRECALL_NO_KW_BUILTIN_O,
            ExtInstruction::PrecallNoKwIsinstance(_) => Opcode::PRECALL_NO_KW_ISINSTANCE,
            ExtInstruction::PrecallNoKwLen(_) => Opcode::PRECALL_NO_KW_LEN,
            ExtInstruction::PrecallNoKwListAppend(_) => Opcode::PRECALL_NO_KW_LIST_APPEND,
            ExtInstruction::PrecallNoKwMethodDescriptorFast(_) => {
                Opcode::PRECALL_NO_KW_METHOD_DESCRIPTOR_FAST
            }
            ExtInstruction::PrecallNoKwMethodDescriptorNoargs(_) => {
                Opcode::PRECALL_NO_KW_METHOD_DESCRIPTOR_NOARGS
            }
            ExtInstruction::PrecallNoKwMethodDescriptorO(_) => {
                Opcode::PRECALL_NO_KW_METHOD_DESCRIPTOR_O
            }
            ExtInstruction::PrecallNoKwStr1(_) => Opcode::PRECALL_NO_KW_STR_1,
            ExtInstruction::PrecallNoKwTuple1(_) => Opcode::PRECALL_NO_KW_TUPLE_1,
            ExtInstruction::PrecallNoKwType1(_) => Opcode::PRECALL_NO_KW_TYPE_1,
            ExtInstruction::PrecallPyfunc(_) => Opcode::PRECALL_PYFUNC,
            ExtInstruction::ResumeQuick(_) => Opcode::RESUME_QUICK,
            ExtInstruction::StoreAttrAdaptive(_) => Opcode::STORE_ATTR_ADAPTIVE,
            ExtInstruction::StoreAttrInstanceValue(_) => Opcode::STORE_ATTR_INSTANCE_VALUE,
            ExtInstruction::StoreAttrSlot(_) => Opcode::STORE_ATTR_SLOT,
            ExtInstruction::StoreAttrWithHint(_) => Opcode::STORE_ATTR_WITH_HINT,
            ExtInstruction::StoreFastLoadFast(_) => Opcode::STORE_FAST__LOAD_FAST,
            ExtInstruction::StoreFastStoreFast(_) => Opcode::STORE_FAST__STORE_FAST,
            ExtInstruction::StoreSubscrAdaptive(_) => Opcode::STORE_SUBSCR_ADAPTIVE,
            ExtInstruction::StoreSubscrDict(_) => Opcode::STORE_SUBSCR_DICT,
            ExtInstruction::StoreSubscrListInt(_) => Opcode::STORE_SUBSCR_LIST_INT,
            ExtInstruction::UnpackSequenceAdaptive(_) => Opcode::UNPACK_SEQUENCE_ADAPTIVE,
            ExtInstruction::UnpackSequenceList(_) => Opcode::UNPACK_SEQUENCE_LIST,
            ExtInstruction::UnpackSequenceTuple(_) => Opcode::UNPACK_SEQUENCE_TUPLE,
            ExtInstruction::UnpackSequenceTwoTuple(_) => Opcode::UNPACK_SEQUENCE_TWO_TUPLE,
            ExtInstruction::DoTracing(_) => Opcode::DO_TRACING,
            ExtInstruction::InvalidOpcode((opcode, _)) => Opcode::INVALID_OPCODE(*opcode),
        }
    }

    fn get_raw_value(&self) -> Self::Arg {
        match &self {
            ExtInstruction::Cache(n)
            | ExtInstruction::PopTop(n)
            | ExtInstruction::PushNull(n)
            | ExtInstruction::Nop(n)
            | ExtInstruction::UnaryPositive(n)
            | ExtInstruction::UnaryNegative(n)
            | ExtInstruction::UnaryNot(n)
            | ExtInstruction::UnaryInvert(n)
            | ExtInstruction::BinarySubscr(n)
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
            | ExtInstruction::StoreSubscr(n)
            | ExtInstruction::DeleteSubscr(n)
            | ExtInstruction::GetIter(n)
            | ExtInstruction::GetYieldFromIter(n)
            | ExtInstruction::PrintExpr(n)
            | ExtInstruction::LoadBuildClass(n)
            | ExtInstruction::LoadAssertionError(n)
            | ExtInstruction::ReturnGenerator(n)
            | ExtInstruction::ListToTuple(n)
            | ExtInstruction::ReturnValue(n)
            | ExtInstruction::ImportStar(n)
            | ExtInstruction::SetupAnnotations(n)
            | ExtInstruction::YieldValue(n)
            | ExtInstruction::AsyncGenWrap(n)
            | ExtInstruction::PrepReraiseStar(n)
            | ExtInstruction::PopExcept(n) => n.0,
            ExtInstruction::StoreName(name_index)
            | ExtInstruction::DeleteName(name_index)
            | ExtInstruction::LoadName(name_index)
            | ExtInstruction::ImportName(name_index)
            | ExtInstruction::ImportFrom(name_index)
            | ExtInstruction::StoreAttr(name_index)
            | ExtInstruction::DeleteAttr(name_index)
            | ExtInstruction::StoreGlobal(name_index)
            | ExtInstruction::DeleteGlobal(name_index)
            | ExtInstruction::LoadAttr(name_index)
            | ExtInstruction::LoadMethod(name_index) => name_index.index,
            ExtInstruction::LoadGlobal(global_name_index) => global_name_index.index,
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
            | ExtInstruction::MatchClass(n)
            | ExtInstruction::BuildConstKeyMap(n)
            | ExtInstruction::BuildString(n)
            | ExtInstruction::ListExtend(n)
            | ExtInstruction::SetUpdate(n)
            | ExtInstruction::DictMerge(n)
            | ExtInstruction::DictUpdate(n)
            | ExtInstruction::Precall(n)
            | ExtInstruction::Call(n)
            | ExtInstruction::BinaryOpAdaptive(n)
            | ExtInstruction::BinaryOpAddFloat(n)
            | ExtInstruction::BinaryOpAddInt(n)
            | ExtInstruction::BinaryOpAddUnicode(n)
            | ExtInstruction::BinaryOpInplaceAddUnicode(n)
            | ExtInstruction::BinaryOpMultiplyFloat(n)
            | ExtInstruction::BinaryOpMultiplyInt(n)
            | ExtInstruction::BinaryOpSubtractFloat(n)
            | ExtInstruction::BinaryOpSubtractInt(n)
            | ExtInstruction::BinarySubscrAdaptive(n)
            | ExtInstruction::BinarySubscrDict(n)
            | ExtInstruction::BinarySubscrGetitem(n)
            | ExtInstruction::BinarySubscrListInt(n)
            | ExtInstruction::BinarySubscrTupleInt(n)
            | ExtInstruction::CallAdaptive(n)
            | ExtInstruction::CallPyExactArgs(n)
            | ExtInstruction::CallPyWithDefaults(n)
            | ExtInstruction::CompareOpAdaptive(n)
            | ExtInstruction::CompareOpFloatJump(n)
            | ExtInstruction::CompareOpIntJump(n)
            | ExtInstruction::CompareOpStrJump(n)
            | ExtInstruction::LoadAttrAdaptive(n)
            | ExtInstruction::LoadAttrInstanceValue(n)
            | ExtInstruction::LoadAttrModule(n)
            | ExtInstruction::LoadAttrSlot(n)
            | ExtInstruction::LoadAttrWithHint(n)
            | ExtInstruction::LoadConstLoadFast(n)
            | ExtInstruction::LoadFastLoadConst(n)
            | ExtInstruction::LoadFastLoadFast(n)
            | ExtInstruction::LoadGlobalAdaptive(n)
            | ExtInstruction::LoadGlobalBuiltin(n)
            | ExtInstruction::LoadGlobalModule(n)
            | ExtInstruction::LoadMethodAdaptive(n)
            | ExtInstruction::LoadMethodClass(n)
            | ExtInstruction::LoadMethodModule(n)
            | ExtInstruction::LoadMethodNoDict(n)
            | ExtInstruction::LoadMethodWithDict(n)
            | ExtInstruction::LoadMethodWithValues(n)
            | ExtInstruction::PrecallAdaptive(n)
            | ExtInstruction::PrecallBoundMethod(n)
            | ExtInstruction::PrecallBuiltinClass(n)
            | ExtInstruction::PrecallBuiltinFastWithKeywords(n)
            | ExtInstruction::PrecallMethodDescriptorFastWithKeywords(n)
            | ExtInstruction::PrecallNoKwBuiltinFast(n)
            | ExtInstruction::PrecallNoKwBuiltinO(n)
            | ExtInstruction::PrecallNoKwIsinstance(n)
            | ExtInstruction::PrecallNoKwLen(n)
            | ExtInstruction::PrecallNoKwListAppend(n)
            | ExtInstruction::PrecallNoKwMethodDescriptorFast(n)
            | ExtInstruction::PrecallNoKwMethodDescriptorNoargs(n)
            | ExtInstruction::PrecallNoKwMethodDescriptorO(n)
            | ExtInstruction::PrecallNoKwStr1(n)
            | ExtInstruction::PrecallNoKwTuple1(n)
            | ExtInstruction::PrecallNoKwType1(n)
            | ExtInstruction::PrecallPyfunc(n)
            | ExtInstruction::StoreAttrAdaptive(n)
            | ExtInstruction::StoreAttrInstanceValue(n)
            | ExtInstruction::StoreAttrSlot(n)
            | ExtInstruction::StoreAttrWithHint(n)
            | ExtInstruction::StoreFastLoadFast(n)
            | ExtInstruction::StoreFastStoreFast(n)
            | ExtInstruction::StoreSubscrAdaptive(n)
            | ExtInstruction::StoreSubscrDict(n)
            | ExtInstruction::StoreSubscrListInt(n)
            | ExtInstruction::UnpackSequenceAdaptive(n)
            | ExtInstruction::UnpackSequenceList(n)
            | ExtInstruction::UnpackSequenceTuple(n)
            | ExtInstruction::UnpackSequenceTwoTuple(n)
            | ExtInstruction::DoTracing(n) => *n,
            ExtInstruction::ForIter(jump) => jump.index,
            ExtInstruction::LoadConst(const_index) | ExtInstruction::KwNames(const_index) => {
                const_index.index
            }
            ExtInstruction::CompareOp(cmp_op) => cmp_op.into(),
            ExtInstruction::JumpForward(jump)
            | ExtInstruction::JumpIfFalseOrPop(jump)
            | ExtInstruction::JumpIfTrueOrPop(jump)
            | ExtInstruction::PopJumpForwardIfFalse(jump)
            | ExtInstruction::PopJumpForwardIfTrue(jump)
            | ExtInstruction::Send(jump)
            | ExtInstruction::PopJumpForwardIfNotNone(jump)
            | ExtInstruction::PopJumpForwardIfNone(jump)
            | ExtInstruction::JumpBackwardNoInterrupt(jump)
            | ExtInstruction::JumpBackward(jump)
            | ExtInstruction::JumpBackwardQuick(jump)
            | ExtInstruction::PopJumpBackwardIfNotNone(jump)
            | ExtInstruction::PopJumpBackwardIfNone(jump)
            | ExtInstruction::PopJumpBackwardIfFalse(jump)
            | ExtInstruction::PopJumpBackwardIfTrue(jump) => jump.index,
            ExtInstruction::IsOp(invert) | ExtInstruction::ContainsOp(invert) => invert.into(),
            ExtInstruction::Reraise(reraise) => reraise.into(),
            ExtInstruction::BinaryOp(binary_op) => binary_op.into(),
            ExtInstruction::LoadFast(varname_index)
            | ExtInstruction::StoreFast(varname_index)
            | ExtInstruction::DeleteFast(varname_index) => varname_index.index,
            ExtInstruction::RaiseVarargs(raise_var_args) => raise_var_args.into(),
            ExtInstruction::GetAwaitable(awaitable_where) => awaitable_where.into(),
            ExtInstruction::MakeFunction(flags) => flags.bits(),
            ExtInstruction::BuildSlice(slice) => slice.into(),
            ExtInstruction::MakeCell(closure_index)
            | ExtInstruction::LoadClosure(closure_index)
            | ExtInstruction::LoadDeref(closure_index)
            | ExtInstruction::StoreDeref(closure_index)
            | ExtInstruction::DeleteDeref(closure_index)
            | ExtInstruction::LoadClassderef(closure_index) => closure_index.index,
            ExtInstruction::CallFunctionEx(flags) => flags.into(),
            ExtInstruction::Resume(resume_where) | ExtInstruction::ResumeQuick(resume_where) => {
                resume_where.into()
            }
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
