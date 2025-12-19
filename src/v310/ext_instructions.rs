use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut},
};

use store_interval_tree::{Interval, IntervalTree};

use crate::{
    define_default_traits,
    error::Error,
    traits::{
        ExtInstructionAccess, ExtInstructionsOwned, GenericInstruction, InstructionAccess,
        InstructionsOwned, Oparg, SimpleInstructionAccess,
    },
    utils::{get_extended_args_count, UnusedArgument},
    v310::{
        code_objects::{
            AbsoluteJump, CallExFlags, ClosureRefIndex, CompareOperation, ConstIndex, FormatFlag,
            GenKind, Jump, MakeFunctionFlags, NameIndex, OpInversion, RaiseForms, RelativeJump,
            Reraise, SliceCount, VarNameIndex,
        },
        instructions::{Instruction, Instructions},
        opcodes::Opcode,
    },
};

/// Low level representation of a Python bytecode instruction with resolved arguments (extended arg is resolved)
/// We have arguments for every opcode, even if those aren't used. This is so we can have a full representation of the instructions, even if they're invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtInstruction {
    PopTop(UnusedArgument),
    /// Python leaves the ROTN argument after optimizing. See https://github.com/python/cpython/blob/3.10/Python/compile.c#L7522
    RotTwo(UnusedArgument),
    RotThree(UnusedArgument),
    DupTop(UnusedArgument),
    DupTopTwo(UnusedArgument),
    RotFour(UnusedArgument),
    /// Version 3.10 has an unique bug where some NOPs are left with an arg. See https://github.com/python/cpython/issues/89918#issuecomment-1093937041
    Nop(UnusedArgument),
    UnaryPositive(UnusedArgument),
    UnaryNegative(UnusedArgument),
    UnaryNot(UnusedArgument),
    UnaryInvert(UnusedArgument),
    BinaryMatrixMultiply(UnusedArgument),
    InplaceMatrixMultiply(UnusedArgument),
    BinaryPower(UnusedArgument),
    BinaryMultiply(UnusedArgument),
    BinaryModulo(UnusedArgument),
    BinaryAdd(UnusedArgument),
    BinarySubtract(UnusedArgument),
    BinarySubscr(UnusedArgument),
    BinaryFloorDivide(UnusedArgument),
    BinaryTrueDivide(UnusedArgument),
    InplaceFloorDivide(UnusedArgument),
    InplaceTrueDivide(UnusedArgument),
    GetLen(UnusedArgument),
    MatchMapping(UnusedArgument),
    MatchSequence(UnusedArgument),
    MatchKeys(UnusedArgument),
    CopyDictWithoutKeys(UnusedArgument),
    WithExceptStart(UnusedArgument),
    GetAiter(UnusedArgument),
    GetAnext(UnusedArgument),
    BeforeAsyncWith(UnusedArgument),
    EndAsyncFor(UnusedArgument),
    InplaceAdd(UnusedArgument),
    InplaceSubtract(UnusedArgument),
    InplaceMultiply(UnusedArgument),
    InplaceModulo(UnusedArgument),
    StoreSubscr(UnusedArgument),
    DeleteSubscr(UnusedArgument),
    BinaryLshift(UnusedArgument),
    BinaryRshift(UnusedArgument),
    BinaryAnd(UnusedArgument),
    BinaryXor(UnusedArgument),
    BinaryOr(UnusedArgument),
    InplacePower(UnusedArgument),
    GetIter(UnusedArgument),
    GetYieldFromIter(UnusedArgument),
    PrintExpr(UnusedArgument),
    LoadBuildClass(UnusedArgument),
    YieldFrom(UnusedArgument),
    GetAwaitable(UnusedArgument),
    LoadAssertionError(UnusedArgument),
    InplaceLshift(UnusedArgument),
    InplaceRshift(UnusedArgument),
    InplaceAnd(UnusedArgument),
    InplaceXor(UnusedArgument),
    InplaceOr(UnusedArgument),
    ListToTuple(UnusedArgument),
    ReturnValue(UnusedArgument),
    ImportStar(UnusedArgument),
    SetupAnnotations(UnusedArgument),
    YieldValue(UnusedArgument),
    PopBlock(UnusedArgument),
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
    Reraise(Reraise),
    JumpIfNotExcMatch(AbsoluteJump),
    SetupFinally(RelativeJump),
    LoadFast(VarNameIndex),
    StoreFast(VarNameIndex),
    DeleteFast(VarNameIndex),
    GenStart(GenKind),
    RaiseVarargs(RaiseForms),
    CallFunction(u32),
    MakeFunction(MakeFunctionFlags),
    BuildSlice(SliceCount),
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
    InvalidOpcode((u8, UnusedArgument)), // u8 is the opcode number
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

    fn get_jump_value(&self, index: u32) -> Option<Jump> {
        match self.get(index as usize)? {
            ExtInstruction::JumpAbsolute(jump)
            | ExtInstruction::PopJumpIfTrue(jump)
            | ExtInstruction::PopJumpIfFalse(jump)
            | ExtInstruction::JumpIfNotExcMatch(jump)
            | ExtInstruction::JumpIfTrueOrPop(jump)
            | ExtInstruction::JumpIfFalseOrPop(jump) => Some(Jump::Absolute(*jump)),
            ExtInstruction::ForIter(jump)
            | ExtInstruction::JumpForward(jump)
            | ExtInstruction::SetupFinally(jump)
            | ExtInstruction::SetupWith(jump)
            | ExtInstruction::SetupAsyncWith(jump) => Some(Jump::Relative(*jump)),
            _ => None,
        }
    }

    /// Returns the index and the instruction of the jump target. None if the index is not a valid jump.
    fn get_jump_target(&self, index: u32) -> Option<(u32, ExtInstruction)> {
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

impl<T> ExtInstructionAccess<Instruction, ExtInstruction> for T
where
    T: Deref<Target = [ExtInstruction]> + AsRef<[ExtInstruction]>,
{
    type ExtInstructions = ExtInstructions;
    type Instructions = Instructions;

    /// Convert the resolved instructions back into instructions with extended args.
    fn to_instructions(&self) -> Self::Instructions {
        // mapping of original to updated index
        let mut absolute_jump_indexes: BTreeMap<u32, u32> = BTreeMap::new();
        let mut relative_jump_indexes = IntervalTree::<u32, u32>::new(); // (u32, u32) is the from and to index for relative jumps

        self.iter().enumerate().for_each(|(idx, inst)| match inst {
            ExtInstruction::JumpAbsolute(jump)
            | ExtInstruction::PopJumpIfTrue(jump)
            | ExtInstruction::PopJumpIfFalse(jump)
            | ExtInstruction::JumpIfNotExcMatch(jump)
            | ExtInstruction::JumpIfTrueOrPop(jump)
            | ExtInstruction::JumpIfFalseOrPop(jump) => {
                absolute_jump_indexes.insert(jump.index, jump.index);
            }
            ExtInstruction::ForIter(jump)
            | ExtInstruction::JumpForward(jump)
            | ExtInstruction::SetupFinally(jump)
            | ExtInstruction::SetupWith(jump)
            | ExtInstruction::SetupAsyncWith(jump) => {
                relative_jump_indexes.insert(
                    Interval::new(
                        std::ops::Bound::Excluded(idx as u32),
                        std::ops::Bound::Excluded(idx as u32 + jump.index + 1),
                    ),
                    jump.index,
                );
            }
            _ => {}
        });

        // We keep a list of jump indexes that become bigger than 255 while recalculating the jump indexes.
        // We will need to account for those afterwards.
        let mut absolute_jumps_to_update = vec![];
        let mut relative_jumps_to_update = vec![];

        for (index, instruction) in self.iter().enumerate() {
            let arg = instruction.get_raw_value();

            if arg > u8::MAX.into() {
                // Calculate how many extended args an instruction will need
                let extended_arg_count = get_extended_args_count(arg) as u32;

                for (original, new) in absolute_jump_indexes.range_mut((
                    std::ops::Bound::Excluded(index as u32),
                    std::ops::Bound::Unbounded,
                )) {
                    if get_extended_args_count(*new)
                        != get_extended_args_count(*new + extended_arg_count)
                    {
                        absolute_jumps_to_update.push(*original);
                    }

                    *new += extended_arg_count;
                }

                for mut entry in relative_jump_indexes.query_mut(&Interval::point(index as u32)) {
                    let interval_clone = (*entry.interval()).clone();
                    let entry_value = entry.value();

                    if get_extended_args_count(*entry_value)
                        != get_extended_args_count(*entry_value + extended_arg_count)
                    {
                        relative_jumps_to_update.push(interval_clone);
                    }

                    *entry_value += extended_arg_count;
                }
            }
        }

        // Keep updating the offsets until there are no new extended args that need to be accounted for
        while !absolute_jumps_to_update.is_empty() || !relative_jumps_to_update.is_empty() {
            let absolute_clone = absolute_jumps_to_update.clone();
            let relative_clone = relative_jumps_to_update.clone();

            absolute_jumps_to_update.clear();
            relative_jumps_to_update.clear();

            for (index, instruction) in self.iter().enumerate() {
                let arg = match instruction {
                    ExtInstruction::JumpAbsolute(jump)
                    | ExtInstruction::PopJumpIfTrue(jump)
                    | ExtInstruction::PopJumpIfFalse(jump)
                    | ExtInstruction::JumpIfNotExcMatch(jump)
                    | ExtInstruction::JumpIfTrueOrPop(jump)
                    | ExtInstruction::JumpIfFalseOrPop(jump) => {
                        if absolute_clone.contains(&jump.index) {
                            *absolute_jump_indexes
                                .get(&jump.index)
                                .expect("The jump table should always contain all jump indexes")
                        } else {
                            continue;
                        }
                    }
                    ExtInstruction::ForIter(jump)
                    | ExtInstruction::JumpForward(jump)
                    | ExtInstruction::SetupFinally(jump)
                    | ExtInstruction::SetupWith(jump)
                    | ExtInstruction::SetupAsyncWith(jump) => {
                        let interval = Interval::new(
                            std::ops::Bound::Excluded(index as u32),
                            std::ops::Bound::Excluded(index as u32 + jump.index + 1),
                        );

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

                for (original, new) in absolute_jump_indexes.range_mut((
                    std::ops::Bound::Excluded(index as u32),
                    std::ops::Bound::Unbounded,
                )) {
                    if get_extended_args_count(*new)
                        != get_extended_args_count(*new + extended_arg_count)
                    {
                        absolute_jumps_to_update.push(*original);
                    }

                    *new += extended_arg_count;
                }

                for mut entry in relative_jump_indexes.query_mut(&Interval::point(index as u32)) {
                    let interval_clone = (*entry.interval()).clone();
                    let entry_value = entry.value();

                    if get_extended_args_count(*entry_value)
                        != get_extended_args_count(*entry_value + extended_arg_count)
                    {
                        relative_jumps_to_update.push(interval_clone);
                    }

                    *entry_value += extended_arg_count;
                }
            }
        }

        let mut instructions: Instructions = Instructions::with_capacity(self.len() * 2); // This will not be enough this as we dynamically generate EXTENDED_ARGS, but it's better than not reserving any length.

        for (index, instruction) in self.iter().enumerate() {
            let arg = match instruction {
                ExtInstruction::JumpAbsolute(jump)
                | ExtInstruction::PopJumpIfTrue(jump)
                | ExtInstruction::PopJumpIfFalse(jump)
                | ExtInstruction::JumpIfNotExcMatch(jump)
                | ExtInstruction::JumpIfTrueOrPop(jump)
                | ExtInstruction::JumpIfFalseOrPop(jump) => absolute_jump_indexes[&jump.index],
                ExtInstruction::ForIter(jump)
                | ExtInstruction::JumpForward(jump)
                | ExtInstruction::SetupFinally(jump)
                | ExtInstruction::SetupWith(jump)
                | ExtInstruction::SetupAsyncWith(jump) => {
                    let interval = Interval::new(
                        std::ops::Bound::Excluded(index as u32),
                        std::ops::Bound::Excluded(index as u32 + jump.index + 1),
                    );
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
                for ext in ExtInstruction::get_extended_args(arg) {
                    instructions.append_instruction(ext);
                }
            }

            instructions.append_instruction((instruction.get_opcode(), (arg & 0xff) as u8).into());
        }

        instructions
    }

    fn from_instructions(instructions: &[Instruction]) -> Result<Self::ExtInstructions, Error> {
        if !instructions.find_ext_arg_jumps().is_empty() {
            return Err(Error::ExtendedArgJump);
        }

        let mut extended_arg = 0; // Used to keep track of extended arguments between instructions
        let mut absolute_jump_indexes: BTreeMap<u32, u32> = BTreeMap::new();
        let mut relative_jump_indexes: IntervalTree<u32, u32> = IntervalTree::new();

        for (index, instruction) in instructions.iter().enumerate() {
            match instruction {
                Instruction::ExtendedArg(arg) => {
                    let arg = *arg as u32 | extended_arg;
                    extended_arg = arg << 8;
                    continue;
                }
                Instruction::JumpAbsolute(arg)
                | Instruction::PopJumpIfTrue(arg)
                | Instruction::PopJumpIfFalse(arg)
                | Instruction::JumpIfNotExcMatch(arg)
                | Instruction::JumpIfTrueOrPop(arg)
                | Instruction::JumpIfFalseOrPop(arg) => {
                    let arg = *arg as u32 | extended_arg;
                    absolute_jump_indexes.insert(arg, arg);
                }
                Instruction::ForIter(arg)
                | Instruction::JumpForward(arg)
                | Instruction::SetupFinally(arg)
                | Instruction::SetupWith(arg)
                | Instruction::SetupAsyncWith(arg) => {
                    let arg = *arg as u32 | extended_arg;
                    relative_jump_indexes.insert(
                        Interval::new(
                            std::ops::Bound::Excluded(index as u32),
                            std::ops::Bound::Excluded(index as u32 + arg + 1),
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
                absolute_jump_indexes
                    .range_mut((
                        std::ops::Bound::Excluded(index as u32),
                        std::ops::Bound::Unbounded,
                    ))
                    .for_each(|(_, updated_index)| *updated_index -= 1);

                for mut entry in relative_jump_indexes.query_mut(&Interval::point(index as u32)) {
                    *entry.value() -= 1
                }
            }
        }

        let mut ext_instructions = ExtInstructions::with_capacity(instructions.len());

        for (index, instruction) in instructions.iter().enumerate() {
            match instruction {
                Instruction::ExtendedArg(arg) => {
                    let arg = *arg as u32 | extended_arg;
                    extended_arg = arg << 8;
                    continue;
                }
                Instruction::JumpAbsolute(arg)
                | Instruction::PopJumpIfTrue(arg)
                | Instruction::PopJumpIfFalse(arg)
                | Instruction::JumpIfNotExcMatch(arg)
                | Instruction::JumpIfTrueOrPop(arg)
                | Instruction::JumpIfFalseOrPop(arg) => {
                    ext_instructions.append_instruction(
                        (
                            instruction.get_opcode(),
                            *absolute_jump_indexes
                                .get(&(*arg as u32 | extended_arg))
                                .expect("The jump table should always contain all jump indexes"),
                        )
                            .try_into()
                            .expect("This will never error, as we know it's not an EXTENDED_ARG"),
                    );
                }
                Instruction::ForIter(arg)
                | Instruction::JumpForward(arg)
                | Instruction::SetupFinally(arg)
                | Instruction::SetupWith(arg)
                | Instruction::SetupAsyncWith(arg) => {
                    let interval = Interval::new(
                        std::ops::Bound::Excluded(index as u32),
                        std::ops::Bound::Excluded(index as u32 + (*arg as u32 | extended_arg) + 1),
                    );
                    ext_instructions.append_instruction(
                        (
                            instruction.get_opcode(),
                            *relative_jump_indexes
                                .query(&interval)
                                .find(|e| *e.interval() == interval)
                                .expect("The jump table should always contain all jump indexes")
                                .value(),
                        )
                            .try_into()
                            .expect("This will never error, as we know it's not an EXTENDED_ARG"),
                    );
                }
                _ => ext_instructions.append_instruction(
                    (
                        instruction.get_opcode(),
                        instruction.get_raw_value().to_u32() | extended_arg,
                    )
                        .try_into()
                        .expect("This will never error, as we know it's not an EXTENDED_ARG"),
                ),
            }

            extended_arg = 0;
        }

        Ok(ext_instructions)
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
                ExtInstruction::JumpAbsolute(jump)
                | ExtInstruction::PopJumpIfTrue(jump)
                | ExtInstruction::PopJumpIfFalse(jump)
                | ExtInstruction::JumpIfNotExcMatch(jump)
                | ExtInstruction::JumpIfTrueOrPop(jump)
                | ExtInstruction::JumpIfFalseOrPop(jump) => {
                    if jump.index as usize >= index {
                        // Update jump indexes that jump to this index or above it
                        jump.index -= 1
                    }
                }
                ExtInstruction::ForIter(jump)
                | ExtInstruction::JumpForward(jump)
                | ExtInstruction::SetupFinally(jump)
                | ExtInstruction::SetupWith(jump)
                | ExtInstruction::SetupAsyncWith(jump) => {
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

    fn insert_instruction(&mut self, index: usize, instruction: ExtInstruction) {
        self.0.iter_mut().enumerate().for_each(|(idx, inst)| {
            match inst {
                ExtInstruction::JumpAbsolute(jump)
                | ExtInstruction::PopJumpIfTrue(jump)
                | ExtInstruction::PopJumpIfFalse(jump)
                | ExtInstruction::JumpIfNotExcMatch(jump)
                | ExtInstruction::JumpIfTrueOrPop(jump)
                | ExtInstruction::JumpIfFalseOrPop(jump) => {
                    if jump.index as usize >= index {
                        // Update jump indexes that jump to this index or above it
                        jump.index += 1
                    }
                }
                ExtInstruction::ForIter(jump)
                | ExtInstruction::JumpForward(jump)
                | ExtInstruction::SetupFinally(jump)
                | ExtInstruction::SetupWith(jump)
                | ExtInstruction::SetupAsyncWith(jump) => {
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
}

impl ExtInstructions {
    pub fn with_capacity(capacity: usize) -> Self {
        ExtInstructions(Vec::with_capacity(capacity))
    }

    pub fn new(instructions: Vec<ExtInstruction>) -> Self {
        ExtInstructions(instructions)
    }

    /// Returns the index and the instruction of the jump target. None if the index is invalid.
    /// This exists so you don't have to supply the index of the jump instruction (only necessary for relative jumps)
    pub fn get_absolute_jump_target(&self, jump: AbsoluteJump) -> Option<(u32, ExtInstruction)> {
        self.0
            .get(jump.index as usize)
            .cloned()
            .map(|target| (jump.index, target))
    }
}

impl TryFrom<(Opcode, u32)> for ExtInstruction {
    type Error = Error;
    fn try_from(value: (Opcode, u32)) -> Result<Self, Self::Error> {
        Ok(match value.0 {
            Opcode::NOP => ExtInstruction::Nop(value.1.into()),
            Opcode::POP_TOP => ExtInstruction::PopTop(value.1.into()),
            Opcode::ROT_TWO => ExtInstruction::RotTwo(value.1.into()),
            Opcode::ROT_THREE => ExtInstruction::RotThree(value.1.into()),
            Opcode::ROT_FOUR => ExtInstruction::RotFour(value.1.into()),
            Opcode::DUP_TOP => ExtInstruction::DupTop(value.1.into()),
            Opcode::DUP_TOP_TWO => ExtInstruction::DupTopTwo(value.1.into()),
            Opcode::UNARY_POSITIVE => ExtInstruction::UnaryPositive(value.1.into()),
            Opcode::UNARY_NEGATIVE => ExtInstruction::UnaryNegative(value.1.into()),
            Opcode::UNARY_NOT => ExtInstruction::UnaryNot(value.1.into()),
            Opcode::UNARY_INVERT => ExtInstruction::UnaryInvert(value.1.into()),
            Opcode::GET_ITER => ExtInstruction::GetIter(value.1.into()),
            Opcode::GET_YIELD_FROM_ITER => ExtInstruction::GetYieldFromIter(value.1.into()),
            Opcode::BINARY_POWER => ExtInstruction::BinaryPower(value.1.into()),
            Opcode::BINARY_MULTIPLY => ExtInstruction::BinaryMultiply(value.1.into()),
            Opcode::BINARY_MATRIX_MULTIPLY => ExtInstruction::BinaryMatrixMultiply(value.1.into()),
            Opcode::BINARY_FLOOR_DIVIDE => ExtInstruction::BinaryFloorDivide(value.1.into()),
            Opcode::BINARY_TRUE_DIVIDE => ExtInstruction::BinaryTrueDivide(value.1.into()),
            Opcode::BINARY_MODULO => ExtInstruction::BinaryModulo(value.1.into()),
            Opcode::BINARY_ADD => ExtInstruction::BinaryAdd(value.1.into()),
            Opcode::BINARY_SUBTRACT => ExtInstruction::BinarySubtract(value.1.into()),
            Opcode::BINARY_SUBSCR => ExtInstruction::BinarySubscr(value.1.into()),
            Opcode::BINARY_LSHIFT => ExtInstruction::BinaryLshift(value.1.into()),
            Opcode::BINARY_RSHIFT => ExtInstruction::BinaryRshift(value.1.into()),
            Opcode::BINARY_AND => ExtInstruction::BinaryAnd(value.1.into()),
            Opcode::BINARY_XOR => ExtInstruction::BinaryXor(value.1.into()),
            Opcode::BINARY_OR => ExtInstruction::BinaryOr(value.1.into()),
            Opcode::INPLACE_POWER => ExtInstruction::InplacePower(value.1.into()),
            Opcode::INPLACE_MULTIPLY => ExtInstruction::InplaceMultiply(value.1.into()),
            Opcode::INPLACE_MATRIX_MULTIPLY => {
                ExtInstruction::InplaceMatrixMultiply(value.1.into())
            }
            Opcode::INPLACE_FLOOR_DIVIDE => ExtInstruction::InplaceFloorDivide(value.1.into()),
            Opcode::INPLACE_TRUE_DIVIDE => ExtInstruction::InplaceTrueDivide(value.1.into()),
            Opcode::INPLACE_MODULO => ExtInstruction::InplaceModulo(value.1.into()),
            Opcode::INPLACE_ADD => ExtInstruction::InplaceAdd(value.1.into()),
            Opcode::INPLACE_SUBTRACT => ExtInstruction::InplaceSubtract(value.1.into()),
            Opcode::INPLACE_LSHIFT => ExtInstruction::InplaceLshift(value.1.into()),
            Opcode::INPLACE_RSHIFT => ExtInstruction::InplaceRshift(value.1.into()),
            Opcode::INPLACE_AND => ExtInstruction::InplaceAnd(value.1.into()),
            Opcode::INPLACE_XOR => ExtInstruction::InplaceXor(value.1.into()),
            Opcode::INPLACE_OR => ExtInstruction::InplaceOr(value.1.into()),
            Opcode::STORE_SUBSCR => ExtInstruction::StoreSubscr(value.1.into()),
            Opcode::DELETE_SUBSCR => ExtInstruction::DeleteSubscr(value.1.into()),
            Opcode::GET_AWAITABLE => ExtInstruction::GetAwaitable(value.1.into()),
            Opcode::GET_AITER => ExtInstruction::GetAiter(value.1.into()),
            Opcode::GET_ANEXT => ExtInstruction::GetAnext(value.1.into()),
            Opcode::END_ASYNC_FOR => ExtInstruction::EndAsyncFor(value.1.into()),
            Opcode::BEFORE_ASYNC_WITH => ExtInstruction::BeforeAsyncWith(value.1.into()),
            Opcode::SETUP_ASYNC_WITH => {
                ExtInstruction::SetupAsyncWith(RelativeJump { index: value.1 })
            }
            Opcode::PRINT_EXPR => ExtInstruction::PrintExpr(value.1.into()),
            Opcode::SET_ADD => ExtInstruction::SetAdd(value.1),
            Opcode::LIST_APPEND => ExtInstruction::ListAppend(value.1),
            Opcode::MAP_ADD => ExtInstruction::MapAdd(value.1),
            Opcode::RETURN_VALUE => ExtInstruction::ReturnValue(value.1.into()),
            Opcode::YIELD_VALUE => ExtInstruction::YieldValue(value.1.into()),
            Opcode::YIELD_FROM => ExtInstruction::YieldFrom(value.1.into()),
            Opcode::SETUP_ANNOTATIONS => ExtInstruction::SetupAnnotations(value.1.into()),
            Opcode::IMPORT_STAR => ExtInstruction::ImportStar(value.1.into()),
            Opcode::POP_BLOCK => ExtInstruction::PopBlock(value.1.into()),
            Opcode::POP_EXCEPT => ExtInstruction::PopExcept(value.1.into()),
            Opcode::RERAISE => ExtInstruction::Reraise(value.1.into()),
            Opcode::WITH_EXCEPT_START => ExtInstruction::WithExceptStart(value.1.into()),
            Opcode::LOAD_ASSERTION_ERROR => ExtInstruction::LoadAssertionError(value.1.into()),
            Opcode::LOAD_BUILD_CLASS => ExtInstruction::LoadBuildClass(value.1.into()),
            Opcode::SETUP_WITH => ExtInstruction::SetupWith(RelativeJump { index: value.1 }),
            Opcode::COPY_DICT_WITHOUT_KEYS => ExtInstruction::CopyDictWithoutKeys(value.1.into()),
            Opcode::GET_LEN => ExtInstruction::GetLen(value.1.into()),
            Opcode::MATCH_MAPPING => ExtInstruction::MatchMapping(value.1.into()),
            Opcode::MATCH_SEQUENCE => ExtInstruction::MatchSequence(value.1.into()),
            Opcode::MATCH_KEYS => ExtInstruction::MatchKeys(value.1.into()),
            Opcode::STORE_NAME => ExtInstruction::StoreName(NameIndex { index: value.1 }),
            Opcode::DELETE_NAME => ExtInstruction::DeleteName(NameIndex { index: value.1 }),
            Opcode::UNPACK_SEQUENCE => ExtInstruction::UnpackSequence(value.1),
            Opcode::UNPACK_EX => ExtInstruction::UnpackEx(value.1),
            Opcode::STORE_ATTR => ExtInstruction::StoreAttr(NameIndex { index: value.1 }),
            Opcode::DELETE_ATTR => ExtInstruction::DeleteAttr(NameIndex { index: value.1 }),
            Opcode::STORE_GLOBAL => ExtInstruction::StoreGlobal(NameIndex { index: value.1 }),
            Opcode::DELETE_GLOBAL => ExtInstruction::DeleteGlobal(NameIndex { index: value.1 }),
            Opcode::LOAD_CONST => ExtInstruction::LoadConst(ConstIndex { index: value.1 }),
            Opcode::LOAD_NAME => ExtInstruction::LoadName(NameIndex { index: value.1 }),
            Opcode::BUILD_TUPLE => ExtInstruction::BuildTuple(value.1),
            Opcode::BUILD_LIST => ExtInstruction::BuildList(value.1),
            Opcode::BUILD_SET => ExtInstruction::BuildSet(value.1),
            Opcode::BUILD_MAP => ExtInstruction::BuildMap(value.1),
            Opcode::BUILD_CONST_KEY_MAP => ExtInstruction::BuildConstKeyMap(value.1),
            Opcode::BUILD_STRING => ExtInstruction::BuildString(value.1),
            Opcode::LIST_TO_TUPLE => ExtInstruction::ListToTuple(value.1.into()),
            Opcode::LIST_EXTEND => ExtInstruction::ListExtend(value.1),
            Opcode::SET_UPDATE => ExtInstruction::SetUpdate(value.1),
            Opcode::DICT_UPDATE => ExtInstruction::DictUpdate(value.1),
            Opcode::DICT_MERGE => ExtInstruction::DictMerge(value.1),
            Opcode::LOAD_ATTR => ExtInstruction::LoadAttr(NameIndex { index: value.1 }),
            Opcode::COMPARE_OP => ExtInstruction::CompareOp(value.1.into()),
            Opcode::IMPORT_NAME => ExtInstruction::ImportName(NameIndex { index: value.1 }),
            Opcode::IMPORT_FROM => ExtInstruction::ImportFrom(NameIndex { index: value.1 }),
            Opcode::JUMP_FORWARD => ExtInstruction::JumpForward(RelativeJump { index: value.1 }),
            Opcode::POP_JUMP_IF_TRUE => {
                ExtInstruction::PopJumpIfTrue(AbsoluteJump { index: value.1 })
            }
            Opcode::POP_JUMP_IF_FALSE => {
                ExtInstruction::PopJumpIfFalse(AbsoluteJump { index: value.1 })
            }
            Opcode::JUMP_IF_NOT_EXC_MATCH => {
                ExtInstruction::JumpIfNotExcMatch(AbsoluteJump { index: value.1 })
            }
            Opcode::JUMP_IF_TRUE_OR_POP => {
                ExtInstruction::JumpIfTrueOrPop(AbsoluteJump { index: value.1 })
            }
            Opcode::JUMP_IF_FALSE_OR_POP => {
                ExtInstruction::JumpIfFalseOrPop(AbsoluteJump { index: value.1 })
            }
            Opcode::JUMP_ABSOLUTE => ExtInstruction::JumpAbsolute(AbsoluteJump { index: value.1 }),
            Opcode::FOR_ITER => ExtInstruction::ForIter(RelativeJump { index: value.1 }),
            Opcode::LOAD_GLOBAL => ExtInstruction::LoadGlobal(NameIndex { index: value.1 }),
            Opcode::IS_OP => ExtInstruction::IsOp(value.1.into()),
            Opcode::CONTAINS_OP => ExtInstruction::ContainsOp(value.1.into()),
            Opcode::SETUP_FINALLY => ExtInstruction::SetupFinally(RelativeJump { index: value.1 }),
            Opcode::LOAD_FAST => ExtInstruction::LoadFast(VarNameIndex { index: value.1 }),
            Opcode::STORE_FAST => ExtInstruction::StoreFast(VarNameIndex { index: value.1 }),
            Opcode::DELETE_FAST => ExtInstruction::DeleteFast(VarNameIndex { index: value.1 }),
            Opcode::LOAD_CLOSURE => ExtInstruction::LoadClosure(ClosureRefIndex { index: value.1 }),
            Opcode::LOAD_DEREF => ExtInstruction::LoadDeref(ClosureRefIndex { index: value.1 }),
            Opcode::LOAD_CLASSDEREF => {
                ExtInstruction::LoadClassderef(ClosureRefIndex { index: value.1 })
            }
            Opcode::STORE_DEREF => ExtInstruction::StoreDeref(ClosureRefIndex { index: value.1 }),
            Opcode::DELETE_DEREF => ExtInstruction::DeleteDeref(ClosureRefIndex { index: value.1 }),
            Opcode::RAISE_VARARGS => ExtInstruction::RaiseVarargs(value.1.into()),
            Opcode::CALL_FUNCTION => ExtInstruction::CallFunction(value.1),
            Opcode::CALL_FUNCTION_KW => ExtInstruction::CallFunctionKW(value.1),
            Opcode::CALL_FUNCTION_EX => ExtInstruction::CallFunctionEx(value.1.into()),
            Opcode::LOAD_METHOD => ExtInstruction::LoadMethod(NameIndex { index: value.1 }),
            Opcode::CALL_METHOD => ExtInstruction::CallMethod(value.1),
            Opcode::MAKE_FUNCTION => {
                ExtInstruction::MakeFunction(MakeFunctionFlags::from_bits_retain(value.1))
            }
            Opcode::BUILD_SLICE => ExtInstruction::BuildSlice(value.1.into()),
            Opcode::FORMAT_VALUE => ExtInstruction::FormatValue(value.1.into()),
            Opcode::MATCH_CLASS => ExtInstruction::MatchClass(value.1),
            Opcode::GEN_START => ExtInstruction::GenStart(value.1.into()),
            Opcode::ROT_N => ExtInstruction::RotN(value.1),
            Opcode::EXTENDED_ARG => return Err(Error::InvalidConversion),
            Opcode::INVALID_OPCODE(opcode) => {
                ExtInstruction::InvalidOpcode((opcode, value.1.into()))
            }
        })
    }
}

impl GenericInstruction for ExtInstruction {
    type OpargType = u32;
    type Opcode = Opcode;

    fn get_opcode(&self) -> Self::Opcode {
        match self {
            ExtInstruction::Nop(_) => Opcode::NOP,
            ExtInstruction::PopTop(_) => Opcode::POP_TOP,
            ExtInstruction::RotTwo(_) => Opcode::ROT_TWO,
            ExtInstruction::RotThree(_) => Opcode::ROT_THREE,
            ExtInstruction::RotFour(_) => Opcode::ROT_FOUR,
            ExtInstruction::DupTop(_) => Opcode::DUP_TOP,
            ExtInstruction::DupTopTwo(_) => Opcode::DUP_TOP_TWO,
            ExtInstruction::UnaryPositive(_) => Opcode::UNARY_POSITIVE,
            ExtInstruction::UnaryNegative(_) => Opcode::UNARY_NEGATIVE,
            ExtInstruction::UnaryNot(_) => Opcode::UNARY_NOT,
            ExtInstruction::UnaryInvert(_) => Opcode::UNARY_INVERT,
            ExtInstruction::GetIter(_) => Opcode::GET_ITER,
            ExtInstruction::GetYieldFromIter(_) => Opcode::GET_YIELD_FROM_ITER,
            ExtInstruction::BinaryPower(_) => Opcode::BINARY_POWER,
            ExtInstruction::BinaryMultiply(_) => Opcode::BINARY_MULTIPLY,
            ExtInstruction::BinaryMatrixMultiply(_) => Opcode::BINARY_MATRIX_MULTIPLY,
            ExtInstruction::BinaryFloorDivide(_) => Opcode::BINARY_FLOOR_DIVIDE,
            ExtInstruction::BinaryTrueDivide(_) => Opcode::BINARY_TRUE_DIVIDE,
            ExtInstruction::BinaryModulo(_) => Opcode::BINARY_MODULO,
            ExtInstruction::BinaryAdd(_) => Opcode::BINARY_ADD,
            ExtInstruction::BinarySubtract(_) => Opcode::BINARY_SUBTRACT,
            ExtInstruction::BinarySubscr(_) => Opcode::BINARY_SUBSCR,
            ExtInstruction::BinaryLshift(_) => Opcode::BINARY_LSHIFT,
            ExtInstruction::BinaryRshift(_) => Opcode::BINARY_RSHIFT,
            ExtInstruction::BinaryAnd(_) => Opcode::BINARY_AND,
            ExtInstruction::BinaryXor(_) => Opcode::BINARY_XOR,
            ExtInstruction::BinaryOr(_) => Opcode::BINARY_OR,
            ExtInstruction::InplacePower(_) => Opcode::INPLACE_POWER,
            ExtInstruction::InplaceMultiply(_) => Opcode::INPLACE_MULTIPLY,
            ExtInstruction::InplaceMatrixMultiply(_) => Opcode::INPLACE_MATRIX_MULTIPLY,
            ExtInstruction::InplaceFloorDivide(_) => Opcode::INPLACE_FLOOR_DIVIDE,
            ExtInstruction::InplaceTrueDivide(_) => Opcode::INPLACE_TRUE_DIVIDE,
            ExtInstruction::InplaceModulo(_) => Opcode::INPLACE_MODULO,
            ExtInstruction::InplaceAdd(_) => Opcode::INPLACE_ADD,
            ExtInstruction::InplaceSubtract(_) => Opcode::INPLACE_SUBTRACT,
            ExtInstruction::InplaceLshift(_) => Opcode::INPLACE_LSHIFT,
            ExtInstruction::InplaceRshift(_) => Opcode::INPLACE_RSHIFT,
            ExtInstruction::InplaceAnd(_) => Opcode::INPLACE_AND,
            ExtInstruction::InplaceXor(_) => Opcode::INPLACE_XOR,
            ExtInstruction::InplaceOr(_) => Opcode::INPLACE_OR,
            ExtInstruction::StoreSubscr(_) => Opcode::STORE_SUBSCR,
            ExtInstruction::DeleteSubscr(_) => Opcode::DELETE_SUBSCR,
            ExtInstruction::GetAwaitable(_) => Opcode::GET_AWAITABLE,
            ExtInstruction::GetAiter(_) => Opcode::GET_AITER,
            ExtInstruction::GetAnext(_) => Opcode::GET_ANEXT,
            ExtInstruction::EndAsyncFor(_) => Opcode::END_ASYNC_FOR,
            ExtInstruction::BeforeAsyncWith(_) => Opcode::BEFORE_ASYNC_WITH,
            ExtInstruction::SetupAsyncWith(_) => Opcode::SETUP_ASYNC_WITH,
            ExtInstruction::PrintExpr(_) => Opcode::PRINT_EXPR,
            ExtInstruction::SetAdd(_) => Opcode::SET_ADD,
            ExtInstruction::ListAppend(_) => Opcode::LIST_APPEND,
            ExtInstruction::MapAdd(_) => Opcode::MAP_ADD,
            ExtInstruction::ReturnValue(_) => Opcode::RETURN_VALUE,
            ExtInstruction::YieldValue(_) => Opcode::YIELD_VALUE,
            ExtInstruction::YieldFrom(_) => Opcode::YIELD_FROM,
            ExtInstruction::SetupAnnotations(_) => Opcode::SETUP_ANNOTATIONS,
            ExtInstruction::ImportStar(_) => Opcode::IMPORT_STAR,
            ExtInstruction::PopBlock(_) => Opcode::POP_BLOCK,
            ExtInstruction::PopExcept(_) => Opcode::POP_EXCEPT,
            ExtInstruction::Reraise(_) => Opcode::RERAISE,
            ExtInstruction::WithExceptStart(_) => Opcode::WITH_EXCEPT_START,
            ExtInstruction::LoadAssertionError(_) => Opcode::LOAD_ASSERTION_ERROR,
            ExtInstruction::LoadBuildClass(_) => Opcode::LOAD_BUILD_CLASS,
            ExtInstruction::SetupWith(_) => Opcode::SETUP_WITH,
            ExtInstruction::CopyDictWithoutKeys(_) => Opcode::COPY_DICT_WITHOUT_KEYS,
            ExtInstruction::GetLen(_) => Opcode::GET_LEN,
            ExtInstruction::MatchMapping(_) => Opcode::MATCH_MAPPING,
            ExtInstruction::MatchSequence(_) => Opcode::MATCH_SEQUENCE,
            ExtInstruction::MatchKeys(_) => Opcode::MATCH_KEYS,
            ExtInstruction::StoreName(_) => Opcode::STORE_NAME,
            ExtInstruction::DeleteName(_) => Opcode::DELETE_NAME,
            ExtInstruction::UnpackSequence(_) => Opcode::UNPACK_SEQUENCE,
            ExtInstruction::UnpackEx(_) => Opcode::UNPACK_EX,
            ExtInstruction::StoreAttr(_) => Opcode::STORE_ATTR,
            ExtInstruction::DeleteAttr(_) => Opcode::DELETE_ATTR,
            ExtInstruction::StoreGlobal(_) => Opcode::STORE_GLOBAL,
            ExtInstruction::DeleteGlobal(_) => Opcode::DELETE_GLOBAL,
            ExtInstruction::LoadConst(_) => Opcode::LOAD_CONST,
            ExtInstruction::LoadName(_) => Opcode::LOAD_NAME,
            ExtInstruction::BuildTuple(_) => Opcode::BUILD_TUPLE,
            ExtInstruction::BuildList(_) => Opcode::BUILD_LIST,
            ExtInstruction::BuildSet(_) => Opcode::BUILD_SET,
            ExtInstruction::BuildMap(_) => Opcode::BUILD_MAP,
            ExtInstruction::BuildConstKeyMap(_) => Opcode::BUILD_CONST_KEY_MAP,
            ExtInstruction::BuildString(_) => Opcode::BUILD_STRING,
            ExtInstruction::ListToTuple(_) => Opcode::LIST_TO_TUPLE,
            ExtInstruction::ListExtend(_) => Opcode::LIST_EXTEND,
            ExtInstruction::SetUpdate(_) => Opcode::SET_UPDATE,
            ExtInstruction::DictUpdate(_) => Opcode::DICT_UPDATE,
            ExtInstruction::DictMerge(_) => Opcode::DICT_MERGE,
            ExtInstruction::LoadAttr(_) => Opcode::LOAD_ATTR,
            ExtInstruction::CompareOp(_) => Opcode::COMPARE_OP,
            ExtInstruction::ImportName(_) => Opcode::IMPORT_NAME,
            ExtInstruction::ImportFrom(_) => Opcode::IMPORT_FROM,
            ExtInstruction::JumpForward(_) => Opcode::JUMP_FORWARD,
            ExtInstruction::PopJumpIfTrue(_) => Opcode::POP_JUMP_IF_TRUE,
            ExtInstruction::PopJumpIfFalse(_) => Opcode::POP_JUMP_IF_FALSE,
            ExtInstruction::JumpIfNotExcMatch(_) => Opcode::JUMP_IF_NOT_EXC_MATCH,
            ExtInstruction::JumpIfTrueOrPop(_) => Opcode::JUMP_IF_TRUE_OR_POP,
            ExtInstruction::JumpIfFalseOrPop(_) => Opcode::JUMP_IF_FALSE_OR_POP,
            ExtInstruction::JumpAbsolute(_) => Opcode::JUMP_ABSOLUTE,
            ExtInstruction::ForIter(_) => Opcode::FOR_ITER,
            ExtInstruction::LoadGlobal(_) => Opcode::LOAD_GLOBAL,
            ExtInstruction::IsOp(_) => Opcode::IS_OP,
            ExtInstruction::ContainsOp(_) => Opcode::CONTAINS_OP,
            ExtInstruction::SetupFinally(_) => Opcode::SETUP_FINALLY,
            ExtInstruction::LoadFast(_) => Opcode::LOAD_FAST,
            ExtInstruction::StoreFast(_) => Opcode::STORE_FAST,
            ExtInstruction::DeleteFast(_) => Opcode::DELETE_FAST,
            ExtInstruction::LoadClosure(_) => Opcode::LOAD_CLOSURE,
            ExtInstruction::LoadDeref(_) => Opcode::LOAD_DEREF,
            ExtInstruction::LoadClassderef(_) => Opcode::LOAD_CLASSDEREF,
            ExtInstruction::StoreDeref(_) => Opcode::STORE_DEREF,
            ExtInstruction::DeleteDeref(_) => Opcode::DELETE_DEREF,
            ExtInstruction::RaiseVarargs(_) => Opcode::RAISE_VARARGS,
            ExtInstruction::CallFunction(_) => Opcode::CALL_FUNCTION,
            ExtInstruction::CallFunctionKW(_) => Opcode::CALL_FUNCTION_KW,
            ExtInstruction::CallFunctionEx(_) => Opcode::CALL_FUNCTION_EX,
            ExtInstruction::LoadMethod(_) => Opcode::LOAD_METHOD,
            ExtInstruction::CallMethod(_) => Opcode::CALL_METHOD,
            ExtInstruction::MakeFunction(_) => Opcode::MAKE_FUNCTION,
            ExtInstruction::BuildSlice(_) => Opcode::BUILD_SLICE,
            ExtInstruction::FormatValue(_) => Opcode::FORMAT_VALUE,
            ExtInstruction::MatchClass(_) => Opcode::MATCH_CLASS,
            ExtInstruction::GenStart(_) => Opcode::GEN_START,
            ExtInstruction::RotN(_) => Opcode::ROT_N,
            ExtInstruction::InvalidOpcode((opcode, _)) => Opcode::INVALID_OPCODE(*opcode),
        }
    }

    fn get_raw_value(&self) -> u32 {
        match &self {
            ExtInstruction::PopTop(unused_arg)
            | ExtInstruction::RotTwo(unused_arg)
            | ExtInstruction::RotThree(unused_arg)
            | ExtInstruction::DupTop(unused_arg)
            | ExtInstruction::DupTopTwo(unused_arg)
            | ExtInstruction::RotFour(unused_arg)
            | ExtInstruction::Nop(unused_arg)
            | ExtInstruction::UnaryPositive(unused_arg)
            | ExtInstruction::UnaryNegative(unused_arg)
            | ExtInstruction::UnaryNot(unused_arg)
            | ExtInstruction::UnaryInvert(unused_arg)
            | ExtInstruction::BinaryMatrixMultiply(unused_arg)
            | ExtInstruction::InplaceMatrixMultiply(unused_arg)
            | ExtInstruction::BinaryPower(unused_arg)
            | ExtInstruction::BinaryMultiply(unused_arg)
            | ExtInstruction::BinaryModulo(unused_arg)
            | ExtInstruction::BinaryAdd(unused_arg)
            | ExtInstruction::BinarySubtract(unused_arg)
            | ExtInstruction::BinarySubscr(unused_arg)
            | ExtInstruction::BinaryFloorDivide(unused_arg)
            | ExtInstruction::BinaryTrueDivide(unused_arg)
            | ExtInstruction::InplaceFloorDivide(unused_arg)
            | ExtInstruction::InplaceTrueDivide(unused_arg)
            | ExtInstruction::GetLen(unused_arg)
            | ExtInstruction::MatchMapping(unused_arg)
            | ExtInstruction::MatchSequence(unused_arg)
            | ExtInstruction::MatchKeys(unused_arg)
            | ExtInstruction::CopyDictWithoutKeys(unused_arg)
            | ExtInstruction::WithExceptStart(unused_arg)
            | ExtInstruction::GetAiter(unused_arg)
            | ExtInstruction::GetAnext(unused_arg)
            | ExtInstruction::BeforeAsyncWith(unused_arg)
            | ExtInstruction::EndAsyncFor(unused_arg)
            | ExtInstruction::InplaceAdd(unused_arg)
            | ExtInstruction::InplaceSubtract(unused_arg)
            | ExtInstruction::InplaceMultiply(unused_arg)
            | ExtInstruction::InplaceModulo(unused_arg)
            | ExtInstruction::StoreSubscr(unused_arg)
            | ExtInstruction::DeleteSubscr(unused_arg)
            | ExtInstruction::BinaryLshift(unused_arg)
            | ExtInstruction::BinaryRshift(unused_arg)
            | ExtInstruction::BinaryAnd(unused_arg)
            | ExtInstruction::BinaryXor(unused_arg)
            | ExtInstruction::BinaryOr(unused_arg)
            | ExtInstruction::InplacePower(unused_arg)
            | ExtInstruction::GetIter(unused_arg)
            | ExtInstruction::GetYieldFromIter(unused_arg)
            | ExtInstruction::PrintExpr(unused_arg)
            | ExtInstruction::LoadBuildClass(unused_arg)
            | ExtInstruction::YieldFrom(unused_arg)
            | ExtInstruction::GetAwaitable(unused_arg)
            | ExtInstruction::LoadAssertionError(unused_arg)
            | ExtInstruction::InplaceLshift(unused_arg)
            | ExtInstruction::InplaceRshift(unused_arg)
            | ExtInstruction::InplaceAnd(unused_arg)
            | ExtInstruction::InplaceXor(unused_arg)
            | ExtInstruction::InplaceOr(unused_arg)
            | ExtInstruction::ListToTuple(unused_arg)
            | ExtInstruction::ReturnValue(unused_arg)
            | ExtInstruction::ImportStar(unused_arg)
            | ExtInstruction::SetupAnnotations(unused_arg)
            | ExtInstruction::YieldValue(unused_arg)
            | ExtInstruction::PopBlock(unused_arg)
            | ExtInstruction::PopExcept(unused_arg)
            | ExtInstruction::InvalidOpcode((_, unused_arg)) => unused_arg.0,
            ExtInstruction::StoreName(name_index)
            | ExtInstruction::DeleteName(name_index)
            | ExtInstruction::StoreAttr(name_index)
            | ExtInstruction::DeleteAttr(name_index)
            | ExtInstruction::StoreGlobal(name_index)
            | ExtInstruction::DeleteGlobal(name_index)
            | ExtInstruction::LoadName(name_index)
            | ExtInstruction::LoadAttr(name_index)
            | ExtInstruction::ImportName(name_index)
            | ExtInstruction::ImportFrom(name_index)
            | ExtInstruction::LoadGlobal(name_index)
            | ExtInstruction::LoadMethod(name_index) => name_index.index,
            ExtInstruction::UnpackSequence(n)
            | ExtInstruction::UnpackEx(n)
            | ExtInstruction::RotN(n)
            | ExtInstruction::BuildTuple(n)
            | ExtInstruction::BuildList(n)
            | ExtInstruction::BuildSet(n)
            | ExtInstruction::BuildMap(n)
            | ExtInstruction::CallFunction(n)
            | ExtInstruction::CallFunctionKW(n)
            | ExtInstruction::ListAppend(n)
            | ExtInstruction::SetAdd(n)
            | ExtInstruction::MapAdd(n)
            | ExtInstruction::MatchClass(n)
            | ExtInstruction::BuildConstKeyMap(n)
            | ExtInstruction::BuildString(n)
            | ExtInstruction::CallMethod(n)
            | ExtInstruction::ListExtend(n)
            | ExtInstruction::SetUpdate(n)
            | ExtInstruction::DictUpdate(n)
            | ExtInstruction::DictMerge(n) => *n,
            ExtInstruction::BuildSlice(slice) => Into::<u32>::into(slice),
            ExtInstruction::ForIter(jump)
            | ExtInstruction::JumpForward(jump)
            | ExtInstruction::SetupFinally(jump)
            | ExtInstruction::SetupWith(jump)
            | ExtInstruction::SetupAsyncWith(jump) => jump.index,
            ExtInstruction::LoadConst(const_index) => const_index.index,
            ExtInstruction::CompareOp(comp_op) => Into::<u32>::into(comp_op),
            ExtInstruction::JumpIfFalseOrPop(jump)
            | ExtInstruction::JumpIfTrueOrPop(jump)
            | ExtInstruction::JumpAbsolute(jump)
            | ExtInstruction::PopJumpIfFalse(jump)
            | ExtInstruction::PopJumpIfTrue(jump)
            | ExtInstruction::JumpIfNotExcMatch(jump) => jump.index,
            ExtInstruction::Reraise(forms) => Into::<u32>::into(forms),
            ExtInstruction::IsOp(op_inv) | ExtInstruction::ContainsOp(op_inv) => {
                Into::<u32>::into(op_inv)
            }
            ExtInstruction::LoadFast(varname_index)
            | ExtInstruction::StoreFast(varname_index)
            | ExtInstruction::DeleteFast(varname_index) => varname_index.index,
            ExtInstruction::GenStart(kind) => Into::<u32>::into(kind),
            ExtInstruction::RaiseVarargs(form) => Into::<u32>::into(form),
            ExtInstruction::MakeFunction(flags) => flags.bits(),
            ExtInstruction::LoadClosure(closure_ref_index)
            | ExtInstruction::LoadDeref(closure_ref_index)
            | ExtInstruction::StoreDeref(closure_ref_index)
            | ExtInstruction::DeleteDeref(closure_ref_index)
            | ExtInstruction::LoadClassderef(closure_ref_index) => closure_ref_index.index,
            ExtInstruction::CallFunctionEx(flags) => Into::<u32>::into(flags),
            ExtInstruction::FormatValue(format_flag) => Into::<u32>::into(format_flag),
        }
    }

    fn get_nop() -> Self {
        ExtInstruction::Nop(UnusedArgument(0))
    }
}

impl ExtInstruction {
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
}

define_default_traits!(v310, ExtInstruction);
