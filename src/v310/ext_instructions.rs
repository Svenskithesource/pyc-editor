use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut},
};

use store_interval_tree::{Interval, IntervalTree};

use crate::{
    error::Error,
    v310::{
        code_objects::{
            AbsoluteJump, CallExFlags, ClosureRefIndex, CompareOperation, ConstIndex, FormatFlag,
            GenKind, Jump, MakeFunctionFlags, NameIndex, OpInversion, RaiseForms, RelativeJump,
            VarNameIndex,
        },
        instructions::{Instruction, Instructions},
        opcodes::Opcode,
    },
};

/// Used to represent opargs for opcodes that don't require arguments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InvalidArgument(u32);

impl From<u32> for InvalidArgument {
    fn from(value: u32) -> Self {
        InvalidArgument(value)
    }
}

/// Low level representation of a Python bytecode instruction with resolved arguments (extended arg is resolved)
/// We have arguments for every opcode, even if those aren't used. This is so we can have a full representation of the instructions, even if they're invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtInstruction {
    PopTop(InvalidArgument),
    /// Python leaves the ROTN argument after optimizing. See https://github.com/python/cpython/blob/3.10/Python/compile.c#L7522
    RotTwo(InvalidArgument),
    RotThree(InvalidArgument),
    DupTop(InvalidArgument),
    DupTopTwo(InvalidArgument),
    RotFour(InvalidArgument),
    /// Version 3.10 has an unique bug where some NOPs are left with an arg. See https://github.com/python/cpython/issues/89918#issuecomment-1093937041
    Nop(InvalidArgument),
    UnaryPositive(InvalidArgument),
    UnaryNegative(InvalidArgument),
    UnaryNot(InvalidArgument),
    UnaryInvert(InvalidArgument),
    BinaryMatrixMultiply(InvalidArgument),
    InplaceMatrixMultiply(InvalidArgument),
    BinaryPower(InvalidArgument),
    BinaryMultiply(InvalidArgument),
    BinaryModulo(InvalidArgument),
    BinaryAdd(InvalidArgument),
    BinarySubtract(InvalidArgument),
    BinarySubscr(InvalidArgument),
    BinaryFloorDivide(InvalidArgument),
    BinaryTrueDivide(InvalidArgument),
    InplaceFloorDivide(InvalidArgument),
    InplaceTrueDivide(InvalidArgument),
    GetLen(InvalidArgument),
    MatchMapping(InvalidArgument),
    MatchSequence(InvalidArgument),
    MatchKeys(InvalidArgument),
    CopyDictWithoutKeys(InvalidArgument),
    WithExceptStart(InvalidArgument),
    GetAiter(InvalidArgument),
    GetAnext(InvalidArgument),
    BeforeAsyncWith(InvalidArgument),
    EndAsyncFor(InvalidArgument),
    InplaceAdd(InvalidArgument),
    InplaceSubtract(InvalidArgument),
    InplaceMultiply(InvalidArgument),
    InplaceModulo(InvalidArgument),
    StoreSubscr(InvalidArgument),
    DeleteSubscr(InvalidArgument),
    BinaryLshift(InvalidArgument),
    BinaryRshift(InvalidArgument),
    BinaryAnd(InvalidArgument),
    BinaryXor(InvalidArgument),
    BinaryOr(InvalidArgument),
    InplacePower(InvalidArgument),
    GetIter(InvalidArgument),
    GetYieldFromIter(InvalidArgument),
    PrintExpr(InvalidArgument),
    LoadBuildClass(InvalidArgument),
    YieldFrom(InvalidArgument),
    GetAwaitable(InvalidArgument),
    LoadAssertionError(InvalidArgument),
    InplaceLshift(InvalidArgument),
    InplaceRshift(InvalidArgument),
    InplaceAnd(InvalidArgument),
    InplaceXor(InvalidArgument),
    InplaceOr(InvalidArgument),
    ListToTuple(InvalidArgument),
    ReturnValue(InvalidArgument),
    ImportStar(InvalidArgument),
    SetupAnnotations(InvalidArgument),
    YieldValue(InvalidArgument),
    PopBlock(InvalidArgument),
    PopExcept(InvalidArgument),
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
    Reraise(RaiseForms),
    JumpIfNotExcMatch(AbsoluteJump),
    SetupFinally(RelativeJump),
    LoadFast(VarNameIndex),
    StoreFast(VarNameIndex),
    DeleteFast(VarNameIndex),
    GenStart(GenKind),
    RaiseVarargs(RaiseForms),
    CallFunction(u32),
    MakeFunction(MakeFunctionFlags),
    BuildSlice(u32),
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
}

/// A list of resolved instructions (extended_arg is resolved)
#[derive(Debug, Clone, PartialEq)]
pub struct ExtInstructions(Vec<ExtInstruction>);

impl ExtInstructions {
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

    pub fn get_instructions(&self) -> &[ExtInstruction] {
        self.deref()
    }

    pub fn get_instructions_mut(&mut self) -> &mut [ExtInstruction] {
        self.deref_mut()
    }

    pub fn get_jump_target(&self, jump: Jump) -> Option<ExtInstruction> {
        match jump {
            Jump::Absolute(AbsoluteJump { index }) | Jump::Relative(RelativeJump { index }) => {
                self.0.get(index as usize).cloned()
            }
        }
    }

    pub fn to_instructions(&self) -> Instructions {
        let mut instructions: Instructions = Instructions::with_capacity(self.0.len() * 2); // This will not be enough this as we dynamically generate EXTENDED_ARGS, but it's better than not reserving any length.

        macro_rules! push_inst {
            ($instruction:expr, $arg:expr) => {{
                let mut arg: u32 = $arg;
                // Emit EXTENDED_ARGs for arguments > 0xFF
                if arg > u8::MAX.into() {
                    // Python bytecode uses EXTENDED_ARG for each additional byte above the lowest.
                    // We need to emit them from most significant to least significant.
                    let mut ext_args = Vec::new();
                    while arg > u8::MAX.into() {
                        ext_args.push(((arg >> 8) & 0xFF) as u8);
                        arg >>= 8;
                    }
                    // Emit EXTENDED_ARGs in reverse order (most significant first)
                    for &ext in ext_args.iter().rev() {
                        instructions.append_instruction((Opcode::EXTENDED_ARG, ext).into());
                    }
                }

                instructions.append_instruction(($instruction.get_opcode(), $arg as u8).into());
            }};
        }

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
                        std::ops::Bound::Included(idx as u32 + 1),
                        std::ops::Bound::Included(idx as u32 + jump.index),
                    ),
                    jump.index,
                );
            }
            _ => {}
        });

        for (index, instruction) in self.iter().enumerate() {
            let arg = instruction.get_raw_value();

            if arg > u8::MAX.into() {
                // Calculate how many extended args an instruction will need
                let extended_arg_count = ((32 - arg.leading_zeros()) + 7) / 8;
                let extended_arg_count = extended_arg_count.saturating_sub(1); // Don't count the instruction itself

                absolute_jump_indexes
                    .range_mut((
                        std::ops::Bound::Excluded(index as u32),
                        std::ops::Bound::Unbounded,
                    ))
                    .for_each(|(_, new)| *new += extended_arg_count);

                for mut entry in relative_jump_indexes.query_mut(&Interval::new(
                    std::ops::Bound::Included(index as u32),
                    std::ops::Bound::Excluded((index + 1) as u32),
                )) {
                    *entry.value() += extended_arg_count
                }
            }
        }

        for (idx, instruction) in self.0.iter().enumerate() {
            match instruction {
                ExtInstruction::JumpAbsolute(jump)
                | ExtInstruction::PopJumpIfTrue(jump)
                | ExtInstruction::PopJumpIfFalse(jump)
                | ExtInstruction::JumpIfNotExcMatch(jump)
                | ExtInstruction::JumpIfTrueOrPop(jump)
                | ExtInstruction::JumpIfFalseOrPop(jump) => {
                    push_inst!(instruction, absolute_jump_indexes[&jump.index]);
                }
                ExtInstruction::ForIter(jump)
                | ExtInstruction::JumpForward(jump)
                | ExtInstruction::SetupFinally(jump)
                | ExtInstruction::SetupWith(jump)
                | ExtInstruction::SetupAsyncWith(jump) => {
                    let interval = Interval::new(
                        std::ops::Bound::Included(idx as u32 + 1),
                        std::ops::Bound::Included(idx as u32 + jump.index),
                    );

                    // dbg!(
                    //     relative_jump_indexes.query(&interval).collect::<Vec<_>>(),
                    //     &interval
                    // );

                    if cfg!(debug_assertions) {
                        let indexes = relative_jump_indexes
                            .query(&interval)
                            .filter(|entry| *entry.interval() == interval)
                            .collect::<Vec<_>>();

                        assert!(indexes.len() == 1);

                        push_inst!(
                            instruction,
                            *indexes
                                .first()
                                .expect("This interval should always exist")
                                .value()
                        );
                    } else {
                        // This is faster, so use for release builds
                        push_inst!(
                            instruction,
                            *relative_jump_indexes
                                .query(&interval)
                                .find(|entry| *entry.interval() == interval)
                                .expect("This interval should always exist")
                                .value()
                        );
                    }
                }
                _ => push_inst!(instruction, instruction.get_raw_value()),
            }
        }

        instructions
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_instructions().to_bytes()
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
        let mut ext_instructions = ExtInstructions(Vec::with_capacity(code.len() / 2));
        let mut extended_arg = 0; // Used to keep track of extended arguments between instructions
        let mut removed_extended_args = vec![]; // Used to offset jump indexes
        let mut removed_count = 0;

        for (index, instruction) in code.iter().enumerate() {
            let opcode = instruction.get_opcode();
            let arg = instruction.get_raw_value();

            match opcode {
                Opcode::EXTENDED_ARG => {
                    removed_extended_args.push(index - removed_count);
                    removed_count += 1;
                    extended_arg = (extended_arg << 8) | arg as u32;
                    continue;
                }
                _ => {
                    let arg = (extended_arg << 8) | arg as u32;

                    ext_instructions.append_instruction((opcode, arg).into());
                }
            }

            extended_arg = 0;
        }

        // mapping of original to updated index
        let mut absolute_jump_indexes: BTreeMap<u32, u32> = BTreeMap::new();
        let mut relative_jump_indexes = IntervalTree::<u32, u32>::new(); // (u32, u32) is the from and to index for relative jumps

        ext_instructions
            .iter()
            .enumerate()
            .for_each(|(idx, inst)| match inst {
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
                            std::ops::Bound::Included(idx as u32 + 1),
                            std::ops::Bound::Included(idx as u32 + jump.index),
                        ),
                        jump.index,
                    );
                }
                _ => {}
            });

        // Update jump offsets to exclude the extended args that were removed
        for index in removed_extended_args {
            absolute_jump_indexes
                .range_mut((
                    std::ops::Bound::Excluded(index as u32),
                    std::ops::Bound::Unbounded,
                ))
                .for_each(|(_, new)| *new -= 1);

            for mut entry in relative_jump_indexes.query_mut(&Interval::new(
                std::ops::Bound::Included(index as u32),
                std::ops::Bound::Included(index as u32),
            )) {
                *entry.value() -= 1
            }
        }

        for (idx, instruction) in ext_instructions.iter_mut().enumerate() {
            match instruction {
                ExtInstruction::JumpAbsolute(jump)
                | ExtInstruction::PopJumpIfTrue(jump)
                | ExtInstruction::PopJumpIfFalse(jump)
                | ExtInstruction::JumpIfNotExcMatch(jump)
                | ExtInstruction::JumpIfTrueOrPop(jump)
                | ExtInstruction::JumpIfFalseOrPop(jump) => {
                    jump.index = absolute_jump_indexes[&jump.index];
                }
                ExtInstruction::ForIter(jump)
                | ExtInstruction::JumpForward(jump)
                | ExtInstruction::SetupFinally(jump)
                | ExtInstruction::SetupWith(jump)
                | ExtInstruction::SetupAsyncWith(jump) => {
                    let interval = Interval::new(
                        std::ops::Bound::Included(idx as u32 + 1),
                        std::ops::Bound::Included(idx as u32 + jump.index),
                    );

                    // dbg!(
                    //     relative_jump_indexes.query(&interval).collect::<Vec<_>>(),
                    //     &interval
                    // );

                    if cfg!(debug_assertions) {
                        let indexes = relative_jump_indexes
                            .query(&interval)
                            .filter(|entry| *entry.interval() == interval)
                            .collect::<Vec<_>>();

                        assert!(indexes.len() == 1);

                        jump.index = *indexes
                            .first()
                            .expect("This interval should always exist")
                            .value();
                    } else {
                        // This is faster, so use for release builds
                        jump.index = *relative_jump_indexes
                            .query(&interval)
                            .find(|entry| *entry.interval() == interval)
                            .expect("This interval should always exist")
                            .value();
                    }
                }
                _ => {}
            }
        }

        ext_instructions
    }
}

impl From<&[ExtInstruction]> for ExtInstructions {
    fn from(value: &[ExtInstruction]) -> Self {
        let mut instructions = ExtInstructions(Vec::with_capacity(value.len()));

        instructions.append_instructions(value);

        instructions
    }
}

impl From<(Opcode, u32)> for ExtInstruction {
    fn from(value: (Opcode, u32)) -> Self {
        match value.0 {
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
            Opcode::BUILD_SLICE => ExtInstruction::BuildSlice(value.1),
            Opcode::FORMAT_VALUE => ExtInstruction::FormatValue(value.1.into()),
            Opcode::MATCH_CLASS => ExtInstruction::MatchClass(value.1),
            Opcode::GEN_START => ExtInstruction::GenStart(value.1.into()),
            Opcode::ROT_N => ExtInstruction::RotN(value.1),
            Opcode::EXTENDED_ARG => panic!(
                "Extended arg can never be turned into an instruction. This should never happen."
            ), // ExtendedArg is handled separately
        }
    }
}

impl ExtInstruction {
    pub fn get_opcode(&self) -> Opcode {
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

    fn get_raw_value(&self) -> u32 {
        match &self {
            ExtInstruction::PopTop(invalid_arg)
            | ExtInstruction::RotTwo(invalid_arg)
            | ExtInstruction::RotThree(invalid_arg)
            | ExtInstruction::DupTop(invalid_arg)
            | ExtInstruction::DupTopTwo(invalid_arg)
            | ExtInstruction::RotFour(invalid_arg)
            | ExtInstruction::Nop(invalid_arg)
            | ExtInstruction::UnaryPositive(invalid_arg)
            | ExtInstruction::UnaryNegative(invalid_arg)
            | ExtInstruction::UnaryNot(invalid_arg)
            | ExtInstruction::UnaryInvert(invalid_arg)
            | ExtInstruction::BinaryMatrixMultiply(invalid_arg)
            | ExtInstruction::InplaceMatrixMultiply(invalid_arg)
            | ExtInstruction::BinaryPower(invalid_arg)
            | ExtInstruction::BinaryMultiply(invalid_arg)
            | ExtInstruction::BinaryModulo(invalid_arg)
            | ExtInstruction::BinaryAdd(invalid_arg)
            | ExtInstruction::BinarySubtract(invalid_arg)
            | ExtInstruction::BinarySubscr(invalid_arg)
            | ExtInstruction::BinaryFloorDivide(invalid_arg)
            | ExtInstruction::BinaryTrueDivide(invalid_arg)
            | ExtInstruction::InplaceFloorDivide(invalid_arg)
            | ExtInstruction::InplaceTrueDivide(invalid_arg)
            | ExtInstruction::GetLen(invalid_arg)
            | ExtInstruction::MatchMapping(invalid_arg)
            | ExtInstruction::MatchSequence(invalid_arg)
            | ExtInstruction::MatchKeys(invalid_arg)
            | ExtInstruction::CopyDictWithoutKeys(invalid_arg)
            | ExtInstruction::WithExceptStart(invalid_arg)
            | ExtInstruction::GetAiter(invalid_arg)
            | ExtInstruction::GetAnext(invalid_arg)
            | ExtInstruction::BeforeAsyncWith(invalid_arg)
            | ExtInstruction::EndAsyncFor(invalid_arg)
            | ExtInstruction::InplaceAdd(invalid_arg)
            | ExtInstruction::InplaceSubtract(invalid_arg)
            | ExtInstruction::InplaceMultiply(invalid_arg)
            | ExtInstruction::InplaceModulo(invalid_arg)
            | ExtInstruction::StoreSubscr(invalid_arg)
            | ExtInstruction::DeleteSubscr(invalid_arg)
            | ExtInstruction::BinaryLshift(invalid_arg)
            | ExtInstruction::BinaryRshift(invalid_arg)
            | ExtInstruction::BinaryAnd(invalid_arg)
            | ExtInstruction::BinaryXor(invalid_arg)
            | ExtInstruction::BinaryOr(invalid_arg)
            | ExtInstruction::InplacePower(invalid_arg)
            | ExtInstruction::GetIter(invalid_arg)
            | ExtInstruction::GetYieldFromIter(invalid_arg)
            | ExtInstruction::PrintExpr(invalid_arg)
            | ExtInstruction::LoadBuildClass(invalid_arg)
            | ExtInstruction::YieldFrom(invalid_arg)
            | ExtInstruction::GetAwaitable(invalid_arg)
            | ExtInstruction::LoadAssertionError(invalid_arg)
            | ExtInstruction::InplaceLshift(invalid_arg)
            | ExtInstruction::InplaceRshift(invalid_arg)
            | ExtInstruction::InplaceAnd(invalid_arg)
            | ExtInstruction::InplaceXor(invalid_arg)
            | ExtInstruction::InplaceOr(invalid_arg)
            | ExtInstruction::ListToTuple(invalid_arg)
            | ExtInstruction::ReturnValue(invalid_arg)
            | ExtInstruction::ImportStar(invalid_arg)
            | ExtInstruction::SetupAnnotations(invalid_arg)
            | ExtInstruction::YieldValue(invalid_arg)
            | ExtInstruction::PopBlock(invalid_arg)
            | ExtInstruction::PopExcept(invalid_arg) => invalid_arg.0,
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
            | ExtInstruction::BuildSlice(n)
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
}
