use std::{collections::HashMap, ops::DerefMut};

use crate::{
    error::Error,
    utils::{self, ExceptionTableEntry, StackEffect},
};

pub trait InstructionAccess<OpargType, I>
where
    Self: AsRef<[Self::Instruction]>,
    OpargType: PartialEq,
{
    type Instruction: GenericInstruction<OpargType> + std::fmt::Debug;
    type Jump;

    fn get_instructions(&self) -> &[Self::Instruction] {
        self.as_ref()
    }

    /// Returns the index and the instruction of the jump target. None if the index is not a valid jump.
    // TODO: Create a bounded version of this function
    fn get_jump_target(&self, index: u32) -> Option<(u32, Self::Instruction)>;

    /// Returns a list of all indexes that jump to the given index
    fn get_jump_xrefs(&self, index: u32) -> Vec<u32> {
        let jump_map = self.get_jump_map();

        jump_map
            .iter()
            .filter(|(_, to)| **to == index)
            .map(|(from, _)| *from)
            .collect()
    }

    /// Returns a hashmap of jump indexes and their jump target
    fn get_jump_map(&self) -> HashMap<u32, u32> {
        let mut jump_map: HashMap<u32, u32> = HashMap::new();

        for index in 0..self.as_ref().len() {
            let jump_target = self.get_jump_target(index as u32);

            if let Some((jump_index, _)) = jump_target {
                jump_map.insert(index as u32, jump_index);
            }
        }

        jump_map
    }

    fn get_jump_value(&self, index: u32) -> Option<Self::Jump>;
}

pub trait SimpleInstructionAccess<I>
where
    Self: InstructionAccess<u8, I> + AsRef<[Self::Instruction]>,
{
    /// This finds jumps that jump to instructions after an extended arg. This is a very unique case.
    /// This kind of bytecode should never be emitted by the Python compiler but it's possible for custom bytecode to have this.
    /// Returns a list of indexes of jump instructions that have a jump target like this.
    fn find_ext_arg_jumps(&self) -> Vec<u32> {
        let mut jumps: Vec<u32> = vec![];

        for (index, instruction) in self.as_ref().iter().enumerate() {
            if instruction.is_jump() {
                let jump_target = self.get_jump_target(index as u32);

                // Jump target is valid
                if let Some(jump) = jump_target {
                    // The jump target has a value bigger than 1 byte, this means we skipped an extended arg
                    if self
                        .get_full_arg(jump.0 as usize)
                        .expect("We know the index is valid")
                        > u8::MAX.into()
                    {
                        jumps.push(index as u32);
                    }
                }
            }
        }

        jumps
    }

    /// Calculates the full argument for an index (keeping in mind extended args). None if the index is not within bounds.
    /// NOTE: If there is a jump skipping the extended arg(s) before this instruction, this will return an incorrect value.
    fn get_full_arg(&self, index: usize) -> Option<u32> {
        if self.as_ref().len() > index {
            let mut curr_index = index;
            let mut extended_args = vec![];

            while curr_index > 0 {
                curr_index -= 1;

                if self.as_ref()[curr_index].is_extended_arg() {
                    extended_args.push(self.as_ref()[curr_index].get_raw_value());
                } else {
                    break;
                }
            }

            let mut extended_arg = 0;

            for arg in extended_args.iter().rev() {
                // We collected them in the reverse order
                let arg = *arg as u32 | extended_arg;
                extended_arg = arg << 8;
            }

            Some(self.as_ref()[index].get_raw_value() as u32 | extended_arg)
        } else {
            None
        }
    }

    /// The same as `get_full_arg` but you can specify a lower limit while searching for extended args.
    /// NOTE: If there is a jump skipping the extended arg(s) before this instruction, this will return an incorrect value.
    fn get_full_arg_bounded(&self, index: usize, lower_bound: usize) -> Option<u32> {
        if self.as_ref().len() > index {
            let mut curr_index = index;
            let mut extended_args = vec![];

            while curr_index > lower_bound {
                curr_index -= 1;

                if self.as_ref()[curr_index].is_extended_arg() {
                    extended_args.push(self.as_ref()[curr_index].get_raw_value());
                } else {
                    break;
                }
            }

            let mut extended_arg = 0;

            for arg in extended_args.iter().rev() {
                // We collected them in the reverse order
                let arg = *arg as u32 | extended_arg;
                extended_arg = arg << 8;
            }

            Some(self.as_ref()[index].get_raw_value() as u32 | extended_arg)
        } else {
            None
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytearray = Vec::with_capacity(self.as_ref().len() * 2);

        for instruction in self.as_ref().iter() {
            bytearray.push(instruction.get_opcode().into());
            bytearray.push(instruction.get_raw_value())
        }

        bytearray
    }

    /// Calculates the maximum stack size the instructions will use. `start_stacksize` is only used for generator code objects that start with a stacksize of 1.
    /// For 3.11+ you should also pass the exception table for correct stack calculation.
    fn max_stack_size(
        &self,
        start_stacksize: u32,
        exception_table: Option<Vec<ExceptionTableEntry>>,
    ) -> Result<u32, Error> {
        // Contains starting indexes of blocks that need to be processed along with their starting stack size.
        let mut block_queue = vec![(start_stacksize, 0usize)];
        let mut max_stack_size: u32 = 0;
        let mut visited: Vec<(u32, usize)> = Vec::new();

        if let Some(exception_entries) = exception_table {
            for exception in exception_entries {
                // dbg!(&exception);
                block_queue.push((
                    exception.depth + 1 + if exception.lasti { 1 } else { 0 },
                    exception.target as usize,
                ));
            }
        }

        while let Some((stack_size, start_index)) = block_queue.pop() {
            if visited.contains(&(stack_size, start_index)) {
                // already analyzed
                continue;
            }

            // dbg!("new block", stack_size, start_index);

            visited.push((stack_size, start_index));

            let mut curr_stack_size = stack_size;

            if curr_stack_size >= max_stack_size {
                max_stack_size = curr_stack_size;
            }

            for instruction_index in start_index..self.as_ref().len() {
                let instruction = self.as_ref().get(instruction_index).unwrap();
                let arg = self
                    .get_full_arg_bounded(instruction_index, start_index)
                    .unwrap();

                if instruction.is_jump() || instruction.stops_execution() {
                    // Not the last instruction
                    if instruction_index != self.as_ref().len() - 1
                        && instruction.is_conditional_jump()
                        && !instruction.stops_execution()
                    {
                        // Block for not taking the jump
                        let stack_effect = instruction.stack_effect(arg, Some(false)).net_total();

                        // dbg!(curr_stack_size, stack_effect, instruction);
                        let (stack_size, indx) = (
                            curr_stack_size.checked_add_signed(stack_effect).ok_or(
                                Error::InvalidStacksize(curr_stack_size as i32 + stack_effect),
                            )?,
                            instruction_index + 1,
                        );

                        block_queue.push((stack_size, indx));
                    }

                    if let Some((jump_index, _)) = self.get_jump_target(instruction_index as u32) {
                        // Block for valid jump target
                        let stack_effect = instruction.stack_effect(arg, Some(true)).net_total();

                        // dbg!(curr_stack_size, stack_effect, instruction, jump_index);
                        let (stack_size, indx) = (
                            curr_stack_size.checked_add_signed(stack_effect).ok_or(
                                Error::InvalidStacksize(curr_stack_size as i32 + stack_effect),
                            )?,
                            jump_index as usize,
                        );

                        block_queue.push((stack_size, indx));
                    }

                    // Process new block, as this one has ended
                    break;
                } else {
                    let stack_effect = instruction.stack_effect(arg, None).net_total();

                    // dbg!(curr_stack_size, stack_effect, instruction);
                    curr_stack_size = curr_stack_size.checked_add_signed(stack_effect).ok_or(
                        Error::InvalidStacksize(curr_stack_size as i32 + stack_effect),
                    )?;

                    if curr_stack_size >= max_stack_size {
                        max_stack_size = curr_stack_size;
                    }
                }
            }
        }

        Ok(max_stack_size)
    }
}

pub trait ExtInstructionAccess<I> {
    type Instructions: SimpleInstructionAccess<I>;

    /// Convert the resolved instructions back into instructions with extended args.
    fn to_instructions(&self) -> Self::Instructions;

    fn to_bytes(&self) -> Vec<u8> {
        self.to_instructions().to_bytes()
    }
}

pub trait InstructionsOwned<T>
where
    Self: DerefMut<Target = [Self::Instruction]>,
    T: Copy,
{
    type Instruction;

    fn push(&mut self, item: T);

    fn get_instructions_mut(&mut self) -> &mut [Self::Instruction] {
        self.deref_mut()
    }

    /// Append multiple instructions at the end
    fn append_instructions(&mut self, instructions: &[T]) {
        for instruction in instructions {
            self.push(*instruction);
        }
    }
    /// Append an instruction at the end
    fn append_instruction(&mut self, instruction: T) {
        self.push(instruction);
    }
}

pub trait ExtInstructionsOwned<T>
where
    Self: DerefMut<Target = [Self::Instruction]>,
    Self::Instruction: Copy,
{
    type Instruction;

    /// Delete instruction at index
    fn delete_instruction(&mut self, index: usize);

    /// Delete instructions in range (ex. 1..10)
    fn delete_instructions(&mut self, range: std::ops::Range<usize>) {
        range
            .into_iter()
            .for_each(|index| self.delete_instruction(index));
    }

    /// Insert instruction at a specific index. It automatically fixes jump offsets in other instructions.
    fn insert_instruction(&mut self, index: usize, instruction: Self::Instruction);

    /// Insert a slice of instructions at an index
    fn insert_instructions(&mut self, index: usize, instructions: &[Self::Instruction]) {
        for (idx, instruction) in instructions.iter().enumerate() {
            self.insert_instruction(index + idx, *instruction);
        }
    }
}

/// Generic opcode functions each version has to implement
pub trait GenericOpcode: PartialEq + Into<u8> {
    fn is_jump(&self) -> bool;
    fn is_absolute_jump(&self) -> bool;
    fn is_relative_jump(&self) -> bool;
    fn is_jump_forwards(&self) -> bool;
    fn is_jump_backwards(&self) -> bool;
    fn is_conditional_jump(&self) -> bool;
    fn stops_execution(&self) -> bool;
    fn is_extended_arg(&self) -> bool;

    /// If the code has a jump target and `jump` is true, `stack_effect()` will return the stack effect of jumping.
    /// If jump is false, it will return the stack effect of not jumping.
    /// And if jump is None, it will return the maximal stack effect of both cases.
    fn stack_effect(&self, oparg: u32, jump: Option<bool>) -> StackEffect;
}

pub trait Oparg<T> {
    const MAX: Self;

    fn new(value: T) -> Self;
}

impl Oparg<u8> for u8 {
    const MAX: Self = u8::MAX;

    fn new(value: u8) -> Self {
        value
    }
}

impl Oparg<u32> for u32 {
    const MAX: Self = u32::MAX;

    fn new(value: u32) -> Self {
        value
    }
}

/// Generic instruction functions used by all versions
pub trait GenericInstruction<OpargType>: PartialEq
where
    OpargType: PartialEq,
{
    type Opcode: GenericOpcode;

    fn get_opcode(&self) -> Self::Opcode;

    fn get_raw_value(&self) -> OpargType;

    /// Relative or absolute jump
    fn is_jump(&self) -> bool {
        self.get_opcode().is_jump()
    }

    fn is_absolute_jump(&self) -> bool {
        self.get_opcode().is_absolute_jump()
    }

    fn is_relative_jump(&self) -> bool {
        self.get_opcode().is_relative_jump()
    }

    fn is_jump_forwards(&self) -> bool {
        self.get_opcode().is_jump_forwards()
    }

    fn is_jump_backwards(&self) -> bool {
        self.get_opcode().is_jump_backwards()
    }

    fn is_conditional_jump(&self) -> bool {
        self.get_opcode().is_conditional_jump()
    }

    fn stops_execution(&self) -> bool {
        self.get_opcode().stops_execution()
    }

    fn is_extended_arg(&self) -> bool {
        self.get_opcode().is_extended_arg()
    }

    /// If the code has a jump target and `jump` is true, `stack_effect()` will return the stack effect of jumping.
    /// If jump is false, it will return the stack effect of not jumping.
    /// And if jump is None, it will return the maximal stack effect of both cases.
    fn stack_effect(&self, oparg: u32, jump: Option<bool>) -> StackEffect {
        self.get_opcode().stack_effect(oparg, jump)
    }
}

#[cfg(all(test, feature = "v311"))]
mod test {
    use crate::{traits::SimpleInstructionAccess, v310::instructions};

    #[test]
    fn test_invalid_extended_arg_jump() {
        let instructions = crate::v311::instructions::Instructions::new(vec![
            crate::v311::instructions::Instruction::JumpForward(1),
            crate::v311::instructions::Instruction::ExtendedArg(1),
            crate::v311::instructions::Instruction::Nop(1),
        ]);

        assert_eq!(instructions.find_ext_arg_jumps().iter().count(), 1)
    }

    #[test]
    fn test_stack_size() {
        let instructions = crate::v311::instructions::Instructions::new(vec![
            crate::v311::instructions::Instruction::Resume(0),
            crate::v311::instructions::Instruction::PushNull(0),
            crate::v311::instructions::Instruction::LoadName(0),
            crate::v311::instructions::Instruction::LoadConst(0),
            crate::v311::instructions::Instruction::Precall(1),
            crate::v311::instructions::Instruction::Cache(0),
            crate::v311::instructions::Instruction::Call(1),
            crate::v311::instructions::Instruction::Cache(0),
            crate::v311::instructions::Instruction::Cache(0),
            crate::v311::instructions::Instruction::Cache(0),
            crate::v311::instructions::Instruction::Cache(0),
            crate::v311::instructions::Instruction::PopTop(0),
            crate::v311::instructions::Instruction::LoadConst(1),
            crate::v311::instructions::Instruction::StoreName(1),
            crate::v311::instructions::Instruction::PushNull(0),
            crate::v311::instructions::Instruction::LoadName(0),
            crate::v311::instructions::Instruction::LoadConst(2),
            crate::v311::instructions::Instruction::LoadName(1),
            crate::v311::instructions::Instruction::FormatValue(2),
            crate::v311::instructions::Instruction::BuildString(2),
            crate::v311::instructions::Instruction::Precall(1),
            crate::v311::instructions::Instruction::Cache(0),
            crate::v311::instructions::Instruction::Call(1),
            crate::v311::instructions::Instruction::Cache(0),
            crate::v311::instructions::Instruction::Cache(0),
            crate::v311::instructions::Instruction::Cache(0),
            crate::v311::instructions::Instruction::Cache(0),
            crate::v311::instructions::Instruction::PopTop(0),
            crate::v311::instructions::Instruction::LoadConst(3),
            crate::v311::instructions::Instruction::ReturnValue(0),
        ]);

        assert_eq!(instructions.max_stack_size(0, None).unwrap(), 4);
    }
}
