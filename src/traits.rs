use std::{collections::HashMap, ops::{Deref, DerefMut}};

pub trait InstructionAccess
where
    Self: AsRef<[Self::Instruction]>,
{
    type Instruction: GenericInstruction;
    type Jump;

    fn get_instructions(&self) -> &[Self::Instruction] {
        self.as_ref()
    }

    fn to_bytes(&self) -> Vec<u8>;

    /// Returns the index and the instruction of the jump target. None if the index is not a valid jump.
    fn get_jump_target(&self, index: u32) -> Option<(u32, Self::Instruction)>;

    /// Returns a list of all indexes that jump to the given index
    fn get_jump_xrefs(&self, index: u32) -> Vec<u32>;

    /// Returns a hashmap of jump indexes and their jump target
    fn get_jump_map(&self) -> HashMap<u32, u32>;

    fn get_jump_value(&self, index: u32) -> Option<Self::Jump>;
}

pub trait SimpleInstructionAccess
where
    Self: Deref<Target = [Self::Instruction]>,
{
    type Instruction: GenericInstruction;

    /// This finds jumps that jump to instructions after an extended arg. This is a very unique case.
    /// This kind of bytecode should never be emitted by the Python compiler but it's possible for custom bytecode to have this.
    /// Returns a list of indexes of jump instructions that have a jump target like this.
    fn find_ext_arg_jumps(instructions: &[Self::Instruction]) -> Vec<u32>;

    /// Calculates the full argument for an index (keeping in mind extended args). None if the index is not within bounds.
    /// NOTE: If there is a jump skipping the extended arg(s) before this instruction, this will return an incorrect value.
    fn get_full_arg(&self, index: usize) -> Option<u32>;
}

pub trait ExtInstructionAccess
{
    type Instructions;

    /// Convert the resolved instructions back into instructions with extended args.
    fn to_instructions(&self) -> Self::Instructions;
}


pub trait InstructionMutAccess
where
    Self: DerefMut<Target = [Self::Instruction]>,
{
    type Instruction;

    fn get_instructions_mut(&mut self) -> &mut [Self::Instruction] {
        self.deref_mut()
    }
}

/// Generic opcode functions each version has to implement
pub trait GenericOpcode {
    fn is_jump(&self) -> bool;
    fn is_absolute_jump(&self) -> bool;
    fn is_relative_jump(&self) -> bool;
}

/// Generic instruction functions used by all versions
pub trait GenericInstruction {
    type Opcode: GenericOpcode;
    type Arg;

    fn get_opcode(&self) -> Self::Opcode;

    fn get_raw_value(&self) -> Self::Arg;

    fn is_jump(&self) -> bool {
        self.get_opcode().is_jump()
    }

    fn is_absolute_jump(&self) -> bool {
        self.get_opcode().is_absolute_jump()
    }

    fn is_relative_jump(&self) -> bool {
        self.get_opcode().is_relative_jump()
    }
}
