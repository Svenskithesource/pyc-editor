use std::ops::{Deref, DerefMut};

pub trait InstructionAccess
where
    Self: Deref<Target = [Self::Instruction]> + DerefMut<Target = [Self::Instruction]>,
{
    type Instruction;

    fn get_instructions(&self) -> &[Self::Instruction] {
        self.deref()
    }

    fn get_instructions_mut(&mut self) -> &mut [Self::Instruction] {
        self.deref_mut()
    }

    fn to_bytes(&self) -> Vec<u8>;
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
