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
pub trait GenericOpcode: PartialEq {
    fn is_jump(&self) -> bool;
    fn is_absolute_jump(&self) -> bool;
    fn is_relative_jump(&self) -> bool;
    fn is_jump_forwards(&self) -> bool;
    fn is_jump_backwards(&self) -> bool;
    fn is_extended_arg(&self) -> bool;
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
pub trait GenericInstruction<OpargType>
where
    OpargType: PartialEq,
{
    type Opcode: GenericOpcode;

    fn get_opcode(&self) -> Self::Opcode;

    fn get_raw_value(&self) -> OpargType;

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

    fn is_extended_arg(&self) -> bool {
        self.get_opcode().is_extended_arg()
    }
}
