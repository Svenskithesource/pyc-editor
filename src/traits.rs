use std::{collections::HashMap, ops::DerefMut};

pub trait InstructionAccess<OpargType, I>
where
    Self: AsRef<[Self::Instruction]>,
    OpargType: PartialEq,
{
    type Instruction: GenericInstruction<OpargType>;
    type Jump;

    fn get_instructions(&self) -> &[Self::Instruction] {
        self.as_ref()
    }

    /// Returns the index and the instruction of the jump target. None if the index is not a valid jump.
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
    fn find_ext_arg_jumps(instructions: &Self) -> Vec<u32> {
        let mut jumps: Vec<u32> = vec![];

        for (index, instruction) in instructions.as_ref().iter().enumerate() {
            if instruction.is_jump() {
                let jump_target = instructions.get_jump_target(index as u32);

                // Jump target is valid
                if let Some(jump) = jump_target {
                    // The jump target has a value bigger than 1 byte, this means we skipped an extended arg
                    if instructions
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

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytearray = Vec::with_capacity(self.as_ref().len() * 2);

        for instruction in self.as_ref().iter() {
            bytearray.push(instruction.get_opcode().into());
            bytearray.push(instruction.get_raw_value())
        }

        bytearray
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
pub trait GenericOpcode: PartialEq + Into<u8> {
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
