use std::collections::HashMap;


/// The amount of extended_args necessary to represent the arg.
/// This is more efficient than `get_extended_args` as we only calculate the count and the actual values.
pub fn get_extended_args_count(arg: u32) -> u8 {
    if arg <= u8::MAX.into() {
        0
    } else if arg <= u16::MAX.into() {
        1
    } else if arg <= 0xffffff {
        2
    } else {
        3
    }
}

/// Used to represent opargs for opcodes that don't require arguments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UnusedArgument(pub u32);

impl From<u32> for UnusedArgument {
    fn from(value: u32) -> Self {
        UnusedArgument(value)
    }
}

/// Used to represent stack operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackEffect {
    pub pushes: u32,
    pub pops: u32,
}

/// Offsets are for instructions (not bytes)
#[derive(Debug, Clone, PartialEq)]
pub struct ExceptionTableEntry {
    pub start: u32,
    pub end: u32,
    pub target: u32,
    /// Stack depth at the start of the try block
    pub depth: u32,
    /// Whether to push the index of the last executed instruction
    pub lasti: bool,
}

impl StackEffect {
    /// Creates a StackEffect with equal pushes and pops.
    pub fn balanced(count: u32) -> Self {
        StackEffect {
            pushes: count,
            pops: count,
        }
    }

    /// Creates a StackEffect when only pushing
    pub fn push(count: u32) -> Self {
        StackEffect {
            pushes: count,
            pops: 0,
        }
    }

    /// Creates a StackEffect when only pushing
    pub fn pop(count: u32) -> Self {
        StackEffect {
            pushes: 0,
            pops: count,
        }
    }

    /// For when there is no stack access
    pub fn zero() -> Self {
        StackEffect { pushes: 0, pops: 0 }
    }

    /// Calculates the net total for the stackeffect
    pub fn net_total(&self) -> i32 {
        self.pushes as i32 - self.pops as i32
    }
}

#[macro_export]
macro_rules! define_default_traits {
    ($variant:ident, Instruction) => {
        impl Deref for $crate::$variant::instructions::Instructions {
            type Target = [$crate::$variant::instructions::Instruction];

            /// Allow the user to get a reference slice to the instructions
            fn deref(&self) -> &Self::Target {
                self.0.deref()
            }
        }

        impl DerefMut for $crate::$variant::instructions::Instructions {
            /// Allow the user to get a mutable reference slice for making modifications to existing instructions.
            fn deref_mut(&mut self) -> &mut [$crate::$variant::instructions::Instruction] {
                self.0.deref_mut()
            }
        }

        impl AsRef<[$crate::$variant::instructions::Instruction]>
            for $crate::$variant::instructions::Instructions
        {
            fn as_ref(&self) -> &[$crate::$variant::instructions::Instruction] {
                &self.0
            }
        }

        impl From<$crate::$variant::instructions::Instructions> for Vec<u8> {
            fn from(val: $crate::$variant::instructions::Instructions) -> Self {
                val.to_bytes()
            }
        }

        impl TryFrom<&[u8]> for $crate::$variant::instructions::Instructions {
            type Error = Error;
            fn try_from(code: &[u8]) -> Result<Self, Self::Error> {
                if code.len() % 2 != 0 {
                    return Err(Error::InvalidBytecodeLength);
                }

                let mut instructions = $crate::$variant::instructions::Instructions(
                    Vec::with_capacity(code.len() / 2),
                );

                for chunk in code.chunks(2) {
                    if chunk.len() != 2 {
                        return Err(Error::InvalidBytecodeLength);
                    }
                    let opcode = Opcode::from(chunk[0]);
                    let arg = chunk[1];

                    instructions.append_instruction((opcode, arg).into());
                }

                Ok(instructions)
            }
        }

        impl From<&[Instruction]> for Instructions {
            fn from(value: &[Instruction]) -> Self {
                $crate::$variant::instructions::Instructions::new(value.to_vec())
            }
        }

        impl InstructionsOwned<$crate::$variant::instructions::Instruction>
            for $crate::$variant::instructions::Instructions
        {
            type Instruction = $crate::$variant::instructions::Instruction;

            fn push(&mut self, item: Self::Instruction) {
                self.0.push(item);
            }
        }

        // impl<T> SimpleInstructionAccess<$crate::$variant::instructions::Instruction> for T where
        //     T: Deref<Target = [Instruction]> + AsRef<[Instruction]>
        // {
        // }
    };

    ($variant:ident, ExtInstruction) => {
        impl Deref for $crate::$variant::ext_instructions::ExtInstructions {
            type Target = [$crate::$variant::ext_instructions::ExtInstruction];

            /// Allow the user to get a reference slice to the instructions
            fn deref(&self) -> &Self::Target {
                self.0.deref()
            }
        }

        impl DerefMut for $crate::$variant::ext_instructions::ExtInstructions {
            /// Allow the user to get a mutable reference slice for making modifications to existing instructions.
            fn deref_mut(&mut self) -> &mut [$crate::$variant::ext_instructions::ExtInstruction] {
                self.0.deref_mut()
            }
        }

        impl AsRef<[$crate::$variant::ext_instructions::ExtInstruction]>
            for $crate::$variant::ext_instructions::ExtInstructions
        {
            fn as_ref(&self) -> &[$crate::$variant::ext_instructions::ExtInstruction] {
                &self.0
            }
        }

        impl From<$crate::$variant::ext_instructions::ExtInstructions> for Vec<u8> {
            fn from(val: $crate::$variant::ext_instructions::ExtInstructions) -> Self {
                val.to_bytes()
            }
        }

        impl TryFrom<&[$crate::$variant::instructions::Instruction]>
            for $crate::$variant::ext_instructions::ExtInstructions
        {
            type Error = Error;

            fn try_from(
                value: &[$crate::$variant::instructions::Instruction],
            ) -> Result<Self, Self::Error> {
                $crate::$variant::ext_instructions::ExtInstructions::from_instructions(value)
            }
        }

        impl From<&[$crate::$variant::ext_instructions::ExtInstruction]>
            for $crate::$variant::ext_instructions::ExtInstructions
        {
            fn from(value: &[$crate::$variant::ext_instructions::ExtInstruction]) -> Self {
                $crate::$variant::ext_instructions::ExtInstructions::new(value.to_vec())
            }
        }
    };
}

pub fn generate_var_name(
    stack_name: &'static str,
    names: &mut HashMap<&'static str, u32>,
) -> String {
    if names.contains_key(stack_name) {
        *names.get_mut(stack_name).unwrap() += 1;
    } else {
        names.insert(stack_name, 0);
    }

    format!("{}_{}", stack_name, names[stack_name])
}
