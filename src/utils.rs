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

#[macro_export]
macro_rules! define_opcodes {
    (
        $(
            $name:ident = $value:expr
        ),* $(,)?
    ) => {
        #[allow(non_camel_case_types)]
        #[allow(clippy::upper_case_acronyms)]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum Opcode {
            $( $name ),*,
            INVALID_OPCODE(u8),
        }

        impl From<u8> for Opcode {
            fn from(value: u8) -> Self {
                match value {
                    $( $value => Opcode::$name, )*
                    _ => Opcode::INVALID_OPCODE(value),
                }
            }
        }

        impl From<Opcode> for u8 {
            fn from(value: Opcode) -> Self {
                match value {
                    $( Opcode::$name => $value , )*
                    Opcode::INVALID_OPCODE(value) => value,
                }
            }
        }

        impl From<(Opcode, u8)> for Instruction {
            fn from(value: (Opcode, u8)) -> Self {
                match value.0 {
                    $(
                        Opcode::$name => define_opcodes!(@instruction $name, value.1),
                    )*
                    Opcode::INVALID_OPCODE(opcode) => {
                        if !cfg!(test) {
                            Instruction::InvalidOpcode((opcode, value.1))
                        } else {
                            panic!("Testing environment should not come across invalid opcodes")
                        }
                    },
                }
            }
        }

        impl Opcode {
            pub fn from_instruction(instruction: &Instruction) -> Self {
                match instruction {
                    $(
                        define_opcodes!(@instruction $name) => Opcode::$name ,
                    )*
                    Instruction::InvalidOpcode((opcode, _)) => Opcode::INVALID_OPCODE(*opcode),
                }
            }
        }
    };

    (@instruction $variant:ident, $val:expr) => {
        paste! { Instruction::[<$variant:camel>]($val) }
    };

    (@instruction $variant:ident) => {
        paste! { Instruction::[<$variant:camel>](_) }
    };
}

#[macro_export]
macro_rules! define_default_traits {
    ($variant:ident, Instruction) => {
        impl Deref for crate::$variant::instructions::Instructions {
            type Target = [crate::$variant::instructions::Instruction];

            /// Allow the user to get a reference slice to the instructions
            fn deref(&self) -> &Self::Target {
                self.0.deref()
            }
        }

        impl DerefMut for crate::$variant::instructions::Instructions {
            /// Allow the user to get a mutable reference slice for making modifications to existing instructions.
            fn deref_mut(&mut self) -> &mut [crate::$variant::instructions::Instruction] {
                self.0.deref_mut()
            }
        }

        impl AsRef<[crate::$variant::instructions::Instruction]>
            for crate::$variant::instructions::Instructions
        {
            fn as_ref(&self) -> &[crate::$variant::instructions::Instruction] {
                &self.0
            }
        }

        impl From<crate::$variant::instructions::Instructions> for Vec<u8> {
            fn from(val: crate::$variant::instructions::Instructions) -> Self {
                val.to_bytes()
            }
        }

        impl TryFrom<&[u8]> for crate::$variant::instructions::Instructions {
            type Error = Error;
            fn try_from(code: &[u8]) -> Result<Self, Self::Error> {
                if code.len() % 2 != 0 {
                    return Err(Error::InvalidBytecodeLength);
                }

                let mut instructions =
                    crate::$variant::instructions::Instructions(Vec::with_capacity(code.len() / 2));

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
                crate::$variant::instructions::Instructions::new(value.to_vec())
            }
        }

        impl InstructionsOwned<crate::$variant::instructions::Instruction>
            for crate::$variant::instructions::Instructions
        {
            type Instruction = crate::$variant::instructions::Instruction;

            fn push(&mut self, item: Self::Instruction) {
                self.0.push(item);
            }
        }

        impl<T>
            SimpleInstructionAccess<
                crate::$variant::instructions::Instruction,
            > for T
        where
            T: Deref<Target = [Instruction]> + AsRef<[Instruction]>,
        {
        }
    };

    ($variant:ident, ExtInstruction) => {
        impl Deref for crate::$variant::ext_instructions::ExtInstructions {
            type Target = [crate::$variant::ext_instructions::ExtInstruction];

            /// Allow the user to get a reference slice to the instructions
            fn deref(&self) -> &Self::Target {
                self.0.deref()
            }
        }

        impl DerefMut for crate::$variant::ext_instructions::ExtInstructions {
            /// Allow the user to get a mutable reference slice for making modifications to existing instructions.
            fn deref_mut(&mut self) -> &mut [crate::$variant::ext_instructions::ExtInstruction] {
                self.0.deref_mut()
            }
        }

        impl AsRef<[crate::$variant::ext_instructions::ExtInstruction]>
            for crate::$variant::ext_instructions::ExtInstructions
        {
            fn as_ref(&self) -> &[crate::$variant::ext_instructions::ExtInstruction] {
                &self.0
            }
        }

        impl From<crate::$variant::ext_instructions::ExtInstructions> for Vec<u8> {
            fn from(val: crate::$variant::ext_instructions::ExtInstructions) -> Self {
                val.to_bytes()
            }
        }

        impl TryFrom<&[crate::$variant::instructions::Instruction]>
            for crate::$variant::ext_instructions::ExtInstructions
        {
            type Error = Error;

            fn try_from(
                value: &[crate::$variant::instructions::Instruction],
            ) -> Result<Self, Self::Error> {
                crate::$variant::ext_instructions::ExtInstructions::from_instructions(value)
            }
        }

        impl From<&[crate::$variant::ext_instructions::ExtInstruction]>
            for crate::$variant::ext_instructions::ExtInstructions
        {
            fn from(value: &[crate::$variant::ext_instructions::ExtInstruction]) -> Self {
                crate::$variant::ext_instructions::ExtInstructions::new(value.to_vec())
            }
        }
    };
}
