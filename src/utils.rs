/// The amount of extended_args necessary to represent the arg.
/// This is more efficient than `get_extended_args` as we only calculate the count and the actual values.
pub fn get_extended_args_count(arg: u32) -> u8 {
    if arg <= u16::MAX.into() {
        1
    } else if arg <= 0xffffff {
        2
    } else {
        3
    }
}

#[macro_export]
macro_rules! define_opcodes {
    (
        $(
            $name:ident = $value:expr
        ),* $(,)?
    ) => {
        #[repr(u8)]
        #[allow(non_camel_case_types)]
        #[allow(clippy::upper_case_acronyms)]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum Opcode {
            $( $name = $value ),*,
            INVALID_OPCODE,
        }

        impl TryFrom<u8> for Opcode {
            type Error = $crate::error::Error;
            fn try_from(value: u8) -> Result<Self, Self::Error> {
                match value {
                    $( $value => Ok(Opcode::$name), )*
                    _ => Err(Self::Error::UnkownOpcode(value)),
                }
            }
        }

        impl From<(Opcode, u8)> for Instruction {
            fn from(value: (Opcode, u8)) -> Self {
                match value.0 {
                    $(
                        Opcode::$name => define_opcodes!(@instruction $name, value.1),
                    )*
                    Opcode::INVALID_OPCODE => Instruction::InvalidOpcode(value.1),
                }
            }
        }

        impl Opcode {
            pub fn from_instruction(instruction: &Instruction) -> Self {
                match instruction {
                    $(
                        define_opcodes!(@instruction $name) => Opcode::$name ,
                    )*
                    Instruction::InvalidOpcode(_) => Opcode::INVALID_OPCODE,
                }
            }
        }
    };

    // Special cases that don't follow the automatic camel case conversion
    (@instruction CALL_FUNCTION_KW, $val:expr) => { Instruction::CallFunctionKW($val) };
    (@instruction CALL_FUNCTION_EX, $val:expr) => { Instruction::CallFunctionEx($val) };

    (@instruction $variant:ident, $val:expr) => {
        paste! { Instruction::[<$variant:camel>]($val) }
    };

    // Special cases that don't follow the automatic camel case conversion
    (@instruction CALL_FUNCTION_KW) => { Instruction::CallFunctionKW(_) };
    (@instruction CALL_FUNCTION_EX) => { Instruction::CallFunctionEx(_) };

    (@instruction $variant:ident) => {
        paste! { Instruction::[<$variant:camel>](_) }
    };

    // Special cases that don't follow the automatic camel case conversion
    (@ext_instruction CALL_FUNCTION_KW) => { ExtInstruction::CallFunctionKW(_) };
    (@ext_instruction CALL_FUNCTION_EX) => { ExtInstruction::CallFunctionEx(_) };

    (@ext_instruction $variant:ident) => {
        paste! { ExtInstruction::[<$variant:camel>](_) }
    };
}
