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
                    Opcode::INVALID_OPCODE(opcode) => Instruction::InvalidOpcode((opcode, value.1)),
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
