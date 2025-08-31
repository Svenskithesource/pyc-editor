pub mod cache;
pub mod code_objects;
pub mod ext_instructions;
pub mod instructions;
pub mod opcodes;

#[cfg(test)]
mod tests {
    use python_marshal::Kind::{ShortAscii, ShortAsciiInterned};
    use python_marshal::{CodeFlags, PyString};

    use crate::v312;
    use crate::v312::code_objects::CompareOperation::Equal;
    use crate::v312::code_objects::{
        Constant, FrozenConstant, LinetableEntry, NameIndex,
    };
    use crate::v312::ext_instructions::{ExtInstruction, ExtInstructions};
    use crate::v312::instructions::{
        get_line_number, starts_line_number, Instruction, Instructions,
    };
    use crate::v312::opcodes::Opcode;
    use crate::{load_pyc, prelude::*};

    use std::fs::File;
    use std::io::BufReader;

    #[test]
    fn test_load_pyc() {
        let file = File::open("tests/test.pyc").unwrap();
        let reader = BufReader::new(file);
        let original_pyc = python_marshal::load_pyc(reader).unwrap();
        let original_pyc = python_marshal::resolver::resolve_all_refs(
            &original_pyc.object,
            &original_pyc.references,
        )
        .0;

        let file = File::open("tests/test.pyc").unwrap();
        let reader = BufReader::new(file);
        let pyc_file = load_pyc(reader).unwrap();

        let pyc: python_marshal::PycFile = pyc_file.into();

        assert_eq!(original_pyc, pyc.object);
    }

    #[test]
    fn test_extended_arg() {
        let mut instructions: ExtInstructions = ([
            (Opcode::LOAD_NAME, 0).try_into().unwrap(),
            (Opcode::LOAD_CONST, 0).try_into().unwrap(),
            ExtInstruction::CompareOp(Equal),
            (Opcode::POP_JUMP_IF_TRUE, 0).try_into().unwrap(), // Jump to load_name
            (Opcode::LOAD_NAME, 1).try_into().unwrap(),
            (Opcode::LOAD_NAME, 2).try_into().unwrap(),
            (Opcode::CALL, 1).try_into().unwrap(),
            (Opcode::POP_TOP, 0).try_into().unwrap(),
            (Opcode::LOAD_CONST, 1).try_into().unwrap(),
            (Opcode::RETURN_VALUE, 0).try_into().unwrap(),
            (Opcode::LOAD_CONST, 1).try_into().unwrap(),
            (Opcode::RETURN_VALUE, 0).try_into().unwrap(),
        ]
        .as_slice())
        .into();

        instructions.insert_instructions(3, &vec![(Opcode::NOP, 0).try_into().unwrap(); 300]);

        let og_target = (304u32, ExtInstruction::LoadName(NameIndex { index: 1 }));

        let resolved = instructions.to_instructions().to_resolved();
        match resolved
            .iter()
            .enumerate()
            .find(|(_, instruction)| instruction.get_opcode() == Opcode::POP_JUMP_IF_TRUE)
            .expect("There must be a jump")
        {
            (index, ExtInstruction::PopJumpIfTrue(_)) => {
                let target = resolved
                    .get_jump_target(index as u32)
                    .expect("Should never fail");

                assert_eq!(og_target, target);
            }
            _ => panic!(),
        }

        assert_eq!(instructions, resolved);
    }

    #[test]
    fn test_relative_jump() {
        // Create a list of instructions that look like this:
        // 0    NOP           1
        // ...
        // 11   JUMP_FORWARD  5 (to 17 = 11 + 5 + 1)
        // 12   EXTENDED_ARG  1
        // 13   NOP           0
        // ...
        // 17  RETURN_VALUE  0
        let mut instructions = Instructions::with_capacity(17);

        // Fill instruction list with nops until index 10 (11 items)
        for _ in instructions.len()..11 {
            instructions.append_instruction(Instruction::Nop(0));
        }

        assert_eq!(instructions.len(), 11);

        instructions.append_instructions(
            vec![
                Instruction::JumpForward(5), // This jumps to index 17
                Instruction::ExtendedArg(1),
            ]
            .as_slice(),
        );

        // Fill instruction list with nops until index 16 (17 items)
        for _ in instructions.len()..17 {
            instructions.append_instruction(Instruction::Nop(0));
        }

        assert_eq!(instructions.len(), 17);

        instructions.append_instruction(Instruction::ReturnValue(0));

        let resolved = instructions.to_resolved();

        assert_eq!(resolved.len(), 17);

        assert_eq!(
            resolved.get_jump_target(11).unwrap().1,
            ExtInstruction::ReturnValue(0.into())
        );

        assert_eq!(
            instructions.get_jump_target(11).unwrap().1,
            Instruction::ReturnValue(0)
        );

        assert_eq!(instructions, resolved.to_instructions());
    }

    #[test]
    fn test_full_arg() {
        let ext_instructions = ExtInstructions::new(vec![
            ExtInstruction::Nop(300.into()), // This will need an extended arg, which will increase the offset above which also causes that to need an extended arg.
        ]);

        let instructions = ext_instructions.to_instructions();

        assert_eq!(
            ext_instructions.first().unwrap().get_raw_value(),
            instructions.get_full_arg(1).unwrap()
        );
    }

    #[test]
    fn test_line_number() {
        // 1.
        // 2. print("line 1")
        // 3. a = 2
        // 4.
        // 5. print(f"line 4, {a=}")

        let code_object = v312::code_objects::Code {
            argcount: 0,
            posonlyargcount: 0,
            kwonlyargcount: 0,
            stacksize: 4,
            flags: CodeFlags::from_bits_retain(0x0),
            code: Instructions::new(vec![
                Instruction::Resume(0),
                Instruction::PushNull(0),
                Instruction::LoadName(0),
                Instruction::LoadConst(0),
                Instruction::Call(1),
                Instruction::Cache(0),
                Instruction::Cache(0),
                Instruction::Cache(0),
                Instruction::PopTop(0),
                Instruction::LoadConst(1),
                Instruction::StoreName(1),
                Instruction::PushNull(0),
                Instruction::LoadName(0),
                Instruction::LoadConst(2),
                Instruction::LoadName(1),
                Instruction::FormatValue(2),
                Instruction::BuildString(2),
                Instruction::Call(1),
                Instruction::Cache(0),
                Instruction::Cache(0),
                Instruction::Cache(0),
                Instruction::PopTop(0),
                Instruction::ReturnConst(3),
            ]),
            consts: vec![
                Constant::FrozenConstant(FrozenConstant::String(PyString {
                    value: "line 1".into(),
                    kind: ShortAscii,
                })),
                Constant::FrozenConstant(FrozenConstant::Long(2.into())),
                Constant::FrozenConstant(FrozenConstant::String(PyString {
                    value: "line 4, a=".into(),
                    kind: ShortAscii,
                })),
                Constant::FrozenConstant(FrozenConstant::None),
            ],
            names: vec![
                PyString {
                    value: "print".into(),
                    kind: ShortAsciiInterned,
                },
                PyString {
                    value: "a".into(),
                    kind: ShortAsciiInterned,
                },
            ],
            localsplusnames: vec![],
            localspluskinds: vec![],
            filename: PyString {
                value: "test.py".into(),
                kind: ShortAscii,
            },
            name: PyString {
                value: "<module>".into(),
                kind: ShortAsciiInterned,
            },
            qualname: PyString {
                value: "<module>".into(),
                kind: ShortAsciiInterned,
            },
            firstlineno: 1,
            linetable: vec![
                240, 3, 1, 1, 1, 225, 0, 5, 128, 104, 132, 15, 216, 4, 5, 128, 1, 225, 0, 5, 136,
                11, 144, 17, 144, 4, 128, 111, 213, 0, 22,
            ],
            exceptiontable: vec![],
        };

        assert_eq!(
            code_object.co_lines().unwrap(),
            vec![
                LinetableEntry {
                    start: 0,
                    end: 2,
                    line_number: Some(0),
                    column_start: Some(1),
                    column_end: Some(1)
                },
                LinetableEntry {
                    start: 2,
                    end: 6,
                    line_number: Some(2),
                    column_start: Some(0),
                    column_end: Some(5)
                },
                LinetableEntry {
                    start: 6,
                    end: 8,
                    line_number: Some(2),
                    column_start: Some(6),
                    column_end: Some(14)
                },
                LinetableEntry {
                    start: 8,
                    end: 18,
                    line_number: Some(2),
                    column_start: Some(0),
                    column_end: Some(15)
                },
                LinetableEntry {
                    start: 18,
                    end: 20,
                    line_number: Some(3),
                    column_start: Some(4),
                    column_end: Some(5)
                },
                LinetableEntry {
                    start: 20,
                    end: 22,
                    line_number: Some(3),
                    column_start: Some(0),
                    column_end: Some(1)
                },
                LinetableEntry {
                    start: 22,
                    end: 26,
                    line_number: Some(5),
                    column_start: Some(0),
                    column_end: Some(5)
                },
                LinetableEntry {
                    start: 26,
                    end: 28,
                    line_number: Some(5),
                    column_start: Some(8),
                    column_end: Some(19)
                },
                LinetableEntry {
                    start: 28,
                    end: 30,
                    line_number: Some(5),
                    column_start: Some(17),
                    column_end: Some(18)
                },
                LinetableEntry {
                    start: 30,
                    end: 32,
                    line_number: Some(5),
                    column_start: Some(16),
                    column_end: Some(20)
                },
                LinetableEntry {
                    start: 32,
                    end: 34,
                    line_number: Some(5),
                    column_start: Some(6),
                    column_end: Some(21)
                },
                LinetableEntry {
                    start: 34,
                    end: 46,
                    line_number: Some(5),
                    column_start: Some(0),
                    column_end: Some(22)
                }
            ]
        );

        assert_eq!(
            starts_line_number(&code_object.co_lines().unwrap(), 1).unwrap(),
            2
        );

        assert_eq!(
            starts_line_number(&code_object.co_lines().unwrap(), 2),
            None
        );

        assert_eq!(
            get_line_number(&code_object.co_lines().unwrap(), 15).unwrap(),
            5
        )
    }
}
