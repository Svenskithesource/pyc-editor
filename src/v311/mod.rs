pub mod code_objects;
pub mod ext_instructions;
pub mod instructions;
pub mod opcodes;

#[cfg(test)]
mod tests {
    use python_marshal::Kind::{ShortAscii, ShortAsciiInterned};
    use python_marshal::{CodeFlags, PyString};

    use crate::v311;
    use crate::v311::code_objects::CompareOperation::Equal;
    use crate::v311::code_objects::{Constant, FrozenConstant, LinetableEntry, NameIndex};
    use crate::v311::ext_instructions::{ExtInstruction, ExtInstructions};
    use crate::v311::instructions::{
        get_line_number, starts_line_number, Instruction, Instructions,
    };
    use crate::v311::opcodes::Opcode;
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
        let mut instructions: v311::ext_instructions::ExtInstructions = ([
            (Opcode::LOAD_NAME, 0).try_into().unwrap(),
            (Opcode::LOAD_CONST, 0).try_into().unwrap(),
            ExtInstruction::CompareOp(Equal),
            (Opcode::POP_JUMP_FORWARD_IF_TRUE, 0).try_into().unwrap(), // Jump to load_name
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
            .find(|(_, instruction)| instruction.get_opcode() == Opcode::POP_JUMP_FORWARD_IF_TRUE)
            .expect("There must be a jump")
        {
            (index, ExtInstruction::PopJumpForwardIfTrue(_)) => {
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
            Instruction::ReturnValue(0.into())
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

        let code_object = v311::code_objects::Code {
            argcount: 0,
            posonlyargcount: 0,
            kwonlyargcount: 0,
            stacksize: 4,
            flags: CodeFlags::from_bits_retain(0x0),
            code: v311::instructions::Instructions::new(vec![
                Instruction::Resume(0),
                Instruction::PushNull(0),
                Instruction::LoadName(0),
                Instruction::LoadConst(0),
                Instruction::Precall(1),
                Instruction::Cache(0),
                Instruction::Call(1),
                Instruction::Cache(0),
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
                Instruction::Precall(1),
                Instruction::Cache(0),
                Instruction::Call(1),
                Instruction::Cache(0),
                Instruction::Cache(0),
                Instruction::Cache(0),
                Instruction::Cache(0),
                Instruction::PopTop(0),
                Instruction::LoadConst(3),
                Instruction::ReturnValue(0),
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
                240, 3, 1, 1, 1, 224, 0, 5, 128, 5, 128, 104, 129, 15, 132, 15, 128, 15, 216, 4, 5,
                128, 1, 224, 0, 5, 128, 5, 128, 111, 144, 17, 128, 111, 128, 111, 209, 0, 22, 212,
                0, 22, 208, 0, 22, 208, 0, 22, 208, 0, 22,
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
                    column_start: 1.into(),
                    column_end: 1.into(),
                },
                LinetableEntry {
                    start: 2,
                    end: 4,
                    line_number: Some(2),
                    column_start: 0.into(),
                    column_end: 5.into(),
                },
                LinetableEntry {
                    start: 4,
                    end: 6,
                    line_number: Some(2),
                    column_start: 0.into(),
                    column_end: 5.into(),
                },
                LinetableEntry {
                    start: 6,
                    end: 8,
                    line_number: Some(2),
                    column_start: 6.into(),
                    column_end: 14.into(),
                },
                LinetableEntry {
                    start: 8,
                    end: 12,
                    line_number: Some(2),
                    column_start: 0.into(),
                    column_end: 15.into(),
                },
                LinetableEntry {
                    start: 12,
                    end: 22,
                    line_number: Some(2),
                    column_start: 0.into(),
                    column_end: 15.into(),
                },
                LinetableEntry {
                    start: 22,
                    end: 24,
                    line_number: Some(2),
                    column_start: 0.into(),
                    column_end: 15.into(),
                },
                LinetableEntry {
                    start: 24,
                    end: 26,
                    line_number: Some(3),
                    column_start: 4.into(),
                    column_end: 5.into(),
                },
                LinetableEntry {
                    start: 26,
                    end: 28,
                    line_number: Some(3),
                    column_start: 0.into(),
                    column_end: 1.into(),
                },
                LinetableEntry {
                    start: 28,
                    end: 30,
                    line_number: Some(5),
                    column_start: 0.into(),
                    column_end: 5.into(),
                },
                LinetableEntry {
                    start: 30,
                    end: 32,
                    line_number: Some(5),
                    column_start: 0.into(),
                    column_end: 5.into(),
                },
                LinetableEntry {
                    start: 32,
                    end: 34,
                    line_number: Some(5),
                    column_start: 6.into(),
                    column_end: 21.into(),
                },
                LinetableEntry {
                    start: 34,
                    end: 36,
                    line_number: Some(5),
                    column_start: 17.into(),
                    column_end: 18.into(),
                },
                LinetableEntry {
                    start: 36,
                    end: 38,
                    line_number: Some(5),
                    column_start: 6.into(),
                    column_end: 21.into(),
                },
                LinetableEntry {
                    start: 38,
                    end: 40,
                    line_number: Some(5),
                    column_start: 6.into(),
                    column_end: 21.into(),
                },
                LinetableEntry {
                    start: 40,
                    end: 44,
                    line_number: Some(5),
                    column_start: 0.into(),
                    column_end: 22.into(),
                },
                LinetableEntry {
                    start: 44,
                    end: 54,
                    line_number: Some(5),
                    column_start: 0.into(),
                    column_end: 22.into(),
                },
                LinetableEntry {
                    start: 54,
                    end: 56,
                    line_number: Some(5),
                    column_start: 0.into(),
                    column_end: 22.into(),
                },
                LinetableEntry {
                    start: 56,
                    end: 58,
                    line_number: Some(5),
                    column_start: 0.into(),
                    column_end: 22.into(),
                },
                LinetableEntry {
                    start: 58,
                    end: 60,
                    line_number: Some(5),
                    column_start: 0.into(),
                    column_end: 22.into(),
                },
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
