pub mod error;
pub mod v310;

use error::Error;
use python_marshal::{self, magic::PyVersion, minimize_references};
use std::io::Read;

#[derive(Debug, Clone)]
pub enum PycFile {
    V310(v310::code_objects::Pyc),
}

#[derive(Debug, Clone, PartialEq)]
pub enum CodeObject {
    V310(v310::code_objects::Code),
}

pub fn load_pyc(data: impl Read) -> Result<PycFile, Error> {
    let pyc_file = python_marshal::load_pyc(data)?;

    match pyc_file.python_version {
        PyVersion {
            major: 3,
            minor: 10,
            ..
        } => {
            let pyc = v310::code_objects::Pyc::try_from(pyc_file)?;
            Ok(PycFile::V310(pyc))
        }
        _ => Err(Error::UnsupportedVersion(pyc_file.python_version)),
    }
}

pub fn dump_pyc(pyc_file: PycFile) -> Result<Vec<u8>, Error> {
    let mut pyc: python_marshal::PycFile = pyc_file.into();

    let (obj, refs) = minimize_references(&pyc.object, pyc.references);

    pyc.object = obj;
    pyc.references = refs;

    python_marshal::dump_pyc(pyc).map_err(|e| e.into())
}

pub fn load_code(mut data: impl Read, python_version: PyVersion) -> Result<CodeObject, Error> {
    match python_version {
        PyVersion {
            major: 3,
            minor: 10,
            ..
        } => {
            let mut buf = Vec::new();
            data.read_to_end(&mut buf)?;
            let code = python_marshal::load_bytes(&buf, python_version)?;
            Ok(CodeObject::V310(code.try_into()?))
        }
        _ => Err(Error::WrongVersion),
    }
}

pub fn dump_code(
    code_object: CodeObject,
    python_version: PyVersion,
    marshal_version: u8,
) -> Result<Vec<u8>, Error> {
    match code_object {
        CodeObject::V310(code) => {
            let object = python_marshal::Object::Code(code.into());
            let (obj, refs) = minimize_references(&object, vec![]);

            Ok(python_marshal::dump_bytes(
                obj,
                Some(refs),
                python_version,
                marshal_version,
            )?)
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::v310::code_objects::CompareOperation::Equal;
    use crate::v310::code_objects::{AbsoluteJump, Jump};
    use crate::v310::ext_instructions::{ExtInstruction, ExtInstructions};
    use crate::v310::instructions::{Instruction, Instructions};
    use crate::v310::opcodes::Opcode;

    use super::*;
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
        let mut instructions: v310::ext_instructions::ExtInstructions = ([
            (Opcode::LOAD_NAME, 0).into(),
            (Opcode::LOAD_CONST, 0).into(),
            ExtInstruction::CompareOp(Equal),
            (Opcode::POP_JUMP_IF_FALSE, 4).into(),
            (Opcode::LOAD_NAME, 1).into(),
            (Opcode::LOAD_NAME, 2).into(),
            (Opcode::CALL_FUNCTION, 1).into(),
            (Opcode::POP_TOP, 0).into(),
            (Opcode::LOAD_CONST, 1).into(),
            (Opcode::RETURN_VALUE, 0).into(),
            (Opcode::LOAD_CONST, 1).into(),
            (Opcode::RETURN_VALUE, 0).into(),
        ]
        .as_slice())
        .into();

        instructions.insert_instructions(3, &vec![(Opcode::NOP, 0).into(); 300]);

        let og_target = instructions
            .get_jump_target(Jump::Absolute(AbsoluteJump::new(304)))
            .expect("Must be to the load name after the jump");

        dbg!(og_target);

        let resolved = instructions.to_instructions().to_resolved();
        match resolved
            .iter()
            .find(|i| i.get_opcode() == Opcode::POP_JUMP_IF_FALSE)
            .expect("There must be a jump")
        {
            ExtInstruction::PopJumpIfFalse(jump) => {
                dbg!(jump);
                let target = resolved
                    .get_jump_target((*jump).into())
                    .expect("Should never fail");

                dbg!(target);

                assert_eq!(og_target, target);
            }
            _ => panic!(),
        }

        assert_eq!(instructions, resolved);
    }

    #[test]
    fn test_absolute_jump() {
        // Create a list of instructions that look like this:
        // 0    EXTENDED_ARG  1
        // 1    JUMP_ABSOLUTE 4 (to 260)
        // 2    NOP           0
        // ...
        // 149  EXTENDED_ARG  1
        // 150  NOP           1
        // ...
        // 259  NOP           1
        // 260  RETURN_VALUE  0
        let mut instructions = Instructions::new(vec![
            Instruction::ExtendedArg(1),
            Instruction::JumpAbsolute(4), // This is a jump to 260 (256 + 4) in reality (via extended arg).
        ]);

        // Fill instruction list with nops until index 149 (150 items)
        for _ in instructions.len()..150 {
            instructions.append_instruction(Instruction::Nop(0));
        }

        assert_eq!(instructions.len(), 150);

        instructions.append_instruction(Instruction::ExtendedArg(1));

        // Fill instruction list with nops until index 259 (260 items)
        for _ in instructions.len()..260 {
            instructions.append_instruction(Instruction::Nop(0));
        }

        assert_eq!(instructions.len(), 260);

        instructions.append_instruction(Instruction::ReturnValue(0));

        let resolved = instructions.to_resolved();

        let jump = resolved.first().unwrap().get_raw_value();

        assert_eq!(
            resolved.get(jump as usize).unwrap(),
            &ExtInstruction::ReturnValue(0.into())
        );

        assert_eq!(resolved.len(), 259);

        assert_eq!(instructions, resolved.to_instructions());
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

        let jump = resolved.get(11).unwrap().get_raw_value();

        assert_eq!(resolved.len(), 17);

        assert_eq!(
            resolved.get(jump as usize + 11 + 1).unwrap(),
            &ExtInstruction::ReturnValue(0.into())
        );

        assert_eq!(instructions, resolved.to_instructions());
    }

    #[test]
    fn test_extra_extended_arg() {
        let ext_instructions = ExtInstructions::new(vec![
            ExtInstruction::JumpAbsolute(255.into()),
            ExtInstruction::JumpAbsolute(300.into()), // This will need an extended arg, which will increase the offset above which also causes that to need an extended arg.
        ]);

        let instructions = ext_instructions.to_instructions();

        assert_eq!(
            instructions,
            Instructions::new(vec![
                Instruction::ExtendedArg(1),
                Instruction::JumpAbsolute(1),
                Instruction::ExtendedArg(1),
                Instruction::JumpAbsolute(46)
            ])
        )
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
}
