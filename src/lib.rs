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
    use python_marshal::Kind::ShortAsciiInterned;
    use python_marshal::{CodeFlags, PyString};

    use crate::v310::code_objects::CompareOperation::Equal;
    use crate::v310::code_objects::Instruction::{self};
    use crate::v310::code_objects::{
        AbsoluteJump, Constant, FrozenConstant, Jump,
    };
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
        let mut instructions: v310::code_objects::Instructions = ([
            (Opcode::LOAD_NAME, 0).into(),
            (Opcode::LOAD_CONST, 0).into(),
            Instruction::CompareOp(Equal),
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

        let code = CodeObject::V310(v310::code_objects::Code {
            argcount: 0,
            posonlyargcount: 0,
            kwonlyargcount: 0,
            nlocals: 0,
            stacksize: 2,
            flags: CodeFlags::from_bits_truncate(CodeFlags::NOFREE.bits()),
            code: instructions,
            consts: vec![
                Constant::FrozenConstant(FrozenConstant::Long(1.into())),
                Constant::FrozenConstant(FrozenConstant::None),
            ],
            names: vec![
                PyString {
                    value: "x".into(),
                    kind: ShortAsciiInterned,
                },
                PyString {
                    value: "print".into(),
                    kind: ShortAsciiInterned,
                },
                PyString {
                    value: "b".into(),
                    kind: ShortAsciiInterned,
                },
            ],
            varnames: vec![],
            freevars: vec![],
            cellvars: vec![],
            filename: PyString {
                value: "<string>".into(),
                kind: ShortAsciiInterned,
            },
            name: PyString {
                value: "<module>".into(),
                kind: ShortAsciiInterned,
            },
            firstlineno: 1,
            lnotab: vec![24, 0],
        });

        let dumped = dump_code(code.clone(), (3, 10).into(), 4).unwrap();

        println!("{:?}", &dumped);

        let new_code =
            load_code(std::io::Cursor::new(dumped), (3, 10).into()).expect("Should never fail");

        match new_code {
            CodeObject::V310(ref code) => {
                dbg!(&code.code, &code.code.len());
                match code
                    .code
                    .iter()
                    .find(|i| i.get_opcode() == Opcode::POP_JUMP_IF_FALSE)
                    .expect("There must be a jump")
                {
                    Instruction::PopJumpIfFalse(jump) => {
                        dbg!(jump);
                        let target = code
                            .code
                            .get_jump_target(jump.clone().into())
                            .expect("Should never fail");

                        dbg!(target);

                        assert_eq!(og_target, target);
                    }
                    _ => panic!(),
                }
            }
        }

        assert_eq!(code, new_code);
    }
}
