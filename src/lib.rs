pub mod error;
pub mod prelude;
pub mod traits;
mod utils;
pub mod v310;
pub mod v311;
pub mod v312;

use error::Error;
use python_marshal::{self, magic::PyVersion, minimize_references};
use std::io::Read;

#[derive(Debug, Clone)]
pub enum PycFile {
    V310(v310::code_objects::Pyc),
    V311(v311::code_objects::Pyc),
}

impl From<PycFile> for python_marshal::PycFile {
    fn from(val: PycFile) -> Self {
        match val.clone() {
            PycFile::V310(pyc) => {
                python_marshal::PycFile {
                    python_version: pyc.python_version,
                    timestamp: Some(pyc.timestamp),
                    hash: pyc.hash,
                    object: python_marshal::Object::Code(pyc.code_object.into()),
                    references: Vec::new(), // All references are resolved in this editor.
                }
            }
            PycFile::V311(pyc) => {
                python_marshal::PycFile {
                    python_version: pyc.python_version,
                    timestamp: Some(pyc.timestamp),
                    hash: pyc.hash,
                    object: python_marshal::Object::Code(pyc.code_object.into()),
                    references: Vec::new(), // All references are resolved in this editor.
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CodeObject {
    V310(v310::code_objects::Code),
    V311(v311::code_objects::Code),
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
        PyVersion {
            major: 3,
            minor: 11,
            ..
        } => {
            let pyc = v311::code_objects::Pyc::try_from(pyc_file)?;
            Ok(PycFile::V311(pyc))
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
        PyVersion {
            major: 3,
            minor: 11,
            ..
        } => {
            let mut buf = Vec::new();
            data.read_to_end(&mut buf)?;
            let code = python_marshal::load_bytes(&buf, python_version)?;
            Ok(CodeObject::V311(code.try_into()?))
        }
        _ => Err(Error::UnsupportedVersion(python_version)),
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
        CodeObject::V311(code) => {
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
