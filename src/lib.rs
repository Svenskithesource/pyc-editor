pub mod error;
pub mod v310;

use error::Error;
use python_marshal::{self, magic::PyVersion, minimize_references};
use std::io::Read;

#[derive(Debug, Clone)]
pub enum PycFile {
    V310(v310::code_objects::Pyc),
}

#[derive(Debug, Clone)]
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

        dbg!(&pyc_file);

        let pyc: python_marshal::PycFile = pyc_file.into();

        assert_eq!(original_pyc, pyc.object);
    }
}
