pub mod error;
pub mod v310;

use error::Error;
use python_marshal::{self, magic::PyVersion};
use std::{io::Read, io::Write};

#[derive(Debug, Clone)]
pub enum PycFile {
    V310(v310::code_objects::Pyc),
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

pub fn dump_pyc(writer: &mut impl Write, pyc_file: PycFile) -> Result<(), Error> {
    let pyc: python_marshal::PycFile = pyc_file.into();

    python_marshal::dump_pyc(writer, pyc)?;

    Ok(())
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
            original_pyc.object,
            original_pyc.references,
        )
        .unwrap()
        .0;

        let file = File::open("tests/test.pyc").unwrap();
        let reader = BufReader::new(file);
        let pyc_file = load_pyc(reader).unwrap();

        dbg!(&pyc_file);

        let pyc: python_marshal::PycFile = pyc_file.into();

        assert_eq!(original_pyc, pyc.object);
    }
}
