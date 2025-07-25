#[derive(Debug)]
pub enum Error {
    WrongVersion,
    UnkownOpcode(u8),
    InvalidBytecodeLength,
    InvalidConstant(python_marshal::Object),
    UnsupportedVersion(python_marshal::magic::PyVersion),
    PythonMarshalError(python_marshal::error::Error),
    RecursiveReference(&'static str),
}

impl From<python_marshal::error::Error> for Error {
    fn from(err: python_marshal::error::Error) -> Self {
        Error::PythonMarshalError(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::PythonMarshalError(python_marshal::error::Error::InvalidData(value))
    }
}
