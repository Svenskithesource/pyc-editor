use std::fmt;

#[derive(Debug)]
pub enum Error {
    UnkownOpcode(u8),
    InvalidBytecodeLength,
    InvalidLinetable,
    InvalidConstant(python_marshal::Object),
    UnsupportedVersion(python_marshal::magic::PyVersion),
    PythonMarshalError(python_marshal::error::Error),
    RecursiveReference(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnkownOpcode(op) => write!(f, "Unknown opcode: {}", op),
            Error::InvalidBytecodeLength => write!(f, "Invalid bytecode length"),
            Error::InvalidLinetable => write!(f, "Invalid linetable"),
            Error::InvalidConstant(obj) => write!(f, "Invalid constant: {:?}", obj),
            Error::UnsupportedVersion(ver) => write!(f, "Unsupported Python version: {:?}", ver),
            Error::PythonMarshalError(err) => write!(f, "Python marshal error: {}", err),
            Error::RecursiveReference(s) => write!(f, "Recursive reference: {}", s),
        }
    }
}

impl std::error::Error for Error {}

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
