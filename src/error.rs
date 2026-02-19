use std::fmt;

#[derive(Debug)]
pub enum Error {
    UnkownOpcode(u8),
    InvalidBytecodeLength,
    InvalidLinetable,
    InvalidConversion,
    InvalidConstant(python_marshal::Object),
    InvalidExceptionTable,
    InvalidStacksize(i32),
    UnexpectedExceptiontable,
    UnsupportedVersion(python_marshal::magic::PyVersion),
    PythonMarshalError(python_marshal::error::Error),
    ExtendedArgJump,
    RecursiveReference(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnkownOpcode(op) => write!(f, "Unknown opcode: {}", op),
            Error::InvalidBytecodeLength => write!(f, "Invalid bytecode length"),
            Error::InvalidLinetable => write!(f, "Invalid linetable"),
            Error::InvalidConversion => write!(
                f,
                "Invalid conversion from instruction to resolved instruction"
            ),
            Error::InvalidConstant(obj) => write!(f, "Invalid constant: {:?}", obj),
            Error::InvalidExceptionTable => write!(f, "Invalid exception table"),
            Error::InvalidStacksize(size) => write!(f, "Invalid stack size: {:?}", size),
            Error::UnexpectedExceptiontable => write!(
                f,
                "Received an exception table for a version that doesn't have one"
            ),
            Error::UnsupportedVersion(ver) => write!(
                f,
                "Unsupported Python version: {:?}. Did you forget to enable the feature for this version?",
                ver
            ),
            Error::PythonMarshalError(err) => write!(f, "Python marshal error: {}", err),
            Error::ExtendedArgJump => write!(
                f,
                "There is a jump skipping over an extended arg. We cannot convert to resolved instructions because of this."
            ),
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
