use bitflags::bitflags;

use hashable::HashableHashSet;
use indexmap::IndexSet;
use num_bigint::BigInt;
use num_complex::Complex;
use ordered_float::OrderedFloat;
use python_marshal::{extract_object, resolver::resolve_all_refs, CodeFlags, Object, PyString};

use crate::{error::Error, utils::ExceptionTableEntry, v311::instructions::Instructions};
use std::fmt;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum FrozenConstant {
    None,
    StopIteration,
    Ellipsis,
    Bool(bool),
    Long(BigInt),
    Float(OrderedFloat<f64>),
    Complex(Complex<OrderedFloat<f64>>),
    Bytes(Vec<u8>),
    String(PyString),
    Tuple(Vec<FrozenConstant>),
    List(Vec<FrozenConstant>),
    FrozenSet(HashableHashSet<FrozenConstant>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    FrozenConstant(FrozenConstant),
    CodeObject(Code),
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant::FrozenConstant(fc) => write!(f, "{fc}"),
            Constant::CodeObject(code) => write!(f, "{}", code),
        }
    }
}

impl fmt::Display for FrozenConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrozenConstant::None => write!(f, "None"),
            FrozenConstant::StopIteration => write!(f, "StopIteration"),
            FrozenConstant::Ellipsis => write!(f, "Ellipsis"),
            FrozenConstant::Bool(b) => write!(f, "{b}"),
            FrozenConstant::Long(l) => write!(f, "{l}"),
            FrozenConstant::Float(fl) => write!(f, "{fl}"),
            FrozenConstant::Complex(c) => write!(f, "{}+{}j", c.re, c.im),
            FrozenConstant::Bytes(b) => write!(f, "b{:?}", b),
            FrozenConstant::String(s) => write!(f, "\'{}\'", s.value),
            FrozenConstant::Tuple(t) => {
                write!(f, "(")?;

                let text = t
                    .iter()
                    .map(|c| format!("{c}"))
                    .collect::<Vec<_>>()
                    .join(", ");

                write!(f, "{text}")?;

                write!(f, ")")
            }
            FrozenConstant::List(l) => {
                write!(f, "[")?;

                let text = l
                    .iter()
                    .map(|c| format!("{c}"))
                    .collect::<Vec<_>>()
                    .join(", ");

                write!(f, "{text}")?;

                write!(f, "]")
            }
            FrozenConstant::FrozenSet(fs) => {
                write!(f, "frozenset({{")?;

                let text = fs
                    .iter()
                    .map(|c| format!("{c}"))
                    .collect::<Vec<_>>()
                    .join(", ");

                write!(f, "{text}")?;

                write!(f, "}})")
            }
        }
    }
}

impl From<Constant> for python_marshal::Object {
    fn from(val: Constant) -> Self {
        match val {
            Constant::CodeObject(code) => python_marshal::Object::Code(code.into()),
            Constant::FrozenConstant(constant) => constant.into(),
        }
    }
}

impl TryFrom<python_marshal::Object> for FrozenConstant {
    type Error = Error;

    fn try_from(value: python_marshal::Object) -> Result<Self, Self::Error> {
        match value {
            python_marshal::Object::None => Ok(FrozenConstant::None),
            python_marshal::Object::StopIteration => Ok(FrozenConstant::StopIteration),
            python_marshal::Object::Ellipsis => Ok(FrozenConstant::Ellipsis),
            python_marshal::Object::Bool(b) => Ok(FrozenConstant::Bool(b)),
            python_marshal::Object::Long(l) => Ok(FrozenConstant::Long(l)),
            python_marshal::Object::Float(f) => Ok(FrozenConstant::Float(f)),
            python_marshal::Object::Complex(c) => {
                Ok(FrozenConstant::Complex(Complex { re: c.re, im: c.im }))
            }
            python_marshal::Object::Bytes(b) => Ok(FrozenConstant::Bytes(b)),
            python_marshal::Object::String(s) => Ok(FrozenConstant::String(s)),
            python_marshal::Object::Tuple(t) => {
                let constants = t
                    .into_iter()
                    .map(FrozenConstant::try_from)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(FrozenConstant::Tuple(constants))
            }
            python_marshal::Object::List(l) => {
                let constants = l
                    .into_iter()
                    .map(FrozenConstant::try_from)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(FrozenConstant::List(constants))
            }
            python_marshal::Object::FrozenSet(fs) => {
                let constants = fs
                    .into_iter()
                    .map(python_marshal::Object::from)
                    .map(FrozenConstant::try_from)
                    .collect::<Result<HashableHashSet<_>, _>>()?;
                Ok(FrozenConstant::FrozenSet(constants))
            }
            _ => Err(Error::InvalidConstant(value)),
        }
    }
}

impl From<FrozenConstant> for python_marshal::Object {
    fn from(val: FrozenConstant) -> Self {
        match val {
            FrozenConstant::Bool(value) => python_marshal::ObjectHashable::Bool(value).into(),
            FrozenConstant::None => python_marshal::ObjectHashable::None.into(),
            FrozenConstant::StopIteration => python_marshal::ObjectHashable::StopIteration.into(),
            FrozenConstant::Ellipsis => python_marshal::ObjectHashable::Ellipsis.into(),
            FrozenConstant::Long(value) => python_marshal::ObjectHashable::Long(value).into(),
            FrozenConstant::Float(value) => python_marshal::ObjectHashable::Float(value).into(),
            FrozenConstant::Complex(value) => python_marshal::ObjectHashable::Complex(value).into(),
            FrozenConstant::Bytes(value) => python_marshal::ObjectHashable::Bytes(value).into(),
            FrozenConstant::String(value) => python_marshal::ObjectHashable::String(value).into(),
            FrozenConstant::Tuple(values) => python_marshal::Object::Tuple(
                values
                    .into_iter()
                    .map(Into::<python_marshal::Object>::into)
                    .collect(),
            ),
            FrozenConstant::List(values) => python_marshal::Object::List(
                values
                    .into_iter()
                    .map(Into::<python_marshal::Object>::into)
                    .collect(),
            ),
            FrozenConstant::FrozenSet(values) => {
                python_marshal::Object::FrozenSet(
                    values
                        .into_iter()
                        .cloned()
                        .map(Into::<python_marshal::Object>::into)
                        .map(TryInto::<python_marshal::ObjectHashable>::try_into)
                        .map(Result::unwrap) // The frozen set can only contain values we know for sure are hashable
                        .collect::<IndexSet<_, _>>(),
                )
            }
        }
    }
}

impl TryFrom<python_marshal::Object> for Constant {
    type Error = Error;

    fn try_from(value: python_marshal::Object) -> Result<Self, Self::Error> {
        match value {
            python_marshal::Object::Code(code) => match code {
                python_marshal::Code::V310(_) => Err(Error::UnsupportedVersion((3, 10).into())),
                python_marshal::Code::V311(code) => {
                    let code = Code::try_from(code)?;
                    Ok(Constant::CodeObject(code))
                }
                python_marshal::Code::V312(_) => Err(Error::UnsupportedVersion((3, 12).into())),
                python_marshal::Code::V313(_) => Err(Error::UnsupportedVersion((3, 13).into())),
            },
            _ => {
                let frozen_constant = FrozenConstant::try_from(value)?;
                Ok(Constant::FrozenConstant(frozen_constant))
            }
        }
    }
}

/// Low level representation of a Python code object
#[derive(Debug, Clone, PartialEq)]
pub struct Code {
    pub argcount: u32,
    pub posonlyargcount: u32,
    pub kwonlyargcount: u32,
    pub stacksize: u32,
    pub flags: CodeFlags,
    pub code: Instructions,
    pub consts: Vec<Constant>,
    pub names: Vec<PyString>,
    pub localsplusnames: Vec<PyString>,
    pub localspluskinds: Vec<u8>,
    pub filename: PyString,
    pub name: PyString,
    pub qualname: PyString,
    pub firstlineno: u32,
    /// NOTE: https://peps.python.org/pep-0626/
    pub linetable: Vec<u8>,
    pub exceptiontable: Vec<u8>,
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "<code object {}, file \"{}\", line {}>",
            self.name.value, self.filename.value, self.firstlineno
        )
    }
}

/// Offsets are for bytes (not instructions)
#[derive(Debug, Clone, PartialEq)]
pub struct LinetableEntry {
    pub start: u32,
    pub end: u32,
    pub line_number: Option<u32>,
    pub column_start: Option<u32>,
    pub column_end: Option<u32>,
}

impl Code {
    pub fn co_lines(&self) -> Result<Vec<LinetableEntry>, Error> {
        // See https://github.com/python/cpython/blob/3.11/Objects/locations.md

        let mut entries = Vec::new();
        let mut offset = 0u32; // Byte offset in code
        let mut line = self.firstlineno; // Current line number
        let mut i = 0; // Index in linetable

        while i < self.linetable.len() {
            let first_byte = self.linetable[i];

            // Check if most significant bit is set (location entry)
            if first_byte & 0x80 == 0 {
                return Err(Error::InvalidLinetable);
            }

            // Extract code and length from first byte
            let code = (first_byte >> 3) & 0x0F; // Bits 3-6
            let length = (first_byte & 0x07) + 1; // Bits 0-2, plus 1

            i += 1;

            let (delta_line, start_col, end_col) = match code {
                // Short forms (0-9)
                0..=9 => {
                    if i >= self.linetable.len() {
                        return Err(Error::InvalidLinetable);
                    }
                    let second_byte = self.linetable[i];
                    i += 1;

                    let start_column = (code * 8) + ((second_byte >> 4) & 7);
                    let end_column = start_column + (second_byte & 15);
                    (0i32, Some(start_column as u32), Some(end_column as u32))
                }

                // One line forms (10-12)
                10..=12 => {
                    if i + 1 >= self.linetable.len() {
                        return Err(Error::InvalidLinetable);
                    }
                    let start_col = self.linetable[i] as u32;
                    let end_col = self.linetable[i + 1] as u32;
                    i += 2;

                    let delta = (code - 10) as i32;
                    (delta, Some(start_col), Some(end_col))
                }

                // No column info (13)
                13 => {
                    let (delta, bytes_read) = Self::read_location_svarint(&self.linetable[i..])?;
                    i += bytes_read;
                    (delta, None, None)
                }

                // Long form (14)
                14 => {
                    let (delta_line, bytes_read1) =
                        Self::read_location_svarint(&self.linetable[i..])?;
                    i += bytes_read1;

                    let (_delta_end_line, bytes_read2) =
                        Self::read_location_varint(&self.linetable[i..])?;
                    i += bytes_read2;

                    let (start_col, bytes_read3) =
                        Self::read_location_varint(&self.linetable[i..])?;
                    i += bytes_read3;

                    let (end_col, bytes_read4) = Self::read_location_varint(&self.linetable[i..])?;
                    i += bytes_read4;

                    // For now, we ignore delta_end_line and just use delta_line
                    // TODO: Handle end line properly
                    (delta_line, Some(start_col), Some(end_col))
                }

                // No location (15)
                15 => (0i32, None, None),

                _ => return Err(Error::InvalidLinetable),
            };

            // Update line number
            if delta_line != 0 {
                line = (line as i32 + delta_line) as u32;
            }

            // Create entry
            let entry = LinetableEntry {
                start: offset,
                end: offset + length as u32 * 2, // length is in code units (2 bytes each)
                line_number: if code == 15 { None } else { Some(line) },
                column_start: start_col,
                column_end: end_col,
            };

            entries.push(entry);
            offset += length as u32 * 2;
        }

        Ok(entries)
    }

    // Helper function to read variable-length unsigned integer (varint) for location information
    fn read_location_varint(data: &[u8]) -> Result<(u32, usize), Error> {
        let mut result = 0u32;
        let mut shift = 0;
        let mut i = 0;

        while i < data.len() {
            let byte = data[i];
            result |= ((byte & 0x3F) as u32) << shift;
            i += 1;

            if byte & 0x40 == 0 {
                // Last chunk
                break;
            }

            shift += 6;
            if shift >= 32 {
                return Err(Error::InvalidLinetable);
            }
        }

        Ok((result, i))
    }

    // Helper function to read variable-length signed integer (svarint) for location information
    fn read_location_svarint(data: &[u8]) -> Result<(i32, usize), Error> {
        let (unsigned_val, bytes_read) = Self::read_location_varint(data)?;

        // Convert unsigned to signed according to CPython spec
        let signed_val = if unsigned_val & 1 != 0 {
            -((unsigned_val >> 1) as i32)
        } else {
            (unsigned_val >> 1) as i32
        };

        Ok((signed_val, bytes_read))
    }

    // Helper function to read variable-length unsigned integer (varint) for exception information
    fn read_exception_varint(data: &[u8]) -> Result<(u32, usize), Error> {
        let mut iter = data.iter();
        let mut consumed = 0;

        let first = *iter.next().ok_or(Error::InvalidExceptionTable)?; // return None if no bytes
        consumed += 1;

        let mut val: u32 = (first & 63) as u32;
        let mut b = first;

        while b & 64 != 0 {
            let next_byte = *iter.next().ok_or(Error::InvalidExceptionTable)?; // return None if input ended early
            consumed += 1;

            val <<= 6;
            val |= (next_byte & 63) as u32;

            b = next_byte;
        }

        Ok((val, consumed))
    }

    pub fn exception_table(&self) -> Result<Vec<ExceptionTableEntry>, Error> {
        let mut exception_entries = vec![];
        let mut i = 0; // Offset in exception table

        while i < self.exceptiontable.len() {
            let (start, bytes_read) = Self::read_exception_varint(&self.exceptiontable[i..])?;
            i += bytes_read;

            let (length, bytes_read) = Self::read_exception_varint(&self.exceptiontable[i..])?;
            i += bytes_read;

            let end = start + length;

            let (target, bytes_read) = Self::read_exception_varint(&self.exceptiontable[i..])?;
            i += bytes_read;

            let (dl, bytes_read) = Self::read_exception_varint(&self.exceptiontable[i..])?;
            i += bytes_read;

            let depth = dl >> 1;
            let lasti = (dl & 1) != 0;

            exception_entries.push(ExceptionTableEntry {
                start,
                end,
                target,
                depth,
                lasti,
            });
        }

        Ok(exception_entries)
    }
}

impl TryFrom<(python_marshal::Object, Vec<Object>)> for Code {
    type Error = Error;

    fn try_from(
        (code_object, refs): (python_marshal::Object, Vec<Object>),
    ) -> Result<Self, Self::Error> {
        let (code_object, refs) = resolve_all_refs(&code_object, &refs);

        if !refs.is_empty() {
            return Err(Error::RecursiveReference(
                "This pyc file contains references that cannot be resolved. This should never happen on a valid pyc file generated by Python.",
            ));
        }

        let code_object = extract_object!(Some(code_object), python_marshal::Object::Code(code) => code, python_marshal::error::Error::UnexpectedObject)?;

        match code_object {
            python_marshal::Code::V310(_) => Err(Error::UnsupportedVersion((3, 10).into())),
            python_marshal::Code::V311(code) => Ok(Code::try_from(code)?),
            python_marshal::Code::V312(_) => Err(Error::UnsupportedVersion((3, 12).into())),
            python_marshal::Code::V313(_) => Err(Error::UnsupportedVersion((3, 13).into())),
        }
    }
}

impl From<Code> for python_marshal::Code {
    fn from(val: Code) -> Self {
        python_marshal::Code::V311(python_marshal::code_objects::Code311 {
            argcount: val.argcount,
            posonlyargcount: val.posonlyargcount,
            kwonlyargcount: val.kwonlyargcount,
            stacksize: val.stacksize,
            flags: val.flags,
            code: python_marshal::Object::Bytes(val.code.into()).into(),
            consts: python_marshal::Object::Tuple(
                val.consts.into_iter().map(|c| c.into()).collect(),
            )
            .into(),
            names: python_marshal::Object::Tuple(
                val.names
                    .into_iter()
                    .map(python_marshal::Object::String)
                    .collect(),
            )
            .into(),
            localsplusnames: python_marshal::Object::Tuple(
                val.localsplusnames
                    .into_iter()
                    .map(python_marshal::Object::String)
                    .collect(),
            )
            .into(),
            localspluskinds: python_marshal::Object::Bytes(val.localspluskinds).into(),
            filename: python_marshal::Object::String(val.filename).into(),
            name: python_marshal::Object::String(val.name).into(),
            qualname: python_marshal::Object::String(val.qualname).into(),
            firstlineno: val.firstlineno,
            linetable: python_marshal::Object::Bytes(val.linetable).into(),
            exceptiontable: python_marshal::Object::Bytes(val.exceptiontable).into(),
        })
    }
}

macro_rules! extract_strings_tuple {
    ($objs:expr, $refs:expr) => {
        $objs
            .iter()
            .map(|o| match o {
                python_marshal::Object::String(string) => Ok(string.clone()),
                _ => Err(python_marshal::error::Error::UnexpectedObject),
            })
            .collect::<Result<Vec<_>, _>>()
    };
}

impl TryFrom<python_marshal::code_objects::Code311> for Code {
    type Error = crate::error::Error;

    fn try_from(code: python_marshal::code_objects::Code311) -> Result<Self, Self::Error> {
        let co_code = extract_object!(Some(*code.code), python_marshal::Object::Bytes(bytes) => bytes, python_marshal::error::Error::NullInTuple)?;
        let co_consts = extract_object!(Some(*code.consts), python_marshal::Object::Tuple(objs) => objs, python_marshal::error::Error::NullInTuple)?;
        let co_names = extract_strings_tuple!(
            extract_object!(Some(*code.names), python_marshal::Object::Tuple(objs) => objs, python_marshal::error::Error::NullInTuple)?,
            self.references
        )?;
        let co_localsplusnames = extract_strings_tuple!(
            extract_object!(Some(*code.localsplusnames), python_marshal::Object::Tuple(objs) => objs, python_marshal::error::Error::NullInTuple)?,
            self.references
        )?;
        let co_localspluskinds = extract_object!(Some(*code.localspluskinds), python_marshal::Object::Bytes(bytes) => bytes, python_marshal::error::Error::NullInTuple)?;

        let co_filename = extract_object!(Some(*code.filename), python_marshal::Object::String(string) => string, python_marshal::error::Error::NullInTuple)?;
        let co_name = extract_object!(Some(*code.name), python_marshal::Object::String(string) => string, python_marshal::error::Error::NullInTuple)?;
        let co_qualname = extract_object!(Some(*code.qualname), python_marshal::Object::String(string) => string, python_marshal::error::Error::NullInTuple)?;
        let co_linetable = extract_object!(Some(*code.linetable), python_marshal::Object::Bytes(bytes) => bytes, python_marshal::error::Error::NullInTuple)?;
        let co_exceptiontable = extract_object!(Some(*code.exceptiontable), python_marshal::Object::Bytes(bytes) => bytes, python_marshal::error::Error::NullInTuple)?;

        Ok(Code {
            argcount: code.argcount,
            posonlyargcount: code.posonlyargcount,
            kwonlyargcount: code.kwonlyargcount,
            stacksize: code.stacksize,
            flags: code.flags,
            code: Instructions::try_from(co_code.as_slice())?,
            consts: co_consts
                .iter()
                .map(|obj| Constant::try_from(obj.clone()))
                .collect::<Result<Vec<_>, _>>()?,
            names: co_names.to_vec(),
            localsplusnames: co_localsplusnames,
            localspluskinds: co_localspluskinds,
            filename: co_filename.clone(),
            name: co_name.clone(),
            qualname: co_qualname.clone(),
            firstlineno: code.firstlineno,
            linetable: co_linetable.to_vec(),
            exceptiontable: co_exceptiontable.to_vec(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jump {
    Relative(RelativeJump),
}

impl From<RelativeJump> for Jump {
    fn from(value: RelativeJump) -> Self {
        Self::Relative(value)
    }
}

/// Indicates jump direction for a relative jump
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JumpDirection {
    Forward,
    Backward,
}

/// Represents a relative jump offset from the current instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelativeJump {
    pub index: u32,
    pub direction: JumpDirection,
}

impl RelativeJump {
    pub fn new(index: u32, direction: JumpDirection) -> Self {
        RelativeJump { index, direction }
    }
}

/// Represents an absolute jump target (a byte offset from the start of the code).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbsoluteJump {
    pub index: u32,
}

impl AbsoluteJump {
    pub fn new(index: u32) -> Self {
        AbsoluteJump { index }
    }
}

impl From<u32> for AbsoluteJump {
    fn from(value: u32) -> Self {
        AbsoluteJump { index: value }
    }
}

/// Holds an index into co_names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NameIndex {
    pub index: u32,
}

/// Holds an index into co_names. LOAD_GLOBAL is a special case where it will look for (index >> 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalNameIndex {
    pub index: u32,
}

/// Holds an index into co_varnames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarNameIndex {
    pub index: u32,
}

impl VarNameIndex {
    pub fn get<'a>(&self, co_varnames: &'a [PyString]) -> Option<&'a PyString> {
        co_varnames.get(self.index as usize)
    }
}

/// Holds an index into co_consts. Has helper functions to get the actual constant at the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstIndex {
    pub index: u32,
}

impl ConstIndex {
    pub fn get<'a>(&self, co_consts: &'a [Constant]) -> Option<&'a Constant> {
        co_consts.get(self.index as usize)
    }
}

/// All binary operations that can be passed as an argument
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperation {
    Add = 0,
    And = 1,
    FloorDivide = 2,
    Lshift = 3,
    MatrixMultiply = 4,
    Multiply = 5,
    Remainder = 6,
    Or = 7,
    Power = 8,
    Rshift = 9,
    Subtract = 10,
    TrueDivide = 11,
    Xor = 12,
    InplaceAdd = 13,
    InplaceAnd = 14,
    InplaceFloorDivide = 15,
    InplaceLshift = 16,
    InplaceMatrixMultiply = 17,
    InplaceMultiply = 18,
    InplaceRemainder = 19,
    InplaceOr = 20,
    InplacePower = 21,
    InplaceRshift = 22,
    InplaceSubtract = 23,
    InplaceTrueDivide = 24,
    InplaceXor = 25,
    Invalid(u32),
}

impl std::fmt::Display for BinaryOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOperation::Add => write!(f, "+"),
            BinaryOperation::And => write!(f, "&"),
            BinaryOperation::FloorDivide => write!(f, "//"),
            BinaryOperation::Lshift => write!(f, "<<"),
            BinaryOperation::MatrixMultiply => write!(f, "@"),
            BinaryOperation::Multiply => write!(f, "*"),
            BinaryOperation::Remainder => write!(f, "%"),
            BinaryOperation::Or => write!(f, "|"),
            BinaryOperation::Power => write!(f, "**"),
            BinaryOperation::Rshift => write!(f, ">>"),
            BinaryOperation::Subtract => write!(f, "-"),
            BinaryOperation::TrueDivide => write!(f, "/"),
            BinaryOperation::Xor => write!(f, "^"),
            BinaryOperation::InplaceAdd => write!(f, "+="),
            BinaryOperation::InplaceAnd => write!(f, "&="),
            BinaryOperation::InplaceFloorDivide => write!(f, "//="),
            BinaryOperation::InplaceLshift => write!(f, "<<="),
            BinaryOperation::InplaceMatrixMultiply => write!(f, "@="),
            BinaryOperation::InplaceMultiply => write!(f, "*="),
            BinaryOperation::InplaceRemainder => write!(f, "%="),
            BinaryOperation::InplaceOr => write!(f, "|="),
            BinaryOperation::InplacePower => write!(f, "**="),
            BinaryOperation::InplaceRshift => write!(f, ">>="),
            BinaryOperation::InplaceSubtract => write!(f, "-="),
            BinaryOperation::InplaceTrueDivide => write!(f, "/="),
            BinaryOperation::InplaceXor => write!(f, "^="),
            BinaryOperation::Invalid(v) => write!(f, "Invalid({})", v),
        }
    }
}

impl From<u32> for BinaryOperation {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Add,
            1 => Self::And,
            2 => Self::FloorDivide,
            3 => Self::Lshift,
            4 => Self::MatrixMultiply,
            5 => Self::Multiply,
            6 => Self::Remainder,
            7 => Self::Or,
            8 => Self::Power,
            9 => Self::Rshift,
            10 => Self::Subtract,
            11 => Self::TrueDivide,
            12 => Self::Xor,
            13 => Self::InplaceAdd,
            14 => Self::InplaceAnd,
            15 => Self::InplaceFloorDivide,
            16 => Self::InplaceLshift,
            17 => Self::InplaceMatrixMultiply,
            18 => Self::InplaceMultiply,
            19 => Self::InplaceRemainder,
            20 => Self::InplaceOr,
            21 => Self::InplacePower,
            22 => Self::InplaceRshift,
            23 => Self::InplaceSubtract,
            24 => Self::InplaceTrueDivide,
            25 => Self::InplaceXor,
            v => Self::Invalid(v),
        }
    }
}

impl From<&BinaryOperation> for u32 {
    fn from(val: &BinaryOperation) -> Self {
        match val {
            BinaryOperation::Add => 0,
            BinaryOperation::And => 1,
            BinaryOperation::FloorDivide => 2,
            BinaryOperation::Lshift => 3,
            BinaryOperation::MatrixMultiply => 4,
            BinaryOperation::Multiply => 5,
            BinaryOperation::Remainder => 6,
            BinaryOperation::Or => 7,
            BinaryOperation::Power => 8,
            BinaryOperation::Rshift => 9,
            BinaryOperation::Subtract => 10,
            BinaryOperation::TrueDivide => 11,
            BinaryOperation::Xor => 12,
            BinaryOperation::InplaceAdd => 13,
            BinaryOperation::InplaceAnd => 14,
            BinaryOperation::InplaceFloorDivide => 15,
            BinaryOperation::InplaceLshift => 16,
            BinaryOperation::InplaceMatrixMultiply => 17,
            BinaryOperation::InplaceMultiply => 18,
            BinaryOperation::InplaceRemainder => 19,
            BinaryOperation::InplaceOr => 20,
            BinaryOperation::InplacePower => 21,
            BinaryOperation::InplaceRshift => 22,
            BinaryOperation::InplaceSubtract => 23,
            BinaryOperation::InplaceTrueDivide => 24,
            BinaryOperation::InplaceXor => 25,
            BinaryOperation::Invalid(v) => *v,
        }
    }
}

/// awaitable locations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AwaitableWhere {
    NoLocation,
    AfterAenter,
    AfterAexit,
    Invalid(u32),
}

impl From<u32> for AwaitableWhere {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::NoLocation,
            1 => Self::AfterAenter,
            2 => Self::AfterAexit,
            v => Self::Invalid(v),
        }
    }
}

impl From<&AwaitableWhere> for u32 {
    fn from(val: &AwaitableWhere) -> Self {
        match val {
            AwaitableWhere::NoLocation => 0,
            AwaitableWhere::AfterAenter => 1,
            AwaitableWhere::AfterAexit => 2,
            AwaitableWhere::Invalid(v) => *v,
        }
    }
}

/// resume locations. Used purely for internal tracing, debugging and optimization checks in the python runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeWhere {
    StartFunction,
    AfterYield,
    AfterYieldFrom,
    AfterAwait,
    Invalid(u32),
}

impl From<u32> for ResumeWhere {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::StartFunction,
            1 => Self::AfterYield,
            2 => Self::AfterYieldFrom,
            3 => Self::AfterAwait,
            _ => Self::Invalid(value),
        }
    }
}

impl From<&ResumeWhere> for u32 {
    fn from(val: &ResumeWhere) -> Self {
        match val {
            ResumeWhere::StartFunction => 0,
            ResumeWhere::AfterYield => 1,
            ResumeWhere::AfterYieldFrom => 2,
            ResumeWhere::AfterAwait => 3,
            ResumeWhere::Invalid(v) => *v,
        }
    }
}

/// Represents a resolved reference to a variable in the "fast locals" storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClosureRefIndex {
    pub index: u32,
}

/// Used to represent the different comparison operations for COMPARE_OP
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOperation {
    Smaller,
    SmallerOrEqual,
    Equal,
    NotEqual,
    Bigger,
    BiggerOrEqual,
    /// We try to support invalid bytecode, so we have to represent it somehow.
    Invalid(u32),
}

impl fmt::Display for CompareOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompareOperation::Smaller => write!(f, "<"),
            CompareOperation::SmallerOrEqual => write!(f, "<="),
            CompareOperation::Equal => write!(f, "=="),
            CompareOperation::NotEqual => write!(f, "!="),
            CompareOperation::Bigger => write!(f, ">"),
            CompareOperation::BiggerOrEqual => write!(f, ">="),
            CompareOperation::Invalid(v) => write!(f, "Invalid({})", v),
        }
    }
}

impl From<u32> for CompareOperation {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Smaller,
            1 => Self::SmallerOrEqual,
            2 => Self::Equal,
            3 => Self::NotEqual,
            4 => Self::Bigger,
            5 => Self::BiggerOrEqual,
            _ => Self::Invalid(value),
        }
    }
}

impl From<&CompareOperation> for u32 {
    fn from(val: &CompareOperation) -> Self {
        match val {
            CompareOperation::Smaller => 0,
            CompareOperation::SmallerOrEqual => 1,
            CompareOperation::Equal => 2,
            CompareOperation::NotEqual => 3,
            CompareOperation::Bigger => 4,
            CompareOperation::BiggerOrEqual => 5,
            CompareOperation::Invalid(v) => *v,
        }
    }
}

/// Whether *_OP is inverted or not
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpInversion {
    NoInvert,
    Invert,
    Invalid(u32),
}

impl fmt::Display for OpInversion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpInversion::NoInvert => write!(f, ""),
            OpInversion::Invert => write!(f, "not"),
            OpInversion::Invalid(v) => write!(f, "Invalid({})", v),
        }
    }
}

impl From<u32> for OpInversion {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::NoInvert,
            1 => Self::Invert,
            _ => Self::Invalid(value),
        }
    }
}

impl From<&OpInversion> for u32 {
    fn from(val: &OpInversion) -> Self {
        match val {
            OpInversion::NoInvert => 0,
            OpInversion::Invert => 1,
            OpInversion::Invalid(v) => *v,
        }
    }
}

/// The different types of raising forms. See https://docs.python.org/3.10/library/dis.html#opcode-RAISE_VARARGS
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaiseForms {
    ReraisePrev,
    RaiseTOS,
    RaiseTOS1FromTOS,
    Invalid(u32),
}

impl fmt::Display for RaiseForms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RaiseForms::ReraisePrev => write!(f, "reraise previous exception"),
            RaiseForms::RaiseTOS => write!(f, "raise TOS"),
            RaiseForms::RaiseTOS1FromTOS => {
                write!(f, "raise exception at TOS1 with __cause__ set to TOS")
            }
            RaiseForms::Invalid(v) => write!(f, "Invalid({})", v),
        }
    }
}

impl From<u32> for RaiseForms {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::ReraisePrev,
            1 => Self::RaiseTOS,
            2 => Self::RaiseTOS1FromTOS,
            _ => Self::Invalid(value),
        }
    }
}

impl From<&RaiseForms> for u32 {
    fn from(val: &RaiseForms) -> Self {
        match val {
            RaiseForms::ReraisePrev => 0,
            RaiseForms::RaiseTOS => 1,
            RaiseForms::RaiseTOS1FromTOS => 2,
            RaiseForms::Invalid(v) => *v,
        }
    }
}

/// The different types of reraising. See https://docs.python.org/3.10/library/dis.html#opcode-RERAISE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reraise {
    ReraiseTOS,
    ReraiseTOSAndSetLasti(u32), // If oparg is non-zero, restores f_lasti of the current frame to its value when the exception was raised.
}

impl fmt::Display for Reraise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reraise::ReraiseTOS => write!(f, "reraise TOS"),
            Reraise::ReraiseTOSAndSetLasti(_) => write!(f, "raise TOS and set the last_i of the current frame to its value when the exception was raised."),
        }
    }
}

impl From<u32> for Reraise {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::ReraiseTOS,
            v => Self::ReraiseTOSAndSetLasti(v),
        }
    }
}

impl From<&Reraise> for u32 {
    fn from(val: &Reraise) -> Self {
        match val {
            Reraise::ReraiseTOS => 0,
            Reraise::ReraiseTOSAndSetLasti(v) => *v,
        }
    }
}

/// Describes the configuration for a CALL_FUNCTION_EX instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallExFlags {
    /// The call has positional arguments only.
    /// Stack layout (top to bottom):
    /// - Positional args (an iterable)
    /// - Callable
    PositionalOnly,

    /// The call has both positional and keyword arguments.
    /// Stack layout (top to bottom):
    /// - Keyword args (a mapping)
    /// - Positional args (an iterable)
    /// - Callable
    WithKeywords,
    Invalid(u32),
}

impl fmt::Display for CallExFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallExFlags::PositionalOnly => write!(f, "positional args only"),
            CallExFlags::WithKeywords => write!(f, "args with keywords"),
            CallExFlags::Invalid(v) => write!(f, "Invalid({})", v),
        }
    }
}

impl From<u32> for CallExFlags {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::PositionalOnly,
            1 => Self::WithKeywords,
            _ => Self::Invalid(value),
        }
    }
}

impl From<&CallExFlags> for u32 {
    fn from(val: &CallExFlags) -> Self {
        match val {
            CallExFlags::PositionalOnly => 0,
            CallExFlags::WithKeywords => 1,
            CallExFlags::Invalid(v) => *v,
        }
    }
}

bitflags! {
    /// Describes which optional data for a new function is present on the stack.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MakeFunctionFlags: u32 { // Or u8 if the arg is always a byte
        /// A tuple of default values for positional args.
        const POS_DEFAULTS = 0x01;
        /// A dictionary of keyword-only default values.
        const KW_DEFAULTS  = 0x02;
        /// A tuple of parameter annotations.
        const ANNOTATIONS  = 0x04;
        /// A tuple of cells for free variables (a closure).
        const CLOSURE      = 0x08;
    }
}

impl fmt::Display for MakeFunctionFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.contains(MakeFunctionFlags::POS_DEFAULTS) {
            parts.push("POS_DEFAULTS");
        }
        if self.contains(MakeFunctionFlags::KW_DEFAULTS) {
            parts.push("KW_DEFAULTS");
        }
        if self.contains(MakeFunctionFlags::ANNOTATIONS) {
            parts.push("ANNOTATIONS");
        }
        if self.contains(MakeFunctionFlags::CLOSURE) {
            parts.push("CLOSURE");
        }

        write!(f, "{}", parts.join(", "))
    }
}

/// BUILD_SLICE gets an argc but it must be 2 or 3. See https://docs.python.org/3.10/library/dis.html#opcode-BUILD_SLICE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceCount {
    Two,
    Three,
    Invalid(u32),
}

impl From<u32> for SliceCount {
    fn from(value: u32) -> Self {
        match value {
            2 => Self::Two,
            3 => Self::Three,
            _ => Self::Invalid(value),
        }
    }
}

impl From<&SliceCount> for u32 {
    fn from(val: &SliceCount) -> Self {
        match val {
            SliceCount::Two => 2,
            SliceCount::Three => 3,
            SliceCount::Invalid(v) => *v,
        }
    }
}

bitflags! {
    /// Represents the conversion to apply to a value before f-string formatting.
    /// From https://github.com/python/cpython/blob/3.10/Python/compile.c#L4349C5-L4361C7
    ///  Our oparg encodes 2 pieces of information: the conversion
    ///    character, and whether or not a format_spec was provided.

    ///    Convert the conversion char to 3 bits:
    ///        : 000  0x0  FVC_NONE   The default if nothing specified.
    ///    !s  : 001  0x1  FVC_STR
    ///    !r  : 010  0x2  FVC_REPR
    ///    !a  : 011  0x3  FVC_ASCII

    ///    next bit is whether or not we have a format spec:
    ///    yes : 100  0x4
    ///    no  : 000  0x0
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct FormatFlag: u8 {
        /// No conversion (default)
        const FVC_NONE = 0x0;
        /// !s conversion
        const FVC_STR = 0x1;
        /// !r conversion
        const FVC_REPR = 0x2;
        /// !a conversion
        const FVC_ASCII = 0x3;
        /// Format spec is present
        const FVS_MASK = 0x4;
    }
}

impl fmt::Display for FormatFlag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let conv = match self.bits() & 0x3 {
            0x0 => "",
            0x1 => "str",
            0x2 => "repr",
            0x3 => "ascii",
            _ => "invalid",
        };

        if self.contains(FormatFlag::FVS_MASK) {
            write!(f, "{} with format", conv)
        } else {
            write!(f, "{}", conv)
        }
    }
}

impl From<u32> for FormatFlag {
    fn from(value: u32) -> Self {
        FormatFlag::from_bits_retain(value as u8)
    }
}

impl From<&FormatFlag> for u32 {
    fn from(val: &FormatFlag) -> Self {
        val.bits() as u32
    }
}

/// Generator kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenKind {
    Generator,
    Coroutine,
    AsyncGenerator,
    Invalid(u32),
}

impl fmt::Display for GenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenKind::Generator => write!(f, "generator"),
            GenKind::Coroutine => write!(f, "coroutine"),
            GenKind::AsyncGenerator => write!(f, "async generator"),
            GenKind::Invalid(v) => write!(f, "Invalid({})", v),
        }
    }
}

impl From<u32> for GenKind {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Generator,
            1 => Self::Coroutine,
            2 => Self::AsyncGenerator,
            _ => Self::Invalid(value),
        }
    }
}

impl From<&GenKind> for u32 {
    fn from(val: &GenKind) -> Self {
        match val {
            GenKind::Generator => 0,
            GenKind::Coroutine => 1,
            GenKind::AsyncGenerator => 2,
            GenKind::Invalid(v) => *v,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Pyc {
    pub python_version: python_marshal::magic::PyVersion,
    pub timestamp: u32,
    pub hash: u64,
    pub code_object: Code,
}

impl TryFrom<python_marshal::PycFile> for Pyc {
    type Error = Error;

    fn try_from(pyc: python_marshal::PycFile) -> Result<Self, Self::Error> {
        Ok(Pyc {
            python_version: pyc.python_version,
            timestamp: pyc
                .timestamp
                .ok_or(Error::UnsupportedVersion(pyc.python_version))?,
            hash: pyc.hash,
            code_object: (pyc.object, pyc.references).try_into()?,
        })
    }
}
