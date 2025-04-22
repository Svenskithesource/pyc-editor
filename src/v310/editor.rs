use python_marshal::extract_strings_tuple;
use python_marshal::{extract_object, resolve_object_ref};

use crate::error::Error;

use super::code_objects::Pyc;
use super::code_objects::{Code, Instruction};
use super::opcodes::Opcode;

