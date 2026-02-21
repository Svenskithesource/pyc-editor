#![allow(non_snake_case)]

use crate::error::Error;
use crate::traits::BranchReasonTrait;
use crate::traits::GenericOpcode;
use crate::traits::StackEffectTrait;
use crate::utils::StackEffect;
use crate::v312::instructions::Instruction;

use python_instruction_dsl_proc::define_opcodes;

// From https://github.com/python/cpython/blob/3.12/Include/opcode.h
define_opcodes!(
    // Custom syntax to specify what values get pushed when an exception is raised
    *EXCEPTION ( -- lasti[if lasti && jump {1} else {0}], exc[if jump {1} else {0}]),
    CACHE = 0 ( -- ),
    POP_TOP = 1 (value -- ),
    PUSH_NULL = 2 (-- res),
    INTERPRETER_EXIT = 3 (return_value --),
    END_FOR = 4 (first, second -- ),
    END_SEND = 5 (receiver, value -- value),
    NOP = 9 ( -- ),
    UNARY_NEGATIVE = 11 (value -- res),
    UNARY_NOT = 12 (value -- res),
    UNARY_INVERT = 15 (value -- res),
    RESERVED = 17 ( -- ),
    BINARY_SUBSCR = 25 (container, sub -- res),
    BINARY_SLICE = 26 (container, start, stop -- res),
    STORE_SLICE = 27 (value, container, start, stop -- ),
    GET_LEN = 30 (obj -- obj, length),
    MATCH_MAPPING = 31 (subject -- subject, res),
    MATCH_SEQUENCE = 32 (subject -- subject, res),
    MATCH_KEYS = 33 (subject, keys -- subject, keys, values_or_none),
    PUSH_EXC_INFO = 35 (new_exc -- prev_exc, new_exc),
    CHECK_EXC_MATCH = 36 (left, right -- left, boolean),
    CHECK_EG_MATCH = 37 (exc_value, match_type -- rest, match_group),
    WITH_EXCEPT_START = 49 (exit_func, lasti, unused, val -- exit_func, lasti, unused, val, res),
    GET_AITER = 50 (obj -- iter),
    GET_ANEXT = 51 (aiter -- aiter, awaitable),
    BEFORE_ASYNC_WITH = 52 (mgr -- exit, res),
    BEFORE_WITH = 53 (mgr -- exit, res),
    END_ASYNC_FOR = 54 (awaitable, exc -- ),
    CLEANUP_THROW = 55 (sub_iter, last_sent_val, exc_value -- none, value),
    STORE_SUBSCR = 60 (value, container, sub -- ),
    DELETE_SUBSCR = 61 (container, sub -- ),
    GET_ITER = 68 (iterable -- iter),
    GET_YIELD_FROM_ITER = 69 (iterable -- iter),
    LOAD_BUILD_CLASS = 71 ( -- bc),
    LOAD_ASSERTION_ERROR = 74 ( -- value),
    RETURN_GENERATOR = 75 ( -- ),
    RETURN_VALUE = 83 (retval -- ),
    SETUP_ANNOTATIONS = 85 ( -- ),
    LOAD_LOCALS = 87 ( -- locals),
    POP_EXCEPT = 89 (exc_value -- ),
    STORE_NAME = 90 (value -- ),
    DELETE_NAME = 91 ( -- ),
    UNPACK_SEQUENCE = 92 (seq -- unpacked[oparg]),
    FOR_ITER = 93 (iter -- iter, next),
    UNPACK_EX = 94 (seq -- before[oparg & 0xFF], leftover, after[oparg >> 8]),
    STORE_ATTR = 95 (value, owner --),
    DELETE_ATTR = 96 (owner --),
    STORE_GLOBAL = 97 (value -- ),
    DELETE_GLOBAL = 98 ( -- ),
    SWAP = 99 (bottom, unused[oparg-2], top -- top, unused[oparg-2], bottom),
    LOAD_CONST = 100 ( -- value),
    LOAD_NAME = 101 ( -- value),
    BUILD_TUPLE = 102 (values[oparg] -- tup),
    BUILD_LIST = 103 (values[oparg] -- list),
    BUILD_SET = 104 (values[oparg] -- set),
    BUILD_MAP = 105 (values[oparg*2] -- map),
    LOAD_ATTR = 106 (owner -- res2[if oparg & 1 != 0 {1} else {0}], res),
    COMPARE_OP = 107 (left, right -- res),
    IMPORT_NAME = 108 (level, fromlist -- res),
    IMPORT_FROM = 109 (from -- from, res),
    JUMP_FORWARD = 110 ( -- ),
    POP_JUMP_IF_FALSE = 114 (condition -- ),
    POP_JUMP_IF_TRUE = 115 (condition -- ),
    LOAD_GLOBAL = 116 ( -- null[if oparg & 1 != 0 {1} else {0}], v),
    IS_OP = 117 (left, right -- boolean),
    CONTAINS_OP = 118 (left, right -- boolean),
    RERAISE = 119 (values[oparg], exc -- values[oparg]),
    COPY = 120 (bottom, unused[oparg-1] -- bottom, unused[oparg-1], top),
    RETURN_CONST = 121 ( -- ),
    BINARY_OP = 122 (left, right -- res),
    SEND = 123 (receiver, value -- receiver, return_value),
    LOAD_FAST = 124 ( -- value),
    STORE_FAST = 125 (value -- ),
    DELETE_FAST = 126 ( -- ),
    LOAD_FAST_CHECK = 127 ( -- value),
    POP_JUMP_IF_NOT_NONE = 128 (value -- ),
    POP_JUMP_IF_NONE = 129 (value -- ),
    RAISE_VARARGS = 130 (args[oparg] -- ),
    GET_AWAITABLE = 131 (iterable -- iter),
    MAKE_FUNCTION = 132 (defaults[if oparg & 0x01 != 0 {1} else {0}],
                        kwdefaults[if oparg & 0x02 != 0 {1} else {0}],
                        annotations[if oparg & 0x04 != 0 {1} else {0}],
                        closure[if oparg & 0x08 != 0 {1} else {0}],
                        code_obj -- func),
    BUILD_SLICE = 133 (start, stop, step[if oparg == 3 {1} else {0}] -- slice),
    JUMP_BACKWARD_NO_INTERRUPT = 134 ( -- ),
    MAKE_CELL = 135 ( -- ),
    LOAD_CLOSURE = 136 ( -- value),
    LOAD_DEREF = 137 ( -- value),
    STORE_DEREF = 138 (value -- ),
    DELETE_DEREF = 139 ( -- ),
    JUMP_BACKWARD = 140 ( -- ),
    LOAD_SUPER_ATTR = 141 (global_super, class, self_ -- res2[if oparg & 1 != 0 {1} else {0}], res),
    CALL_FUNCTION_EX = 142 (unused, callable, args, kwargs[if oparg & 0x01 != 0 {1} else {0}] -- res),
    LOAD_FAST_AND_CLEAR = 143 ( -- value),
    EXTENDED_ARG = 144 ( -- ),
    LIST_APPEND = 145 (list, unused[oparg-1], value -- list, unused[oparg-1]),
    SET_ADD = 146 (set, unused[oparg-1], v -- set, unused[oparg-1]),
    MAP_ADD = 147 (key, value -- ),
    COPY_FREE_VARS = 149 ( -- ),
    YIELD_VALUE = 150 (retval -- received_value),
    RESUME = 151 ( -- ),
    MATCH_CLASS = 152 (subject, cmp_type, names -- attrs_or_none),
    // FVS_MASK = FVS_HAVE_SPEC = 0x4
    FORMAT_VALUE = 155 (value, fmt_spec[if (oparg & 0x4) == 0x4 {1} else {0}] -- result),
    BUILD_CONST_KEY_MAP = 156 (values[oparg], keys -- map),
    BUILD_STRING = 157 (pieces[oparg] -- string),
    LIST_EXTEND = 162 (list, unused[oparg-1], iterable -- list, unused[oparg-1]),
    SET_UPDATE = 163 (set, unused[oparg-1], iterable -- set, unused[oparg-1]),
    DICT_MERGE = 164 (update -- ),
    DICT_UPDATE = 165 (update -- ),
    CALL = 171 (method_or_null, self_or_callable, args[oparg] -- res),
    KW_NAMES = 172 ( -- ),
    CALL_INTRINSIC_1 = 173 (value -- res),
    CALL_INTRINSIC_2 = 174 (value2, value1 -- res),
    LOAD_FROM_DICT_OR_GLOBALS = 175 (mod_or_class_dict -- v),
    LOAD_FROM_DICT_OR_DEREF = 176 (class_dict -- value),
    // Specialized ops
    INSTRUMENTED_LOAD_SUPER_ATTR = 237 ( / ),
    INSTRUMENTED_POP_JUMP_IF_NONE = 238 ( / ),
    INSTRUMENTED_POP_JUMP_IF_NOT_NONE = 239 ( / ),
    INSTRUMENTED_RESUME = 240 ( / ),
    INSTRUMENTED_CALL = 241 ( / ),
    INSTRUMENTED_RETURN_VALUE = 242 ( / ),
    INSTRUMENTED_YIELD_VALUE = 243 ( / ),
    INSTRUMENTED_CALL_FUNCTION_EX = 244 ( / ),
    INSTRUMENTED_JUMP_FORWARD = 245 ( / ),
    INSTRUMENTED_JUMP_BACKWARD = 246 ( / ),
    INSTRUMENTED_RETURN_CONST = 247 ( / ),
    INSTRUMENTED_FOR_ITER = 248 ( / ),
    INSTRUMENTED_POP_JUMP_IF_FALSE = 249 ( / ),
    INSTRUMENTED_POP_JUMP_IF_TRUE = 250 ( / ),
    INSTRUMENTED_END_FOR = 251 ( / ),
    INSTRUMENTED_END_SEND = 252 ( / ),
    INSTRUMENTED_INSTRUCTION = 253 ( / ),
    INSTRUMENTED_LINE = 254 ( / ),
    // We skip psuedo opcodes as they can never appear in actual bytecode
    // MIN_PSEUDO_OPCODE = 256,
    // SETUP_FINALLY = 256,
    // SETUP_CLEANUP = 257,
    // SETUP_WITH = 258,
    // POP_BLOCK = 259,
    // JUMP = 260,
    // JUMP_NO_INTERRUPT = 261,
    // LOAD_METHOD = 262,
    // LOAD_SUPER_METHOD = 263,
    // LOAD_ZERO_SUPER_METHOD = 264,
    // LOAD_ZERO_SUPER_ATTR = 265,
    // STORE_FAST_MAYBE_NULL = 266,
    // MAX_PSEUDO_OPCODE = 266,
    BINARY_OP_ADD_FLOAT = 6 ( / ),
    BINARY_OP_ADD_INT = 7 ( / ),
    BINARY_OP_ADD_UNICODE = 8 ( / ),
    BINARY_OP_INPLACE_ADD_UNICODE = 10 ( / ),
    BINARY_OP_MULTIPLY_FLOAT = 13 ( / ),
    BINARY_OP_MULTIPLY_INT = 14 ( / ),
    BINARY_OP_SUBTRACT_FLOAT = 16 ( / ),
    BINARY_OP_SUBTRACT_INT = 18 ( / ),
    BINARY_SUBSCR_DICT = 19 ( / ),
    BINARY_SUBSCR_GETITEM = 20 ( / ),
    BINARY_SUBSCR_LIST_INT = 21 ( / ),
    BINARY_SUBSCR_TUPLE_INT = 22 ( / ),
    CALL_PY_EXACT_ARGS = 23 ( / ),
    CALL_PY_WITH_DEFAULTS = 24 ( / ),
    CALL_BOUND_METHOD_EXACT_ARGS = 28 ( / ),
    CALL_BUILTIN_CLASS = 29 ( / ),
    CALL_BUILTIN_FAST_WITH_KEYWORDS = 34 ( / ),
    CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS = 38 ( / ),
    CALL_NO_KW_BUILTIN_FAST = 39 ( / ),
    CALL_NO_KW_BUILTIN_O = 40 ( / ),
    CALL_NO_KW_ISINSTANCE = 41 ( / ),
    CALL_NO_KW_LEN = 42 ( / ),
    CALL_NO_KW_LIST_APPEND = 43 ( / ),
    CALL_NO_KW_METHOD_DESCRIPTOR_FAST = 44 ( / ),
    CALL_NO_KW_METHOD_DESCRIPTOR_NOARGS = 45 ( / ),
    CALL_NO_KW_METHOD_DESCRIPTOR_O = 46 ( / ),
    CALL_NO_KW_STR_1 = 47 ( / ),
    CALL_NO_KW_TUPLE_1 = 48 ( / ),
    CALL_NO_KW_TYPE_1 = 56 ( / ),
    COMPARE_OP_FLOAT = 57 ( / ),
    COMPARE_OP_INT = 58 ( / ),
    COMPARE_OP_STR = 59 ( / ),
    FOR_ITER_LIST = 62 ( / ),
    FOR_ITER_TUPLE = 63 ( / ),
    FOR_ITER_RANGE = 64 ( / ),
    FOR_ITER_GEN = 65 ( / ),
    LOAD_SUPER_ATTR_ATTR = 66 ( / ),
    LOAD_SUPER_ATTR_METHOD = 67 ( / ),
    LOAD_ATTR_CLASS = 70 ( / ),
    LOAD_ATTR_GETATTRIBUTE_OVERRIDDEN = 72 ( / ),
    LOAD_ATTR_INSTANCE_VALUE = 73 ( / ),
    LOAD_ATTR_MODULE = 76 ( / ),
    LOAD_ATTR_PROPERTY = 77 ( / ),
    LOAD_ATTR_SLOT = 78 ( / ),
    LOAD_ATTR_WITH_HINT = 79 ( / ),
    LOAD_ATTR_METHOD_LAZY_DICT = 80 ( / ),
    LOAD_ATTR_METHOD_NO_DICT = 81 ( / ),
    LOAD_ATTR_METHOD_WITH_VALUES = 82 ( / ),
    LOAD_CONST__LOAD_FAST = 84 ( / ),
    LOAD_FAST__LOAD_CONST = 86 ( / ),
    LOAD_FAST__LOAD_FAST = 88 ( / ),
    LOAD_GLOBAL_BUILTIN = 111 ( / ),
    LOAD_GLOBAL_MODULE = 112 ( / ),
    STORE_ATTR_INSTANCE_VALUE = 113 ( / ),
    STORE_ATTR_SLOT = 148 ( / ),
    STORE_ATTR_WITH_HINT = 153 ( / ),
    STORE_FAST__LOAD_FAST = 154 ( / ),
    STORE_FAST__STORE_FAST = 158 ( / ),
    STORE_SUBSCR_DICT = 159 ( / ),
    STORE_SUBSCR_LIST_INT = 160 ( / ),
    UNPACK_SEQUENCE_LIST = 161 ( / ),
    UNPACK_SEQUENCE_TUPLE = 166 ( / ),
    UNPACK_SEQUENCE_TWO_TUPLE = 167 ( / ),
    SEND_GEN = 168 ( / ),
);

impl GenericOpcode for Opcode {
    /// There are no absolute jumps in 3.12. Only an exception unwind can trigger an absolute jump.
    fn is_absolute_jump(&self) -> bool {
        false
    }

    /// From: https://github.com/python/cpython/blob/3.11/Lib/opcode.py#L28
    fn is_relative_jump(&self) -> bool {
        matches!(
            self,
            Opcode::FOR_ITER
                | Opcode::JUMP_FORWARD
                | Opcode::POP_JUMP_IF_FALSE
                | Opcode::POP_JUMP_IF_TRUE
                | Opcode::SEND
                | Opcode::POP_JUMP_IF_NOT_NONE
                | Opcode::POP_JUMP_IF_NONE
                | Opcode::JUMP_BACKWARD_NO_INTERRUPT
                | Opcode::JUMP_BACKWARD
                | Opcode::FOR_ITER_RANGE
                | Opcode::FOR_ITER_LIST
                | Opcode::FOR_ITER_GEN
                | Opcode::FOR_ITER_TUPLE
                | Opcode::INSTRUMENTED_FOR_ITER
                | Opcode::INSTRUMENTED_POP_JUMP_IF_NONE
                | Opcode::INSTRUMENTED_POP_JUMP_IF_NOT_NONE
                | Opcode::INSTRUMENTED_JUMP_FORWARD
                | Opcode::INSTRUMENTED_JUMP_BACKWARD
                | Opcode::INSTRUMENTED_POP_JUMP_IF_FALSE
                | Opcode::INSTRUMENTED_POP_JUMP_IF_TRUE
        )
    }

    fn is_jump_forwards(&self) -> bool {
        matches!(
            self,
            Opcode::FOR_ITER
                | Opcode::JUMP_FORWARD
                | Opcode::POP_JUMP_IF_FALSE
                | Opcode::POP_JUMP_IF_TRUE
                | Opcode::SEND
                | Opcode::POP_JUMP_IF_NOT_NONE
                | Opcode::POP_JUMP_IF_NONE
                | Opcode::FOR_ITER_RANGE
                | Opcode::FOR_ITER_LIST
                | Opcode::FOR_ITER_GEN
                | Opcode::FOR_ITER_TUPLE
                | Opcode::INSTRUMENTED_FOR_ITER
                | Opcode::INSTRUMENTED_POP_JUMP_IF_NONE
                | Opcode::INSTRUMENTED_POP_JUMP_IF_NOT_NONE
                | Opcode::INSTRUMENTED_JUMP_FORWARD
                | Opcode::INSTRUMENTED_POP_JUMP_IF_FALSE
                | Opcode::INSTRUMENTED_POP_JUMP_IF_TRUE
        )
    }

    fn is_jump_backwards(&self) -> bool {
        matches!(
            self,
            Opcode::JUMP_BACKWARD_NO_INTERRUPT
                | Opcode::JUMP_BACKWARD
                | Opcode::INSTRUMENTED_JUMP_BACKWARD
        )
    }

    /// Relative or absolute jump
    fn is_jump(&self) -> bool {
        self.is_absolute_jump() | self.is_relative_jump()
    }

    fn is_conditional_jump(&self) -> bool {
        matches!(
            self,
            Opcode::POP_JUMP_IF_FALSE
                | Opcode::POP_JUMP_IF_TRUE
                | Opcode::POP_JUMP_IF_NOT_NONE
                | Opcode::POP_JUMP_IF_NONE
                | Opcode::INSTRUMENTED_POP_JUMP_IF_FALSE
                | Opcode::INSTRUMENTED_POP_JUMP_IF_TRUE
                | Opcode::INSTRUMENTED_POP_JUMP_IF_NONE
                | Opcode::INSTRUMENTED_POP_JUMP_IF_NOT_NONE
                | Opcode::FOR_ITER
                | Opcode::FOR_ITER_GEN
                | Opcode::FOR_ITER_LIST
                | Opcode::FOR_ITER_RANGE
                | Opcode::FOR_ITER_TUPLE
                | Opcode::INSTRUMENTED_FOR_ITER
                | Opcode::SEND // If the send call raises StopIteration, it jumps
        )
    }

    fn stops_execution(&self) -> bool {
        matches!(
            self,
            Opcode::RETURN_VALUE | Opcode::RETURN_CONST | Opcode::RAISE_VARARGS | Opcode::RERAISE
        )
    }

    fn is_extended_arg(&self) -> bool {
        matches!(self, Opcode::EXTENDED_ARG)
    }

    fn get_nop() -> Self {
        Opcode::NOP
    }
}

#[derive(Clone, Debug)]
pub enum BranchReason {
    Opcode(Opcode),
    /// Bool is the `lasti` field of the `ExceptionTableEntry`
    Exception(bool),
}

impl std::fmt::Display for BranchReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BranchReason::Opcode(opcode) => write!(f, "{:#?}", opcode),
            BranchReason::Exception(lasti) => {
                write!(
                    f,
                    "EXCEPTION({})",
                    if *lasti { "lasti" } else { "no lasti" }
                )
            }
        }
    }
}

impl BranchReasonTrait for BranchReason {
    type Opcode = Opcode;

    fn from_exception(lasti: bool) -> Result<Self, Error> {
        Ok(BranchReason::Exception(lasti))
    }

    fn from_opcode(opcode: Opcode) -> Result<Self, Error> {
        Ok(BranchReason::Opcode(opcode))
    }

    fn is_opcode(&self) -> bool {
        matches!(self, BranchReason::Opcode(_))
    }

    fn is_exception(&self) -> bool {
        matches!(self, BranchReason::Exception(_))
    }

    fn get_opcode(&self) -> Option<&Opcode> {
        match self {
            BranchReason::Opcode(opcode) => Some(opcode),
            BranchReason::Exception(_) => None,
        }
    }

    fn get_lasti(&self) -> Option<bool> {
        match self {
            BranchReason::Opcode(_) => None,
            BranchReason::Exception(lasti) => Some(*lasti),
        }
    }
}
