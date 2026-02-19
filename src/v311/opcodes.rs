#![allow(non_snake_case)]

use crate::error::Error;
use crate::traits::BranchReasonTrait;
use crate::traits::GenericOpcode;
use crate::traits::StackEffectTrait;
use crate::utils::StackEffect;
use crate::v311::instructions::Instruction;

use python_instruction_dsl_proc::define_opcodes;

// From https://github.com/python/cpython/blob/3.11/Include/opcode.h
define_opcodes!(
    CACHE = 0 ( -- ),
    POP_TOP = 1 (top --),
    PUSH_NULL = 2 ( -- null),
    NOP = 9 ( -- ),
    UNARY_POSITIVE = 10 (value -- res),
    UNARY_NEGATIVE = 11 (value -- res),
    UNARY_NOT = 12 (value -- res),
    UNARY_INVERT = 15 (value -- res),
    BINARY_SUBSCR = 25 (container, sub -- res),
    GET_LEN = 30 (obj -- obj, length),
    MATCH_MAPPING = 31 (subject -- subject, res),
    MATCH_SEQUENCE = 32 (subject -- subject, res),
    MATCH_KEYS = 33 (subject, keys -- subject, keys, values_or_none),
    PUSH_EXC_INFO = 35 ( -- exc),
    CHECK_EXC_MATCH = 36 (left_exc, right_exc -- left_exc, boolean),
    CHECK_EG_MATCH = 37 (exc_value, match_type -- rest, match_group),
    WITH_EXCEPT_START = 49 (exit_func, lasti, unused, val -- exit_func, lasti, unused, val, res),
    GET_AITER = 50 (obj -- iter),
    GET_ANEXT = 51 (aiter -- aiter, awaitable),
    BEFORE_ASYNC_WITH = 52 (mgr -- exit, res),
    BEFORE_WITH = 53 (mgr -- exit, res),
    END_ASYNC_FOR = 54 (awaitable, exc -- ),
    STORE_SUBSCR = 60 (value, container, sub --),
    DELETE_SUBSCR = 61 (container, sub --),
    GET_ITER = 68 (iterable -- iter),
    GET_YIELD_FROM_ITER = 69 (iterable -- iter),
    PRINT_EXPR = 70 (value --),
    LOAD_BUILD_CLASS = 71 (-- bc),
    LOAD_ASSERTION_ERROR = 74 ( -- value),
    RETURN_GENERATOR = 75 ( -- ),
    LIST_TO_TUPLE = 82 (list -- tuple),
    RETURN_VALUE = 83 (return_value --),
    IMPORT_STAR = 84 (from --),
    SETUP_ANNOTATIONS = 85 ( -- ),
    YIELD_VALUE = 86 (value -- received_value),
    ASYNC_GEN_WRAP = 87 (value -- wrapped),
    PREP_RERAISE_STAR = 88 (org_exc_group, reraised_group -- exc_to_reraise_or_none),
    POP_EXCEPT = 89 (exc_value -- ),
    STORE_NAME = 90 (value -- ),
    DELETE_NAME = 91 ( -- ),
    UNPACK_SEQUENCE = 92 (seq -- unpacked[oparg]),
    FOR_ITER = 93 (iter -- iter[if jump || calculate_max {0} else {1}], next[if jump || calculate_max {0} else {1}]),
    UNPACK_EX = 94 (seq -- before[oparg & 0xFF], leftover, after[oparg >> 8]),
    STORE_ATTR = 95 (value, owner --),
    DELETE_ATTR = 96 (owner --),
    STORE_GLOBAL = 97 (value --),
    DELETE_GLOBAL = 98 (--),
    SWAP = 99 (bottom, unused[oparg-2], top -- top, unused[oparg-2], bottom),
    LOAD_CONST = 100 (-- value),
    LOAD_NAME = 101 (-- value),
    BUILD_TUPLE = 102 (values[oparg] -- tuple),
    BUILD_LIST = 103 (values[oparg] -- list),
    BUILD_SET = 104 (values[oparg] -- set),
    BUILD_MAP = 105 (values[oparg*2] -- map),
    LOAD_ATTR = 106 (owner -- res),
    COMPARE_OP = 107 (left, right -- res),
    IMPORT_NAME = 108 (level, fromlist -- res),
    IMPORT_FROM = 109 (from -- from, res),
    JUMP_FORWARD = 110 ( -- ),
    JUMP_IF_FALSE_OR_POP = 111 (condition -- condition[if jump && !calculate_max {1} else {0}]),
    JUMP_IF_TRUE_OR_POP = 112 (condition -- condition[if jump && !calculate_max {1} else {0}]),
    POP_JUMP_FORWARD_IF_FALSE = 114 (condition -- ),
    POP_JUMP_FORWARD_IF_TRUE = 115 (condition -- ),
    LOAD_GLOBAL = 116 (-- null[if oparg & 0x01 != 0 {1} else {0}], value),
    IS_OP = 117 (left, right -- boolean),
    CONTAINS_OP = 118 (left, right -- boolean),
    RERAISE = 119 (values[oparg] ,exc -- values[oparg]),
    COPY = 120 (bottom, unused[oparg-1] -- bottom, unused[oparg-1], top),
    BINARY_OP = 122 (left, right -- res),
    SEND = 123 (receiver, value -- receiver[if jump && !calculate_max {0} else {1}], return_value),
    LOAD_FAST = 124 (-- value),
    STORE_FAST = 125 (value --),
    DELETE_FAST = 126 ( -- ),
    POP_JUMP_FORWARD_IF_NOT_NONE = 128 (condition -- ),
    POP_JUMP_FORWARD_IF_NONE = 129 (condition -- ),
    // oparg must be 2 or 1
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
    CALL_FUNCTION_EX = 142 (unused, callable, args, kwargs[if oparg & 0x01 != 0 {1} else {0}] -- res),
    EXTENDED_ARG = 144 ( -- ),
    LIST_APPEND = 145 (list, unused[oparg-1], value -- list, unused[oparg-1]),
    SET_ADD = 146 (set, unused[oparg-1], value -- set, unused[oparg-1]),
    MAP_ADD = 147 (map, unused[oparg-1], key, value -- map, unused[oparg-1]),
    LOAD_CLASSDEREF = 148 (-- value),
    COPY_FREE_VARS = 149 ( -- ),
    RESUME = 151 ( -- ),
    MATCH_CLASS = 152 (subject, cmp_type, names -- attrs_or_none),
    // FVS_MASK = FVS_HAVE_SPEC = 0x4
    FORMAT_VALUE = 155 (value, fmt_spec[if (oparg & 0x4) == 0x4 {1} else {0}] -- result),
    BUILD_CONST_KEY_MAP = 156 (values[oparg], keys -- map),
    BUILD_STRING = 157 (pieces[oparg] -- string),
    LOAD_METHOD = 160 (obj -- null_or_method, method_or_self),
    LIST_EXTEND = 162 (list, unused[oparg-1], iterable -- list, unused[oparg-1]),
    SET_UPDATE = 163 (set, unused[oparg-1], iterable -- set, unused[oparg-1]),
    DICT_MERGE = 164 (update --),
    DICT_UPDATE = 165 (update --),
    PRECALL = 166 ( -- ),
    CALL = 171 (method_or_null, self_or_callable, args[oparg] -- res),
    KW_NAMES = 172 ( -- ),
    POP_JUMP_BACKWARD_IF_NOT_NONE = 173 (condition -- ),
    POP_JUMP_BACKWARD_IF_NONE = 174 (condition -- ),
    POP_JUMP_BACKWARD_IF_FALSE = 175 (condition -- ),
    POP_JUMP_BACKWARD_IF_TRUE = 176 (condition -- ),
    //
    BINARY_OP_ADAPTIVE = 3 ( / ),
    BINARY_OP_ADD_FLOAT = 4 ( / ),
    BINARY_OP_ADD_INT = 5 ( / ),
    BINARY_OP_ADD_UNICODE = 6 ( / ),
    BINARY_OP_INPLACE_ADD_UNICODE = 7 ( / ),
    BINARY_OP_MULTIPLY_FLOAT = 8 ( / ),
    BINARY_OP_MULTIPLY_INT = 13 ( / ),
    BINARY_OP_SUBTRACT_FLOAT = 14 ( / ),
    BINARY_OP_SUBTRACT_INT = 16 ( / ),
    BINARY_SUBSCR_ADAPTIVE = 17 ( / ),
    BINARY_SUBSCR_DICT = 18 ( / ),
    BINARY_SUBSCR_GETITEM = 19 ( / ),
    BINARY_SUBSCR_LIST_INT = 20 ( / ),
    BINARY_SUBSCR_TUPLE_INT = 21 ( / ),
    CALL_ADAPTIVE = 22 ( / ),
    CALL_PY_EXACT_ARGS = 23 ( / ),
    CALL_PY_WITH_DEFAULTS = 24 ( / ),
    COMPARE_OP_ADAPTIVE = 26 ( / ),
    COMPARE_OP_FLOAT_JUMP = 27 ( / ),
    COMPARE_OP_INT_JUMP = 28 ( / ),
    COMPARE_OP_STR_JUMP = 29 ( / ),
    EXTENDED_ARG_QUICK = 34 ( / ),
    JUMP_BACKWARD_QUICK = 38 ( / ),
    LOAD_ATTR_ADAPTIVE = 39 ( / ),
    LOAD_ATTR_INSTANCE_VALUE = 40 ( / ),
    LOAD_ATTR_MODULE = 41 ( / ),
    LOAD_ATTR_SLOT = 42 ( / ),
    LOAD_ATTR_WITH_HINT = 43 ( / ),
    LOAD_CONST__LOAD_FAST = 44 ( / ),
    LOAD_FAST__LOAD_CONST = 45 ( / ),
    LOAD_FAST__LOAD_FAST = 46 ( / ),
    LOAD_GLOBAL_ADAPTIVE = 47 ( / ),
    LOAD_GLOBAL_BUILTIN = 48 ( / ),
    LOAD_GLOBAL_MODULE = 55 ( / ),
    LOAD_METHOD_ADAPTIVE = 56 ( / ),
    LOAD_METHOD_CLASS = 57 ( / ),
    LOAD_METHOD_MODULE = 58 ( / ),
    LOAD_METHOD_NO_DICT = 59 ( / ),
    LOAD_METHOD_WITH_DICT = 62 ( / ),
    LOAD_METHOD_WITH_VALUES = 63 ( / ),
    PRECALL_ADAPTIVE = 64 ( / ),
    PRECALL_BOUND_METHOD = 65 ( / ),
    PRECALL_BUILTIN_CLASS = 66 ( / ),
    PRECALL_BUILTIN_FAST_WITH_KEYWORDS = 67 ( / ),
    PRECALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS = 72 ( / ),
    PRECALL_NO_KW_BUILTIN_FAST = 73 ( / ),
    PRECALL_NO_KW_BUILTIN_O = 76 ( / ),
    PRECALL_NO_KW_ISINSTANCE = 77 ( / ),
    PRECALL_NO_KW_LEN = 78 ( / ),
    PRECALL_NO_KW_LIST_APPEND = 79 ( / ),
    PRECALL_NO_KW_METHOD_DESCRIPTOR_FAST = 80 ( / ),
    PRECALL_NO_KW_METHOD_DESCRIPTOR_NOARGS = 81 ( / ),
    PRECALL_NO_KW_METHOD_DESCRIPTOR_O = 113 ( / ),
    PRECALL_NO_KW_STR_1 = 121 ( / ),
    PRECALL_NO_KW_TUPLE_1 = 127 ( / ),
    PRECALL_NO_KW_TYPE_1 = 141 ( / ),
    PRECALL_PYFUNC = 143 ( / ),
    RESUME_QUICK = 150 ( / ),
    STORE_ATTR_ADAPTIVE = 153 ( / ),
    STORE_ATTR_INSTANCE_VALUE = 154 ( / ),
    STORE_ATTR_SLOT = 158 ( / ),
    STORE_ATTR_WITH_HINT = 159 ( / ),
    STORE_FAST__LOAD_FAST = 161 ( / ),
    STORE_FAST__STORE_FAST = 167 ( / ),
    STORE_SUBSCR_ADAPTIVE = 168 ( / ),
    STORE_SUBSCR_DICT = 169 ( / ),
    STORE_SUBSCR_LIST_INT = 170 ( / ),
    UNPACK_SEQUENCE_ADAPTIVE = 177 ( / ),
    UNPACK_SEQUENCE_LIST = 178 ( / ),
    UNPACK_SEQUENCE_TUPLE = 179 ( / ),
    UNPACK_SEQUENCE_TWO_TUPLE = 180 ( / ),
    DO_TRACING = 255 ( / ),
);

impl GenericOpcode for Opcode {
    /// There are no absolute jumps in 3.11. Only an exception unwind can trigger an absolute jump.
    fn is_absolute_jump(&self) -> bool {
        false
    }

    /// From: https://github.com/python/cpython/blob/3.11/Lib/opcode.py#L28
    fn is_relative_jump(&self) -> bool {
        matches!(
            self,
            Opcode::FOR_ITER
                | Opcode::JUMP_FORWARD
                | Opcode::JUMP_IF_FALSE_OR_POP
                | Opcode::JUMP_IF_TRUE_OR_POP
                | Opcode::POP_JUMP_FORWARD_IF_FALSE
                | Opcode::POP_JUMP_FORWARD_IF_TRUE
                | Opcode::SEND
                | Opcode::POP_JUMP_FORWARD_IF_NOT_NONE
                | Opcode::POP_JUMP_FORWARD_IF_NONE
                | Opcode::JUMP_BACKWARD_NO_INTERRUPT
                | Opcode::JUMP_BACKWARD
                | Opcode::JUMP_BACKWARD_QUICK
                | Opcode::POP_JUMP_BACKWARD_IF_NOT_NONE
                | Opcode::POP_JUMP_BACKWARD_IF_NONE
                | Opcode::POP_JUMP_BACKWARD_IF_FALSE
                | Opcode::POP_JUMP_BACKWARD_IF_TRUE
        )
    }

    fn is_jump_forwards(&self) -> bool {
        matches!(
            self,
            Opcode::FOR_ITER
                | Opcode::JUMP_FORWARD
                | Opcode::JUMP_IF_FALSE_OR_POP
                | Opcode::JUMP_IF_TRUE_OR_POP
                | Opcode::POP_JUMP_FORWARD_IF_FALSE
                | Opcode::POP_JUMP_FORWARD_IF_TRUE
                | Opcode::SEND
                | Opcode::POP_JUMP_FORWARD_IF_NOT_NONE
                | Opcode::POP_JUMP_FORWARD_IF_NONE
        )
    }

    fn is_jump_backwards(&self) -> bool {
        matches!(
            self,
            Opcode::JUMP_BACKWARD_NO_INTERRUPT
                | Opcode::JUMP_BACKWARD
                | Opcode::JUMP_BACKWARD_QUICK
                | Opcode::POP_JUMP_BACKWARD_IF_NOT_NONE
                | Opcode::POP_JUMP_BACKWARD_IF_NONE
                | Opcode::POP_JUMP_BACKWARD_IF_FALSE
                | Opcode::POP_JUMP_BACKWARD_IF_TRUE
        )
    }

    fn is_jump(&self) -> bool {
        self.is_absolute_jump() | self.is_relative_jump()
    }

    fn is_conditional_jump(&self) -> bool {
        matches!(
            self,
            Opcode::FOR_ITER
                | Opcode::JUMP_IF_FALSE_OR_POP
                | Opcode::JUMP_IF_TRUE_OR_POP
                | Opcode::POP_JUMP_FORWARD_IF_FALSE
                | Opcode::POP_JUMP_FORWARD_IF_TRUE
                | Opcode::POP_JUMP_FORWARD_IF_NOT_NONE
                | Opcode::POP_JUMP_FORWARD_IF_NONE
                | Opcode::POP_JUMP_BACKWARD_IF_NOT_NONE
                | Opcode::POP_JUMP_BACKWARD_IF_NONE
                | Opcode::POP_JUMP_BACKWARD_IF_FALSE
                | Opcode::POP_JUMP_BACKWARD_IF_TRUE
                | Opcode::SEND // If the send call raises StopIteration, it jumps
        )
    }

    fn stops_execution(&self) -> bool {
        matches!(
            self,
            Opcode::RETURN_VALUE | Opcode::RAISE_VARARGS | Opcode::RERAISE
        )
    }

    fn is_extended_arg(&self) -> bool {
        matches!(self, Opcode::EXTENDED_ARG | Opcode::EXTENDED_ARG_QUICK)
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
