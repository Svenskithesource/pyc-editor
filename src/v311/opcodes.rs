#![allow(non_snake_case)]

use crate::define_opcodes;
use crate::traits::GenericOpcode;
use crate::v311::instructions::Instruction;
use paste::paste;

// From https://github.com/python/cpython/blob/3.11/Include/opcode.h
define_opcodes!(
    CACHE = 0,
    POP_TOP = 1,
    PUSH_NULL = 2,
    NOP = 9,
    UNARY_POSITIVE = 10,
    UNARY_NEGATIVE = 11,
    UNARY_NOT = 12,
    UNARY_INVERT = 15,
    BINARY_SUBSCR = 25,
    GET_LEN = 30,
    MATCH_MAPPING = 31,
    MATCH_SEQUENCE = 32,
    MATCH_KEYS = 33,
    PUSH_EXC_INFO = 35,
    CHECK_EXC_MATCH = 36,
    CHECK_EG_MATCH = 37,
    WITH_EXCEPT_START = 49,
    GET_AITER = 50,
    GET_ANEXT = 51,
    BEFORE_ASYNC_WITH = 52,
    BEFORE_WITH = 53,
    END_ASYNC_FOR = 54,
    STORE_SUBSCR = 60,
    DELETE_SUBSCR = 61,
    GET_ITER = 68,
    GET_YIELD_FROM_ITER = 69,
    PRINT_EXPR = 70,
    LOAD_BUILD_CLASS = 71,
    LOAD_ASSERTION_ERROR = 74,
    RETURN_GENERATOR = 75,
    LIST_TO_TUPLE = 82,
    RETURN_VALUE = 83,
    IMPORT_STAR = 84,
    SETUP_ANNOTATIONS = 85,
    YIELD_VALUE = 86,
    ASYNC_GEN_WRAP = 87,
    PREP_RERAISE_STAR = 88,
    POP_EXCEPT = 89,
    STORE_NAME = 90,
    DELETE_NAME = 91,
    UNPACK_SEQUENCE = 92,
    FOR_ITER = 93,
    UNPACK_EX = 94,
    STORE_ATTR = 95,
    DELETE_ATTR = 96,
    STORE_GLOBAL = 97,
    DELETE_GLOBAL = 98,
    SWAP = 99,
    LOAD_CONST = 100,
    LOAD_NAME = 101,
    BUILD_TUPLE = 102,
    BUILD_LIST = 103,
    BUILD_SET = 104,
    BUILD_MAP = 105,
    LOAD_ATTR = 106,
    COMPARE_OP = 107,
    IMPORT_NAME = 108,
    IMPORT_FROM = 109,
    JUMP_FORWARD = 110,
    JUMP_IF_FALSE_OR_POP = 111,
    JUMP_IF_TRUE_OR_POP = 112,
    POP_JUMP_FORWARD_IF_FALSE = 114,
    POP_JUMP_FORWARD_IF_TRUE = 115,
    LOAD_GLOBAL = 116,
    IS_OP = 117,
    CONTAINS_OP = 118,
    RERAISE = 119,
    COPY = 120,
    BINARY_OP = 122,
    SEND = 123,
    LOAD_FAST = 124,
    STORE_FAST = 125,
    DELETE_FAST = 126,
    POP_JUMP_FORWARD_IF_NOT_NONE = 128,
    POP_JUMP_FORWARD_IF_NONE = 129,
    RAISE_VARARGS = 130,
    GET_AWAITABLE = 131,
    MAKE_FUNCTION = 132,
    BUILD_SLICE = 133,
    JUMP_BACKWARD_NO_INTERRUPT = 134,
    MAKE_CELL = 135,
    LOAD_CLOSURE = 136,
    LOAD_DEREF = 137,
    STORE_DEREF = 138,
    DELETE_DEREF = 139,
    JUMP_BACKWARD = 140,
    CALL_FUNCTION_EX = 142,
    EXTENDED_ARG = 144,
    LIST_APPEND = 145,
    SET_ADD = 146,
    MAP_ADD = 147,
    LOAD_CLASSDEREF = 148,
    COPY_FREE_VARS = 149,
    RESUME = 151,
    MATCH_CLASS = 152,
    FORMAT_VALUE = 155,
    BUILD_CONST_KEY_MAP = 156,
    BUILD_STRING = 157,
    LOAD_METHOD = 160,
    LIST_EXTEND = 162,
    SET_UPDATE = 163,
    DICT_MERGE = 164,
    DICT_UPDATE = 165,
    PRECALL = 166,
    CALL = 171,
    KW_NAMES = 172,
    POP_JUMP_BACKWARD_IF_NOT_NONE = 173,
    POP_JUMP_BACKWARD_IF_NONE = 174,
    POP_JUMP_BACKWARD_IF_FALSE = 175,
    POP_JUMP_BACKWARD_IF_TRUE = 176,
    BINARY_OP_ADAPTIVE = 3,
    BINARY_OP_ADD_FLOAT = 4,
    BINARY_OP_ADD_INT = 5,
    BINARY_OP_ADD_UNICODE = 6,
    BINARY_OP_INPLACE_ADD_UNICODE = 7,
    BINARY_OP_MULTIPLY_FLOAT = 8,
    BINARY_OP_MULTIPLY_INT = 13,
    BINARY_OP_SUBTRACT_FLOAT = 14,
    BINARY_OP_SUBTRACT_INT = 16,
    BINARY_SUBSCR_ADAPTIVE = 17,
    BINARY_SUBSCR_DICT = 18,
    BINARY_SUBSCR_GETITEM = 19,
    BINARY_SUBSCR_LIST_INT = 20,
    BINARY_SUBSCR_TUPLE_INT = 21,
    CALL_ADAPTIVE = 22,
    CALL_PY_EXACT_ARGS = 23,
    CALL_PY_WITH_DEFAULTS = 24,
    COMPARE_OP_ADAPTIVE = 26,
    COMPARE_OP_FLOAT_JUMP = 27,
    COMPARE_OP_INT_JUMP = 28,
    COMPARE_OP_STR_JUMP = 29,
    EXTENDED_ARG_QUICK = 34,
    JUMP_BACKWARD_QUICK = 38,
    LOAD_ATTR_ADAPTIVE = 39,
    LOAD_ATTR_INSTANCE_VALUE = 40,
    LOAD_ATTR_MODULE = 41,
    LOAD_ATTR_SLOT = 42,
    LOAD_ATTR_WITH_HINT = 43,
    LOAD_CONST__LOAD_FAST = 44,
    LOAD_FAST__LOAD_CONST = 45,
    LOAD_FAST__LOAD_FAST = 46,
    LOAD_GLOBAL_ADAPTIVE = 47,
    LOAD_GLOBAL_BUILTIN = 48,
    LOAD_GLOBAL_MODULE = 55,
    LOAD_METHOD_ADAPTIVE = 56,
    LOAD_METHOD_CLASS = 57,
    LOAD_METHOD_MODULE = 58,
    LOAD_METHOD_NO_DICT = 59,
    LOAD_METHOD_WITH_DICT = 62,
    LOAD_METHOD_WITH_VALUES = 63,
    PRECALL_ADAPTIVE = 64,
    PRECALL_BOUND_METHOD = 65,
    PRECALL_BUILTIN_CLASS = 66,
    PRECALL_BUILTIN_FAST_WITH_KEYWORDS = 67,
    PRECALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS = 72,
    PRECALL_NO_KW_BUILTIN_FAST = 73,
    PRECALL_NO_KW_BUILTIN_O = 76,
    PRECALL_NO_KW_ISINSTANCE = 77,
    PRECALL_NO_KW_LEN = 78,
    PRECALL_NO_KW_LIST_APPEND = 79,
    PRECALL_NO_KW_METHOD_DESCRIPTOR_FAST = 80,
    PRECALL_NO_KW_METHOD_DESCRIPTOR_NOARGS = 81,
    PRECALL_NO_KW_METHOD_DESCRIPTOR_O = 113,
    PRECALL_NO_KW_STR_1 = 121,
    PRECALL_NO_KW_TUPLE_1 = 127,
    PRECALL_NO_KW_TYPE_1 = 141,
    PRECALL_PYFUNC = 143,
    RESUME_QUICK = 150,
    STORE_ATTR_ADAPTIVE = 153,
    STORE_ATTR_INSTANCE_VALUE = 154,
    STORE_ATTR_SLOT = 158,
    STORE_ATTR_WITH_HINT = 159,
    STORE_FAST__LOAD_FAST = 161,
    STORE_FAST__STORE_FAST = 167,
    STORE_SUBSCR_ADAPTIVE = 168,
    STORE_SUBSCR_DICT = 169,
    STORE_SUBSCR_LIST_INT = 170,
    UNPACK_SEQUENCE_ADAPTIVE = 177,
    UNPACK_SEQUENCE_LIST = 178,
    UNPACK_SEQUENCE_TUPLE = 179,
    UNPACK_SEQUENCE_TWO_TUPLE = 180,
    DO_TRACING = 255,
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

    /// Relative or absolute jump
    fn is_jump(&self) -> bool {
        self.is_absolute_jump() | self.is_relative_jump()
    }
}
