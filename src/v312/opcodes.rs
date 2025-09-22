#![allow(non_snake_case)]

use crate::define_opcodes;
use crate::traits::GenericOpcode;
use crate::utils::StackEffect;
use crate::v312::instructions::Instruction;
use paste::paste;

// From https://github.com/python/cpython/blob/3.12/Include/opcode.h
define_opcodes!(
    CACHE = 0,
    POP_TOP = 1,
    PUSH_NULL = 2,
    INTERPRETER_EXIT = 3,
    END_FOR = 4,
    END_SEND = 5,
    NOP = 9,
    UNARY_NEGATIVE = 11,
    UNARY_NOT = 12,
    UNARY_INVERT = 15,
    RESERVED = 17,
    BINARY_SUBSCR = 25,
    BINARY_SLICE = 26,
    STORE_SLICE = 27,
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
    CLEANUP_THROW = 55,
    STORE_SUBSCR = 60,
    DELETE_SUBSCR = 61,
    GET_ITER = 68,
    GET_YIELD_FROM_ITER = 69,
    LOAD_BUILD_CLASS = 71,
    LOAD_ASSERTION_ERROR = 74,
    RETURN_GENERATOR = 75,
    RETURN_VALUE = 83,
    SETUP_ANNOTATIONS = 85,
    LOAD_LOCALS = 87,
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
    POP_JUMP_IF_FALSE = 114,
    POP_JUMP_IF_TRUE = 115,
    LOAD_GLOBAL = 116,
    IS_OP = 117,
    CONTAINS_OP = 118,
    RERAISE = 119,
    COPY = 120,
    RETURN_CONST = 121,
    BINARY_OP = 122,
    SEND = 123,
    LOAD_FAST = 124,
    STORE_FAST = 125,
    DELETE_FAST = 126,
    LOAD_FAST_CHECK = 127,
    POP_JUMP_IF_NOT_NONE = 128,
    POP_JUMP_IF_NONE = 129,
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
    LOAD_SUPER_ATTR = 141,
    CALL_FUNCTION_EX = 142,
    LOAD_FAST_AND_CLEAR = 143,
    EXTENDED_ARG = 144,
    LIST_APPEND = 145,
    SET_ADD = 146,
    MAP_ADD = 147,
    COPY_FREE_VARS = 149,
    YIELD_VALUE = 150,
    RESUME = 151,
    MATCH_CLASS = 152,
    FORMAT_VALUE = 155,
    BUILD_CONST_KEY_MAP = 156,
    BUILD_STRING = 157,
    LIST_EXTEND = 162,
    SET_UPDATE = 163,
    DICT_MERGE = 164,
    DICT_UPDATE = 165,
    CALL = 171,
    KW_NAMES = 172,
    CALL_INTRINSIC_1 = 173,
    CALL_INTRINSIC_2 = 174,
    LOAD_FROM_DICT_OR_GLOBALS = 175,
    LOAD_FROM_DICT_OR_DEREF = 176,
    // Specialized ops
    INSTRUMENTED_LOAD_SUPER_ATTR = 237,
    INSTRUMENTED_POP_JUMP_IF_NONE = 238,
    INSTRUMENTED_POP_JUMP_IF_NOT_NONE = 239,
    INSTRUMENTED_RESUME = 240,
    INSTRUMENTED_CALL = 241,
    INSTRUMENTED_RETURN_VALUE = 242,
    INSTRUMENTED_YIELD_VALUE = 243,
    INSTRUMENTED_CALL_FUNCTION_EX = 244,
    INSTRUMENTED_JUMP_FORWARD = 245,
    INSTRUMENTED_JUMP_BACKWARD = 246,
    INSTRUMENTED_RETURN_CONST = 247,
    INSTRUMENTED_FOR_ITER = 248,
    INSTRUMENTED_POP_JUMP_IF_FALSE = 249,
    INSTRUMENTED_POP_JUMP_IF_TRUE = 250,
    INSTRUMENTED_END_FOR = 251,
    INSTRUMENTED_END_SEND = 252,
    INSTRUMENTED_INSTRUCTION = 253,
    INSTRUMENTED_LINE = 254,
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
    BINARY_OP_ADD_FLOAT = 6,
    BINARY_OP_ADD_INT = 7,
    BINARY_OP_ADD_UNICODE = 8,
    BINARY_OP_INPLACE_ADD_UNICODE = 10,
    BINARY_OP_MULTIPLY_FLOAT = 13,
    BINARY_OP_MULTIPLY_INT = 14,
    BINARY_OP_SUBTRACT_FLOAT = 16,
    BINARY_OP_SUBTRACT_INT = 18,
    BINARY_SUBSCR_DICT = 19,
    BINARY_SUBSCR_GETITEM = 20,
    BINARY_SUBSCR_LIST_INT = 21,
    BINARY_SUBSCR_TUPLE_INT = 22,
    CALL_PY_EXACT_ARGS = 23,
    CALL_PY_WITH_DEFAULTS = 24,
    CALL_BOUND_METHOD_EXACT_ARGS = 28,
    CALL_BUILTIN_CLASS = 29,
    CALL_BUILTIN_FAST_WITH_KEYWORDS = 34,
    CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS = 38,
    CALL_NO_KW_BUILTIN_FAST = 39,
    CALL_NO_KW_BUILTIN_O = 40,
    CALL_NO_KW_ISINSTANCE = 41,
    CALL_NO_KW_LEN = 42,
    CALL_NO_KW_LIST_APPEND = 43,
    CALL_NO_KW_METHOD_DESCRIPTOR_FAST = 44,
    CALL_NO_KW_METHOD_DESCRIPTOR_NOARGS = 45,
    CALL_NO_KW_METHOD_DESCRIPTOR_O = 46,
    CALL_NO_KW_STR_1 = 47,
    CALL_NO_KW_TUPLE_1 = 48,
    CALL_NO_KW_TYPE_1 = 56,
    COMPARE_OP_FLOAT = 57,
    COMPARE_OP_INT = 58,
    COMPARE_OP_STR = 59,
    FOR_ITER_LIST = 62,
    FOR_ITER_TUPLE = 63,
    FOR_ITER_RANGE = 64,
    FOR_ITER_GEN = 65,
    LOAD_SUPER_ATTR_ATTR = 66,
    LOAD_SUPER_ATTR_METHOD = 67,
    LOAD_ATTR_CLASS = 70,
    LOAD_ATTR_GETATTRIBUTE_OVERRIDDEN = 72,
    LOAD_ATTR_INSTANCE_VALUE = 73,
    LOAD_ATTR_MODULE = 76,
    LOAD_ATTR_PROPERTY = 77,
    LOAD_ATTR_SLOT = 78,
    LOAD_ATTR_WITH_HINT = 79,
    LOAD_ATTR_METHOD_LAZY_DICT = 80,
    LOAD_ATTR_METHOD_NO_DICT = 81,
    LOAD_ATTR_METHOD_WITH_VALUES = 82,
    LOAD_CONST__LOAD_FAST = 84,
    LOAD_FAST__LOAD_CONST = 86,
    LOAD_FAST__LOAD_FAST = 88,
    LOAD_GLOBAL_BUILTIN = 111,
    LOAD_GLOBAL_MODULE = 112,
    STORE_ATTR_INSTANCE_VALUE = 113,
    STORE_ATTR_SLOT = 148,
    STORE_ATTR_WITH_HINT = 153,
    STORE_FAST__LOAD_FAST = 154,
    STORE_FAST__STORE_FAST = 158,
    STORE_SUBSCR_DICT = 159,
    STORE_SUBSCR_LIST_INT = 160,
    UNPACK_SEQUENCE_LIST = 161,
    UNPACK_SEQUENCE_TUPLE = 166,
    UNPACK_SEQUENCE_TWO_TUPLE = 167,
    SEND_GEN = 168,
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

    fn stack_effect(&self, oparg: u32, jump: Option<bool>) -> StackEffect {
        match self {
            Opcode::NOP
            | Opcode::RESUME
            | Opcode::INSTRUMENTED_RESUME
            | Opcode::EXTENDED_ARG
            | Opcode::CACHE
            | Opcode::RESERVED => StackEffect::zero(),

            Opcode::LOAD_CLOSURE
            | Opcode::LOAD_FAST_CHECK
            | Opcode::LOAD_FAST
            | Opcode::LOAD_FAST_AND_CLEAR
            | Opcode::LOAD_CONST => StackEffect::push(1),

            Opcode::STORE_FAST => StackEffect::pop(1),

            Opcode::LOAD_FAST__LOAD_FAST
            | Opcode::LOAD_FAST__LOAD_CONST
            | Opcode::LOAD_CONST__LOAD_FAST => StackEffect { pops: 0, pushes: 2 },

            Opcode::STORE_FAST__LOAD_FAST => StackEffect { pops: 1, pushes: 1 },
            Opcode::STORE_FAST__STORE_FAST => StackEffect::pop(2),

            Opcode::POP_TOP => StackEffect::pop(1),
            Opcode::PUSH_NULL => StackEffect::push(1),

            Opcode::END_FOR | Opcode::INSTRUMENTED_END_FOR => StackEffect::pop(2),

            Opcode::END_SEND | Opcode::INSTRUMENTED_END_SEND => StackEffect { pops: 2, pushes: 1 },

            Opcode::UNARY_NEGATIVE | Opcode::UNARY_NOT | Opcode::UNARY_INVERT => {
                StackEffect::balanced(1)
            }

            Opcode::BINARY_OP_MULTIPLY_INT
            | Opcode::BINARY_OP_MULTIPLY_FLOAT
            | Opcode::BINARY_OP_SUBTRACT_INT
            | Opcode::BINARY_OP_SUBTRACT_FLOAT
            | Opcode::BINARY_OP_ADD_UNICODE
            | Opcode::BINARY_OP_ADD_FLOAT
            | Opcode::BINARY_OP_ADD_INT
            | Opcode::BINARY_OP => StackEffect { pops: 2, pushes: 1 },

            Opcode::BINARY_OP_INPLACE_ADD_UNICODE => StackEffect::pop(2),

            Opcode::BINARY_SUBSCR
            | Opcode::BINARY_SUBSCR_LIST_INT
            | Opcode::BINARY_SUBSCR_TUPLE_INT
            | Opcode::BINARY_SUBSCR_DICT
            | Opcode::BINARY_SUBSCR_GETITEM => StackEffect { pops: 2, pushes: 1 },

            Opcode::BINARY_SLICE => StackEffect { pops: 3, pushes: 1 },
            Opcode::STORE_SLICE => StackEffect::pop(4),

            Opcode::LIST_APPEND | Opcode::SET_ADD => StackEffect {
                pops: (oparg - 1) + 2,
                pushes: (oparg - 1) + 1,
            },

            Opcode::STORE_SUBSCR | Opcode::STORE_SUBSCR_LIST_INT | Opcode::STORE_SUBSCR_DICT => {
                StackEffect::pop(3)
            }

            Opcode::DELETE_SUBSCR => StackEffect::pop(2),

            Opcode::CALL_INTRINSIC_1 => StackEffect::balanced(1),
            Opcode::CALL_INTRINSIC_2 => StackEffect { pops: 2, pushes: 1 },

            Opcode::RAISE_VARARGS => StackEffect::pop(oparg),

            Opcode::INTERPRETER_EXIT | Opcode::RETURN_VALUE | Opcode::INSTRUMENTED_RETURN_VALUE => {
                StackEffect::pop(1)
            }

            Opcode::RETURN_CONST | Opcode::INSTRUMENTED_RETURN_CONST | Opcode::RETURN_GENERATOR => {
                StackEffect::zero()
            }

            Opcode::GET_AITER => StackEffect::balanced(1),
            Opcode::GET_ANEXT => StackEffect { pops: 1, pushes: 2 },
            Opcode::GET_AWAITABLE => StackEffect::balanced(1),

            Opcode::SEND | Opcode::SEND_GEN => StackEffect { pops: 2, pushes: 2 },

            Opcode::INSTRUMENTED_YIELD_VALUE | Opcode::YIELD_VALUE => StackEffect::balanced(1),

            Opcode::POP_EXCEPT => StackEffect::pop(1),

            Opcode::RERAISE => StackEffect {
                pops: oparg + 1,
                pushes: oparg,
            },

            Opcode::END_ASYNC_FOR => StackEffect::pop(2),

            Opcode::CLEANUP_THROW => StackEffect { pops: 3, pushes: 2 },

            Opcode::LOAD_ASSERTION_ERROR | Opcode::LOAD_BUILD_CLASS | Opcode::LOAD_LOCALS => {
                StackEffect::push(1)
            }

            Opcode::STORE_NAME
            | Opcode::STORE_GLOBAL
            | Opcode::STORE_DEREF
            | Opcode::DELETE_ATTR => StackEffect::pop(1),

            Opcode::STORE_ATTR => StackEffect::pop(2),

            Opcode::DELETE_NAME
            | Opcode::DELETE_GLOBAL
            | Opcode::DELETE_FAST
            | Opcode::DELETE_DEREF
            | Opcode::MAKE_CELL
            | Opcode::COPY_FREE_VARS
            | Opcode::SETUP_ANNOTATIONS => StackEffect::zero(),

            Opcode::UNPACK_SEQUENCE
            | Opcode::UNPACK_SEQUENCE_TWO_TUPLE
            | Opcode::UNPACK_SEQUENCE_TUPLE
            | Opcode::UNPACK_SEQUENCE_LIST => StackEffect {
                pops: 1,
                pushes: oparg,
            },

            Opcode::UNPACK_EX => StackEffect {
                pops: 1,
                pushes: (oparg & 0xFF) + (oparg >> 8) + 1,
            },

            Opcode::LOAD_FROM_DICT_OR_GLOBALS => StackEffect::balanced(1),

            Opcode::LOAD_NAME | Opcode::LOAD_DEREF => StackEffect::push(1),

            Opcode::LOAD_FROM_DICT_OR_DEREF => StackEffect::balanced(1),

            Opcode::LOAD_GLOBAL | Opcode::LOAD_GLOBAL_MODULE | Opcode::LOAD_GLOBAL_BUILTIN => {
                StackEffect {
                    pops: 0,
                    pushes: if (oparg & 1) != 0 { 2 } else { 1 },
                }
            }

            Opcode::BUILD_STRING | Opcode::BUILD_TUPLE | Opcode::BUILD_LIST | Opcode::BUILD_SET => {
                StackEffect {
                    pops: oparg,
                    pushes: 1,
                }
            }

            Opcode::LIST_EXTEND | Opcode::SET_UPDATE => StackEffect {
                pops: (oparg - 1) + 2,
                pushes: (oparg - 1) + 1,
            },

            Opcode::BUILD_MAP => StackEffect {
                pops: oparg * 2,
                pushes: 1,
            },
            Opcode::BUILD_CONST_KEY_MAP => StackEffect {
                pops: oparg + 1,
                pushes: 1,
            },

            Opcode::DICT_UPDATE | Opcode::DICT_MERGE => StackEffect::pop(1),

            Opcode::MAP_ADD => StackEffect::pop(2),

            Opcode::INSTRUMENTED_LOAD_SUPER_ATTR
            | Opcode::LOAD_SUPER_ATTR
            | Opcode::LOAD_SUPER_ATTR_ATTR => StackEffect {
                pops: 3,
                pushes: if (oparg & 1) != 0 { 2 } else { 1 },
            },
            Opcode::LOAD_ATTR
            | Opcode::LOAD_ATTR_INSTANCE_VALUE
            | Opcode::LOAD_ATTR_MODULE
            | Opcode::LOAD_ATTR_WITH_HINT
            | Opcode::LOAD_ATTR_SLOT
            | Opcode::LOAD_ATTR_CLASS
            | Opcode::LOAD_ATTR_PROPERTY
            | Opcode::LOAD_ATTR_GETATTRIBUTE_OVERRIDDEN
            | Opcode::LOAD_ATTR_METHOD_WITH_VALUES
            | Opcode::LOAD_ATTR_METHOD_NO_DICT
            | Opcode::LOAD_ATTR_METHOD_LAZY_DICT => StackEffect {
                pops: 1,
                pushes: if (oparg & 1) != 0 { 2 } else { 1 },
            },

            Opcode::LOAD_SUPER_ATTR_METHOD => StackEffect { pops: 3, pushes: 2 },

            Opcode::STORE_ATTR_INSTANCE_VALUE
            | Opcode::STORE_ATTR_WITH_HINT
            | Opcode::STORE_ATTR_SLOT => StackEffect::pop(2),

            Opcode::COMPARE_OP
            | Opcode::COMPARE_OP_FLOAT
            | Opcode::COMPARE_OP_INT
            | Opcode::COMPARE_OP_STR
            | Opcode::IS_OP
            | Opcode::CONTAINS_OP => StackEffect { pops: 2, pushes: 1 },

            Opcode::CHECK_EG_MATCH | Opcode::CHECK_EXC_MATCH => StackEffect::balanced(2),

            Opcode::IMPORT_NAME => StackEffect { pops: 2, pushes: 1 },
            Opcode::IMPORT_FROM => StackEffect { pops: 1, pushes: 2 },

            Opcode::JUMP_FORWARD
            | Opcode::JUMP_BACKWARD
            | Opcode::JUMP_BACKWARD_NO_INTERRUPT
            | Opcode::INSTRUMENTED_JUMP_FORWARD
            | Opcode::INSTRUMENTED_JUMP_BACKWARD
            | Opcode::INSTRUMENTED_POP_JUMP_IF_TRUE // Python internally flags this as 0 stack effect for some reason.
            | Opcode::INSTRUMENTED_POP_JUMP_IF_FALSE
            | Opcode::INSTRUMENTED_POP_JUMP_IF_NONE
            | Opcode::INSTRUMENTED_POP_JUMP_IF_NOT_NONE => StackEffect::zero(),

            Opcode::POP_JUMP_IF_FALSE
            | Opcode::POP_JUMP_IF_TRUE
            | Opcode::POP_JUMP_IF_NOT_NONE
            | Opcode::POP_JUMP_IF_NONE
             => StackEffect::pop(1),

            Opcode::GET_LEN => StackEffect { pops: 1, pushes: 2 },

            Opcode::MATCH_CLASS => StackEffect { pops: 3, pushes: 1 },
            Opcode::MATCH_MAPPING | Opcode::MATCH_SEQUENCE => StackEffect { pops: 1, pushes: 2 },
            Opcode::MATCH_KEYS => StackEffect { pops: 2, pushes: 3 },

            Opcode::GET_ITER | Opcode::GET_YIELD_FROM_ITER => StackEffect::balanced(1),

            Opcode::FOR_ITER
            | Opcode::FOR_ITER_LIST
            | Opcode::FOR_ITER_TUPLE
            | Opcode::FOR_ITER_RANGE
            | Opcode::FOR_ITER_GEN => StackEffect { pops: 1, pushes: 2 },

            Opcode::INSTRUMENTED_FOR_ITER => StackEffect::zero(),

            Opcode::BEFORE_ASYNC_WITH | Opcode::BEFORE_WITH => StackEffect { pops: 1, pushes: 2 },

            Opcode::WITH_EXCEPT_START => StackEffect { pops: 4, pushes: 5 },
            Opcode::PUSH_EXC_INFO => StackEffect { pops: 1, pushes: 2 },

            Opcode::KW_NAMES
            | Opcode::INSTRUMENTED_CALL
            | Opcode::INSTRUMENTED_CALL_FUNCTION_EX
            | Opcode::INSTRUMENTED_INSTRUCTION => StackEffect::zero(),

            Opcode::CALL
            | Opcode::CALL_BOUND_METHOD_EXACT_ARGS
            | Opcode::CALL_PY_EXACT_ARGS
            | Opcode::CALL_PY_WITH_DEFAULTS
            | Opcode::CALL_NO_KW_TYPE_1
            | Opcode::CALL_NO_KW_STR_1
            | Opcode::CALL_NO_KW_TUPLE_1
            | Opcode::CALL_BUILTIN_CLASS
            | Opcode::CALL_NO_KW_BUILTIN_O
            | Opcode::CALL_NO_KW_BUILTIN_FAST
            | Opcode::CALL_BUILTIN_FAST_WITH_KEYWORDS
            | Opcode::CALL_NO_KW_LEN
            | Opcode::CALL_NO_KW_ISINSTANCE
            | Opcode::CALL_NO_KW_LIST_APPEND
            | Opcode::CALL_NO_KW_METHOD_DESCRIPTOR_O
            | Opcode::CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS
            | Opcode::CALL_NO_KW_METHOD_DESCRIPTOR_NOARGS
            | Opcode::CALL_NO_KW_METHOD_DESCRIPTOR_FAST => StackEffect {
                pops: oparg + 2,
                pushes: 1,
            },

            Opcode::CALL_FUNCTION_EX => StackEffect {
                pops:  if (oparg & 1) != 0 {4} else {3},
                pushes: 1,
            },

            Opcode::MAKE_FUNCTION => StackEffect {
                pops: ((oparg & 0b1111).count_ones() + 1) as u32,
                pushes: 1,
            },

            Opcode::BUILD_SLICE => StackEffect {
                pops: if oparg == 3 { 3 } else { 2 },
                pushes: 1,
            },

            Opcode::FORMAT_VALUE => {
                // If there's a fmt_spec on the stack, we go from 2->1, else 1->1.
                const FVS_MASK: u32 = 0x4;
                const FVS_HAVE_SPEC: u32 = 0x4;
                if (oparg & FVS_MASK) == FVS_HAVE_SPEC {
                    StackEffect { pops: 2, pushes: 1 } // value + format_spec -> formatted
                } else {
                    StackEffect::balanced(1) // value -> formatted
                }
            }

            Opcode::COPY => StackEffect {
                pops: oparg,
                pushes: oparg + 1,
            },

            Opcode::SWAP => StackEffect::balanced(oparg),

            Opcode::INVALID_OPCODE(_) => StackEffect::zero(),

            _ => unimplemented!("stack_effect not implemented for {:?}", self),
        }
    }
}
