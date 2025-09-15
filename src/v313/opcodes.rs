#![allow(non_snake_case)]

use crate::define_opcodes;
use crate::traits::GenericOpcode;
use crate::utils::StackEffect;
use crate::v313::instructions::Instruction;
use paste::paste;

// From https://github.com/python/cpython/blob/3.12/Include/opcode.h
define_opcodes!(
    CACHE = 0,
    BEFORE_ASYNC_WITH = 1,
    BEFORE_WITH = 2,
    // Specialized op
    BINARY_OP_INPLACE_ADD_UNICODE = 3,
    BINARY_SLICE = 4,
    BINARY_SUBSCR = 5,
    CHECK_EG_MATCH = 6,
    CHECK_EXC_MATCH = 7,
    CLEANUP_THROW = 8,
    DELETE_SUBSCR = 9,
    END_ASYNC_FOR = 10,
    END_FOR = 11,
    END_SEND = 12,
    EXIT_INIT_CHECK = 13,
    FORMAT_SIMPLE = 14,
    FORMAT_WITH_SPEC = 15,
    GET_AITER = 16,
    RESERVED = 17,
    GET_ANEXT = 18,
    GET_ITER = 19,
    GET_LEN = 20,
    GET_YIELD_FROM_ITER = 21,
    INTERPRETER_EXIT = 22,
    LOAD_ASSERTION_ERROR = 23,
    LOAD_BUILD_CLASS = 24,
    LOAD_LOCALS = 25,
    MAKE_FUNCTION = 26,
    MATCH_KEYS = 27,
    MATCH_MAPPING = 28,
    MATCH_SEQUENCE = 29,
    NOP = 30,
    POP_EXCEPT = 31,
    POP_TOP = 32,
    PUSH_EXC_INFO = 33,
    PUSH_NULL = 34,
    RETURN_GENERATOR = 35,
    RETURN_VALUE = 36,
    SETUP_ANNOTATIONS = 37,
    STORE_SLICE = 38,
    STORE_SUBSCR = 39,
    TO_BOOL = 40,
    UNARY_INVERT = 41,
    UNARY_NEGATIVE = 42,
    UNARY_NOT = 43,
    WITH_EXCEPT_START = 44,
    BINARY_OP = 45,
    BUILD_CONST_KEY_MAP = 46,
    BUILD_LIST = 47,
    BUILD_MAP = 48,
    BUILD_SET = 49,
    BUILD_SLICE = 50,
    BUILD_STRING = 51,
    BUILD_TUPLE = 52,
    CALL = 53,
    CALL_FUNCTION_EX = 54,
    CALL_INTRINSIC_1 = 55,
    CALL_INTRINSIC_2 = 56,
    CALL_KW = 57,
    COMPARE_OP = 58,
    CONTAINS_OP = 59,
    CONVERT_VALUE = 60,
    COPY = 61,
    COPY_FREE_VARS = 62,
    DELETE_ATTR = 63,
    DELETE_DEREF = 64,
    DELETE_FAST = 65,
    DELETE_GLOBAL = 66,
    DELETE_NAME = 67,
    DICT_MERGE = 68,
    DICT_UPDATE = 69,
    ENTER_EXECUTOR = 70,
    EXTENDED_ARG = 71,
    FOR_ITER = 72,
    GET_AWAITABLE = 73,
    IMPORT_FROM = 74,
    IMPORT_NAME = 75,
    IS_OP = 76,
    JUMP_BACKWARD = 77,
    JUMP_BACKWARD_NO_INTERRUPT = 78,
    JUMP_FORWARD = 79,
    LIST_APPEND = 80,
    LIST_EXTEND = 81,
    LOAD_ATTR = 82,
    LOAD_CONST = 83,
    LOAD_DEREF = 84,
    LOAD_FAST = 85,
    LOAD_FAST_AND_CLEAR = 86,
    LOAD_FAST_CHECK = 87,
    LOAD_FAST_LOAD_FAST = 88,
    LOAD_FROM_DICT_OR_DEREF = 89,
    LOAD_FROM_DICT_OR_GLOBALS = 90,
    LOAD_GLOBAL = 91,
    LOAD_NAME = 92,
    LOAD_SUPER_ATTR = 93,
    MAKE_CELL = 94,
    MAP_ADD = 95,
    MATCH_CLASS = 96,
    POP_JUMP_IF_FALSE = 97,
    POP_JUMP_IF_NONE = 98,
    POP_JUMP_IF_NOT_NONE = 99,
    POP_JUMP_IF_TRUE = 100,
    RAISE_VARARGS = 101,
    RERAISE = 102,
    RETURN_CONST = 103,
    SEND = 104,
    SET_ADD = 105,
    SET_FUNCTION_ATTRIBUTE = 106,
    SET_UPDATE = 107,
    STORE_ATTR = 108,
    STORE_DEREF = 109,
    STORE_FAST = 110,
    STORE_FAST_LOAD_FAST = 111,
    STORE_FAST_STORE_FAST = 112,
    STORE_GLOBAL = 113,
    STORE_NAME = 114,
    SWAP = 115,
    UNPACK_EX = 116,
    UNPACK_SEQUENCE = 117,
    YIELD_VALUE = 118,
    RESUME = 149,
    // Specialized ops
    BINARY_OP_ADD_FLOAT = 150,
    BINARY_OP_ADD_INT = 151,
    BINARY_OP_ADD_UNICODE = 152,
    BINARY_OP_MULTIPLY_FLOAT = 153,
    BINARY_OP_MULTIPLY_INT = 154,
    BINARY_OP_SUBTRACT_FLOAT = 155,
    BINARY_OP_SUBTRACT_INT = 156,
    BINARY_SUBSCR_DICT = 157,
    BINARY_SUBSCR_GETITEM = 158,
    BINARY_SUBSCR_LIST_INT = 159,
    BINARY_SUBSCR_STR_INT = 160,
    BINARY_SUBSCR_TUPLE_INT = 161,
    CALL_ALLOC_AND_ENTER_INIT = 162,
    CALL_BOUND_METHOD_EXACT_ARGS = 163,
    CALL_BOUND_METHOD_GENERAL = 164,
    CALL_BUILTIN_CLASS = 165,
    CALL_BUILTIN_FAST = 166,
    CALL_BUILTIN_FAST_WITH_KEYWORDS = 167,
    CALL_BUILTIN_O = 168,
    CALL_ISINSTANCE = 169,
    CALL_LEN = 170,
    CALL_LIST_APPEND = 171,
    CALL_METHOD_DESCRIPTOR_FAST = 172,
    CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS = 173,
    CALL_METHOD_DESCRIPTOR_NOARGS = 174,
    CALL_METHOD_DESCRIPTOR_O = 175,
    CALL_NON_PY_GENERAL = 176,
    CALL_PY_EXACT_ARGS = 177,
    CALL_PY_GENERAL = 178,
    CALL_STR_1 = 179,
    CALL_TUPLE_1 = 180,
    CALL_TYPE_1 = 181,
    COMPARE_OP_FLOAT = 182,
    COMPARE_OP_INT = 183,
    COMPARE_OP_STR = 184,
    CONTAINS_OP_DICT = 185,
    CONTAINS_OP_SET = 186,
    FOR_ITER_GEN = 187,
    FOR_ITER_LIST = 188,
    FOR_ITER_RANGE = 189,
    FOR_ITER_TUPLE = 190,
    LOAD_ATTR_CLASS = 191,
    LOAD_ATTR_GETATTRIBUTE_OVERRIDDEN = 192,
    LOAD_ATTR_INSTANCE_VALUE = 193,
    LOAD_ATTR_METHOD_LAZY_DICT = 194,
    LOAD_ATTR_METHOD_NO_DICT = 195,
    LOAD_ATTR_METHOD_WITH_VALUES = 196,
    LOAD_ATTR_MODULE = 197,
    LOAD_ATTR_NONDESCRIPTOR_NO_DICT = 198,
    LOAD_ATTR_NONDESCRIPTOR_WITH_VALUES = 199,
    LOAD_ATTR_PROPERTY = 200,
    LOAD_ATTR_SLOT = 201,
    LOAD_ATTR_WITH_HINT = 202,
    LOAD_GLOBAL_BUILTIN = 203,
    LOAD_GLOBAL_MODULE = 204,
    LOAD_SUPER_ATTR_ATTR = 205,
    LOAD_SUPER_ATTR_METHOD = 206,
    RESUME_CHECK = 207,
    SEND_GEN = 208,
    STORE_ATTR_INSTANCE_VALUE = 209,
    STORE_ATTR_SLOT = 210,
    STORE_ATTR_WITH_HINT = 211,
    STORE_SUBSCR_DICT = 212,
    STORE_SUBSCR_LIST_INT = 213,
    TO_BOOL_ALWAYS_TRUE = 214,
    TO_BOOL_BOOL = 215,
    TO_BOOL_INT = 216,
    TO_BOOL_LIST = 217,
    TO_BOOL_NONE = 218,
    TO_BOOL_STR = 219,
    UNPACK_SEQUENCE_LIST = 220,
    UNPACK_SEQUENCE_TUPLE = 221,
    UNPACK_SEQUENCE_TWO_TUPLE = 222,
    INSTRUMENTED_RESUME = 236,
    INSTRUMENTED_END_FOR = 237,
    INSTRUMENTED_END_SEND = 238,
    INSTRUMENTED_RETURN_VALUE = 239,
    INSTRUMENTED_RETURN_CONST = 240,
    INSTRUMENTED_YIELD_VALUE = 241,
    INSTRUMENTED_LOAD_SUPER_ATTR = 242,
    INSTRUMENTED_FOR_ITER = 243,
    INSTRUMENTED_CALL = 244,
    INSTRUMENTED_CALL_KW = 245,
    INSTRUMENTED_CALL_FUNCTION_EX = 246,
    INSTRUMENTED_INSTRUCTION = 247,
    INSTRUMENTED_JUMP_FORWARD = 248,
    INSTRUMENTED_JUMP_BACKWARD = 249,
    INSTRUMENTED_POP_JUMP_IF_TRUE = 250,
    INSTRUMENTED_POP_JUMP_IF_FALSE = 251,
    INSTRUMENTED_POP_JUMP_IF_NONE = 252,
    INSTRUMENTED_POP_JUMP_IF_NOT_NONE = 253,
    INSTRUMENTED_LINE = 254,
    // We skip psuedo opcodes as they can never appear in actual bytecode
    // JUMP = 256,
    // JUMP_NO_INTERRUPT = 257,
    // LOAD_CLOSURE = 258,
    // LOAD_METHOD = 259,
    // LOAD_SUPER_METHOD = 260,
    // LOAD_ZERO_SUPER_ATTR = 261,
    // LOAD_ZERO_SUPER_METHOD = 262,
    // POP_BLOCK = 263,
    // SETUP_CLEANUP = 264,
    // SETUP_FINALLY = 265,
    // SETUP_WITH = 266,
    // STORE_FAST_MAYBE_NULL = 267,
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

    fn is_extended_arg(&self) -> bool {
        matches!(self, Opcode::EXTENDED_ARG)
    }

    fn stack_effect(&self, oparg: u32, jump: Option<bool>) -> StackEffect {
        match self {
            // BEFORE/AFTER / with
            Opcode::BEFORE_ASYNC_WITH => StackEffect { pops: 1, pushes: 2 },
            Opcode::BEFORE_WITH => StackEffect { pops: 1, pushes: 2 },

            // Binary ops
            Opcode::BINARY_OP
            | Opcode::BINARY_OP_ADD_FLOAT
            | Opcode::BINARY_OP_ADD_INT
            | Opcode::BINARY_OP_ADD_UNICODE
            | Opcode::BINARY_OP_MULTIPLY_FLOAT
            | Opcode::BINARY_OP_MULTIPLY_INT
            | Opcode::BINARY_OP_SUBTRACT_FLOAT
            | Opcode::BINARY_OP_SUBTRACT_INT
            | Opcode::BINARY_SUBSCR
            | Opcode::BINARY_SUBSCR_DICT
            | Opcode::BINARY_SUBSCR_GETITEM
            | Opcode::BINARY_SUBSCR_LIST_INT
            | Opcode::BINARY_SUBSCR_STR_INT
            | Opcode::BINARY_SUBSCR_TUPLE_INT => StackEffect { pops: 2, pushes: 1 },

            Opcode::BINARY_OP_INPLACE_ADD_UNICODE => StackEffect::pop(2),

            Opcode::BINARY_SLICE => StackEffect { pops: 3, pushes: 1 },

            // Builds
            Opcode::BUILD_CONST_KEY_MAP => StackEffect {
                pops: 1 + oparg,
                pushes: 1,
            },
            Opcode::BUILD_LIST | Opcode::BUILD_SET | Opcode::BUILD_STRING | Opcode::BUILD_TUPLE => {
                StackEffect {
                    pops: oparg,
                    pushes: 1,
                }
            }
            Opcode::BUILD_MAP => StackEffect {
                pops: oparg * 2,
                pushes: 1,
            },
            Opcode::BUILD_SLICE => StackEffect {
                pops: if oparg == 3 { 3 } else { 2 },
                pushes: 1,
            },

            // Cache/metadata/no-op
            Opcode::CACHE
            | Opcode::EXTENDED_ARG
            | Opcode::NOP
            | Opcode::RESUME
            | Opcode::RESUME_CHECK
            | Opcode::RESERVED => StackEffect::zero(),

            // Calls (many variants)
            Opcode::CALL
            | Opcode::CALL_ALLOC_AND_ENTER_INIT
            | Opcode::CALL_BUILTIN_CLASS
            | Opcode::CALL_BUILTIN_FAST
            | Opcode::CALL_BUILTIN_FAST_WITH_KEYWORDS
            | Opcode::CALL_BUILTIN_O
            | Opcode::CALL_ISINSTANCE
            | Opcode::CALL_LEN
            | Opcode::CALL_METHOD_DESCRIPTOR_FAST
            | Opcode::CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS
            | Opcode::CALL_METHOD_DESCRIPTOR_NOARGS
            | Opcode::CALL_METHOD_DESCRIPTOR_O
            | Opcode::CALL_NON_PY_GENERAL
            | Opcode::CALL_PY_EXACT_ARGS
            | Opcode::CALL_PY_GENERAL => {
                // C: return 2 + oparg;
                StackEffect {
                    pops: 2 + oparg,
                    pushes: 1,
                }
            }

            Opcode::CALL_BOUND_METHOD_EXACT_ARGS | Opcode::CALL_BOUND_METHOD_GENERAL => {
                StackEffect::pop(2 + oparg)
            }

            Opcode::CALL_KW => StackEffect {
                pops: 3 + oparg,
                pushes: 1,
            },

            Opcode::CALL_FUNCTION_EX => {
                // C: return 3 + (oparg & 1);
                StackEffect {
                    pops: if (oparg & 1) != 0 { 4 } else { 3 },
                    pushes: 1,
                }
            }

            Opcode::CALL_INTRINSIC_1 => StackEffect::balanced(1),
            Opcode::CALL_INTRINSIC_2 => StackEffect { pops: 2, pushes: 1 },

            Opcode::CALL_LIST_APPEND
            | Opcode::CALL_STR_1
            | Opcode::CALL_TUPLE_1
            | Opcode::CALL_TYPE_1 => StackEffect { pops: 3, pushes: 1 },

            // Compare / contains / matches
            Opcode::CHECK_EG_MATCH | Opcode::CHECK_EXC_MATCH => StackEffect::balanced(2),

            Opcode::CLEANUP_THROW => StackEffect { pushes: 2, pops: 3 },

            Opcode::COMPARE_OP
            | Opcode::COMPARE_OP_FLOAT
            | Opcode::COMPARE_OP_INT
            | Opcode::COMPARE_OP_STR
            | Opcode::CONTAINS_OP
            | Opcode::CONTAINS_OP_DICT
            | Opcode::CONTAINS_OP_SET => StackEffect { pops: 2, pushes: 1 },

            Opcode::CONVERT_VALUE => StackEffect::balanced(1),

            Opcode::COPY => StackEffect {
                pops: 1 + (oparg - 1),
                pushes: 2 + (oparg - 1),
            },

            Opcode::COPY_FREE_VARS => StackEffect::zero(),

            // Deletes
            Opcode::DELETE_ATTR => StackEffect::pop(1),
            Opcode::DELETE_DEREF
            | Opcode::DELETE_FAST
            | Opcode::DELETE_GLOBAL
            | Opcode::DELETE_NAME => StackEffect::zero(),
            Opcode::DELETE_SUBSCR => StackEffect::pop(2),

            // Dict ops
            Opcode::DICT_MERGE => StackEffect {
                pops: 5 + (oparg - 1),
                pushes: 4 + (oparg - 1),
            },
            Opcode::DICT_UPDATE => StackEffect {
                pops: 2 + (oparg - 1),
                pushes: 1 + (oparg - 1),
            },

            Opcode::END_ASYNC_FOR => StackEffect::pop(2),
            Opcode::END_FOR => StackEffect::pop(1),
            Opcode::END_SEND => StackEffect { pops: 2, pushes: 1 },

            Opcode::ENTER_EXECUTOR => StackEffect::zero(),
            Opcode::EXIT_INIT_CHECK => StackEffect::pop(1),

            Opcode::FORMAT_SIMPLE => StackEffect::balanced(1),
            Opcode::FORMAT_WITH_SPEC => StackEffect { pops: 2, pushes: 1 },

            Opcode::FOR_ITER
            | Opcode::FOR_ITER_LIST
            | Opcode::FOR_ITER_RANGE
            | Opcode::FOR_ITER_TUPLE => StackEffect { pops: 1, pushes: 2 },

            Opcode::FOR_ITER_GEN => StackEffect::balanced(1),

            Opcode::GET_AITER
            | Opcode::GET_AWAITABLE
            | Opcode::GET_ITER
            | Opcode::GET_YIELD_FROM_ITER => StackEffect::balanced(1),
            Opcode::GET_ANEXT => StackEffect { pops: 1, pushes: 2 },
            Opcode::GET_LEN => StackEffect { pops: 1, pushes: 2 },

            Opcode::IMPORT_FROM => StackEffect { pops: 1, pushes: 2 },
            Opcode::IMPORT_NAME => StackEffect { pops: 2, pushes: 1 },

            Opcode::INSTRUMENTED_CALL
            | Opcode::INSTRUMENTED_CALL_FUNCTION_EX
            | Opcode::INSTRUMENTED_CALL_KW => StackEffect::zero(),
            Opcode::INSTRUMENTED_END_FOR | Opcode::INSTRUMENTED_END_SEND => {
                StackEffect { pops: 2, pushes: 1 }
            }
            Opcode::INSTRUMENTED_FOR_ITER
            | Opcode::INSTRUMENTED_INSTRUCTION
            | Opcode::INSTRUMENTED_JUMP_BACKWARD
            | Opcode::INSTRUMENTED_JUMP_FORWARD => StackEffect::zero(),
            Opcode::INSTRUMENTED_LOAD_SUPER_ATTR => StackEffect {
                pops: 3,
                pushes: if (oparg & 1) != 0 { 2 } else { 1 },
            },

            Opcode::INSTRUMENTED_POP_JUMP_IF_FALSE
            | Opcode::INSTRUMENTED_POP_JUMP_IF_NONE
            | Opcode::INSTRUMENTED_POP_JUMP_IF_NOT_NONE
            | Opcode::INSTRUMENTED_POP_JUMP_IF_TRUE
            | Opcode::INSTRUMENTED_RESUME
            | Opcode::INSTRUMENTED_RETURN_CONST => StackEffect::zero(),
            Opcode::INSTRUMENTED_RETURN_VALUE => StackEffect::pop(1),
            Opcode::INSTRUMENTED_YIELD_VALUE => StackEffect::balanced(1),

            Opcode::INTERPRETER_EXIT => StackEffect::pop(1),

            Opcode::IS_OP => StackEffect { pops: 2, pushes: 1 },

            Opcode::JUMP_BACKWARD | Opcode::JUMP_BACKWARD_NO_INTERRUPT | Opcode::JUMP_FORWARD => {
                StackEffect::zero()
            }

            Opcode::LIST_APPEND | Opcode::LIST_EXTEND => StackEffect {
                pops: 2 + (oparg - 1),
                pushes: 1 + (oparg - 1),
            },

            Opcode::LOAD_ASSERTION_ERROR => StackEffect::push(1),
            Opcode::LOAD_ATTR
            | Opcode::LOAD_ATTR_CLASS
            | Opcode::LOAD_ATTR_INSTANCE_VALUE
            | Opcode::LOAD_ATTR_MODULE
            | Opcode::LOAD_ATTR_SLOT
            | Opcode::LOAD_ATTR_WITH_HINT => StackEffect {
                pops: 1,
                pushes: if (oparg & 1) != 0 { 2 } else { 1 },
            },
            Opcode::LOAD_ATTR_GETATTRIBUTE_OVERRIDDEN => StackEffect::balanced(1),
            Opcode::LOAD_ATTR_METHOD_LAZY_DICT
            | Opcode::LOAD_ATTR_METHOD_NO_DICT
            | Opcode::LOAD_ATTR_METHOD_WITH_VALUES => StackEffect { pops: 1, pushes: 2 },

            Opcode::LOAD_ATTR_NONDESCRIPTOR_NO_DICT
            | Opcode::LOAD_ATTR_NONDESCRIPTOR_WITH_VALUES
            | Opcode::LOAD_ATTR_PROPERTY => StackEffect::balanced(1),

            Opcode::LOAD_BUILD_CLASS
            | Opcode::LOAD_CONST
            | Opcode::LOAD_DEREF
            | Opcode::LOAD_FAST
            | Opcode::LOAD_FAST_AND_CLEAR
            | Opcode::LOAD_FAST_CHECK => StackEffect::push(1),
            Opcode::LOAD_FAST_LOAD_FAST => StackEffect::push(2),
            Opcode::LOAD_FROM_DICT_OR_DEREF | Opcode::LOAD_FROM_DICT_OR_GLOBALS => {
                StackEffect::balanced(1)
            }
            Opcode::LOAD_GLOBAL | Opcode::LOAD_GLOBAL_BUILTIN | Opcode::LOAD_GLOBAL_MODULE => {
                StackEffect {
                    pops: 0,
                    pushes: if (oparg & 1) != 0 { 2 } else { 1 },
                }
            }
            Opcode::LOAD_LOCALS | Opcode::LOAD_NAME => StackEffect::push(1),

            Opcode::LOAD_SUPER_ATTR => StackEffect {
                pops: 3,
                pushes: if (oparg & 1) != 0 { 2 } else { 1 },
            },

            Opcode::LOAD_SUPER_ATTR_ATTR => StackEffect { pushes: 3, pops: 1 },

            Opcode::LOAD_SUPER_ATTR_METHOD => StackEffect { pops: 3, pushes: 2 },

            Opcode::MAKE_CELL => StackEffect::zero(),
            Opcode::MAKE_FUNCTION => StackEffect::balanced(1),
            Opcode::MAP_ADD => StackEffect {
                pops: 3 + (oparg - 1),
                pushes: 1 + (oparg - 1),
            },

            Opcode::MATCH_CLASS => StackEffect { pops: 3, pushes: 1 },
            Opcode::MATCH_KEYS => StackEffect { pops: 2, pushes: 3 },
            Opcode::MATCH_MAPPING => StackEffect { pops: 1, pushes: 2 },
            Opcode::MATCH_SEQUENCE => StackEffect { pops: 1, pushes: 2 },

            Opcode::POP_EXCEPT => StackEffect::pop(1),
            Opcode::POP_JUMP_IF_FALSE
            | Opcode::POP_JUMP_IF_NONE
            | Opcode::POP_JUMP_IF_NOT_NONE
            | Opcode::POP_JUMP_IF_TRUE => StackEffect::pop(1),
            Opcode::POP_TOP => StackEffect::pop(1),
            Opcode::PUSH_EXC_INFO => StackEffect { pops: 1, pushes: 2 },
            Opcode::PUSH_NULL => StackEffect::push(1),

            Opcode::RAISE_VARARGS => StackEffect::pop(oparg),
            Opcode::RERAISE => StackEffect {
                pops: 1 + oparg,
                pushes: oparg,
            },

            Opcode::RETURN_CONST => StackEffect::zero(),
            Opcode::RETURN_GENERATOR => StackEffect::push(1),
            Opcode::RETURN_VALUE => StackEffect::pop(1),

            Opcode::SEND | Opcode::SEND_GEN => StackEffect::balanced(2),

            Opcode::SETUP_ANNOTATIONS => StackEffect::zero(),

            Opcode::SET_ADD => StackEffect {
                pops: 2 + (oparg - 1),
                pushes: 1 + (oparg - 1),
            },
            Opcode::SET_FUNCTION_ATTRIBUTE => StackEffect { pops: 2, pushes: 1 },
            Opcode::SET_UPDATE => StackEffect {
                pops: 2 + (oparg - 1),
                pushes: 1 + (oparg - 1),
            },

            Opcode::STORE_ATTR
            | Opcode::STORE_ATTR_INSTANCE_VALUE
            | Opcode::STORE_ATTR_SLOT
            | Opcode::STORE_ATTR_WITH_HINT => StackEffect::pop(2),

            Opcode::STORE_DEREF => StackEffect::pop(1),
            Opcode::STORE_FAST => StackEffect::pop(1),
            Opcode::STORE_FAST_LOAD_FAST => StackEffect::balanced(1),
            Opcode::STORE_FAST_STORE_FAST => StackEffect::pop(2),
            Opcode::STORE_GLOBAL | Opcode::STORE_NAME => StackEffect::pop(1),

            Opcode::STORE_SLICE => StackEffect::pop(4),
            Opcode::STORE_SUBSCR | Opcode::STORE_SUBSCR_DICT | Opcode::STORE_SUBSCR_LIST_INT => {
                StackEffect::pop(3)
            }

            Opcode::SWAP => StackEffect::balanced(2 + (oparg - 2)),

            Opcode::TO_BOOL
            | Opcode::TO_BOOL_ALWAYS_TRUE
            | Opcode::TO_BOOL_BOOL
            | Opcode::TO_BOOL_INT
            | Opcode::TO_BOOL_LIST
            | Opcode::TO_BOOL_NONE
            | Opcode::TO_BOOL_STR
            | Opcode::UNARY_INVERT
            | Opcode::UNARY_NEGATIVE
            | Opcode::UNARY_NOT => StackEffect::balanced(1),

            Opcode::UNPACK_EX => StackEffect {
                pops: 1,
                pushes: 1 + (oparg >> 8) + (oparg & 0xFF),
            },
            Opcode::UNPACK_SEQUENCE
            | Opcode::UNPACK_SEQUENCE_LIST
            | Opcode::UNPACK_SEQUENCE_TUPLE => StackEffect {
                pops: 1,
                pushes: oparg,
            },
            Opcode::UNPACK_SEQUENCE_TWO_TUPLE => StackEffect { pops: 1, pushes: 2 },

            Opcode::WITH_EXCEPT_START => StackEffect { pops: 4, pushes: 5 },

            Opcode::YIELD_VALUE => StackEffect::balanced(1),

            Opcode::INVALID_OPCODE(_) => StackEffect::zero(),

            // Fallback
            _ => unimplemented!("stack_effect not implemented for {:?}", self),
        }
    }
}
