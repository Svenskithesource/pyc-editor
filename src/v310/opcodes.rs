#![allow(non_snake_case)]

use crate::define_opcodes;
use crate::traits::GenericOpcode;
use crate::utils::StackEffect;
use crate::v310::instructions::Instruction;
use paste::paste;

define_opcodes!(
    POP_TOP = 1,
    ROT_TWO = 2,
    ROT_THREE = 3,
    DUP_TOP = 4,
    DUP_TOP_TWO = 5,
    ROT_FOUR = 6,
    NOP = 9,
    UNARY_POSITIVE = 10,
    UNARY_NEGATIVE = 11,
    UNARY_NOT = 12,
    UNARY_INVERT = 15,
    BINARY_MATRIX_MULTIPLY = 16,
    INPLACE_MATRIX_MULTIPLY = 17,
    BINARY_POWER = 19,
    BINARY_MULTIPLY = 20,
    BINARY_MODULO = 22,
    BINARY_ADD = 23,
    BINARY_SUBTRACT = 24,
    BINARY_SUBSCR = 25,
    BINARY_FLOOR_DIVIDE = 26,
    BINARY_TRUE_DIVIDE = 27,
    INPLACE_FLOOR_DIVIDE = 28,
    INPLACE_TRUE_DIVIDE = 29,
    GET_LEN = 30,
    MATCH_MAPPING = 31,
    MATCH_SEQUENCE = 32,
    MATCH_KEYS = 33,
    COPY_DICT_WITHOUT_KEYS = 34,
    WITH_EXCEPT_START = 49,
    GET_AITER = 50,
    GET_ANEXT = 51,
    BEFORE_ASYNC_WITH = 52,
    END_ASYNC_FOR = 54,
    INPLACE_ADD = 55,
    INPLACE_SUBTRACT = 56,
    INPLACE_MULTIPLY = 57,
    INPLACE_MODULO = 59,
    STORE_SUBSCR = 60,
    DELETE_SUBSCR = 61,
    BINARY_LSHIFT = 62,
    BINARY_RSHIFT = 63,
    BINARY_AND = 64,
    BINARY_XOR = 65,
    BINARY_OR = 66,
    INPLACE_POWER = 67,
    GET_ITER = 68,
    GET_YIELD_FROM_ITER = 69,
    PRINT_EXPR = 70,
    LOAD_BUILD_CLASS = 71,
    YIELD_FROM = 72,
    GET_AWAITABLE = 73,
    LOAD_ASSERTION_ERROR = 74,
    INPLACE_LSHIFT = 75,
    INPLACE_RSHIFT = 76,
    INPLACE_AND = 77,
    INPLACE_XOR = 78,
    INPLACE_OR = 79,
    LIST_TO_TUPLE = 82,
    RETURN_VALUE = 83,
    IMPORT_STAR = 84,
    SETUP_ANNOTATIONS = 85,
    YIELD_VALUE = 86,
    POP_BLOCK = 87,
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
    ROT_N = 99,
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
    JUMP_ABSOLUTE = 113,
    POP_JUMP_IF_FALSE = 114,
    POP_JUMP_IF_TRUE = 115,
    LOAD_GLOBAL = 116,
    IS_OP = 117,
    CONTAINS_OP = 118,
    RERAISE = 119,
    JUMP_IF_NOT_EXC_MATCH = 121,
    SETUP_FINALLY = 122,
    LOAD_FAST = 124,
    STORE_FAST = 125,
    DELETE_FAST = 126,
    GEN_START = 129,
    RAISE_VARARGS = 130,
    CALL_FUNCTION = 131,
    MAKE_FUNCTION = 132,
    BUILD_SLICE = 133,
    LOAD_CLOSURE = 135,
    LOAD_DEREF = 136,
    STORE_DEREF = 137,
    DELETE_DEREF = 138,
    CALL_FUNCTION_KW = 141,
    CALL_FUNCTION_EX = 142,
    SETUP_WITH = 143,
    EXTENDED_ARG = 144,
    LIST_APPEND = 145,
    SET_ADD = 146,
    MAP_ADD = 147,
    LOAD_CLASSDEREF = 148,
    MATCH_CLASS = 152,
    SETUP_ASYNC_WITH = 154,
    FORMAT_VALUE = 155,
    BUILD_CONST_KEY_MAP = 156,
    BUILD_STRING = 157,
    LOAD_METHOD = 160,
    CALL_METHOD = 161,
    LIST_EXTEND = 162,
    SET_UPDATE = 163,
    DICT_MERGE = 164,
    DICT_UPDATE = 165,
);

impl GenericOpcode for Opcode {
    /// From (by removing relative jumps): https://github.com/python/cpython/blob/fdc9d214c01cb4588f540cfa03726bbf2a33fc15/Include/opcode.h#L149-L158
    fn is_absolute_jump(&self) -> bool {
        matches!(
            self,
            Opcode::JUMP_IF_FALSE_OR_POP
                | Opcode::JUMP_IF_TRUE_OR_POP
                | Opcode::JUMP_ABSOLUTE
                | Opcode::POP_JUMP_IF_FALSE
                | Opcode::POP_JUMP_IF_TRUE
                | Opcode::JUMP_IF_NOT_EXC_MATCH
        )
    }

    /// From: https://github.com/python/cpython/blob/fdc9d214c01cb4588f540cfa03726bbf2a33fc15/Include/opcode.h#L139-L148
    fn is_relative_jump(&self) -> bool {
        matches!(
            self,
            Opcode::FOR_ITER
                | Opcode::JUMP_FORWARD
                | Opcode::SETUP_FINALLY
                | Opcode::SETUP_WITH
                | Opcode::SETUP_ASYNC_WITH
        )
    }

    fn is_jump_forwards(&self) -> bool {
        true
    }

    /// Impossible to jump backwards in 3.10
    fn is_jump_backwards(&self) -> bool {
        false
    }

    /// Relative or absolute jump
    fn is_jump(&self) -> bool {
        self.is_absolute_jump() | self.is_relative_jump()
    }

    fn is_extended_arg(&self) -> bool {
        matches!(self, Opcode::EXTENDED_ARG)
    }

    fn stack_effect(&self, oparg: u32, jump: Option<bool>) -> StackEffect {
        // See https://github.com/python/cpython/blob/3.10/Python/compile.c#L956
        match &self {
            Opcode::NOP | Opcode::EXTENDED_ARG => StackEffect::zero(),

            Opcode::ROT_TWO => StackEffect::balanced(2),
            Opcode::ROT_THREE => StackEffect::balanced(3),
            Opcode::ROT_FOUR => StackEffect::balanced(4),
            Opcode::UNARY_POSITIVE
            | Opcode::UNARY_NEGATIVE
            | Opcode::UNARY_NOT
            | Opcode::UNARY_INVERT => StackEffect::balanced(1),

            Opcode::POP_TOP => StackEffect::pop(1),

            Opcode::DUP_TOP => StackEffect { pops: 1, pushes: 2 },
            Opcode::DUP_TOP_TWO => StackEffect { pops: 1, pushes: 3 },

            Opcode::SET_ADD | Opcode::LIST_APPEND => StackEffect { pops: (oparg - 1) + 2, pushes: (oparg - 1) + 1},

            Opcode::MAP_ADD => StackEffect::pop(2),

            Opcode::BINARY_POWER
            | Opcode::BINARY_MULTIPLY
            | Opcode::BINARY_MATRIX_MULTIPLY
            | Opcode::BINARY_MODULO
            | Opcode::BINARY_ADD
            | Opcode::BINARY_SUBTRACT
            | Opcode::BINARY_SUBSCR
            | Opcode::BINARY_FLOOR_DIVIDE
            | Opcode::BINARY_TRUE_DIVIDE => StackEffect { pops: 2, pushes: 1 },

            Opcode::INPLACE_FLOOR_DIVIDE | Opcode::INPLACE_TRUE_DIVIDE => {
                StackEffect { pops: 2, pushes: 1 }
            }

            Opcode::INPLACE_ADD
            | Opcode::INPLACE_SUBTRACT
            | Opcode::INPLACE_MULTIPLY
            | Opcode::INPLACE_MATRIX_MULTIPLY
            | Opcode::INPLACE_MODULO => StackEffect { pops: 2, pushes: 1 },

            Opcode::STORE_SUBSCR => StackEffect::pop(3),

            Opcode::DELETE_SUBSCR => StackEffect::pop(2),

            Opcode::BINARY_LSHIFT
            | Opcode::BINARY_RSHIFT
            | Opcode::BINARY_AND
            | Opcode::BINARY_XOR
            | Opcode::BINARY_OR => StackEffect { pops: 2, pushes: 1 },

            Opcode::INPLACE_POWER => StackEffect { pops: 2, pushes: 1 },

            Opcode::GET_ITER => StackEffect::balanced(1),

            Opcode::PRINT_EXPR => StackEffect::pop(1),

            Opcode::LOAD_BUILD_CLASS => StackEffect::push(1),

            Opcode::INPLACE_LSHIFT
            | Opcode::INPLACE_RSHIFT
            | Opcode::INPLACE_AND
            | Opcode::INPLACE_XOR
            | Opcode::INPLACE_OR => StackEffect { pops: 2, pushes: 1 },

            Opcode::SETUP_WITH => StackEffect {
                pops: 1,
                pushes: match jump {
                    Some(true) => 7,
                    _ => 2,
                },
            },
            Opcode::RETURN_VALUE => StackEffect::pop(1),
            Opcode::IMPORT_STAR => StackEffect::pop(1),
            Opcode::SETUP_ANNOTATIONS => StackEffect::zero(),
            Opcode::YIELD_VALUE => StackEffect::pop(0),
            Opcode::YIELD_FROM => StackEffect { pushes: 1, pops: 2}, // Pops the generator/iterator to delegate to
            Opcode::POP_BLOCK => StackEffect::zero(), // Just modifies the block stack, not value stack
            Opcode::POP_EXCEPT => StackEffect::pop(3), // Pops exception type, value, and traceback

            Opcode::STORE_NAME => StackEffect::pop(1), // Pops value to store
            Opcode::DELETE_NAME => StackEffect::zero(), // Just removes name from namespace
            Opcode::UNPACK_SEQUENCE => StackEffect { 
                pops: 1, 
                pushes: oparg as u32 
            }, // Pops sequence, pushes oparg items
            Opcode::UNPACK_EX => StackEffect {
                pops: 1,
                pushes: ((oparg & 0xff) + (oparg >> 8) + 1) as u32
            }, // Pops sequence, pushes low + high + 1 items
            Opcode::FOR_ITER => {
                // At end of iterator: pops iterator, pushes nothing (-1)
                // Continue iterating: pops iterator, pushes iterator + next value (+1)
                match jump {
                    Some(true) => StackEffect::pop(1), // End of iteration
                    _ => StackEffect { pops: 1, pushes: 2 }, // Continue iteration
                }
            }

            Opcode::STORE_ATTR => StackEffect::pop(2), // Pops object and value
            Opcode::DELETE_ATTR => StackEffect::pop(1), // Pops object
            Opcode::STORE_GLOBAL => StackEffect::pop(1), // Pops value to store
            Opcode::DELETE_GLOBAL => StackEffect::zero(), // Just removes global
            Opcode::LOAD_CONST => StackEffect::push(1), // Pushes constant value
            Opcode::LOAD_NAME => StackEffect::push(1), // Pushes name value
            Opcode::BUILD_TUPLE | Opcode::BUILD_LIST | Opcode::BUILD_SET | Opcode::BUILD_STRING => {
                StackEffect { 
                    pops: oparg as u32, 
                    pushes: 1 
                } // Pops oparg items, pushes 1 container
            }
            Opcode::BUILD_MAP => StackEffect { 
                pops: (2 * oparg) as u32, 
                pushes: 1 
            }, // Pops 2*oparg items (key-value pairs), pushes 1 map
            Opcode::BUILD_CONST_KEY_MAP => StackEffect { 
                pops: (oparg + 1) as u32, 
                pushes: 1 
            }, // Pops oparg values + 1 key tuple, pushes 1 map
            Opcode::LOAD_ATTR => StackEffect::balanced(1), // Pops object, pushes attribute value
            Opcode::COMPARE_OP | Opcode::IS_OP | Opcode::CONTAINS_OP => StackEffect { 
                pops: 2, 
                pushes: 1 
            }, // Pops 2 operands, pushes comparison result
            Opcode::JUMP_IF_NOT_EXC_MATCH => StackEffect::pop(2), // Pops exception and match target
            Opcode::IMPORT_NAME => StackEffect { 
                pops: 2, 
                pushes: 1 
            }, // Pops level and fromlist, pushes module
            Opcode::IMPORT_FROM => StackEffect::push(1), // Keeps module on stack, pushes imported name

            Opcode::JUMP_FORWARD | Opcode::JUMP_ABSOLUTE => StackEffect::zero(), // No stack effect

            Opcode::JUMP_IF_TRUE_OR_POP | Opcode::JUMP_IF_FALSE_OR_POP => match jump {
                Some(true) => StackEffect::zero(), // Jump taken, value stays on stack
                _ => StackEffect::pop(1), // Jump not taken, value is popped
            },

            Opcode::POP_JUMP_IF_FALSE | Opcode::POP_JUMP_IF_TRUE => StackEffect::pop(1), // Always pops test value

            Opcode::LOAD_GLOBAL => StackEffect::push(1), // Pushes global value

            Opcode::SETUP_FINALLY => match jump {
                Some(true) => StackEffect { pops: 0, pushes: 6 }, // Sets up exception handler with 6 values
                _ => StackEffect::zero(), // Normal flow
            },
            Opcode::RERAISE => StackEffect::pop(3), // Pops exception type, value, traceback

            Opcode::WITH_EXCEPT_START => StackEffect::push(1), // Pushes result of __exit__

            Opcode::LOAD_FAST => StackEffect::push(1), // Pushes local variable value
            Opcode::STORE_FAST => StackEffect::pop(1), // Pops value to store in local
            Opcode::DELETE_FAST => StackEffect::zero(), // Just clears local variable

            Opcode::RAISE_VARARGS => StackEffect::pop(oparg as u32), // Pops oparg exception arguments

            Opcode::CALL_FUNCTION => StackEffect { 
                pops: (oparg + 1) as u32, 
                pushes: 1 
            }, // Pops function + oparg arguments, pushes result
            Opcode::CALL_METHOD => StackEffect { 
                pops: (oparg + 2) as u32, 
                pushes: 1 
            }, // Pops self + method + oparg arguments, pushes result
            Opcode::CALL_FUNCTION_KW => StackEffect { 
                pops: (oparg + 2) as u32, 
                pushes: 1 
            }, // Pops function + oparg arguments + kwargs tuple, pushes result
            Opcode::CALL_FUNCTION_EX => {
                if (oparg & 0x01) != 0 {
                    StackEffect { pops: 3, pushes: 1 } // Function + args + kwargs
                } else {
                    StackEffect { pops: 2, pushes: 1 } // Function + args
                }
            }
            Opcode::MAKE_FUNCTION => StackEffect { 
                pops: (2 + (oparg & 0b1111).count_ones()) as u32, 
                pushes: 1 
            }, // Pops name + code + obj for each flag, pushes function
            Opcode::BUILD_SLICE => {
                if oparg == 3 {
                    StackEffect { pops: 3, pushes: 1 } // start, stop, step
                } else {
                    StackEffect { pops: 2, pushes: 1 } // start, stop
                }
            }

            Opcode::LOAD_CLOSURE => StackEffect::push(1), // Pushes closure variable
            Opcode::LOAD_DEREF | Opcode::LOAD_CLASSDEREF => StackEffect::push(1), // Pushes dereferenced variable
            Opcode::STORE_DEREF => StackEffect::pop(1), // Pops value to store in closure
            Opcode::DELETE_DEREF => StackEffect::zero(), // Just clears closure variable

            Opcode::GET_AWAITABLE => StackEffect::balanced(1), // Transforms object to awaitable
            Opcode::SETUP_ASYNC_WITH => {
                match jump {
                    // Restore the stack position to the position before the result
                    // of __aenter__ and push 6 values before jumping to the handler
                    // if an exception be raised.
                    Some(true) => StackEffect { pops: 1, pushes: 6 },
                    // Normal flow
                    _ => StackEffect::zero(),
                }
            }
            Opcode::BEFORE_ASYNC_WITH => StackEffect::push(1), // Pushes result of __aenter__
            Opcode::GET_AITER => StackEffect::balanced(1), // Transforms to async iterator
            Opcode::GET_ANEXT => StackEffect { pops: 1, pushes: 2}, // Pushes next value from async iterator
            Opcode::GET_YIELD_FROM_ITER => StackEffect::balanced(1), // Ensures object is iterator
            Opcode::END_ASYNC_FOR => StackEffect::pop(7), // Cleans up async for loop stack
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
            Opcode::LOAD_METHOD => StackEffect { pops: 1, pushes: 2}, // Pushes method of TOS
            Opcode::LOAD_ASSERTION_ERROR => StackEffect::push(1), // Pushes AssertionError class
            Opcode::LIST_TO_TUPLE => StackEffect::balanced(1), // Converts list to tuple
            Opcode::GEN_START => StackEffect::pop(1), // Pops generator argument
            Opcode::LIST_EXTEND | Opcode::SET_UPDATE | Opcode::DICT_MERGE | Opcode::DICT_UPDATE => {
                StackEffect { pops: 2, pushes: 1} // Pops the iterable/mapping to extend/update with
            }
            Opcode::COPY_DICT_WITHOUT_KEYS => StackEffect::balanced(2),
            Opcode::MATCH_CLASS => StackEffect { 
                pops: 3, 
                pushes: 2 
            },
            Opcode::GET_LEN | Opcode::MATCH_MAPPING | Opcode::MATCH_SEQUENCE => StackEffect { pops: 1, pushes: 2}, // Pushes length or match result
            Opcode::MATCH_KEYS => StackEffect { 
                pops: 2, 
                pushes: 4 
            },
            Opcode::ROT_N => StackEffect::zero(), // Rotates top N items without changing count

            Opcode::INVALID_OPCODE(_) => StackEffect::zero(), // Unknown opcode, assume no effect

            _ => unimplemented!("stack_effect not implemented for {:?}", self),
        }
    }
}
