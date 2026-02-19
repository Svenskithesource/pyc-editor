#![allow(non_snake_case)]

use crate::traits::{GenericOpcode, StackEffectTrait};
use crate::utils::StackEffect;
use crate::v310::instructions::Instruction;

use python_instruction_dsl_proc::define_opcodes;

// See https://github.com/python/cpython/blob/main/Tools/cases_generator/interpreter_definition.md#syntax for the grammar of the stack manipulation
define_opcodes!(
    POP_TOP = 1 (top -- ),
    ROT_TWO = 2 (second, first -- first, second),
    ROT_THREE = 3 (third, second, first -- first, second, third),
    DUP_TOP = 4 (top -- top, top),
    DUP_TOP_TWO = 5 (top -- top, top, top),
    ROT_FOUR = 6 (fourth, third, second, first -- first, third, second, fourth),
    NOP = 9 ( -- ),
    UNARY_POSITIVE = 10 (value -- res),
    UNARY_NEGATIVE = 11 (value -- res),
    UNARY_NOT = 12 (value -- res),
    UNARY_INVERT = 15 (value -- res),
    BINARY_MATRIX_MULTIPLY = 16 (left, right -- res),
    INPLACE_MATRIX_MULTIPLY = 17 (left, right -- res),
    BINARY_POWER = 19 (base, exp -- res),
    BINARY_MULTIPLY = 20 (left, right -- res),
    BINARY_MODULO = 22 (dividend, divisor -- res),
    BINARY_ADD = 23 (left, right -- sum),
    BINARY_SUBTRACT = 24 (left, right -- diff),
    BINARY_SUBSCR = 25 (container, sub -- res),
    BINARY_FLOOR_DIVIDE = 26 (dividend, divisor -- quotient),
    BINARY_TRUE_DIVIDE = 27 (dividend, divisor -- quotient),
    INPLACE_FLOOR_DIVIDE = 28 (dividend, divisor -- quotient),
    INPLACE_TRUE_DIVIDE = 29 (dividend, divisor -- quotient),
    GET_LEN = 30 (obj -- obj, length),
    MATCH_MAPPING = 31 (subject -- subject, res),
    MATCH_SEQUENCE = 32 (subject -- subject, res),
    MATCH_KEYS = 33 (subject, keys -- subject, keys, values_or_none, true_or_false),
    COPY_DICT_WITHOUT_KEYS = 34 (subject, keys -- subject, rest),
    // "unused[3]" here are the 3 exception values of the previously raised exception
    WITH_EXCEPT_START = 49 (exit_func, unused[3], tb, val, exc -- exit_func, unused[3], tb, val, exc, res),
    GET_AITER = 50 (obj -- iter),
    GET_ANEXT = 51 (aiter -- aiter, awaitable),
    BEFORE_ASYNC_WITH = 52 (mgr -- exit, res),
    END_ASYNC_FOR = 54 (awaitable, prev_tb, prev_val, prev_exc, tb, val, exc -- ),
    INPLACE_ADD = 55 (left, right -- sum),
    INPLACE_SUBTRACT = 56 (left, right -- diff),
    INPLACE_MULTIPLY = 57 (left, right -- res),
    INPLACE_MODULO = 59 (left, right -- modulo),
    STORE_SUBSCR = 60 (value, container, sub -- ),
    DELETE_SUBSCR = 61 (container, sub -- ),
    BINARY_LSHIFT = 62 (left, right -- res),
    BINARY_RSHIFT = 63 (left, right -- res),
    BINARY_AND = 64 (left, right -- res),
    BINARY_XOR = 65 (left, right -- res),
    BINARY_OR = 66 (left, right -- res),
    INPLACE_POWER = 67 (base, exp -- res),
    GET_ITER = 68 (iterable -- iter),
    GET_YIELD_FROM_ITER = 69 (iterable -- iter),
    PRINT_EXPR = 70 (value -- ),
    LOAD_BUILD_CLASS = 71 ( -- bc),
    YIELD_FROM = 72 (receiver, value -- retval),
    GET_AWAITABLE = 73 (iterable -- iter),
    LOAD_ASSERTION_ERROR = 74 ( -- value),
    INPLACE_LSHIFT = 75 (left, right -- res),
    INPLACE_RSHIFT = 76 (left, right -- res),
    INPLACE_AND = 77 (left, right -- res),
    INPLACE_XOR = 78 (left, right -- res),
    INPLACE_OR = 79 (left, right -- res),
    LIST_TO_TUPLE = 82 (list -- tuple),
    RETURN_VALUE = 83 (return_value -- ),
    IMPORT_STAR = 84 (from -- ),
    SETUP_ANNOTATIONS = 85 ( -- ),
    YIELD_VALUE = 86 (value -- received_value),
    POP_BLOCK = 87 ( -- ),
    POP_EXCEPT = 89 (exc_tb, exc_value, exc_type -- ),
    STORE_NAME = 90 (value -- ),
    DELETE_NAME = 91 ( -- ),
    UNPACK_SEQUENCE = 92 (seq -- unpacked[oparg]),
    FOR_ITER = 93 (iter -- iter[if jump || calculate_max {0} else {1}], next[if jump || calculate_max {0} else {1}]),
    UNPACK_EX = 94 (seq -- before[oparg & 0xFF], leftover, after[oparg >> 8]),
    STORE_ATTR = 95 (value, owner -- ),
    DELETE_ATTR = 96 (owner -- ),
    STORE_GLOBAL = 97 (value -- ),
    DELETE_GLOBAL = 98 ( -- ),
    ROT_N = 99 (bottom, array[oparg-2], top -- top, array[oparg - 2], bottom),
    LOAD_CONST = 100 ( -- value),
    LOAD_NAME = 101 ( -- value),
    BUILD_TUPLE = 102 (values[oparg] -- tuple),
    BUILD_LIST = 103 (values[oparg] -- list),
    BUILD_SET = 104 (values[oparg] -- set),
    BUILD_MAP = 105 (values[oparg*2] -- map),
    LOAD_ATTR = 106 (owner -- res),
    COMPARE_OP = 107 (left, right -- res),
    IMPORT_NAME = 108 (level, fromlist -- res),
    IMPORT_FROM = 109 (from -- from, res),
    JUMP_FORWARD = 110 ( -- ),
    // Condition is only kept on stack if jump is taken
    JUMP_IF_FALSE_OR_POP = 111 (condition -- condition[if jump && !calculate_max {1} else {0}]),
    JUMP_IF_TRUE_OR_POP = 112 (condition -- condition[if jump && !calculate_max {1} else {0}]),
    JUMP_ABSOLUTE = 113 ( -- ),
    POP_JUMP_IF_FALSE = 114 (condition --),
    POP_JUMP_IF_TRUE = 115 (condition --),
    LOAD_GLOBAL = 116 ( -- value),
    IS_OP = 117 (left, right -- boolean),
    CONTAINS_OP = 118 (left, right -- boolean),
    RERAISE = 119 (values[oparg], tb, val, exc -- values[oparg]),
    JUMP_IF_NOT_EXC_MATCH = 121 (left, right --),
    // When the jump is taken (exception raised) the stack looks like this ( -- prev_tb, prev_value, prev_type, curr_tb, curr_value, curr_type)
    SETUP_FINALLY = 122 ( -- exceptions[if jump || calculate_max {6} else {0}]),
    LOAD_FAST = 124 ( -- value),
    STORE_FAST = 125 (value -- ),
    DELETE_FAST = 126 ( -- ),
    GEN_START = 129 (none -- ),
    // oparg must be 2 or 1
    RAISE_VARARGS = 130 (args[oparg] -- ),
    CALL_FUNCTION = 131 (callable, pos_args[oparg] -- res),
    MAKE_FUNCTION = 132 (defaults[if oparg & 0x01 != 0 {1} else {0}],
                        kwdefaults[if oparg & 0x02 != 0 {1} else {0}],
                        annotations[if oparg & 0x04 != 0 {1} else {0}],
                        closure[if oparg & 0x08 != 0 {1} else {0}],
                        code_obj, qualname -- func),
    BUILD_SLICE = 133 (start, stop, step[if oparg == 3 {1} else {0}] -- slice),
    LOAD_CLOSURE = 135 ( -- value),
    LOAD_DEREF = 136 ( -- value),
    STORE_DEREF = 137 (value -- ),
    DELETE_DEREF = 138 ( -- ),
    CALL_FUNCTION_KW = 141 (callable, args[oparg], name_tuple -- res),
    CALL_FUNCTION_EX = 142 (callable, args, kwargs[if oparg & 0x01 != 0 {1} else {0}] -- res),
    SETUP_WITH = 143 (mgr -- exit, exc[if jump || calculate_max {3 + 3} else {0}], enter[if jump || calculate_max {0} else {1}]),
    EXTENDED_ARG = 144 ( -- ),
    LIST_APPEND = 145 (list, unused[oparg-1], value -- list, unused[oparg-1]),
    SET_ADD = 146 (set, unused[oparg-1], value -- set, unused[oparg-1]),
    MAP_ADD = 147 (map, unused[oparg-1], key, value -- map, unused[oparg-1]),
    LOAD_CLASSDEREF = 148 ( -- value),
    MATCH_CLASS = 152 (subject, cmp_type, names -- attrs_or_none, boolean),
    SETUP_ASYNC_WITH = 154 (res -- res[if jump || calculate_max {0} else {1}], excs[if jump || calculate_max {6} else {0}]),
    // FVS_MASK = FVS_HAVE_SPEC = 0x4
    FORMAT_VALUE = 155 (value, fmt_spec[if (oparg & 0x4) == 0x4 {1} else {0}] -- result),
    BUILD_CONST_KEY_MAP = 156 (values[oparg], keys -- map),
    BUILD_STRING = 157 (pieces[oparg] -- string),
    LOAD_METHOD = 160 (obj -- null_or_method, method_or_self),
    CALL_METHOD = 161 (null_or_method, method_or_self, pos_args[oparg] -- res),
    LIST_EXTEND = 162 (list, unused[oparg-1], iterable -- list, unused[oparg-1]),
    SET_UPDATE = 163 (set, unused[oparg-1], iterable -- set, unused[oparg-1]),
    DICT_MERGE = 164 (update --),
    DICT_UPDATE = 165 (update --),
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

    fn is_conditional_jump(&self) -> bool {
        matches!(
            self,
            Opcode::JUMP_IF_FALSE_OR_POP
                | Opcode::JUMP_IF_TRUE_OR_POP
                | Opcode::POP_JUMP_IF_FALSE
                | Opcode::POP_JUMP_IF_TRUE
                | Opcode::FOR_ITER
                | Opcode::JUMP_IF_NOT_EXC_MATCH
                | Opcode::SETUP_FINALLY
                | Opcode::SETUP_WITH
                | Opcode::SETUP_ASYNC_WITH
        )
    }

    fn stops_execution(&self) -> bool {
        matches!(
            self,
            Opcode::RETURN_VALUE | Opcode::RAISE_VARARGS | Opcode::RERAISE
        )
    }

    fn is_extended_arg(&self) -> bool {
        matches!(self, Opcode::EXTENDED_ARG)
    }

    fn get_nop() -> Self {
        Opcode::NOP
    }
}
