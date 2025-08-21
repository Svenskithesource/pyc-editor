use crate::v313::opcodes::Opcode;

/// Cache layout of the LOAD_GLOBAL instruction
/// See https://github.com/python/cpython/blob/3.13/Include/internal/pycore_code.h#L90
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LoadGlobalCache {
    pub counter: u16,
    pub module_keys_version: u16,
    pub builtin_keys_version: u16,
    pub index: u16,
}

/// Cache layout of the BINARY_OP, UNPACK_SEQUENCE, COMPARE_OP, CONTAINS_OP, BINARY_SUBCR, FOR_ITER, LOAD_SUPER_ATTR, STORE_SUBSCR SEND, JUMP_BACKWARD, POP_JUMP_IF_TRUE, POP_JUMP_IF_FALSE, POP_JUMP_IF_NONE, POP_JUMP_IF_NOT_NONE instruction
/// See https://github.com/python/cpython/blob/3.13/Include/internal/pycore_code.h#L99
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CounterCache {
    pub counter: u16,
}

/// Union for the LOAD_ATTR cache middle field
#[repr(C)]
#[derive(Clone, Copy)]
pub union LoadAttrCacheUnion {
    pub keys_version: (u16, u16),
    pub dict_offset: u16,
}

/// Cache layout of the LOAD_ATTR instruction
/// See https://github.com/python/cpython/blob/3.13/Include/internal/pycore_code.h#L136
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LoadAttrCache {
    pub counter: u16,
    pub version: (u16, u16),
    pub union_field: LoadAttrCacheUnion,
    pub descr: (u16, u16, u16, u16),
}

/// Cache layout of the STORE_ATTR instruction
/// See https://github.com/python/cpython/blob/3.13/Include/internal/pycore_code.h#L130
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StoreAttrCache {
    pub counter: u16,
    pub version: (u16, u16),
    pub index: u16,
}

/// Cache layout of the CALL instruction
/// See https://github.com/python/cpython/blob/3.13/Include/internal/pycore_code.h#L152
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CallCache {
    pub counter: u16,
    pub version: (u16, u16),
    pub index: u16,
}

/// Cache layout of the TO_BOOL instruction
/// See https://github.com/python/cpython/blob/3.13/Include/internal/pycore_code.h#L177
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ToBoolCache {
    pub counter: u16,
    pub version: (u16, u16),
}

/// Returns the amount of cache instructions necessary to represent the cache information for the given opcode.
/// See
pub fn get_cache_count(opcode: Opcode) -> Option<usize> {
    let byte_size = match opcode {
        Opcode::LOAD_GLOBAL => std::mem::size_of::<LoadGlobalCache>(),
        Opcode::BINARY_OP
        | Opcode::UNPACK_SEQUENCE
        | Opcode::COMPARE_OP
        | Opcode::CONTAINS_OP
        | Opcode::BINARY_SUBSCR
        | Opcode::FOR_ITER
        | Opcode::LOAD_SUPER_ATTR
        | Opcode::STORE_SUBSCR
        | Opcode::SEND
        | Opcode::JUMP_BACKWARD
        | Opcode::POP_JUMP_IF_TRUE
        | Opcode::POP_JUMP_IF_FALSE
        | Opcode::POP_JUMP_IF_NONE
        | Opcode::POP_JUMP_IF_NOT_NONE => std::mem::size_of::<CounterCache>(),
        Opcode::LOAD_ATTR => std::mem::size_of::<LoadAttrCache>(),
        Opcode::STORE_ATTR => std::mem::size_of::<StoreAttrCache>(),
        Opcode::CALL => std::mem::size_of::<CallCache>(),
        _ => return None,
    };

    // u16 is the size of a (opcode, oparg)
    Some(byte_size / std::mem::size_of::<u16>())
}

#[cfg(test)]
mod tests {
    use crate::v313::{cache::get_cache_count, opcodes::Opcode};

    #[test]
    fn test_cache_count() {
        assert_eq!(get_cache_count(Opcode::LOAD_GLOBAL), Some(4));
        assert_eq!(get_cache_count(Opcode::LOAD_ATTR), Some(9));
    }
}
