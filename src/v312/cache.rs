use crate::v312::opcodes::Opcode;

/// Cache layout of the LOAD_GLOBAL instruction
/// See https://github.com/python/cpython/blob/3.12/Include/internal/pycore_code.h#L20
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LoadGlobalCache {
    pub counter: u16,
    pub index: u16,
    pub module_keys_version: u16,
    pub builtin_keys_version: u16,
}

/// Cache layout of the BINARY_OP, UNPACK_SEQUENCE, COMPARE_OP, BINARY_SUBCR, FOR_ITER, LOAD_SUPER_ATTR, STORE_SUBSCR and SEND instruction
/// See https://github.com/python/cpython/blob/3.12/Include/internal/pycore_code.h#L29
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CounterCache {
    pub counter: u16,
}

/// Cache layout of the LOAD_ATTR instruction
/// See https://github.com/python/cpython/blob/3.12/Include/internal/pycore_code.h#L66
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LoadAttrCache {
    pub counter: u16,
    pub version: (u16, u16),
    pub keys_version: (u16, u16),
    pub descr: (u16, u16, u16, u16),
}

/// Cache layout of the STORE_ATTR instruction
/// See https://github.com/python/cpython/blob/3.12/Include/internal/pycore_code.h#L60
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StoreAttrCache {
    pub counter: u16,
    pub version: (u16, u16),
    pub index: u16,
}

/// Cache layout of the CALL instruction
/// See https://github.com/python/cpython/blob/3.12/Include/internal/pycore_code.h#L60
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CallCache {
    pub counter: u16,
    pub version: (u16, u16),
    pub index: u16,
}

/// Returns the amount of cache instructions necessary to represent the cache information for the given opcode.
/// See
pub fn get_cache_count(opcode: Opcode) -> Option<usize> {
    let byte_size = match opcode {
        Opcode::LOAD_GLOBAL => std::mem::size_of::<LoadGlobalCache>(),
        Opcode::BINARY_OP
        | Opcode::UNPACK_SEQUENCE
        | Opcode::COMPARE_OP
        | Opcode::BINARY_SUBSCR
        | Opcode::FOR_ITER
        | Opcode::LOAD_SUPER_ATTR
        | Opcode::STORE_SUBSCR
        | Opcode::SEND => std::mem::size_of::<CounterCache>(),
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
    use crate::v312::{cache::get_cache_count, opcodes::Opcode};

    #[test]
    fn test_cache_count() {
        assert_eq!(get_cache_count(Opcode::LOAD_GLOBAL), Some(4));
        assert_eq!(get_cache_count(Opcode::LOAD_ATTR), Some(9));
    }
}