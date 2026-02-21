use std::collections::{HashMap, VecDeque};

/// The amount of extended_args necessary to represent the arg.
/// This is more efficient than `get_extended_args` as we only calculate the count and the actual values.
pub fn get_extended_args_count(arg: u32) -> u8 {
    if arg <= u8::MAX.into() {
        0
    } else if arg <= u16::MAX.into() {
        1
    } else if arg <= 0xffffff {
        2
    } else {
        3
    }
}

/// Used to represent opargs for opcodes that don't require arguments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UnusedArgument(pub u32);

impl From<u32> for UnusedArgument {
    fn from(value: u32) -> Self {
        UnusedArgument(value)
    }
}

/// Used to represent stack operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackEffect {
    pub pushes: u32,
    pub pops: u32,
}

/// Offsets are for instructions (not bytes)
#[derive(Debug, Clone, PartialEq)]
pub struct ExceptionTableEntry {
    /// Inclusive offset
    pub start: u32,
    /// Exclusive offset
    pub end: u32,
    pub target: u32,
    /// Stack depth at the start of the try block
    pub depth: u32,
    /// Whether to push the index of the last executed instruction
    pub lasti: bool,
}

impl StackEffect {
    /// Creates a StackEffect with equal pushes and pops.
    pub fn balanced(count: u32) -> Self {
        StackEffect {
            pushes: count,
            pops: count,
        }
    }

    /// Creates a StackEffect when only pushing
    pub fn push(count: u32) -> Self {
        StackEffect {
            pushes: count,
            pops: 0,
        }
    }

    /// Creates a StackEffect when only pushing
    pub fn pop(count: u32) -> Self {
        StackEffect {
            pushes: 0,
            pops: count,
        }
    }

    /// For when there is no stack access
    pub fn zero() -> Self {
        StackEffect { pushes: 0, pops: 0 }
    }

    /// Calculates the net total for the stackeffect
    pub fn net_total(&self) -> i32 {
        self.pushes as i32 - self.pops as i32
    }
}

#[macro_export]
macro_rules! define_default_traits {
    ($variant:ident, Instruction) => {
        impl Deref for $crate::$variant::instructions::Instructions {
            type Target = [$crate::$variant::instructions::Instruction];

            /// Allow the user to get a reference slice to the instructions
            fn deref(&self) -> &Self::Target {
                self.0.deref()
            }
        }

        impl DerefMut for $crate::$variant::instructions::Instructions {
            /// Allow the user to get a mutable reference slice for making modifications to existing instructions.
            fn deref_mut(&mut self) -> &mut [$crate::$variant::instructions::Instruction] {
                self.0.deref_mut()
            }
        }

        impl AsRef<[$crate::$variant::instructions::Instruction]>
            for $crate::$variant::instructions::Instructions
        {
            fn as_ref(&self) -> &[$crate::$variant::instructions::Instruction] {
                &self.0
            }
        }

        impl From<$crate::$variant::instructions::Instructions> for Vec<u8> {
            fn from(val: $crate::$variant::instructions::Instructions) -> Self {
                val.to_bytes()
            }
        }

        impl TryFrom<&[u8]> for $crate::$variant::instructions::Instructions {
            type Error = Error;
            fn try_from(code: &[u8]) -> Result<Self, Self::Error> {
                if code.len() % 2 != 0 {
                    return Err(Error::InvalidBytecodeLength);
                }

                let mut instructions = $crate::$variant::instructions::Instructions(
                    Vec::with_capacity(code.len() / 2),
                );

                for chunk in code.chunks(2) {
                    if chunk.len() != 2 {
                        return Err(Error::InvalidBytecodeLength);
                    }
                    let opcode = Opcode::from(chunk[0]);
                    let arg = chunk[1];

                    instructions.append_instruction((opcode, arg).into());
                }

                Ok(instructions)
            }
        }

        impl From<&[Instruction]> for Instructions {
            fn from(value: &[Instruction]) -> Self {
                $crate::$variant::instructions::Instructions::new(value.to_vec())
            }
        }

        impl InstructionsOwned<$crate::$variant::instructions::Instruction>
            for $crate::$variant::instructions::Instructions
        {
            type Instruction = $crate::$variant::instructions::Instruction;

            fn push(&mut self, item: Self::Instruction) {
                self.0.push(item);
            }
        }

        // impl<T> SimpleInstructionAccess<$crate::$variant::instructions::Instruction> for T where
        //     T: Deref<Target = [Instruction]> + AsRef<[Instruction]>
        // {
        // }
    };

    ($variant:ident, ExtInstruction) => {
        impl Deref for $crate::$variant::ext_instructions::ExtInstructions {
            type Target = [$crate::$variant::ext_instructions::ExtInstruction];

            /// Allow the user to get a reference slice to the instructions
            fn deref(&self) -> &Self::Target {
                self.0.deref()
            }
        }

        impl DerefMut for $crate::$variant::ext_instructions::ExtInstructions {
            /// Allow the user to get a mutable reference slice for making modifications to existing instructions.
            fn deref_mut(&mut self) -> &mut [$crate::$variant::ext_instructions::ExtInstruction] {
                self.0.deref_mut()
            }
        }

        impl AsRef<[$crate::$variant::ext_instructions::ExtInstruction]>
            for $crate::$variant::ext_instructions::ExtInstructions
        {
            fn as_ref(&self) -> &[$crate::$variant::ext_instructions::ExtInstruction] {
                &self.0
            }
        }

        impl From<$crate::$variant::ext_instructions::ExtInstructions> for Vec<u8> {
            fn from(val: $crate::$variant::ext_instructions::ExtInstructions) -> Self {
                val.to_bytes()
            }
        }

        impl TryFrom<&[$crate::$variant::instructions::Instruction]>
            for $crate::$variant::ext_instructions::ExtInstructions
        {
            type Error = Error;

            fn try_from(
                value: &[$crate::$variant::instructions::Instruction],
            ) -> Result<Self, Self::Error> {
                $crate::$variant::ext_instructions::ExtInstructions::from_instructions(value)
            }
        }

        impl From<&[$crate::$variant::ext_instructions::ExtInstruction]>
            for $crate::$variant::ext_instructions::ExtInstructions
        {
            fn from(value: &[$crate::$variant::ext_instructions::ExtInstruction]) -> Self {
                $crate::$variant::ext_instructions::ExtInstructions::new(value.to_vec())
            }
        }
    };
}

pub fn generate_var_name(
    stack_name: &'static str,
    names: &mut HashMap<&'static str, u32>,
) -> String {
    if names.contains_key(stack_name) {
        *names.get_mut(stack_name).unwrap() += 1;
    } else {
        names.insert(stack_name, 0);
    }

    format!("{}_{}", stack_name, names[stack_name])
}

/// A vector that allows for negative indexes and automatically fills elements with an "empty" value when inserting at an arbitrary index below 0.
#[derive(Debug, Clone)]
pub struct InfiniteVec<T>
where
    T: Clone + std::fmt::Debug,
{
    data: VecDeque<Option<T>>,
    /// This is the offset that indicates where index "0" is really at
    negative_offset: usize,
}

impl<T> InfiniteVec<T>
where
    T: Clone + std::fmt::Debug,
{
    pub fn new() -> Self {
        InfiniteVec {
            data: vec![].into(),
            negative_offset: 0,
        }
    }

    pub fn insert(&mut self, index: isize, value: T) {
        let real_index = index + self.negative_offset as isize;

        if real_index < 0 {
            for _ in 0..(real_index.abs() - 1) {
                self.data.push_front(None)
            }

            self.data.push_front(Some(value));

            self.negative_offset += real_index.abs() as usize;
        } else {
            self.data.insert(real_index as usize, Some(value));
        }
    }

    pub fn push(&mut self, value: T) {
        self.data.push_back(Some(value));
    }

    pub fn get(&self, index: isize) -> Option<&Option<T>> {
        let real_index = index + self.negative_offset as isize;

        if real_index < 0 {
            None
        } else {
            self.data.get(real_index as usize)
        }
    }

    pub fn get_mut(&mut self, index: isize) -> Option<&mut Option<T>> {
        let real_index = index + self.negative_offset as isize;

        if real_index < 0 {
            None
        } else {
            self.data.get_mut(real_index as usize)
        }
    }

    pub fn remove(&mut self, index: isize) {
        let real_index = index + self.negative_offset as isize;

        if index < 0 {
            self.negative_offset -= 1;
        }

        self.data.remove(real_index.try_into().unwrap());
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn positive_len(&self) -> usize {
        self.data.len() - self.negative_offset
    }

    pub fn negative_len(&self) -> usize {
        debug_assert!(self.data.len() >= self.negative_offset);

        self.negative_offset
    }

    /// Collects the values with Some() value and their index
    pub fn collect_pairs(&self) -> Vec<(isize, &T)> {
        self.data
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_some())
            .map(|(i, value)| {
                (
                    i as isize - self.negative_offset as isize,
                    value.as_ref().unwrap(),
                )
            })
            .collect()
    }

    /// Tells us whether negative items were used
    pub fn no_negative_items(&self) -> bool {
        self.negative_offset == 0
    }

    pub fn iter(&self) -> std::collections::vec_deque::Iter<'_, Option<T>> {
        self.data.iter()
    }

    /// Only iter over negative indexed values
    pub fn iter_negative(
        &self,
    ) -> std::iter::Take<std::collections::vec_deque::Iter<'_, Option<T>>> {
        self.data.iter().take(self.negative_offset)
    }
}

impl<T> From<Vec<T>> for InfiniteVec<T>
where
    T: Clone + std::fmt::Debug,
{
    fn from(value: Vec<T>) -> Self {
        InfiniteVec {
            data: value.into_iter().map(|e| Some(e)).collect(),
            negative_offset: 0,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::utils::InfiniteVec;

    #[test]
    fn test_infinite_vec() {
        let mut infinite_vec = InfiniteVec::new();

        infinite_vec.push(1);
        infinite_vec.insert(-5, 5);

        assert_eq!(
            infinite_vec.iter().collect::<Vec<_>>(),
            vec![Some(5), None, None, None, None, Some(1)]
                .iter()
                .collect::<Vec<_>>()
        );

        assert_eq!(infinite_vec.get(0).unwrap(), &Some(1));
    }
}
