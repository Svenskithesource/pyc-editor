use crate::traits::{ExtInstructionsOwned, SimpleInstructionAccess};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockIndex {
    Index(usize),
    /// For jumps with invalid jump targets
    InvalidIndex(usize),
    /// For blocks without a target
    NoIndex,
}

/// Represents a block in the control flow graph
pub struct Block<T> {
    pub instructions: T,
    /// Index to block for conditional jump
    pub branch_block: BlockIndex,
    /// Index to default block (unconditional)
    pub default_block: BlockIndex,
}

impl<T> Block<T> {
    pub fn is_terminating(&self) -> bool {
        matches!(self.default_block, BlockIndex::NoIndex)
    }

    /// Whether the block has a conditional jump or not
    pub fn is_conditional(&self) -> bool {
        matches!(self.branch_block, BlockIndex::Index(_))
    }
}

pub struct ControlFlowGraph<T> {
    pub blocks: Vec<Block<T>>,
    pub start_index: BlockIndex,
}

pub fn create_cfg<T, I>(instructions: T) -> ControlFlowGraph<T>
where
    T: SimpleInstructionAccess<I>,
{
    // Used for keeping track of finished blocks
    let mut blocks: Vec<Block<T>> = vec![];
    // Keeps indexes of instructions that start new blocks and still need to be processed.
    let mut block_queue = vec![];

    ControlFlowGraph::<T> {
        blocks: vec![],
        start_index: BlockIndex::NoIndex,
    }
}
