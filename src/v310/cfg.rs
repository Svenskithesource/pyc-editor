use crate::v310::code_objects::Instructions;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockIndex {
    Index(usize),
    /// For jumps with invalid jump targets
    InvalidIndex(usize),
    /// For blocks without a target
    NoIndex,
}

/// Represents a block in the control flow graph
pub struct Block {
    instructions: Instructions,
    /// Index to block for conditional jump
    branch_block: BlockIndex,
    /// Index to default block (unconditional)
    default_block: BlockIndex,
}

impl Block {
    pub fn is_terminating(&self) -> bool {
        matches!(self.default_block, BlockIndex::NoIndex)
    }

    /// Whether the block has a conditional jump or not
    pub fn is_conditional(&self) -> bool {
        matches!(self.branch_block, BlockIndex::Index(_))
    }
}

pub struct ControlFlowGraph {
    blocks: Vec<Block>,
    start_index: BlockIndex,
}

impl From<Instructions> for ControlFlowGraph {
    fn from(value: Instructions) -> Self {
        let mut blocks = vec![];

        ControlFlowGraph {
            blocks,
            start_index: BlockIndex::NoIndex,
        }
    }
}
