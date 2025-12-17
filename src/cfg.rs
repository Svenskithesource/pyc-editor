use std::{collections::VecDeque, fmt::Debug, ops::Deref};

use crate::traits::{GenericInstruction, InstructionAccess, Oparg, SimpleInstructionAccess};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockIndex {
    /// Index of the block in the `blocks` list of the CFG
    Index(usize),
    /// For jumps with invalid jump targets (the value is the invalid jump index)
    InvalidIndex(usize),
    /// For blocks without a target
    NoIndex,
}

#[derive(Debug, PartialEq, Eq)]
/// Represents a block in the control flow graph
pub struct Block<I> {
    pub instructions: Vec<I>,
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

#[derive(Debug, PartialEq, Eq)]
pub struct ControlFlowGraph<I> {
    pub blocks: Vec<Block<I>>,
    pub start_index: BlockIndex,
}

impl<I, T> SimpleInstructionAccess<I> for T where
    T: Deref<Target = [I]> + AsRef<[I]> + InstructionAccess<u8, I>
{
}

pub fn create_cfg<OpargType, I>(instructions: Vec<I>) -> ControlFlowGraph<I>
where
    OpargType: Oparg,
    I: GenericInstruction<OpargType>,
    Vec<I>: InstructionAccess<OpargType, I>,
{
    // Used for keeping track of finished blocks
    let mut blocks: Vec<Block<I>> = vec![];
    // Keeps indexes of instructions that start new blocks and still need to be processed.
    let mut block_queue: VecDeque<usize> = VecDeque::new();
    block_queue.push_front(0);

    let mut curr_block_index = 0;

    while !block_queue.is_empty() {
        let mut index = block_queue.pop_back().expect("queue is not empty");
        let mut curr_block = vec![];

        loop {
            let instruction = instructions
                .get(index)
                .expect("index was previously checked")
                .clone();

            curr_block.push(instruction.clone());

            if instruction.is_jump() {
                let next_instruction = if index + 1 < instructions.len() {
                    Some(index + 1)
                } else {
                    None
                };

                let jump_instruction =
                    if let Some((jump_index, _)) = instructions.get_jump_target(index as u32) {
                        Some(jump_index)
                    } else {
                        None
                    };

                blocks.push(Block {
                    instructions: curr_block,
                    branch_block: if let Some(jump_index) = jump_instruction {
                        block_queue.push_front(jump_index as usize);

                        BlockIndex::Index(curr_block_index + 1)
                    } else {
                        BlockIndex::InvalidIndex(
                            instructions
                                .get_full_arg(index)
                                .expect("Index is within bounds")
                                as usize,
                        )
                    },
                    default_block: if let Some(next_index) = next_instruction {
                        block_queue.push_front(next_index);

                        // If branch block also exists add another 1
                        BlockIndex::Index(
                            curr_block_index + 1 + if jump_instruction.is_some() { 1 } else { 0 },
                        )
                    } else {
                        BlockIndex::NoIndex
                    },
                });

                break;
            } else if instruction.stops_execution() {
                blocks.push(Block {
                    instructions: curr_block,
                    branch_block: BlockIndex::NoIndex,
                    default_block: BlockIndex::NoIndex,
                });

                break;
            }

            if index + 1 < instructions.len() {
                index += 1;
            } else {
                blocks.push(Block {
                    instructions: curr_block,
                    branch_block: BlockIndex::NoIndex,
                    default_block: BlockIndex::NoIndex,
                });

                break;
            }
        }

        curr_block_index += 1;
    }

    ControlFlowGraph::<I> {
        start_index: if !blocks.is_empty() {
            BlockIndex::Index(0)
        } else {
            BlockIndex::NoIndex
        },

        blocks: blocks,
    }
}

#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use petgraph::{
        dot::{Config, Dot},
        graph::NodeIndex,
    };

    use crate::{
        cfg::{create_cfg, Block, BlockIndex},
        traits::GenericInstruction,
        v311::instructions::{Instruction, Instructions},
    };

    fn add_block<I: GenericInstruction<u8>>(
        graph: &mut petgraph::Graph<String, &str>,
        blocks: &[Block<I>],
        block_index: &BlockIndex,
    ) -> Option<NodeIndex> {
        let block = match block_index {
            BlockIndex::Index(index) => blocks.get(*index).unwrap(),
            _ => return None,
        };

        let text = block
            .instructions
            .iter()
            .map(|i| format!("{:#?} {:#?}", i.get_opcode(), i.get_raw_value()))
            .collect::<Vec<_>>()
            .join("\n");

        let index = graph.add_node(text);

        let branch_index = add_block(graph, blocks, &block.branch_block);

        let default_index = add_block(graph, blocks, &block.default_block);

        if let Some(to_index) = branch_index {
            graph.add_edge(index, to_index, "branch");
        }

        if let Some(to_index) = default_index {
            graph.add_edge(index, to_index, "fallthrough");
        }

        Some(index)
    }

    #[test]
    fn make_dot_graph() {
        let instructions = Instructions::new(vec![
            Instruction::LoadConst(0),
            Instruction::LoadConst(1),
            Instruction::CompareOp(0),
            Instruction::PopJumpForwardIfTrue(2),
            Instruction::LoadConst(2),
            Instruction::ReturnValue(0),
            Instruction::LoadConst(3),
            Instruction::ReturnValue(0),
        ]);

        let cfg = create_cfg(instructions.to_vec());

        dbg!(&cfg);

        let mut graph = petgraph::Graph::<String, &str>::new();

        add_block(&mut graph, &cfg.blocks, &cfg.start_index);

        println!(
            "{:#?}",
            Dot::with_attr_getters(
                &graph,
                &[Config::NodeNoLabel],
                &|_, _| "".to_string(),
                &|_, (_, s)| format!(r#"label = "{}""#, s),
            )
        );
    }
}
