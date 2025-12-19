use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
    ops::Deref,
};

use crate::{
    error::Error,
    traits::{
        ExtInstructionAccess, GenericInstruction, GenericOpcode, InstructionAccess, Oparg,
        SimpleInstructionAccess,
    },
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BlockIndex {
    /// Index of the block in the `blocks` list of the CFG
    Index(usize),
    /// For jumps with invalid jump targets (the value is the invalid jump index)
    InvalidIndex(usize),
    /// For blocks without a target
    NoIndex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BranchBlockIndex<O>
where
    O: GenericOpcode,
{
    Edge(BranchEdge<O>),
    /// For blocks without a target
    NoIndex,
}

/// Used to represent the opcode that was used for this branch and the block index it's jumping to.
/// We do this so the value of the branch instruction cannot represent a wrong index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchEdge<O>
where
    O: GenericOpcode,
{
    opcode: O,
    block_index: BlockIndex,
}

#[derive(Debug, PartialEq, Eq)]
/// Represents a block in the control flow graph
pub struct Block<I>
where
    I: GenericInstruction,
{
    pub instructions: Vec<I>,
    /// Index to block for conditional jump
    pub branch_block: BranchBlockIndex<I::Opcode>,
    /// Index to default block (unconditional)
    pub default_block: BlockIndex,
}

impl<T> Block<T>
where
    T: GenericInstruction,
{
    pub fn is_terminating(&self) -> bool {
        matches!(self.default_block, BlockIndex::NoIndex)
    }

    /// Whether the block has a conditional jump or not
    pub fn is_conditional(&self) -> bool {
        matches!(
            self.branch_block,
            BranchBlockIndex::Edge(BranchEdge {
                block_index: BlockIndex::Index(_),
                ..
            })
        )
    }
}

#[derive(Debug, PartialEq)]
pub struct ControlFlowGraph<I>
where
    I: GenericInstruction,
{
    pub blocks: Vec<Block<I>>,
    pub start_index: BlockIndex,
}

impl<I, T> SimpleInstructionAccess<I> for T where
    T: Deref<Target = [I]> + AsRef<[I]> + InstructionAccess<u8, I>
{
}

/// This "fixes" a unique pattern:
///
/// LOAD_CONST 0
/// POP_JUMP_FORWARD_IF_TRUE 1
/// EXTENDED_ARG
/// STORE_NAME 0
///
/// We will convert this to
///
/// LOAD_CONST 0
/// POP_JUMP_FORWARD_IF_TRUE 3
/// EXTENDED_ARG
/// STORE_NAME 0
/// JUMP_FORWARD 1
/// STORE_NAME 0
///
/// `blocks_to_fix` contains a list of blocks indexes that jump over an EXTENDED_ARG
fn fix_extended_args<I>(cfg: &mut ControlFlowGraph<I>, blocks_to_fix: &[BlockIndex])
where
    I: GenericInstruction,
{
    for block_to_fix in blocks_to_fix {
        match block_to_fix {
            BlockIndex::Index(index) => {
                let default_block_index = match cfg.blocks[*index].default_block {
                    BlockIndex::Index(default_index) => default_index,
                    _ => unreachable!(),
                };

                let branch_block_index = match cfg.blocks[*index].branch_block {
                    BranchBlockIndex::Edge(BranchEdge {
                        block_index: BlockIndex::Index(branch_index),
                        ..
                    }) => branch_index,
                    _ => unreachable!(),
                };

                // The instruction that the EXTENDED_ARG should be applied to is at the start of this one
                let mut instructions = cfg.blocks[branch_block_index].instructions.clone();

                // If there are multiple extended args, this should indicate that
                let instructions_to_copy = instructions
                    .iter()
                    .take_while(|i| i.is_extended_arg())
                    .count()
                    // The instruction itself
                    + 1;

                // Remove rest of instructions in the branch block
                cfg.blocks
                    .get_mut(branch_block_index)
                    .expect("index is valid here")
                    .instructions = instructions
                    .iter()
                    .take(instructions_to_copy)
                    .cloned()
                    .collect();

                // Copy the instruction(s) behind the EXTENDED_ARG
                cfg.blocks
                    .get_mut(default_block_index)
                    .expect("index is valid here")
                    .instructions
                    .extend_from_slice(
                        &instructions
                            .iter()
                            .take(instructions_to_copy)
                            .cloned()
                            .collect::<Vec<_>>(),
                    );

                // Remove first instruction(s) in the branch block that we will copy into the new branch block
                let instructions = instructions.split_off(instructions_to_copy);

                // Create new block that has the remaining instructions of the branch block
                cfg.blocks.push(Block {
                    instructions: instructions,
                    branch_block: cfg.blocks[branch_block_index].branch_block.clone(),
                    default_block: cfg.blocks[branch_block_index].default_block.clone(),
                });

                let new_block_index = cfg.blocks.len() - 1;

                cfg.blocks
                    .get_mut(branch_block_index)
                    .expect("index is valid here")
                    .default_block = BlockIndex::Index(new_block_index);

                cfg.blocks
                    .get_mut(branch_block_index)
                    .expect("index is valid here")
                    .branch_block = BranchBlockIndex::NoIndex;

                cfg.blocks
                    .get_mut(default_block_index)
                    .expect("index is valid here")
                    .default_block = BlockIndex::Index(new_block_index);

                // Should never happen but just in case we will add a NOP.
                if cfg.blocks[new_block_index].instructions.is_empty() {
                    cfg.blocks
                        .get_mut(new_block_index)
                        .expect("index is valid here")
                        .instructions
                        .push(I::get_nop());
                }
            }
            _ => unreachable!(),
        }
    }
}

pub fn create_cfg<OpargType, I>(instructions: Vec<I>) -> ControlFlowGraph<I>
where
    OpargType: Oparg,
    I: GenericInstruction,
    Vec<I>: InstructionAccess<OpargType, I>,
{
    // Used for keeping track of finished blocks
    let mut blocks: Vec<Block<I>> = vec![];

    // Maps instruction index to block index
    let mut block_map: HashMap<usize, BlockIndex> = HashMap::new();

    // Keeps indexes of instructions that start new blocks and still need to be processed.
    let mut block_queue: VecDeque<usize> = VecDeque::new();
    block_queue.push_front(0);

    // A list of block indexes that will be used to fix a unique EXTENDED_ARG pattern. See `fix_extended_args`.
    let mut block_indexes_to_fix = vec![];

    let mut curr_block_index = 0;

    while !block_queue.is_empty() {
        let mut index = block_queue.pop_back().expect("queue is not empty");

        let mut curr_block = vec![];

        let start_index = index;

        loop {
            if block_map.contains_key(&index) {
                if !curr_block.is_empty() {
                    blocks.push(Block {
                        instructions: curr_block,
                        branch_block: BranchBlockIndex::NoIndex,
                        default_block: block_map[&index].clone(),
                    });
                }

                break;
            }

            let instruction = instructions[index].clone();

            if instruction.is_jump() {
                let next_instruction =
                    if index + 1 < instructions.len() && instruction.is_conditional_jump() {
                        Some(index + 1)
                    } else {
                        None
                    };

                let jump_instruction =
                    if let Some((jump_index, _)) = instructions.get_jump_target(index as u32) {
                        if instruction.is_conditional_jump() {
                            if let Some(instruction) = instructions.get(jump_index as usize - 1)
                                && instruction.is_extended_arg()
                            {
                                block_indexes_to_fix.push(BlockIndex::Index(curr_block_index));
                            }
                        }
                        Some(jump_index)
                    } else {
                        None
                    };

                blocks.push(Block {
                    instructions: curr_block,
                    branch_block: BranchBlockIndex::Edge(BranchEdge {
                        opcode: instruction.get_opcode(),
                        block_index: if let Some(jump_index) = jump_instruction {
                            if block_map.contains_key(&(jump_index as usize)) {
                                block_map[&(jump_index as usize)].clone()
                            } else {
                                block_queue.push_front(jump_index as usize);

                                BlockIndex::Index(curr_block_index + 1)
                            }
                        } else {
                            BlockIndex::InvalidIndex(
                                instructions
                                    .get_full_arg(index)
                                    .expect("Index is within bounds")
                                    as usize,
                            )
                        },
                    }),
                    default_block: if let Some(next_index) = next_instruction {
                        if block_map.contains_key(&next_index) {
                            block_map[&next_index].clone()
                        } else {
                            block_queue.push_front(next_index);

                            // If branch block also exists add another 1
                            BlockIndex::Index(
                                curr_block_index
                                    + 1
                                    + if jump_instruction.is_some() { 1 } else { 0 },
                            )
                        }
                    } else {
                        BlockIndex::NoIndex
                    },
                });

                break;
            }

            curr_block.push(instruction.clone());

            if instruction.stops_execution() {
                blocks.push(Block {
                    instructions: curr_block,
                    branch_block: BranchBlockIndex::NoIndex,
                    default_block: BlockIndex::NoIndex,
                });

                break;
            }

            if index + 1 < instructions.len() {
                index += 1;
            } else {
                blocks.push(Block {
                    instructions: curr_block,
                    branch_block: BranchBlockIndex::NoIndex,
                    default_block: BlockIndex::NoIndex,
                });

                break;
            }
        }

        if !block_map.contains_key(&index) {
            block_map.insert(start_index, BlockIndex::Index(curr_block_index));

            curr_block_index += 1;
        }
    }

    let mut cfg = ControlFlowGraph::<I> {
        start_index: if !blocks.is_empty() {
            BlockIndex::Index(0)
        } else {
            BlockIndex::NoIndex
        },

        blocks: blocks,
    };

    fix_extended_args(&mut cfg, &block_indexes_to_fix);

    cfg
}

// Convert a cfg that consists of simple instructions to a cfg where the extended args are resolved.
pub fn simple_cfg_to_ext_cfg<SimpleI, ExtI, ExtInstructions>(
    simple_cfg: &ControlFlowGraph<SimpleI>,
) -> Result<ControlFlowGraph<ExtI>, Error>
where
    SimpleI: GenericInstruction<OpargType = u8>,
    ExtI: GenericInstruction<OpargType = u32, Opcode = SimpleI::Opcode>,
    ExtInstructions: ExtInstructionAccess<SimpleI, ExtI>,
{
    let mut blocks = vec![];

    for block in &simple_cfg.blocks {
        let ext_instructions = ExtInstructions::from_instructions(&block.instructions)?.to_vec();

        blocks.push(Block {
            instructions: ext_instructions,
            branch_block: match &block.branch_block {
                BranchBlockIndex::Edge(edge) => BranchBlockIndex::Edge(BranchEdge {
                    opcode: edge.opcode.clone(),
                    block_index: edge.block_index.clone(),
                }),
                BranchBlockIndex::NoIndex => BranchBlockIndex::NoIndex,
            },
            default_block: block.default_block.clone(),
        });
    }

    Ok(ControlFlowGraph::<ExtI> {
        blocks,
        start_index: simple_cfg.start_index.clone(),
    })
}

#[cfg(test)]
mod test {
    use std::{collections::HashMap, fmt::Debug};

    use petgraph::{
        dot::{Config, Dot},
        graph::NodeIndex,
    };

    use crate::{
        cfg::{
            Block, BlockIndex, BranchBlockIndex, BranchEdge, ControlFlowGraph, create_cfg,
            simple_cfg_to_ext_cfg,
        },
        traits::GenericInstruction,
        v311::{
            ext_instructions::{ExtInstruction, ExtInstructions},
            instructions::{Instruction, Instructions},
        },
    };

    fn add_block<I: GenericInstruction>(
        graph: &mut petgraph::Graph<String, String>,
        blocks: &[Block<I>],
        block_index: &BlockIndex,
        block_map: &mut HashMap<BlockIndex, NodeIndex>,
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

        let index = if block_map.contains_key(block_index) {
            block_map[block_index]
        } else {
            let index = graph.add_node(text);
            block_map.insert(block_index.clone(), index);

            index
        };

        dbg!(&block.branch_block);

        let (block_index, opcode) = match &block.branch_block {
            BranchBlockIndex::Edge(BranchEdge {
                block_index,
                opcode,
            }) => (block_index, Some(opcode.clone())),
            _ => (&BlockIndex::NoIndex, None),
        };

        dbg!(&opcode);

        let branch_index = add_block(graph, blocks, block_index, block_map);

        let default_index = add_block(graph, blocks, &block.default_block, block_map);

        if let Some(to_index) = branch_index {
            let text = format!("{:#?}", opcode.unwrap());
            graph.add_edge(index, to_index, text);
        }

        if let Some(to_index) = default_index {
            graph.add_edge(index, to_index, "fallthrough".to_string());
        }

        Some(index)
    }

    #[test]
    fn simple_instructions() {
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

        make_dot_graph(&cfg);
    }

    #[test]
    fn simple_to_ext_instructions() {
        let instructions = Instructions::new(vec![
            Instruction::ExtendedArg(1),
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

        make_dot_graph(&cfg);

        let cfg: ControlFlowGraph<ExtInstruction> =
            simple_cfg_to_ext_cfg::<Instruction, ExtInstruction, ExtInstructions>(&cfg).unwrap();

        make_dot_graph(&cfg);
    }

    #[test]
    fn ext_trick() {
        let instructions = Instructions::new(vec![
            Instruction::LoadConst(0),
            Instruction::PopJumpForwardIfTrue(1),
            Instruction::ExtendedArg(0),
            Instruction::ExtendedArg(0),
            Instruction::StoreName(0),
        ]);

        let cfg = create_cfg(instructions.to_vec());

        make_dot_graph(&cfg);
    }

    fn make_dot_graph<I>(cfg: &ControlFlowGraph<I>)
    where
        I: GenericInstruction,
    {
        let mut graph = petgraph::Graph::<String, String>::new();

        add_block(
            &mut graph,
            &cfg.blocks,
            &cfg.start_index,
            &mut HashMap::new(),
        );

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
