use crate::{
    cfg::{
        Block, BlockIndex, BlockIndexInfo, BranchEdge, ControlFlowGraph, ExceptionBlock,
        NormalBlock, replace_block_index,
    },
    traits::FinalizeCFG,
    v310::{ext_instructions::ExtInstruction, instructions::Instruction, opcodes::Opcode},
};

macro_rules! generate_cfg_finalize {
    ($instruction:ident, $self:ident) => {
        let start_block_index =
            if let BlockIndex::Index(index) = $self.start_index.get_block_index().unwrap() {
                *index
            } else {
                return Ok(());
            };

        // (index of the block in self.blocks, block)
        let mut temp_blocks: Vec<(usize, Block<$instruction>)> = vec![];

        let mut visited_blocks: Vec<usize> = Vec::with_capacity($self.blocks.len());

        // Maps old block indexes to the new one
        let mut block_index_map: nohash_hasher::IntMap<usize, usize> =
            nohash_hasher::IntMap::default();

        /// The index is an index into `temp_exception_blocks`
        #[derive(Debug, Clone)]
        struct ExceptionBlockIndex(usize);

        // (block index, exception_block_indexes)
        let mut visit_queue: Vec<(usize, Vec<ExceptionBlockIndex>)> =
            vec![(start_block_index, vec![])];

        let mut current_block_index = $self.blocks.len();

        // Visit blocks by exploring the branches
        while let Some((block_index, mut exception_block_indexes)) = visit_queue.pop() {
            if visited_blocks.contains(&block_index) {
                continue;
            }

            let block = $self.blocks.get_mut(block_index).unwrap();
            visited_blocks.push(block_index);

            let original_branch_block = block.get_branch_block();
            let original_default_block = block.get_default_block();

            let original_branch_index = match block.get_branch_block().get_block_index() {
                Some(BlockIndex::Index(index)) => Some(*index),
                _ => None,
            };

            let original_default_index = match block.get_default_block().get_block_index() {
                Some(BlockIndex::Index(index)) => Some(*index),
                _ => None,
            };

            // It's not possible to have exception blocks here yet, since we're still creating them
            assert!(matches!(block, crate::cfg::Block::NormalBlock(_)));

            let pop_block_indexes = block
                .get_instructions_slice()
                .unwrap()
                .iter()
                .enumerate()
                .filter_map(|(index, instruction)| {
                    matches!(instruction, $instruction::PopBlock(_)).then_some(index)
                })
                .collect::<Vec<_>>();

            let last_block_index = if !pop_block_indexes.is_empty() {
                // We will split the instructions up, based on the POP_BLOCK indexes.
                // Then make them fallthrough to each other.

                let mut current_instruction_index = 0;
                let instructions = block.clone().get_instructions().unwrap();

                for (i, pop_block_index) in pop_block_indexes.into_iter().enumerate() {
                    let new_instructions =
                        instructions[current_instruction_index..=pop_block_index].to_vec();

                    if pop_block_index != instructions.len() - 1 {
                        // Not last block
                        let new_block = Block::NormalBlock(NormalBlock {
                            instructions: new_instructions,
                            branch_block: BlockIndexInfo::NoIndex,
                            default_block: BlockIndexInfo::Fallthrough(BlockIndex::Index(
                                current_block_index + if i != 0 { 1 } else { 0 },
                            )),
                        });

                        let current_block_index = if i == 0 {
                            *block = new_block;

                            block_index
                        } else {
                            temp_blocks.push((current_block_index, new_block));

                            current_block_index += 1;

                            current_block_index - 1
                        };

                        // Populate block index lists of current scope
                        for exception_block_index in &exception_block_indexes {
                            let (_, block) = temp_blocks.get_mut(exception_block_index.0).unwrap();

                            assert!(matches!(block, Block::ExceptionBlock(_)));
                            match block {
                                Block::ExceptionBlock(block) => {
                                    block.block_indexes.push(current_block_index)
                                }
                                _ => unreachable!(),
                            };
                        }
                    } else {
                        break;
                    }

                    match exception_block_indexes.pop() {
                        None => return Err(crate::Error::InvalidBlockStackUsage),
                        _ => {}
                    };

                    current_instruction_index = pop_block_index + 1
                }

                // The last instructions left
                let new_instructions = instructions[current_instruction_index..].to_vec();

                temp_blocks.push((
                    current_block_index,
                    Block::NormalBlock(NormalBlock {
                        instructions: new_instructions,
                        branch_block: original_branch_block,
                        default_block: original_default_block,
                    }),
                ));

                current_block_index += 1;

                current_block_index - 1
            } else {
                block_index
            };

            // Populate block index lists of current scope
            for exception_block_index in &exception_block_indexes {
                let (_, block) = temp_blocks.get_mut(exception_block_index.0).unwrap();

                assert!(matches!(block, Block::ExceptionBlock(_)));
                match block {
                    Block::ExceptionBlock(block) => block.block_indexes.push(last_block_index),
                    _ => unreachable!(),
                };
            }

            // Check if we're starting an exception block
            match block.get_branch_block() {
                BlockIndexInfo::Edge(BranchEdge {
                    reason: Opcode::SETUP_FINALLY,
                    block_index: branch_block_index,
                }) => {
                    // Add an ExceptionBlock and "reroute" jumps to the target block to the exception block instead (this happens later)

                    temp_blocks.push((
                        current_block_index,
                        Block::ExceptionBlock(ExceptionBlock {
                            block_indexes: vec![],
                            exception_handler: BranchEdge {
                                reason: Opcode::SETUP_FINALLY,
                                block_index: branch_block_index,
                            },
                            default_block: block.get_default_block(),
                        }),
                    ));

                    current_block_index += 1;

                    let exception_block_index = current_block_index - 1;

                    match block {
                        Block::NormalBlock(block) => {
                            block.branch_block = BlockIndexInfo::NoIndex;

                            block.default_block = BlockIndexInfo::Fallthrough(BlockIndex::Index(
                                // Points to the exception block we just pushed
                                exception_block_index,
                            ));

                            block_index_map.insert(block_index, exception_block_index);
                        }
                        Block::ExceptionBlock(_) => unreachable!(),
                    }

                    if let Some(index) = original_branch_index {
                        visit_queue.push((index, exception_block_indexes.clone()))
                    }

                    exception_block_indexes.push(ExceptionBlockIndex(temp_blocks.len() - 1));
                }
                _ => {
                    if let Some(index) = original_branch_index {
                        visit_queue.push((index, exception_block_indexes.clone()))
                    }
                }
            }

            if let Some(index) = original_default_index {
                visit_queue.push((index, exception_block_indexes))
            }
        }

        $self.blocks.extend(temp_blocks.into_iter().map(|(_, b)| b));

        // Replace the old block indexes with their new indexes
        for block in $self.blocks.iter_mut() {
            match block {
                Block::NormalBlock(block) => {
                    replace_block_index(&mut block.branch_block, &block_index_map);
                    replace_block_index(&mut block.default_block, &block_index_map);
                }
                Block::ExceptionBlock(block) => {
                    if let BranchEdge {
                        block_index: BlockIndex::Index(block_index),
                        ..
                    } = &mut block.exception_handler
                    {
                        if let Some(mapped_index) = block_index_map.get(block_index) {
                            *block_index = *mapped_index;
                        }
                    }

                    replace_block_index(&mut block.default_block, &block_index_map);

                    block.block_indexes.iter_mut().for_each(|i| {
                        if let Some(mapped_index) = block_index_map.get(i) {
                            *i = *mapped_index;
                        }
                    });
                }
            }
        }
    };
}

impl FinalizeCFG<Instruction> for ControlFlowGraph<Instruction> {
    fn finalize_cfg(&mut self) -> Result<(), crate::error::Error> {
        generate_cfg_finalize!(Instruction, self);

        Ok(())
    }
}

impl FinalizeCFG<ExtInstruction> for ControlFlowGraph<ExtInstruction> {
    fn finalize_cfg(&mut self) -> Result<(), crate::error::Error> {
        generate_cfg_finalize!(ExtInstruction, self);

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_finalize_cfg() {
        let mut cfg = ControlFlowGraph {
            blocks: vec![
                Block::NormalBlock(NormalBlock {
                    instructions: vec![Instruction::LoadConst(0), Instruction::StoreName(0)],
                    branch_block: BlockIndexInfo::Edge(BranchEdge {
                        reason: Opcode::SETUP_FINALLY,
                        block_index: BlockIndex::Index(1),
                    }),
                    default_block: BlockIndexInfo::Edge(BranchEdge {
                        reason: Opcode::SETUP_FINALLY,
                        block_index: BlockIndex::Index(2),
                    }),
                }),
                Block::NormalBlock(NormalBlock {
                    instructions: vec![
                        Instruction::PopTop(0),
                        Instruction::PopTop(0),
                        Instruction::PopTop(0),
                        Instruction::LoadName(1),
                        Instruction::LoadConst(2),
                        Instruction::CallFunction(1),
                        Instruction::PopTop(0),
                        Instruction::PopExcept(0),
                        Instruction::LoadConst(3),
                        Instruction::ReturnValue(0),
                    ],
                    branch_block: BlockIndexInfo::NoIndex,
                    default_block: BlockIndexInfo::NoIndex,
                }),
                Block::NormalBlock(NormalBlock {
                    instructions: vec![
                        Instruction::LoadName(0),
                        Instruction::LoadConst(1),
                        Instruction::CompareOp(2),
                    ],
                    branch_block: BlockIndexInfo::Edge(BranchEdge {
                        reason: Opcode::POP_JUMP_IF_FALSE,
                        block_index: BlockIndex::Index(3),
                    }),
                    default_block: BlockIndexInfo::Edge(BranchEdge {
                        reason: Opcode::POP_JUMP_IF_FALSE,
                        block_index: BlockIndex::Index(4),
                    }),
                }),
                Block::NormalBlock(NormalBlock {
                    instructions: vec![
                        Instruction::LoadName(1),
                        Instruction::LoadName(2),
                        Instruction::CallFunction(1),
                        Instruction::PopTop(0),
                        Instruction::PopBlock(0),
                        Instruction::LoadConst(3),
                        Instruction::ReturnValue(0),
                    ],
                    branch_block: BlockIndexInfo::NoIndex,
                    default_block: BlockIndexInfo::NoIndex,
                }),
                Block::NormalBlock(NormalBlock {
                    instructions: vec![
                        Instruction::LoadName(1),
                        Instruction::LoadName(0),
                        Instruction::CallFunction(1),
                        Instruction::PopTop(0),
                        Instruction::PopBlock(0),
                        Instruction::LoadConst(3),
                        Instruction::ReturnValue(0),
                    ],
                    branch_block: BlockIndexInfo::NoIndex,
                    default_block: BlockIndexInfo::NoIndex,
                }),
            ],
            start_index: BlockIndexInfo::Fallthrough(BlockIndex::Index(0)),
        };

        cfg.finalize_cfg().unwrap();

        println!("{}", cfg.make_dot_graph());

        insta::assert_debug_snapshot!(cfg);
    }
}
