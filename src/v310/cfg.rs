use crate::{
    cfg::{
        Block, BlockIndex, BlockIndexInfo, BranchEdge, ControlFlowGraph, ExceptionBlock,
        NormalBlock,
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

        let mut visited_blocks: Vec<usize> = Vec::with_capacity($self.blocks.len());

        /// The index is an index into `temp_exception_blocks`
        #[derive(Debug, Clone)]
        struct ExceptionBlockIndex(usize);

        // (block index, exception_block_indexes, real_block)
        // real_block is used to show that in the exception handler itself, this does not behave like a normal block item
        let mut visit_queue: Vec<(usize, Vec<(ExceptionBlockIndex, bool)>)> =
            vec![(start_block_index, vec![], )];

        let mut current_block_index = $self.blocks.len();

        // Visit blocks by exploring the branches
        while let Some((block_index, mut exception_block_indexes)) = visit_queue.pop() {
            if visited_blocks.contains(&block_index) {
                continue;
            }

            let block = &$self.blocks.get(block_index).unwrap().clone();
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

            let instructions = block.get_instructions_slice().unwrap();

            // We view other instructions that pop from the block stack as pop_blocks
            let pop_block_indexes = instructions
                .iter()
                .enumerate()
                .filter_map(|(index, instruction)| {
                    matches!(instruction, $instruction::PopBlock(_) | $instruction::PopExcept(_) | $instruction::EndAsyncFor(_)).then_some(index)
                })
                .collect::<Vec<_>>();

            let last_block_index = if !pop_block_indexes.is_empty() {
                // We will split the instructions up, based on the POP_BLOCK indexes.
                // Then make them fallthrough to each other.

                let mut current_instruction_index = 0;

                for (i, pop_block_index) in pop_block_indexes.iter().enumerate() {
                    let new_instructions =
                        instructions[current_instruction_index..=*pop_block_index].to_vec();

                    if *pop_block_index != instructions.len() - 1 {
                        // Not last block
                        let new_block = Block::NormalBlock(NormalBlock {
                            instructions: new_instructions,
                            branch_block: BlockIndexInfo::NoIndex,
                            default_block: BlockIndexInfo::Fallthrough(BlockIndex::Index(
                                current_block_index + if i != 0 { 1 } else { 0 },
                            )),
                        });

                        let temp_block_index = if i == 0 {
                            *$self.blocks.get_mut(block_index).unwrap() = new_block;

                            block_index
                        } else {
                            $self.blocks.push(new_block);

                            current_block_index += 1;

                            current_block_index - 1
                        };

                        // Populate block index lists of current scope
                        for exception_block_index in &exception_block_indexes {
                            let block = $self.blocks.get_mut(exception_block_index.0.0).unwrap();

                            assert!(matches!(block, Block::ExceptionBlock(_)));
                            match block {
                                Block::ExceptionBlock(block) => {
                                    if exception_block_index.1 {
                                        block.block_indexes.push(temp_block_index)
                                    }
                                }
                                _ => unreachable!(),
                            };
                        }

                        visited_blocks.push(temp_block_index);
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

                let new_block = Block::NormalBlock(NormalBlock {
                    instructions: new_instructions,
                    branch_block: original_branch_block.clone(),
                    default_block: original_default_block,
                });

                if let Some(pop_block_index) = pop_block_indexes.last() && *pop_block_index == instructions.len() - 1 && pop_block_indexes.len() == 1 {
                    // If there is only one pop block, at the end of the basic block, then we just replace the block inplace.
                    *$self.blocks.get_mut(block_index).unwrap() = new_block;

                    block_index
                } else {
                    $self.blocks.push(new_block);

                    current_block_index += 1;

                    current_block_index - 1
                }
            } else {
                block_index
            };

            // Populate block index lists of current scope
            for exception_block_index in &exception_block_indexes {
                let block = $self.blocks.get_mut(exception_block_index.0.0).unwrap();

                assert!(matches!(block, Block::ExceptionBlock(_)));
                match block {
                    Block::ExceptionBlock(block) => {
                        if exception_block_index.1 {
                            block.block_indexes.push(last_block_index)
                        }
                    }
                    _ => unreachable!(),
                };
            }

            if let Some(pop_block_index) = pop_block_indexes.last() && *pop_block_index == instructions.len() - 1 {
                // We still have to pop one, since we didn't do it above
                match exception_block_indexes.pop() {
                        None => return Err(crate::Error::InvalidBlockStackUsage),
                        _ => {}
                };
            }

            // Check if we're starting an exception block
            match original_branch_block {
                BlockIndexInfo::Edge(BranchEdge {
                    reason: reason @ (Opcode::SETUP_FINALLY | Opcode::SETUP_WITH | Opcode::SETUP_ASYNC_WITH),
                    block_index: branch_block_index,
                }) => {
                    // Add an ExceptionBlock and "reroute" jumps to the target block to the exception block instead (this happens later)

                    $self.blocks.push(
                        Block::ExceptionBlock(ExceptionBlock {
                            block_indexes: vec![],
                            exception_handler: BranchEdge {
                                reason,
                                block_index: branch_block_index,
                            },
                            default_block: block.get_default_block(),
                        }
                    ));

                    current_block_index += 1;

                    let exception_block_index = current_block_index - 1;

                    match $self.blocks.get_mut(last_block_index).unwrap() {
                        Block::NormalBlock(block) => {
                            block.branch_block = BlockIndexInfo::NoIndex;

                            block.default_block = BlockIndexInfo::Fallthrough(BlockIndex::Index(
                                // Points to the exception block we just pushed
                                exception_block_index,
                            ));
                        }
                        Block::ExceptionBlock(_) => unreachable!(),
                    }

                    if let Some(index) = original_branch_index {
                        exception_block_indexes.push((ExceptionBlockIndex(exception_block_index), false));

                        // The branch edge needs to be visited with the bool set to false
                        visit_queue.push((index, exception_block_indexes.clone()));

                        exception_block_indexes.pop();

                        // The default edge with the bool set to true
                        exception_block_indexes.push((ExceptionBlockIndex(exception_block_index), true));
                    } else {
                        unreachable!();
                    }

                }
                _ => {
                    if let Some(index) = original_branch_index {
                        visit_queue.push((index, exception_block_indexes.clone()))
                    }
                }
            }

            if let Some(index) = original_default_index {
                visit_queue.push((index, exception_block_indexes.clone()))
            }

            if original_branch_index.is_none() && original_default_index.is_none() && !exception_block_indexes.is_empty() {
                if exception_block_indexes.iter().any(|(_, is_real)| *is_real) {
                    if instructions.is_empty()  {
                        return Err(crate::Error::NonEmptyBlockStack);
                    }

                    if let Some(instruction) = instructions.last() && !matches!(instruction, $instruction::Reraise(_) | $instruction::RaiseVarargs(_)) {
                        return Err(crate::Error::NonEmptyBlockStack);
                    }
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

    #[test]
    fn test_finalize_cfg_with_pop_block_last() {
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
                    ],
                    branch_block: BlockIndexInfo::NoIndex,
                    default_block: BlockIndexInfo::Fallthrough(BlockIndex::Index(5)),
                }),
                Block::NormalBlock(NormalBlock {
                    instructions: vec![
                        Instruction::LoadName(1),
                        Instruction::LoadName(0),
                        Instruction::CallFunction(1),
                        Instruction::PopTop(0),
                        Instruction::PopBlock(0),
                    ],
                    branch_block: BlockIndexInfo::NoIndex,
                    default_block: BlockIndexInfo::Fallthrough(BlockIndex::Index(5)),
                }),
                Block::NormalBlock(NormalBlock {
                    instructions: vec![Instruction::LoadConst(3), Instruction::ReturnValue(0)],
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
