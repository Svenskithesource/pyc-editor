use crate::{
    sir::{AuxVar, Call, SIRExpression},
    traits::SIRCFGPass,
    utils::replace_var_in_statement,
    v310::opcodes::{Opcode, sir::SIRNode},
};

pub struct RemoveStackOperations;

impl Default for RemoveStackOperations {
    fn default() -> Self {
        Self::new()
    }
}

impl RemoveStackOperations {
    pub fn new() -> Self {
        RemoveStackOperations {}
    }

    fn remove_pop_tops(&self, cfg: &mut crate::sir::SIRControlFlowGraph<SIRNode>) {
        for block in cfg.blocks.iter_mut() {
            if let Some(nodes) = block.get_nodes_mut() {
                nodes.0.retain_mut(|node| match node {
                    crate::sir::SIRStatement::DisregardCall(Call {
                        node:
                            SIRNode {
                                opcode: Opcode::POP_TOP,
                                ..
                            },
                        stack_inputs,
                    }) => {
                        assert!(stack_inputs.len() == 1);

                        // Don't keep the POP_TOP
                        false
                    }
                    _ => true,
                });
            }
        }
    }

    fn replace_dup(&self, cfg: &mut crate::sir::SIRControlFlowGraph<SIRNode>) {
        let mut items_left = true;

        while items_left {
            items_left = false;
            for block in cfg.blocks.iter_mut() {
                if let Some(nodes) = block.get_nodes_mut() {
                    let mut replacements: Vec<(Vec<AuxVar>, AuxVar)> = vec![];

                    nodes.0.retain(|node| match node {
                        crate::sir::SIRStatement::TupleAssignment(
                            outputs,
                            SIRExpression::Call(Call {
                                node:
                                    SIRNode {
                                        opcode: Opcode::DUP_TOP,
                                        ..
                                    },
                                stack_inputs,
                            }),
                        ) => {
                            assert!(outputs.len() == 2);
                            assert!(stack_inputs.len() == 1);

                            let input_var = match stack_inputs.first() {
                                Some(SIRExpression::AuxVar(input_var)) => input_var.clone(),
                                _ => unreachable!(),
                            };

                            if replacements.iter().any(|(saved_outputs, _): &(_, _)| {
                                saved_outputs.contains(&input_var)
                            }) {
                                // We will already replace this variable once, we will have to process this one in a next iteration
                                items_left = true;

                                true
                            } else {
                                replacements.push((outputs.clone(), input_var));

                                false
                            }
                        }
                        crate::sir::SIRStatement::TupleAssignment(
                            outputs,
                            SIRExpression::Call(Call {
                                node:
                                    SIRNode {
                                        opcode: Opcode::DUP_TOP_TWO,
                                        ..
                                    },
                                stack_inputs,
                            }),
                        ) => {
                            assert!(outputs.len() == 3);
                            assert!(stack_inputs.len() == 1);

                            let input_var = match stack_inputs.first() {
                                Some(SIRExpression::AuxVar(input_var)) => input_var.clone(),
                                _ => unreachable!(),
                            };

                            if replacements.iter().any(|(saved_outputs, _): &(_, _)| {
                                saved_outputs.contains(&input_var)
                            }) {
                                // We will already replace this variable once, we will have to process this one in a next iteration
                                items_left = true;

                                true
                            } else {
                                replacements.push((outputs.clone(), input_var));

                                false
                            }
                        }
                        _ => true,
                    });

                    // Apply replacements
                    for (outputs, input_var) in replacements {
                        nodes.0.iter_mut().for_each(|inner_node| {
                            for out in outputs.iter() {
                                replace_var_in_statement(inner_node, out, &input_var);
                            }
                        });
                    }
                }
            }
        }
    }

    fn replace_rot(&self, cfg: &mut crate::sir::SIRControlFlowGraph<SIRNode>) {
        let mut items_left = true;

        while items_left {
            items_left = false;
            for block in cfg.blocks.iter_mut() {
                if let Some(nodes) = block.get_nodes_mut() {
                    let mut replacements: Vec<(Vec<AuxVar>, Vec<AuxVar>)> = vec![];

                    nodes.0.retain(|node| match node {
                        crate::sir::SIRStatement::TupleAssignment(
                            outputs,
                            SIRExpression::Call(Call {
                                node:
                                    SIRNode {
                                        opcode: Opcode::ROT_TWO,
                                        ..
                                    },
                                stack_inputs,
                            }),
                        ) => {
                            assert!(outputs.len() == 2);
                            assert!(stack_inputs.len() == 2);

                            let input_vars = stack_inputs
                                .iter()
                                .map(|stack_input| match stack_input {
                                    SIRExpression::AuxVar(input_var) => input_var.clone(),
                                    _ => unreachable!(),
                                })
                                .collect::<Vec<_>>();

                            let mut input_vars = input_vars.clone();

                            if replacements.iter().any(|(saved_outputs, _): &(_, _)| {
                                // Check if this variable has already been used before
                                input_vars.iter().any(|e| saved_outputs.contains(e))
                            }) {
                                // We will already replace this variable once, we will have to process this one in a next iteration
                                items_left = true;

                                true
                            } else {
                                // Simulate SWAP behaviour
                                input_vars.swap(0, 1);

                                replacements.push((outputs.clone(), input_vars));

                                false
                            }
                        }
                        crate::sir::SIRStatement::TupleAssignment(
                            outputs,
                            SIRExpression::Call(Call {
                                node:
                                    SIRNode {
                                        opcode: Opcode::ROT_THREE,
                                        ..
                                    },
                                stack_inputs,
                            }),
                        ) => {
                            assert!(outputs.len() == 3);
                            assert!(stack_inputs.len() == 3);

                            let input_vars = stack_inputs
                                .iter()
                                .map(|stack_input| match stack_input {
                                    SIRExpression::AuxVar(input_var) => input_var.clone(),
                                    _ => unreachable!(),
                                })
                                .collect::<Vec<_>>();

                            let mut input_vars = input_vars.clone();

                            if replacements.iter().any(|(saved_outputs, _): &(_, _)| {
                                // Check if this variable has already been used before
                                input_vars.iter().any(|e| saved_outputs.contains(e))
                            }) {
                                // We will already replace this variable once, we will have to process this one in a next iteration
                                items_left = true;

                                true
                            } else {
                                // Simulate ROT_THREE behaviour
                                let top = input_vars.pop().unwrap();
                                input_vars.insert(0, top);

                                replacements.push((outputs.clone(), input_vars));

                                false
                            }
                        }
                        crate::sir::SIRStatement::TupleAssignment(
                            outputs,
                            SIRExpression::Call(Call {
                                node:
                                    SIRNode {
                                        opcode: Opcode::ROT_FOUR,
                                        ..
                                    },
                                stack_inputs,
                            }),
                        ) => {
                            assert!(outputs.len() == 4);
                            assert!(stack_inputs.len() == 4);

                            let input_vars = stack_inputs
                                .iter()
                                .map(|stack_input| match stack_input {
                                    SIRExpression::AuxVar(input_var) => input_var.clone(),
                                    _ => unreachable!(),
                                })
                                .collect::<Vec<_>>();

                            let mut input_vars = input_vars.clone();

                            if replacements.iter().any(|(saved_outputs, _): &(_, _)| {
                                // Check if this variable has already been used before
                                input_vars.iter().any(|e| saved_outputs.contains(e))
                            }) {
                                // We will already replace this variable once, we will have to process this one in a next iteration
                                items_left = true;

                                true
                            } else {
                                // Simulate ROT_FOUR behaviour
                                let top = input_vars.pop().unwrap();
                                input_vars.insert(0, top);

                                replacements.push((outputs.clone(), input_vars));

                                false
                            }
                        }
                        crate::sir::SIRStatement::TupleAssignment(
                            outputs,
                            SIRExpression::Call(Call {
                                node:
                                    SIRNode {
                                        opcode: Opcode::ROT_N,
                                        ..
                                    },
                                stack_inputs,
                            }),
                        ) => {
                            assert!(outputs.len() == 4);
                            assert!(stack_inputs.len() == 4);

                            let input_vars = stack_inputs
                                .iter()
                                .map(|stack_input| match stack_input {
                                    SIRExpression::AuxVar(input_var) => input_var.clone(),
                                    _ => unreachable!(),
                                })
                                .collect::<Vec<_>>();

                            let mut input_vars = input_vars.clone();

                            if replacements.iter().any(|(saved_outputs, _): &(_, _)| {
                                // Check if this variable has already been used before
                                input_vars.iter().any(|e| saved_outputs.contains(e))
                            }) {
                                // We will already replace this variable once, we will have to process this one in a next iteration
                                items_left = true;

                                true
                            } else {
                                // Simulate ROT_N behaviour
                                let top = input_vars.pop().unwrap();
                                input_vars.insert(0, top);

                                replacements.push((outputs.clone(), input_vars));

                                false
                            }
                        }
                        _ => true,
                    });

                    // Apply replacements
                    for (outputs, input_vars) in replacements {
                        nodes.0.iter_mut().for_each(|inner_node| {
                            for (out_var, in_var) in outputs.iter().zip(input_vars.iter()) {
                                replace_var_in_statement(inner_node, out_var, in_var);
                            }
                        });
                    }
                }
            }
        }
    }
}

impl SIRCFGPass<SIRNode> for RemoveStackOperations {
    fn run_on(&self, cfg: &mut crate::sir::SIRControlFlowGraph<SIRNode>) {
        self.remove_pop_tops(cfg);
        self.replace_dup(cfg);
        self.replace_rot(cfg);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        sir::{
            AuxVar, Call, SIRBlock, SIRControlFlowGraph, SIRExpression, SIRNormalBlock,
            SIRStatement,
        },
        traits::SIRCFGPass,
        v310::{
            opcodes::{Opcode, sir::SIRNode},
            sir_passes::RemoveStackOperations,
        },
    };

    #[test]
    fn test_replace_dup_top() {
        let mut cfg = SIRControlFlowGraph::<SIRNode> {
            blocks: vec![SIRBlock::<SIRNode>::NormalBlock(SIRNormalBlock {
                nodes: vec![
                    SIRStatement::<SIRNode>::Assignment(
                        AuxVar {
                            name: "value_0".to_string(),
                        },
                        crate::sir::SIRExpression::Call(Call {
                            node: SIRNode {
                                opcode: Opcode::LOAD_CONST,
                                oparg: 0,
                                input: vec![], // This is not true, but it doesn't matter for the test
                                output: vec![],
                                net_stack_delta: 1,
                            },
                            stack_inputs: vec![],
                        }),
                    ),
                    SIRStatement::<SIRNode>::TupleAssignment(
                        vec![
                            AuxVar {
                                name: "top_0".to_string(),
                            },
                            AuxVar {
                                name: "top_1".to_string(),
                            },
                        ],
                        crate::sir::SIRExpression::Call(Call {
                            node: SIRNode {
                                opcode: Opcode::DUP_TOP,
                                oparg: 0,
                                input: vec![], // This is not true, but it doesn't matter for the test
                                output: vec![],
                                net_stack_delta: 1,
                            },
                            stack_inputs: vec![SIRExpression::AuxVar(AuxVar {
                                name: "value_0".to_string(),
                            })],
                        }),
                    ),
                    SIRStatement::<SIRNode>::UseVar(AuxVar {
                        name: "top_0".to_string(),
                    }),
                    SIRStatement::<SIRNode>::UseVar(AuxVar {
                        name: "top_1".to_string(),
                    }),
                ]
                .into(),
                default_block: crate::sir::SIRBlockIndexInfo::NoIndex,
                branch_block: crate::sir::SIRBlockIndexInfo::NoIndex,
            })],
            start_index: crate::sir::SIRBlockIndexInfo::Fallthrough(crate::cfg::BlockIndex::Index(
                0,
            )),
        };

        RemoveStackOperations::new().run_on(&mut cfg);

        assert_eq!(
            cfg.blocks[0].get_nodes_ref(),
            Some(
                &vec![
                    SIRStatement::Assignment(
                        AuxVar {
                            name: "value_0".into()
                        },
                        SIRExpression::Call(Call {
                            node: SIRNode {
                                opcode: Opcode::LOAD_CONST,
                                oparg: 0,
                                input: vec![],
                                output: vec![],
                                net_stack_delta: 1,
                            },
                            stack_inputs: vec![],
                        },),
                    ),
                    SIRStatement::UseVar(AuxVar {
                        name: "value_0".into()
                    },),
                    SIRStatement::UseVar(AuxVar {
                        name: "value_0".into()
                    },),
                ]
                .into()
            )
        )
    }

    #[test]
    fn test_replace_dup_top_two() {
        let mut cfg = SIRControlFlowGraph::<SIRNode> {
            blocks: vec![SIRBlock::<SIRNode>::NormalBlock(SIRNormalBlock {
                nodes: vec![
                    SIRStatement::<SIRNode>::Assignment(
                        AuxVar {
                            name: "value_0".to_string(),
                        },
                        crate::sir::SIRExpression::Call(Call {
                            node: SIRNode {
                                opcode: Opcode::LOAD_CONST,
                                oparg: 0,
                                input: vec![], // This is not true, but it doesn't matter for the test
                                output: vec![],
                                net_stack_delta: 1,
                            },
                            stack_inputs: vec![],
                        }),
                    ),
                    SIRStatement::<SIRNode>::TupleAssignment(
                        vec![
                            AuxVar {
                                name: "top_0".to_string(),
                            },
                            AuxVar {
                                name: "top_1".to_string(),
                            },
                            AuxVar {
                                name: "top_2".to_string(),
                            },
                        ],
                        crate::sir::SIRExpression::Call(Call {
                            node: SIRNode {
                                opcode: Opcode::DUP_TOP_TWO,
                                oparg: 0,
                                input: vec![], // This is not true, but it doesn't matter for the test
                                output: vec![],
                                net_stack_delta: 2,
                            },
                            stack_inputs: vec![SIRExpression::AuxVar(AuxVar {
                                name: "value_0".to_string(),
                            })],
                        }),
                    ),
                    SIRStatement::<SIRNode>::UseVar(AuxVar {
                        name: "top_0".to_string(),
                    }),
                    SIRStatement::<SIRNode>::UseVar(AuxVar {
                        name: "top_1".to_string(),
                    }),
                    SIRStatement::<SIRNode>::UseVar(AuxVar {
                        name: "top_2".to_string(),
                    }),
                ]
                .into(),
                default_block: crate::sir::SIRBlockIndexInfo::NoIndex,
                branch_block: crate::sir::SIRBlockIndexInfo::NoIndex,
            })],
            start_index: crate::sir::SIRBlockIndexInfo::Fallthrough(crate::cfg::BlockIndex::Index(
                0,
            )),
        };

        RemoveStackOperations::new().run_on(&mut cfg);

        assert_eq!(
            cfg.blocks[0].get_nodes_ref(),
            Some(
                &vec![
                    SIRStatement::Assignment(
                        AuxVar {
                            name: "value_0".into()
                        },
                        SIRExpression::Call(Call {
                            node: SIRNode {
                                opcode: Opcode::LOAD_CONST,
                                oparg: 0,
                                input: vec![],
                                output: vec![],
                                net_stack_delta: 1,
                            },
                            stack_inputs: vec![],
                        },),
                    ),
                    SIRStatement::UseVar(AuxVar {
                        name: "value_0".into()
                    },),
                    SIRStatement::UseVar(AuxVar {
                        name: "value_0".into()
                    },),
                    SIRStatement::UseVar(AuxVar {
                        name: "value_0".into()
                    },),
                ]
                .into()
            )
        )
    }

    #[test]
    fn test_replace_rot_three() {
        let mut cfg = SIRControlFlowGraph::<SIRNode> {
            blocks: vec![SIRBlock::<SIRNode>::NormalBlock(SIRNormalBlock {
                nodes: vec![
                    SIRStatement::<SIRNode>::Assignment(
                        AuxVar {
                            name: "value_0".to_string(),
                        },
                        crate::sir::SIRExpression::Call(Call {
                            node: SIRNode {
                                opcode: Opcode::LOAD_CONST,
                                oparg: 0,
                                input: vec![], // This is not true, but it doesn't matter for the test
                                output: vec![],
                                net_stack_delta: 1,
                            },
                            stack_inputs: vec![],
                        }),
                    ),
                    SIRStatement::<SIRNode>::Assignment(
                        AuxVar {
                            name: "value_1".to_string(),
                        },
                        crate::sir::SIRExpression::Call(Call {
                            node: SIRNode {
                                opcode: Opcode::LOAD_CONST,
                                oparg: 0,
                                input: vec![], // This is not true, but it doesn't matter for the test
                                output: vec![],
                                net_stack_delta: 1,
                            },
                            stack_inputs: vec![],
                        }),
                    ),
                    SIRStatement::<SIRNode>::Assignment(
                        AuxVar {
                            name: "value_2".to_string(),
                        },
                        crate::sir::SIRExpression::Call(Call {
                            node: SIRNode {
                                opcode: Opcode::LOAD_CONST,
                                oparg: 0,
                                input: vec![], // This is not true, but it doesn't matter for the test
                                output: vec![],
                                net_stack_delta: 1,
                            },
                            stack_inputs: vec![],
                        }),
                    ),
                    SIRStatement::<SIRNode>::TupleAssignment(
                        vec![
                            AuxVar {
                                name: "first".to_string(),
                            },
                            AuxVar {
                                name: "second".to_string(),
                            },
                            AuxVar {
                                name: "third".to_string(),
                            },
                        ],
                        crate::sir::SIRExpression::Call(Call {
                            node: SIRNode {
                                opcode: Opcode::ROT_THREE,
                                oparg: 0,
                                input: vec![], // This is not true, but it doesn't matter for the test
                                output: vec![],
                                net_stack_delta: 0,
                            },
                            stack_inputs: vec![
                                SIRExpression::AuxVar(AuxVar {
                                    name: "value_0".to_string(),
                                }),
                                SIRExpression::AuxVar(AuxVar {
                                    name: "value_1".to_string(),
                                }),
                                SIRExpression::AuxVar(AuxVar {
                                    name: "value_2".to_string(),
                                }),
                            ],
                        }),
                    ),
                    SIRStatement::<SIRNode>::UseVar(AuxVar {
                        name: "first".to_string(),
                    }),
                    SIRStatement::<SIRNode>::UseVar(AuxVar {
                        name: "second".to_string(),
                    }),
                    SIRStatement::<SIRNode>::UseVar(AuxVar {
                        name: "third".to_string(),
                    }),
                ]
                .into(),
                default_block: crate::sir::SIRBlockIndexInfo::NoIndex,
                branch_block: crate::sir::SIRBlockIndexInfo::NoIndex,
            })],
            start_index: crate::sir::SIRBlockIndexInfo::Fallthrough(crate::cfg::BlockIndex::Index(
                0,
            )),
        };

        RemoveStackOperations::new().run_on(&mut cfg);

        assert_eq!(
            cfg.blocks[0].get_nodes_ref(),
            Some(
                &vec![
                    SIRStatement::Assignment(
                        AuxVar {
                            name: "value_0".into(),
                        },
                        SIRExpression::Call(Call {
                            node: SIRNode {
                                opcode: Opcode::LOAD_CONST,
                                oparg: 0,
                                input: vec![],
                                output: vec![],
                                net_stack_delta: 1,
                            },
                            stack_inputs: vec![],
                        }),
                    ),
                    SIRStatement::Assignment(
                        AuxVar {
                            name: "value_1".into(),
                        },
                        SIRExpression::Call(Call {
                            node: SIRNode {
                                opcode: Opcode::LOAD_CONST,
                                oparg: 0,
                                input: vec![],
                                output: vec![],
                                net_stack_delta: 1,
                            },
                            stack_inputs: vec![],
                        }),
                    ),
                    SIRStatement::Assignment(
                        AuxVar {
                            name: "value_2".into(),
                        },
                        SIRExpression::Call(Call {
                            node: SIRNode {
                                opcode: Opcode::LOAD_CONST,
                                oparg: 0,
                                input: vec![],
                                output: vec![],
                                net_stack_delta: 1,
                            },
                            stack_inputs: vec![],
                        }),
                    ),
                    SIRStatement::UseVar(AuxVar {
                        name: "value_2".into(),
                    }),
                    SIRStatement::UseVar(AuxVar {
                        name: "value_0".into(),
                    }),
                    SIRStatement::UseVar(AuxVar {
                        name: "value_1".into(),
                    }),
                ]
                .into()
            )
        )
    }
}
