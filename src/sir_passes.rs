use crate::{
    sir::{AuxVar, SIRControlFlowGraph, SIRExpression, SIRStatement},
    traits::{GenericSIRNode, SIRCFGPass},
};

struct RemoveSinglePhiNodes;

impl RemoveSinglePhiNodes {
    pub fn new() -> Self {
        RemoveSinglePhiNodes {}
    }
}

fn replace_var_in_expression<SIRNode: GenericSIRNode>(
    node: &mut SIRExpression<SIRNode>,
    og_var: &AuxVar,
    new_var: &AuxVar,
) {
    match node {
        SIRExpression::AuxVar(var) => {
            if var == og_var {
                *var = new_var.clone();
            }
        }
        SIRExpression::Call(call) => {
            for stack_input in call.stack_inputs.iter_mut() {
                replace_var_in_expression(stack_input, og_var, new_var);
            }
        }
        SIRExpression::Exception(exc) => {
            for stack_input in exc.stack_inputs.iter_mut() {
                replace_var_in_expression(stack_input, og_var, new_var);
            }
        }
        SIRExpression::PhiNode(values) => {
            for var in values {
                if var == og_var {
                    *var = new_var.clone();
                }
            }
        }
        SIRExpression::GeneratorStart => {}
    }
}

fn replace_var_in_statement<SIRNode: GenericSIRNode>(
    node: &mut SIRStatement<SIRNode>,
    og_var: &AuxVar,
    new_var: &AuxVar,
) {
    match node {
        SIRStatement::Assignment(var, value) => {
            if var == og_var {
                *var = new_var.clone();
            }

            replace_var_in_expression(value, og_var, new_var);
        }
        SIRStatement::DisregardCall(call) => {
            for stack_input in call.stack_inputs.iter_mut() {
                replace_var_in_expression(stack_input, og_var, new_var);
            }
        }
        SIRStatement::TupleAssignment(vars, value) => {
            for var in vars.iter_mut() {
                if var == og_var {
                    *var = new_var.clone();
                }
            }

            replace_var_in_expression(value, og_var, new_var);
        }
        SIRStatement::UseVar(var) => {
            if var == og_var {
                *var = new_var.clone();
            }
        }
    }
}

impl<SIRNode: GenericSIRNode> SIRCFGPass<SIRNode> for RemoveSinglePhiNodes {
    fn run_on(&self, cfg: &mut SIRControlFlowGraph<SIRNode>) {
        for block in cfg.blocks.iter_mut() {
            let mut phi_map: Vec<(AuxVar, AuxVar)> = vec![];

            if let Some(nodes) = block.get_nodes_mut() {
                // Only keep non single phi values
                nodes.0.retain(|e| match e {
                    SIRStatement::Assignment(phi_var, SIRExpression::PhiNode(phi_values)) => {
                        if phi_values.len() == 1 {
                            phi_map.push((phi_var.clone(), phi_values[0].clone()));
                            false
                        } else {
                            true
                        }
                    }
                    _ => true,
                });

                for node in nodes.iter_mut() {
                    for (phi_var, new_var) in phi_map.iter() {
                        replace_var_in_statement(node, phi_var, new_var);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        sir::{AuxVar, SIRBlock, SIRControlFlowGraph, SIRNormalBlock, SIRStatement},
        sir_passes::RemoveSinglePhiNodes,
        traits::SIRCFGPass,
        v311::opcodes::sir::SIRNode,
    };

    #[test]
    fn test_phi_simplification() {
        let mut cfg = SIRControlFlowGraph::<SIRNode> {
            blocks: vec![SIRBlock::<SIRNode>::NormalBlock(SIRNormalBlock {
                nodes: vec![
                    SIRStatement::<SIRNode>::Assignment(
                        AuxVar {
                            name: "phi_0".to_string(),
                        },
                        crate::sir::SIRExpression::PhiNode(vec![AuxVar {
                            name: "value".to_string(),
                        }]),
                    ),
                    SIRStatement::<SIRNode>::UseVar(AuxVar {
                        name: "phi_0".to_string(),
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

        RemoveSinglePhiNodes::new().run_on(&mut cfg);

        assert_eq!(
            cfg.blocks[0].get_nodes_ref(),
            Some(
                &vec![SIRStatement::UseVar(AuxVar {
                    name: "value".to_string(),
                })]
                .into()
            )
        )
    }
}
