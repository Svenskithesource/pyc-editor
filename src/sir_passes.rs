use crate::sir::{SIR, SIRBranchEdge};
use crate::{
    sir::{SIRBlock, SIRBlockIndexInfo, SIRControlFlowGraph, SIRExpression, SIRStatement},
    traits::{GenericSIRNode, SIRCFGPass},
    utils::replace_var_in_statement,
};

pub struct RemoveSinglePhiNodes;

impl RemoveSinglePhiNodes {
    pub fn new() -> Self {
        RemoveSinglePhiNodes {}
    }
}

fn remove_single_nodes<SIRNode: GenericSIRNode>(nodes: &mut SIR<SIRNode>) {
    let mut phi_map = vec![];

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

impl<SIRNode: GenericSIRNode> SIRCFGPass<SIRNode> for RemoveSinglePhiNodes {
    fn run_on(&self, cfg: &mut SIRControlFlowGraph<SIRNode>) {
        for block in cfg.blocks.iter_mut() {
            // Apply to main list of nodes

            if let Some(nodes) = block.get_nodes_mut() {
                remove_single_nodes(nodes);
            }

            // Apply to edge statements
            match block {
                SIRBlock::NormalBlock(normal_block) => {
                    if let SIRBlockIndexInfo::Edge(SIRBranchEdge {
                            statements: Some(nodes),
                            ..
                        }) = &mut normal_block.branch_block { remove_single_nodes(nodes) }

                    if let SIRBlockIndexInfo::Edge(SIRBranchEdge {
                            statements: Some(nodes),
                            ..
                        }) = &mut normal_block.default_block { remove_single_nodes(nodes) }
                }
                SIRBlock::ExceptionBlock(exception_block) => {
                    if let SIRBranchEdge {
                            statements: Some(nodes),
                            ..
                        } = &mut exception_block.exception_handler { remove_single_nodes(nodes) }

                    if let SIRBlockIndexInfo::Edge(SIRBranchEdge {
                            statements: Some(nodes),
                            ..
                        }) = &mut exception_block.default_block { remove_single_nodes(nodes) }
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
