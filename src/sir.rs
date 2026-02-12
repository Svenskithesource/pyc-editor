use std::{
    collections::{BTreeMap, HashMap},
    ops::Index,
};

use crate::{
    cfg::{BlockIndex, BlockIndexInfo, BranchEdge, ControlFlowGraph},
    traits::{ExtInstructionAccess, GenericInstruction, GenericOpcode, GenericSIRNode, SIROwned},
    utils::generate_var_name,
};

#[derive(PartialEq, Debug, Clone)]
pub struct AuxVar {
    pub name: String,
}

#[derive(PartialEq, Debug, Clone)]
pub struct StackItem {
    pub name: &'static str,
    pub count: u32,
    /// Index of the (first) item on the stack (0 = TOS)
    pub index: u32,
}

#[derive(PartialEq, Debug, Clone)]
pub enum SIRExpression<SIRNode> {
    Call(Call<SIRNode>),
    AuxVar(AuxVar),
    PhiNode(Vec<AuxVar>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct Call<SIRNode> {
    pub node: SIRNode,
    pub stack_inputs: Vec<SIRExpression<SIRNode>>, // Allow direct usage of a call as an input
}

#[derive(PartialEq, Debug, Clone)]
pub enum SIRStatement<SIRNode> {
    Assignment(AuxVar, SIRExpression<SIRNode>),
    TupleAssignment(Vec<AuxVar>, SIRExpression<SIRNode>),
    /// For when there is no output value (or it is not used)
    DisregardCall(Call<SIRNode>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct SIR<SIRNode>(pub Vec<SIRStatement<SIRNode>>);

fn instruction_to_ir<ExtInstruction, SIRNode>(
    opcode: ExtInstruction::Opcode,
    oparg: ExtInstruction::OpargType,
    jump: bool,
    stack: &mut Vec<SIRExpression<SIRNode>>,
    phi_map: &mut BTreeMap<i32, AuxVar>,
    names: &mut HashMap<&'static str, u32>,
) -> Result<Vec<SIRStatement<SIRNode>>, Error>
where
    ExtInstruction: GenericInstruction<OpargType = u32>,
    SIRNode: GenericSIRNode<Opcode = ExtInstruction::Opcode>,
{
    let node = SIRNode::new(opcode.clone(), oparg, jump);

    let mut stack_inputs = vec![];

    let mut statements = vec![];

    for input in node.get_inputs() {
        for count in 0..input.count {
            let index = (stack.len() as i32 - 1) - (input.index + count) as i32;

            if index < 0 {
                // Stack item from other basic block, create an empty phi node that we populate later.

                if phi_map.contains_key(&index) {
                    return Err(Error::StackItemReused);
                }

                let phi = SIRExpression::PhiNode(vec![]);

                let var = AuxVar {
                    name: generate_var_name("phi", names),
                };

                statements.push(SIRStatement::Assignment(var.clone(), phi));
                stack_inputs.push(SIRExpression::AuxVar(var.clone()));
                phi_map.insert(index, var);
            } else {
                stack_inputs.push((*stack.index(index as usize)).clone());
                stack.remove(index as usize);
            }
        }
    }

    let mut stack_outputs = vec![];

    for output in node.get_outputs() {
        for count in 0..output.count {
            let var = AuxVar {
                name: generate_var_name(output.name, names),
            };

            stack_outputs.push(var.clone());

            stack.insert(
                stack.len() - (output.index as i32 + count as i32) as usize,
                SIRExpression::<SIRNode>::AuxVar(var),
            );
        }
    }

    let call = Call::<SIRNode> {
        node: node,
        stack_inputs,
    };

    if stack_outputs.len() > 1 {
        statements.push(SIRStatement::TupleAssignment(
            stack_outputs,
            SIRExpression::Call(call),
        ));
    } else if !stack_outputs.is_empty() {
        statements.push(SIRStatement::Assignment(
            stack_outputs.first().unwrap().clone(),
            SIRExpression::Call(call),
        ))
    } else {
        statements.push(SIRStatement::DisregardCall(call))
    }

    Ok(statements)
}

/// Internal function that is used while converting an Ext CFG to SIR nodes.
/// This function is meant to process a single block.
fn bb_to_ir<ExtInstruction, SIRNode>(
    instructions: &[ExtInstruction],
    names: &mut HashMap<&'static str, u32>,
) -> Result<
    (
        SIR<SIRNode>,
        BTreeMap<i32, AuxVar>,
        Vec<SIRExpression<SIRNode>>,
    ),
    Error,
>
where
    ExtInstruction: GenericInstruction<OpargType = u32>,
    SIR<SIRNode>: SIROwned<SIRNode>,
    SIRNode: GenericSIRNode<Opcode = ExtInstruction::Opcode>,
{
    let mut statements: Vec<SIRStatement<SIRNode>> = vec![];

    // Every basic block starts with an empty stack.
    // When we try to access stack items below 0, we know it's accessing items from a different basic block.
    let mut stack: Vec<SIRExpression<SIRNode>> = vec![];

    // When we assign a phi node to a var we keep track of what stack index this phi node is representing
    let mut phi_map: BTreeMap<i32, AuxVar> = BTreeMap::new();

    for instruction in instructions {
        // In a basic block there shouldn't be any jumps. (the last jump instruction is removed in the cfg)
        debug_assert!(!instruction.is_jump());

        statements.extend_from_slice(&instruction_to_ir::<ExtInstruction, SIRNode>(
            instruction.get_opcode(),
            instruction.get_raw_value(),
            false,
            &mut stack,
            &mut phi_map,
            names,
        )?);
    }

    Ok((SIR::new(statements), phi_map, stack))
}

/// Used to represent the opcode that was used for this branch and the block index it's jumping to.
/// We do this so the value of the branch instruction cannot represent a wrong index.
/// This also shows which inputs and outputs the opcode uses.
#[derive(Debug, Clone, PartialEq)]
pub struct SIRBranchEdge<SIRNode>
where
    SIRNode: GenericSIRNode,
{
    pub opcode: SIRNode::Opcode,
    pub statements: Option<SIR<SIRNode>>,
    pub block_index: BlockIndex,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SIRBlockIndexInfo<SIRNode>
where
    SIRNode: GenericSIRNode,
{
    Edge(SIRBranchEdge<SIRNode>),
    /// For blocks that fallthrough with no opcode (cannot be generated by Python, used by internal algorithms)
    Fallthrough(BlockIndex),
    /// For blocks without a target
    NoIndex,
}

impl<SIRNode> SIRBlockIndexInfo<SIRNode>
where
    SIRNode: GenericSIRNode,
{
    pub fn get_block_index(&self) -> Option<&BlockIndex> {
        match self {
            SIRBlockIndexInfo::Edge(SIRBranchEdge { block_index, .. }) => Some(block_index),
            SIRBlockIndexInfo::Fallthrough(block_index) => Some(block_index),
            SIRBlockIndexInfo::NoIndex => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Represents a block in the control flow graph
pub struct SIRBlock<SIRNode>
where
    SIRNode: GenericSIRNode,
{
    pub nodes: SIR<SIRNode>,
    /// Index to block for conditional jump
    pub branch_block: SIRBlockIndexInfo<SIRNode>,
    /// Index to default block (unconditional)
    pub default_block: SIRBlockIndexInfo<SIRNode>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SIRControlFlowGraph<SIRNode>
where
    SIRNode: GenericSIRNode,
{
    pub blocks: Vec<SIRBlock<SIRNode>>,
    pub start_index: SIRBlockIndexInfo<SIRNode>,
}

#[derive(Clone, Debug)]
pub enum Error {
    InvalidStackAccess,
    StackItemReused,
    PhiNodeNotPopulated,
    NotAllBlocksProcessed,
}

pub fn cfg_to_ir<ExtInstruction, SIRNode>(
    cfg: &ControlFlowGraph<ExtInstruction>,
) -> Result<SIRControlFlowGraph<SIRNode>, Error>
where
    ExtInstruction: GenericInstruction<OpargType = u32>,
    SIRNode: GenericSIRNode<Opcode = ExtInstruction::Opcode>,
    SIR<SIRNode>: SIROwned<SIRNode>,
{
    // Prefill blocks with None that we replace later
    let mut blocks: Vec<
        Option<(
            SIR<SIRNode>,
            BlockIndexInfo<ExtInstruction::Opcode>,
            BlockIndexInfo<ExtInstruction::Opcode>,
            (Vec<SIRStatement<SIRNode>>, BTreeMap<i32, AuxVar>),
            (Vec<SIRStatement<SIRNode>>, BTreeMap<i32, AuxVar>),
        )>,
    > = Vec::with_capacity(cfg.blocks.len());
    blocks.resize_with(cfg.blocks.len(), || None);

    let mut temp_blocks = vec![];
    let mut names = HashMap::new();

    // Create isolated IR blocks
    for block in &cfg.blocks {
        temp_blocks.push(bb_to_ir::<ExtInstruction, SIRNode>(
            &block.instructions,
            &mut names,
        )?);
    }

    let empty_statements = vec![];
    let empty_phi_map = BTreeMap::new();

    // Fill the empty phi nodes with actual values
    let mut queue: Vec<(
        BlockIndexInfo<ExtInstruction::Opcode>,
        (Vec<SIRExpression<SIRNode>>, Vec<SIRExpression<SIRNode>>),
        Option<(bool, usize)>, // shows whether to use the branch or default statement on the specified index block
    )> = vec![(cfg.start_index.clone(), (vec![], vec![]), None)];

    while let Some((index, (mut curr_stack, mut branch_stack), edge_statements)) = queue.pop() {
        // The branch opcode is used to calculate any effects the branch instruction might have
        let index = match index {
            BlockIndexInfo::Fallthrough(BlockIndex::Index(index))
            | BlockIndexInfo::Edge(BranchEdge {
                block_index: BlockIndex::Index(index),
                ..
            }) => index,
            _ => continue,
        };

        let already_analysed = blocks.get(index).unwrap().is_some();

        // Get the branch
        let (branch_statements, branch_phi_map) =
            if let Some((is_branch, block_index)) = edge_statements {
                let block = blocks.get_mut(block_index).unwrap().as_mut().unwrap();
                if is_branch {
                    &mut block.3
                } else {
                    &mut block.4
                }
            } else {
                &mut (empty_statements.clone(), empty_phi_map.clone())
            };

        // For the branch statements only
        for (item_index, phi_var) in branch_phi_map.iter().rev() {
            let mut found = false;
            let item_index = (curr_stack.len() as i32 + item_index) as usize;

            if let Some(item) = curr_stack.get(item_index) {
                // Add the item to the phi node
                // Loop both the branch statements and the actual basic block statements
                for statement in branch_statements.iter_mut() {
                    match (statement, item) {
                        (
                            SIRStatement::Assignment(var, SIRExpression::PhiNode(values)),
                            SIRExpression::AuxVar(item),
                        ) => {
                            if var == phi_var {
                                values.push(item.clone());
                                found = true;
                            }
                        }
                        _ => {}
                    }
                }
            } else {
                return Err(Error::InvalidStackAccess);
            }

            if !found {
                return Err(Error::PhiNodeNotPopulated);
            }
        }

        // Add the branch stack after processing the branch statements
        curr_stack.extend_from_slice(&branch_stack);

        if already_analysed {
            // Already processed this block but we still processed the statements of the edge
            continue;
        }

        let (statements, phi_map, stack) = temp_blocks.get_mut(index).unwrap();

        // Loop over phi map in descending order (ex. -1 -> -2 -> -5 -> ...)
        for (item_index, phi_var) in phi_map.iter().rev() {
            let mut found = false;
            let item_index = (curr_stack.len() as i32 + item_index) as usize;

            if let Some(item) = curr_stack.get(item_index) {
                // Add the item to the phi node
                // Loop both the branch statements and the actual basic block statements
                for statement in statements.0.iter_mut() {
                    match (statement, item) {
                        (
                            SIRStatement::Assignment(var, SIRExpression::PhiNode(values)),
                            SIRExpression::AuxVar(item),
                        ) => {
                            if var == phi_var {
                                values.push(item.clone());
                                found = true;
                            }
                        }
                        _ => {}
                    }
                }
            } else {
                return Err(Error::InvalidStackAccess);
            }

            if !found {
                return Err(Error::PhiNodeNotPopulated);
            } else {
                // Remove item from the stack after it was used
                curr_stack.remove(item_index);
            }
        }

        let new_stack = [curr_stack.to_vec(), stack.to_vec()].concat();

        let mut default_stack = vec![];
        let mut default_phi_map: BTreeMap<i32, AuxVar> = BTreeMap::new();

        let default_statements = match &cfg.blocks.index(index).default_block {
            BlockIndexInfo::Edge(BranchEdge {
                opcode: branch_opcode,
                ..
            }) => {
                instruction_to_ir::<ExtInstruction, SIRNode>(
                    branch_opcode.clone(),
                    0, // The oparg doesn't matter for the stack effect in the case of a branch opcode (oparg is the jump target)
                    false, // Don't take the jump for the default block
                    &mut default_stack,
                    &mut default_phi_map,
                    &mut names,
                )?
            }
            _ => vec![],
        };

        queue.push((
            cfg.blocks.index(index).default_block.clone(),
            (new_stack.clone(), default_stack),
            Some((false, index)),
        ));

        let mut branch_stack = vec![];
        let mut branch_phi_map: BTreeMap<i32, AuxVar> = BTreeMap::new();

        let branch_statements = match &cfg.blocks.index(index).branch_block {
            BlockIndexInfo::Edge(BranchEdge {
                opcode: branch_opcode,
                ..
            }) => {
                instruction_to_ir::<ExtInstruction, SIRNode>(
                    branch_opcode.clone(),
                    0, // The oparg doesn't matter for the stack effect in the case of a branch opcode (oparg is the jump target)
                    false, // Don't take the jump for the default block
                    &mut branch_stack,
                    &mut branch_phi_map,
                    &mut names,
                )?
            }
            _ => vec![],
        };

        queue.push((
            cfg.blocks.index(index).branch_block.clone(),
            (new_stack, branch_stack),
            Some((true, index)),
        ));

        // Mutable so the phi nodes can be populated later on
        blocks[index] = Some((
            statements.clone(),
            cfg.blocks.index(index).branch_block.clone(),
            cfg.blocks.index(index).default_block.clone(),
            (branch_statements, branch_phi_map),
            (default_statements, default_phi_map),
        ));
    }

    let blocks: Vec<_> = blocks
        .into_iter()
        .map(|v| {
            v.ok_or(Error::NotAllBlocksProcessed).map(
                |(
                    statements,
                    branch_block,
                    default_block,
                    (branch_statements, _),
                    (default_statements, _),
                )| {
                    SIRBlock {
                        nodes: statements.clone(),
                        branch_block: branch_block.into_sir(if branch_statements.is_empty() {
                            None
                        } else {
                            Some(branch_statements)
                        }),
                        default_block: default_block.into_sir(if default_statements.is_empty() {
                            None
                        } else {
                            Some(default_statements)
                        }),
                    }
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let sir_cfg = SIRControlFlowGraph::<SIRNode> {
        start_index: SIRBlockIndexInfo::Fallthrough(BlockIndex::Index(0)),
        blocks: blocks,
    };

    Ok(sir_cfg)
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use petgraph::dot::{Config, Dot};
    use petgraph::graph::NodeIndex;

    use crate::cfg::{BlockIndex, ControlFlowGraph, create_cfg, simple_cfg_to_ext_cfg};
    use crate::sir::{
        SIR, SIRBlock, SIRBlockIndexInfo, SIRBranchEdge, SIRControlFlowGraph, bb_to_ir, cfg_to_ir,
    };
    use crate::traits::{GenericSIRNode, SIROwned};
    use crate::v311::ext_instructions::{ExtInstruction, ExtInstructions};
    use crate::v311::instructions::Instruction;
    use crate::v311::opcodes::sir::SIRNode;
    use crate::{CodeObject, v311};

    #[test]
    fn test_simple_ext_to_sir() {
        let ext_instructions = v311::instructions::Instructions::new(vec![
            Instruction::Resume(0),
            Instruction::PushNull(0),
            Instruction::LoadName(0),
            Instruction::LoadConst(0),
            Instruction::Precall(1),
            Instruction::Cache(0),
            Instruction::Call(1),
            Instruction::Cache(0),
            Instruction::Cache(0),
            Instruction::Cache(0),
            Instruction::Cache(0),
            Instruction::PopTop(0),
            Instruction::LoadConst(1),
            Instruction::StoreName(1),
            Instruction::PushNull(0),
            Instruction::LoadName(0),
            Instruction::LoadConst(2),
            Instruction::LoadName(1),
            Instruction::FormatValue(2),
            Instruction::BuildString(2),
            Instruction::Precall(1),
            Instruction::Cache(0),
            Instruction::Call(1),
            Instruction::Cache(0),
            Instruction::Cache(0),
            Instruction::Cache(0),
            Instruction::Cache(0),
            Instruction::PopTop(0),
            Instruction::LoadConst(3),
            Instruction::ReturnValue(0),
        ])
        .to_resolved()
        .unwrap();

        println!(
            "{}",
            bb_to_ir::<ExtInstruction, SIRNode>(&ext_instructions, &mut HashMap::new())
                .unwrap()
                .0
        );
    }

    #[test]
    fn test_simple_cfg_to_sir() {
        let instructions = v311::instructions::Instructions::new(vec![
            Instruction::LoadConst(0), // Extra stack value that is used in the other branches
            Instruction::ExtendedArg(1),
            Instruction::LoadConst(0),
            Instruction::LoadConst(1),
            Instruction::CompareOp(0),
            Instruction::PopJumpForwardIfTrue(3),
            Instruction::PopTop(0),
            Instruction::LoadConst(2),
            Instruction::ReturnValue(0),
            Instruction::PopTop(0),
            Instruction::LoadConst(3),
            Instruction::ReturnValue(0),
        ]);

        let cfg = create_cfg(instructions.to_vec());

        let cfg: ControlFlowGraph<ExtInstruction> =
            simple_cfg_to_ext_cfg::<Instruction, ExtInstruction, ExtInstructions>(&cfg).unwrap();

        make_dot_graph(&cfg_to_ir::<ExtInstruction, SIRNode>(&cfg).unwrap());
    }

    #[test]
    fn test_complex_cfg_to_sir() {
        let program = crate::load_code(&b"\xe3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\xf3\\\x00\x00\x00\x97\x00\x02\x00e\x00d\x00\xa6\x01\x00\x00\xab\x01\x00\x00\x00\x00\x00\x00\x00\x00D\x00]\x1fZ\x01e\x01d\x01k\x02\x00\x00\x00\x00r\x0c\x02\x00e\x02d\x02\xa6\x01\x00\x00\xab\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x8c\x14\x02\x00e\x02d\x03\xa6\x01\x00\x00\xab\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x8c d\x04S\x00)\x05\xe9\n\x00\x00\x00\xe9\t\x00\x00\x00\xda\x03yay\xda\x03nayN)\x03\xda\x05range\xda\x01x\xda\x05print\xa9\x00\xf3\x00\x00\x00\x00z\x08<string>\xfa\x08<module>r\x0b\x00\x00\x00\x01\x00\x00\x00sM\x00\x00\x00\xf0\x03\x01\x01\x01\xe0\t\x0e\x88\x15\x88r\x89\x19\x8c\x19\xf0\x00\x04\x01\x15\xf0\x00\x04\x01\x15\x80A\xd8\x07\x08\x88A\x82v\x80v\xd8\x08\r\x88\x05\x88e\x89\x0c\x8c\x0c\x88\x0c\x88\x0c\xe0\x08\r\x88\x05\x88e\x89\x0c\x8c\x0c\x88\x0c\x88\x0c\xf0\t\x04\x01\x15\xf0\x00\x04\x01\x15r\n\x00\x00\x00"[..], (3, 11).into()).unwrap();

        let instructions = match program {
            CodeObject::V311(code) => code.code,
            _ => unreachable!(),
        };

        let cfg = create_cfg(instructions.to_vec());

        let cfg =
            simple_cfg_to_ext_cfg::<Instruction, ExtInstruction, ExtInstructions>(&cfg).unwrap();

        let ir_cfg = cfg_to_ir::<ExtInstruction, SIRNode>(&cfg).unwrap();

        make_dot_graph(&ir_cfg);
    }

    fn add_block<'a, SIRNode: GenericSIRNode>(
        graph: &mut petgraph::Graph<String, String>,
        blocks: &'a [SIRBlock<SIRNode>],
        block_index: Option<&'a BlockIndex>,
        block_map: &mut HashMap<Option<&'a BlockIndex>, NodeIndex>,
    ) -> Option<NodeIndex>
    where
        SIR<SIRNode>: SIROwned<SIRNode>,
    {
        let block = match block_index {
            Some(BlockIndex::Index(index)) => blocks.get(*index).unwrap(),
            _ => return None,
        };

        let text = format!("{}", block.nodes);

        let index = if block_map.contains_key(&block_index) {
            block_map[&block_index]
        } else {
            let index = graph.add_node(text);
            block_map.insert(block_index.clone(), index);

            index
        };

        let (branch_index, branch_statements, opcode) = match &block.branch_block {
            SIRBlockIndexInfo::Edge(SIRBranchEdge {
                block_index: branch_index,
                statements,
                opcode,
            }) => (Some(branch_index), Some(statements), Some(opcode.clone())),
            SIRBlockIndexInfo::Fallthrough(branch_index) => (Some(branch_index), None, None),
            _ => (None, None, None),
        };

        let branch_index = if block_map.contains_key(&branch_index) {
            Some(block_map[&branch_index])
        } else {
            let index = add_block(graph, blocks, branch_index, block_map);

            let index = if let Some(index) = index {
                block_map.insert(branch_index.clone(), index);
                Some(index)
            } else {
                match branch_index {
                    Some(BlockIndex::InvalidIndex(invalid_index)) => {
                        Some(graph.add_node(format!("invalid jump to index {}", invalid_index)))
                    }
                    Some(BlockIndex::Index(_)) => unreachable!(),
                    None => None,
                }
            };

            index
        };

        let (default_index, default_statements, opcode) = match &block.default_block {
            SIRBlockIndexInfo::Edge(SIRBranchEdge {
                block_index: default_index,
                statements,
                opcode,
            }) => (Some(default_index), Some(statements), Some(opcode.clone())),
            SIRBlockIndexInfo::Fallthrough(branch_index) => (Some(branch_index), None, None),
            _ => (None, None, None),
        };

        let default_index = if block_map.contains_key(&default_index) {
            Some(block_map[&default_index])
        } else {
            let index = add_block(graph, blocks, default_index, block_map);

            let index = if let Some(index) = index {
                block_map.insert(default_index, index);
                Some(index)
            } else {
                match default_index {
                    Some(BlockIndex::InvalidIndex(invalid_index)) => {
                        Some(graph.add_node(format!("invalid jump to index {}", invalid_index)))
                    }
                    Some(BlockIndex::Index(_)) => unreachable!(),
                    None => None,
                }
            };

            index
        };

        if let Some(to_index) = branch_index {
            let text = if let Some(statements) = branch_statements {
                format!("{}", statements.as_ref().unwrap())
            } else {
                "".to_owned()
            };
            graph.add_edge(index, to_index, text);
        }

        if let Some(to_index) = default_index {
            let text = if let Some(statements) = default_statements {
                format!("fallthrough\n{}", statements.as_ref().unwrap())
            } else {
                format!("fallthrough")
            };

            graph.add_edge(index, to_index, text);
        }

        Some(index)
    }

    fn make_dot_graph<SIRNode>(cfg: &SIRControlFlowGraph<SIRNode>)
    where
        SIRNode: GenericSIRNode,
        SIR<SIRNode>: SIROwned<SIRNode>,
    {
        let mut graph = petgraph::Graph::<String, String>::new();

        add_block(
            &mut graph,
            &cfg.blocks,
            cfg.start_index.get_block_index(),
            &mut HashMap::new(),
        );

        println!(
            "{:#?}",
            Dot::with_attr_getters(
                &graph,
                &[Config::NodeNoLabel, Config::EdgeNoLabel],
                &|_, e| {
                    let color = if e.weight().contains("fallthrough") {
                        "green"
                    } else {
                        "red"
                    };

                    format!(r#"label = "{}", color = {}"#, e.weight(), color)
                },
                &|_, (_, s)| format!(r#"label = "{}""#, s),
            )
        );
    }
}
