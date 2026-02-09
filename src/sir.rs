use std::{collections::HashMap, ops::Index};

use crate::{
    cfg::{BlockIndex, BranchBlockIndex, BranchEdge, ControlFlowGraph},
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
    instruction: &ExtInstruction,
    jump: bool,
    stack: &mut Vec<SIRExpression<SIRNode>>,
    phi_map: &mut HashMap<i32, AuxVar>,
    names: &mut HashMap<&'static str, u32>,
) -> Result<Vec<SIRStatement<SIRNode>>, Error>
where
    ExtInstruction: GenericInstruction<OpargType = u32>,
    SIRNode: GenericSIRNode<Opcode = ExtInstruction::Opcode>,
{
    let node = SIRNode::new(instruction.get_opcode(), instruction.get_raw_value(), jump);

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

    let mut inserted_items = 0;
    for output in node.get_outputs() {
        for count in 0..output.count {
            let var = AuxVar {
                name: generate_var_name(output.name, names),
            };

            stack_outputs.push(var.clone());

            stack.insert(
                stack.len() - (output.index + count) as usize + inserted_items,
                SIRExpression::<SIRNode>::AuxVar(var),
            );
            inserted_items += 1;
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
        HashMap<i32, AuxVar>,
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
    let mut phi_map: HashMap<i32, AuxVar> = HashMap::new();

    for instruction in instructions {
        // In a basic block there shouldn't be any jumps. (the last jump instruction is removed in the cfg)
        debug_assert!(!instruction.is_jump());

        statements.extend_from_slice(&instruction_to_ir(
            instruction,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SIRBranchEdge<O>
where
    O: GenericOpcode,
{
    opcode: O,
    block_index: BlockIndex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SIRBranchBlockIndex<O>
where
    O: GenericOpcode,
{
    Edge(SIRBranchEdge<O>),
    /// For blocks without a target
    NoIndex,
}

#[derive(Debug, PartialEq)]
/// Represents a block in the control flow graph
pub struct SIRBlock<SIRNode>
where
    SIRNode: GenericSIRNode,
{
    pub nodes: SIR<SIRNode>,
    /// Index to block for conditional jump
    pub branch_block: SIRBranchBlockIndex<SIRNode::Opcode>,
    /// Index to default block (unconditional)
    pub default_block: BlockIndex,
}

#[derive(Debug, PartialEq)]
pub struct SIRControlFlowGraph<SIRNode>
where
    SIRNode: GenericSIRNode,
{
    pub blocks: Vec<SIRBlock<SIRNode>>,
    pub start_index: BlockIndex,
}

#[derive(Debug)]
pub enum Error {
    InvalidStackAccess,
    StackItemReused,
}

pub fn cfg_to_ir<ExtInstruction, SIRNode>(
    cfg: ControlFlowGraph<ExtInstruction>,
) -> Result<SIRControlFlowGraph<SIRNode>, Error>
where
    ExtInstruction: GenericInstruction<OpargType = u32>,
    SIRNode: GenericSIRNode<Opcode = ExtInstruction::Opcode>,
    SIR<SIRNode>: SIROwned<SIRNode>,
{
    let mut temp_blocks = vec![];
    let mut names = HashMap::new();

    // Create isolated IR blocks
    for block in &cfg.blocks {
        temp_blocks.push(bb_to_ir::<ExtInstruction, SIRNode>(
            &block.instructions,
            &mut names,
        )?);
    }

    // Fill the empty phi nodes with actual values

    // The branch index is used to calculate any effects the branch instruction might have
    let mut queue: Vec<(BlockIndex, Vec<SIRExpression<SIRNode>>)> = vec![(cfg.start_index, vec![])];

    while let Some((index, curr_stack)) = queue.pop() {
        match index {
            BlockIndex::Index(index) => {
                let (statements, phi_map, stack) = temp_blocks.get_mut(index).unwrap();

                for (item_index, phi_var) in phi_map.iter() {
                    if let Some(item) =
                        curr_stack.get((curr_stack.len() as i32 + item_index) as usize)
                    {
                        // Add the item to the phi node
                        for statement in statements.0.iter_mut() {
                            match (statement, item) {
                                (
                                    SIRStatement::Assignment(var, value),
                                    SIRExpression::AuxVar(item),
                                ) => {
                                    if var == phi_var {
                                        match value {
                                            SIRExpression::PhiNode(values) => {
                                                values.push(item.clone());
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    } else {
                    }
                }

                let new_stack = [curr_stack, stack.to_vec()].concat();

                queue.push((
                    cfg.blocks.index(index).default_block.clone(),
                    new_stack.clone(),
                ));

                match &cfg.blocks.index(index).branch_block {
                    BranchBlockIndex::Edge(branch) => {
                        queue.push((branch.block_index.clone(), new_stack));
                    }
                    _ => {}
                }

                println!("{}", statements);
            }
            _ => continue,
        }
    }

    let sir_cfg = SIRControlFlowGraph::<SIRNode> {
        start_index: BlockIndex::NoIndex,
        blocks: vec![],
    };

    Ok(sir_cfg)
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use crate::cfg::{ControlFlowGraph, create_cfg, simple_cfg_to_ext_cfg};
    use crate::sir::{bb_to_ir, cfg_to_ir};
    use crate::v311;
    use crate::v311::ext_instructions::{ExtInstruction, ExtInstructions};
    use crate::v311::instructions::Instruction;
    use crate::v311::opcodes::sir::SIRNode;

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

        cfg_to_ir::<ExtInstruction, SIRNode>(cfg);
    }
}
