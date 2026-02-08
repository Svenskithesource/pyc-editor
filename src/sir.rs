use std::{collections::HashMap, ops::Index};

use crate::{
    traits::{ExtInstructionAccess, GenericInstruction, GenericSIRNode, SIROwned},
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
}

#[derive(PartialEq, Debug, Clone)]
pub struct Call<SIRNode> {
    pub node: SIRNode,
    pub stack_inputs: Vec<SIRExpression<SIRNode>>, // Allow direct usage of a call as an input
}

#[derive(PartialEq, Debug, Clone)]
pub enum SIRStatement<SIRNode> {
    Assignment(AuxVar, Call<SIRNode>),
    TupleAssignment(Vec<AuxVar>, Call<SIRNode>),
    /// For when there is no output value (or it is not used)
    DisregardCall(Call<SIRNode>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct SIR<SIRNode>(pub Vec<SIRStatement<SIRNode>>);

/// Internal function that is used while converting an Ext CFG to SIR nodes.
/// This function is meant to process a single block.
fn to_ir<ExtInstruction, SIRNode>(
    instructions: &[ExtInstruction],
    stack: &mut Vec<SIRExpression<SIRNode>>,
    names: &mut HashMap<&'static str, u32>,
) -> SIR<SIRNode>
where
    ExtInstruction: GenericInstruction<OpargType = u32>,
    SIR<SIRNode>: SIROwned<SIRNode>,
    SIRNode: GenericSIRNode<ExtInstruction::Opcode>,
{
    let mut statements: Vec<SIRStatement<SIRNode>> = vec![];

    for instruction in instructions {
        // In a basic block there shouldn't be any jumps. (the last jump instruction is removed in the cfg)
        debug_assert!(!instruction.is_jump());

        let node = SIRNode::new(instruction.get_opcode(), instruction.get_raw_value(), false);

        let mut stack_inputs = vec![];

        for input in node.get_inputs() {
            for count in 0..input.count {
                dbg!(instruction, &stack, input.index, count);
                let index = (stack.len() - 1) - (input.index + count) as usize;
                stack_inputs.push((*stack.index(index)).clone());
                stack.remove(index);
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
            statements.push(SIRStatement::TupleAssignment(stack_outputs, call));
        } else if !stack_outputs.is_empty() {
            statements.push(SIRStatement::Assignment(
                stack_outputs.first().unwrap().clone(),
                call,
            ))
        } else {
            statements.push(SIRStatement::DisregardCall(call))
        }
    }

    SIR::new(statements)
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use crate::sir::to_ir;
    use crate::v311;
    use crate::v311::ext_instructions::ExtInstruction;
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
            to_ir::<ExtInstruction, SIRNode>(&ext_instructions, &mut vec![], &mut HashMap::new())
        );
    }
}
