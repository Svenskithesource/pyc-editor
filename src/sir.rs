use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    ops::Index,
};

use crate::{
    cfg::{BlockIndex, BlockIndexInfo, BranchEdge, ControlFlowGraph},
    traits::{
        BranchReasonTrait, GenericInstruction, GenericSIRException, GenericSIRNode, SIROwned,
    },
    utils::{InfiniteVec, generate_var_name},
};

#[cfg(feature = "dot")]
use petgraph::{
    dot::{Config, Dot},
    graph::NodeIndex,
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
pub enum SIRExpression<SIRNode, SIRException> {
    Call(Call<SIRNode, SIRException>),
    Exception(ExceptionCall<SIRNode, SIRException>),
    AuxVar(AuxVar),
    PhiNode(Vec<AuxVar>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct Call<SIRNode, SIRException> {
    pub node: SIRNode,
    pub stack_inputs: Vec<SIRExpression<SIRNode, SIRException>>, // Allow direct usage of a call as an input
}

#[derive(PartialEq, Debug, Clone)]
pub struct ExceptionCall<SIRNode, SIRException> {
    pub exception: SIRException,
    pub stack_inputs: Vec<SIRExpression<SIRNode, SIRException>>, // Allow direct usage of a call as an input
}

#[derive(PartialEq, Debug, Clone)]
pub enum SIRStatement<SIRNode, SIRException> {
    Assignment(AuxVar, SIRExpression<SIRNode, SIRException>),
    TupleAssignment(Vec<AuxVar>, SIRExpression<SIRNode, SIRException>),
    /// For when there is no output value (or it is not used)
    DisregardCall(Call<SIRNode, SIRException>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct SIR<SIRNode, SIRException>(pub Vec<SIRStatement<SIRNode, SIRException>>);

fn process_phi_item<SIRNode, SIRException>(
    phi_var: &AuxVar,
    stack_item: &SIRExpression<SIRNode, SIRException>,
    statements: &mut Vec<SIRStatement<SIRNode, SIRException>>,
) -> bool
where
    SIRNode: GenericSIRNode,
    SIRException: GenericSIRException,
{
    let mut found = false;

    // Add the item to the phi node
    // Loop both the branch statements and the actual basic block statements
    for statement in statements.iter_mut() {
        if let (
            SIRStatement::Assignment(var, SIRExpression::PhiNode(values)),
            SIRExpression::AuxVar(item),
        ) = (statement, stack_item)
            && var == phi_var
        {
            values.push(item.clone());
            found = true;
        }
    }

    found
}

fn fill_phi_nodes<SIRNode, SIRException>(
    curr_stack: &mut Vec<SIRExpression<SIRNode, SIRException>>,
    statements: &mut Vec<SIRStatement<SIRNode, SIRException>>,
    phi_map: &[(isize, AuxVar)],
) -> Result<Vec<isize>, Error>
where
    SIRNode: GenericSIRNode,
    SIRException: GenericSIRException,
{
    // Indexes of items we have to delete in the curr stack (needs to be done after fully processing the new stack)
    let mut items_to_delete: Vec<isize> = vec![];

    dbg!(phi_map);

    for (stack_index, aux_var) in phi_map {
        let found;

        let item_index = (curr_stack.len() as i32 + (*stack_index as i32)) as usize;

        dbg!(item_index, &curr_stack);

        if let Some(item) = curr_stack.get(item_index) {
            found = process_phi_item(&aux_var, item, statements);
        } else {
            return Err(Error::InvalidStackAccess);
        }

        if !found {
            return Err(Error::PhiNodeNotPopulated);
        } else {
            items_to_delete.push(*stack_index);
        }
    }

    items_to_delete.sort();

    dbg!(&items_to_delete, &curr_stack);

    Ok(items_to_delete)
}

fn process_stack_effects<SIRNode, SIRException>(
    inputs: &[StackItem],
    outputs: &[StackItem],
    stack: &mut InfiniteVec<SIRExpression<SIRNode, SIRException>>,
    phi_map: &mut Vec<(isize, AuxVar)>,
    names: &mut HashMap<&'static str, u32>,
) -> Result<
    (
        Vec<SIRExpression<SIRNode, SIRException>>,
        Vec<AuxVar>,
        Vec<SIRStatement<SIRNode, SIRException>>,
    ),
    Error,
>
where
    SIRNode: GenericSIRNode,
    SIRException: GenericSIRException,
{
    let mut stack_inputs = vec![];

    let mut statements = vec![];

    // We don't want to offset by items in the current input list
    // So we store those separately here first, and then extend later
    let mut temp_phi_map = vec![];

    // This var shows what index the lowest stack input is at.
    // We need it to determine where to start pushing our outputs.
    let mut lowest_stack_input = stack.positive_len() as isize;

    let original_len = stack.positive_len() as i32;

    // Items we have to delete after using them as input
    let mut to_delete = vec![];

    for input in inputs {
        for count in 0..input.count {
            let index = ((original_len as i32 - 1) - (input.index + count) as i32) as isize;

            // If we pushed a value that replaced a phi item we need to offset that
            let mut og_phi_count = 0;

            loop {
                let mut finished = true;
                for (i, _) in phi_map.iter() {
                    if *i == index - og_phi_count {
                        og_phi_count += 1;

                        finished = false;

                        break;
                    }
                }

                if finished {
                    break;
                }
            }

            let index = if index < 0 {
                index - og_phi_count
            } else {
                index
            };

            if index < lowest_stack_input {
                lowest_stack_input = index;
            }

            dbg!(index);

            if index < 0 && (stack.get(index).is_none() || stack.get(index).unwrap().is_none()) {
                // Stack item from other basic block, create an empty phi node that we populate later.

                let phi_count = temp_phi_map.iter().filter(|(i, _)| *i <= index).count();

                dbg!(og_phi_count, phi_count, index, original_len, &temp_phi_map,);

                for (i, _) in temp_phi_map.iter_mut() {
                    // Move items to the left by the amount necessary
                    *i -= (og_phi_count as usize).saturating_sub(phi_count) as isize;
                }

                if phi_map.iter().any(|(i, _)| *i == index) {
                    dbg!(inputs, outputs);
                    dbg!(stack, phi_map);
                    return Err(Error::StackItemReused);
                }

                let phi = SIRExpression::PhiNode(vec![]);

                let var = AuxVar {
                    name: generate_var_name("phi", names),
                };

                statements.push(SIRStatement::Assignment(var.clone(), phi));
                stack_inputs.push(SIRExpression::AuxVar(var.clone()));
                temp_phi_map.push((index, var));
            } else {
                stack_inputs.push((stack.get(index).unwrap().clone().unwrap()).clone());
                to_delete.push(index);
            }
        }
    }

    let mut stack_outputs = vec![];

    for output in outputs {
        for count in 0..output.count {
            let var = AuxVar {
                name: generate_var_name(output.name, names),
            };

            stack_outputs.push(var.clone());

            let real_index =
                lowest_stack_input as isize + (output.index as i32 + count as i32) as isize;

            dbg!(lowest_stack_input, output);

            if let Some(index) = to_delete.iter().position(|v| v == &real_index) {
                *stack.get_mut(real_index).unwrap() =
                    Some(SIRExpression::<SIRNode, SIRException>::AuxVar(var));
                to_delete.remove(index);
            } else {
                // Update offsets
                if real_index >= 0 {
                    to_delete
                        .iter_mut()
                        .filter(|e| **e > real_index)
                        .for_each(|e| *e += 1);
                } else {
                    to_delete
                        .iter_mut()
                        .filter(|e| **e < real_index)
                        .for_each(|e| *e -= 1);
                }

                stack.insert(
                    real_index,
                    SIRExpression::<SIRNode, SIRException>::AuxVar(var),
                );
            }
        }
    }

    for index in to_delete.iter().rev() {
        if *index < 0 {
            // Don't remove items in the negative part
            *stack.get_mut(*index).unwrap() = None;
        } else {
            stack.remove(*index);
        }
    }

    phi_map.extend(temp_phi_map);

    Ok((stack_inputs, stack_outputs, statements))
}

fn instruction_to_ir<ExtInstruction, SIRNode, SIRException>(
    opcode: ExtInstruction::Opcode,
    oparg: ExtInstruction::OpargType,
    jump: bool,
    stack: &mut InfiniteVec<SIRExpression<SIRNode, SIRException>>,
    phi_map: &mut Vec<(isize, AuxVar)>,
    names: &mut HashMap<&'static str, u32>,
) -> Result<Vec<SIRStatement<SIRNode, SIRException>>, Error>
where
    ExtInstruction: GenericInstruction<OpargType = u32>,
    SIRNode: GenericSIRNode<Opcode = ExtInstruction::Opcode>,
    SIRException: GenericSIRException,
{
    let node = SIRNode::new(opcode.clone(), oparg, jump);

    let (stack_inputs, stack_outputs, mut statements) =
        process_stack_effects(node.get_inputs(), node.get_outputs(), stack, phi_map, names)?;

    let call = Call::<SIRNode, SIRException> { node, stack_inputs };

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

fn exception_to_ir<SIRNode, SIRException>(
    lasti: bool,
    jump: bool,
    stack: &mut InfiniteVec<SIRExpression<SIRNode, SIRException>>,
    phi_map: &mut Vec<(isize, AuxVar)>,
    names: &mut HashMap<&'static str, u32>,
) -> Result<Vec<SIRStatement<SIRNode, SIRException>>, Error>
where
    SIRNode: GenericSIRNode,
    SIRException: GenericSIRException,
{
    let exception = SIRException::new(lasti, jump);

    let (stack_inputs, stack_outputs, mut statements) = process_stack_effects(
        exception.get_inputs(),
        exception.get_outputs(),
        stack,
        phi_map,
        names,
    )?;

    let call = ExceptionCall::<SIRNode, SIRException> {
        exception,
        stack_inputs,
    };

    if stack_outputs.len() > 1 {
        statements.push(SIRStatement::TupleAssignment(
            stack_outputs,
            SIRExpression::Exception(call),
        ));
    } else if !stack_outputs.is_empty() {
        statements.push(SIRStatement::Assignment(
            stack_outputs.first().unwrap().clone(),
            SIRExpression::Exception(call),
        ))
    } else {
        // There was no exception raised
    }

    Ok(statements)
}

/// Internal function that is used while converting an Ext CFG to SIR nodes.
/// This function is meant to process a single block.
fn bb_to_ir<ExtInstruction, SIRNode, SIRException>(
    instructions: &[ExtInstruction],
    names: &mut HashMap<&'static str, u32>,
) -> Result<
    (
        SIR<SIRNode, SIRException>,
        InfiniteVec<SIRExpression<SIRNode, SIRException>>,
        Vec<(isize, AuxVar)>,
    ),
    Error,
>
where
    ExtInstruction: GenericInstruction<OpargType = u32>,
    SIR<SIRNode, SIRException>: SIROwned<SIRNode, SIRException>,
    SIRNode: GenericSIRNode<Opcode = ExtInstruction::Opcode>,
    SIRException: GenericSIRException,
{
    let mut statements: Vec<SIRStatement<SIRNode, SIRException>> = vec![];

    // Every basic block starts with an empty stack.
    // When we try to access stack items below 0, we know it's accessing items from a different basic block.
    let mut stack: InfiniteVec<SIRExpression<SIRNode, SIRException>> = vec![].into();

    // When we assign a phi node to a var we keep track of what stack index this phi node is representing
    let mut phi_map: Vec<(isize, AuxVar)> = vec![];

    for instruction in instructions {
        // In a basic block there shouldn't be any jumps. (the last jump instruction is removed in the cfg)
        debug_assert!(!instruction.is_jump());

        statements.extend_from_slice(&instruction_to_ir::<ExtInstruction, SIRNode, SIRException>(
            instruction.get_opcode(),
            instruction.get_raw_value(),
            false,
            &mut stack,
            &mut phi_map,
            names,
        )?);
    }

    Ok((SIR::new(statements), stack, phi_map))
}

/// Used to represent the opcode that was used for this branch and the block index it's jumping to.
/// We do this so the value of the branch instruction cannot represent a wrong index.
/// This also shows which inputs and outputs the opcode uses.
#[derive(Debug, Clone, PartialEq)]
pub struct SIRBranchEdge<SIRNode, SIRException, BranchReason>
where
    SIRNode: GenericSIRNode,
    SIRException: GenericSIRException,
    BranchReason: BranchReasonTrait,
{
    pub reason: BranchReason,
    pub statements: Option<SIR<SIRNode, SIRException>>,
    pub block_index: BlockIndex,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SIRBlockIndexInfo<SIRNode, SIRException, BranchReason>
where
    SIRNode: GenericSIRNode,
    SIRException: GenericSIRException,
    BranchReason: BranchReasonTrait,
{
    Edge(SIRBranchEdge<SIRNode, SIRException, BranchReason>),
    /// For blocks that fallthrough with no opcode (cannot be generated by Python, used by internal algorithms)
    Fallthrough(BlockIndex),
    /// For blocks without a target
    NoIndex,
}

impl<SIRNode, SIRException, BranchReason> SIRBlockIndexInfo<SIRNode, SIRException, BranchReason>
where
    SIRNode: GenericSIRNode,
    SIRException: GenericSIRException,
    BranchReason: BranchReasonTrait,
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
pub struct SIRBlock<SIRNode, SIRException, BranchReason>
where
    SIRNode: GenericSIRNode,
    SIRException: GenericSIRException,
    BranchReason: BranchReasonTrait,
{
    pub nodes: SIR<SIRNode, SIRException>,
    /// Index to block for conditional jump
    pub branch_block: SIRBlockIndexInfo<SIRNode, SIRException, BranchReason>,
    /// Index to default block (unconditional)
    pub default_block: SIRBlockIndexInfo<SIRNode, SIRException, BranchReason>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SIRControlFlowGraph<SIRNode, SIRException, BranchReason>
where
    SIRNode: GenericSIRNode,
    SIRException: GenericSIRException,
    BranchReason: BranchReasonTrait,
{
    pub blocks: Vec<SIRBlock<SIRNode, SIRException, BranchReason>>,
    pub start_index: SIRBlockIndexInfo<SIRNode, SIRException, BranchReason>,
}

#[cfg(feature = "dot")]
impl<SIRNode, SIRException, BranchReason> SIRControlFlowGraph<SIRNode, SIRException, BranchReason>
where
    SIRNode: GenericSIRNode,
    SIRException: GenericSIRException,
    BranchReason: BranchReasonTrait,
{
    fn add_block<'a>(
        graph: &mut petgraph::Graph<String, String>,
        blocks: &'a [SIRBlock<SIRNode, SIRException, BranchReason>],
        block_index: Option<&'a BlockIndex>,
        block_map: &mut HashMap<Option<&'a BlockIndex>, NodeIndex>,
    ) -> Option<NodeIndex>
    where
        SIR<SIRNode, SIRException>: SIROwned<SIRNode, SIRException>,
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
            block_map.insert(block_index, index);

            index
        };

        let (branch_index, branch_statements) = match &block.branch_block {
            SIRBlockIndexInfo::Edge(SIRBranchEdge {
                block_index: branch_index,
                statements,
                ..
            }) => (Some(branch_index), Some(statements)),
            SIRBlockIndexInfo::Fallthrough(branch_index) => (Some(branch_index), None),
            _ => (None, None),
        };

        let branch_index = if block_map.contains_key(&branch_index) {
            Some(block_map[&branch_index])
        } else {
            let index = Self::add_block(graph, blocks, branch_index, block_map);

            if let Some(index) = index {
                block_map.insert(branch_index, index);
                Some(index)
            } else {
                match branch_index {
                    Some(BlockIndex::InvalidIndex(invalid_index)) => {
                        Some(graph.add_node(format!("invalid jump to index {}", invalid_index)))
                    }
                    Some(BlockIndex::Index(_)) => unreachable!(),
                    None => None,
                }
            }
        };

        let (default_index, default_statements) = match &block.default_block {
            SIRBlockIndexInfo::Edge(SIRBranchEdge {
                block_index: default_index,
                statements,
                ..
            }) => (Some(default_index), Some(statements)),
            SIRBlockIndexInfo::Fallthrough(branch_index) => (Some(branch_index), None),
            _ => (None, None),
        };

        let default_index = if block_map.contains_key(&default_index) {
            Some(block_map[&default_index])
        } else {
            let index = Self::add_block(graph, blocks, default_index, block_map);

            if let Some(index) = index {
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
            }
        };

        if let Some(to_index) = branch_index {
            let text = if let Some(Some(statements)) = branch_statements {
                format!("{}", statements)
            } else {
                "".to_owned()
            };
            graph.add_edge(index, to_index, text);
        }

        if let Some(to_index) = default_index {
            let text = if let Some(Some(statements)) = default_statements {
                format!("fallthrough\n{}", statements)
            } else {
                "fallthrough".to_string()
            };

            graph.add_edge(index, to_index, text);
        }

        Some(index)
    }

    pub fn make_dot_graph(&self) -> String
    where
        SIRNode: GenericSIRNode,
        SIR<SIRNode, SIRException>: SIROwned<SIRNode, SIRException>,
    {
        let mut graph = petgraph::Graph::<String, String>::new();

        Self::add_block(
            &mut graph,
            &self.blocks,
            self.start_index.get_block_index(),
            &mut HashMap::new(),
        );

        format!(
            "{:#?}",
            Dot::with_attr_getters(
                &graph,
                &[Config::NodeNoLabel, Config::EdgeNoLabel],
                &|_, e| {
                    let color = if !e.weight().contains("fallthrough") {
                        "green"
                    } else {
                        "red"
                    };

                    format!(r#"label = "{}", color = {}"#, e.weight(), color)
                },
                &|_, (_, s)| format!(r#"label = "{}""#, s),
            )
        )
    }
}

#[derive(Clone, Debug)]
pub enum Error {
    InvalidStackAccess,
    StackItemReused,
    PhiNodeNotPopulated,
    NotAllBlocksProcessed,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidStackAccess => write!(f, "Invalid stack access"),
            Error::StackItemReused => write!(f, "Stack item reused"),
            Error::PhiNodeNotPopulated => write!(f, "Phi node not populated"),
            Error::NotAllBlocksProcessed => write!(f, "Not all blocks processed"),
        }
    }
}

/// This will extend the vec and overwrite any items that have a corresponding negative value
pub fn extend_merge_stack<SIRNode, SIRException>(
    vec_to_extend: &mut Vec<SIRExpression<SIRNode, SIRException>>,
    infinite_stack: &InfiniteVec<SIRExpression<SIRNode, SIRException>>,
    items_to_delete: Vec<isize>,
) -> Result<(), Error>
where
    SIRNode: GenericSIRNode,
    SIRException: GenericSIRException,
{
    let original_len = vec_to_extend.len();

    let mut items_to_delete = items_to_delete;

    dbg!(&vec_to_extend, infinite_stack);

    let pairs = infinite_stack.collect_pairs();

    for (i, item) in pairs {
        let real_index: usize = (original_len as isize + i).try_into().unwrap();

        if let Some(index) = items_to_delete.iter().position(|v| v == &i) {
            vec_to_extend[real_index] = item.clone();
            items_to_delete.remove(index);
        } else if i >= 0 {
            vec_to_extend.insert(real_index.try_into().unwrap(), item.clone());
        } else {
            todo!("pushing into the previous stack without consuming it")
        }
    }

    // Loop the items from biggest to smallest
    for i in items_to_delete.iter().rev() {
        // Remove item from the stack after it was used

        let real_index: usize = (original_len as isize + i).try_into().unwrap();

        vec_to_extend.remove(real_index);
    }

    Ok(())
}

pub fn cfg_to_ir<ExtInstruction, SIRNode, SIRException, BranchReason>(
    cfg: &ControlFlowGraph<ExtInstruction, BranchReason>,
) -> Result<SIRControlFlowGraph<SIRNode, SIRException, BranchReason>, Error>
where
    ExtInstruction: GenericInstruction<OpargType = u32>,
    SIRNode: GenericSIRNode<Opcode = ExtInstruction::Opcode>,
    SIRException: GenericSIRException,
    BranchReason: BranchReasonTrait<Opcode = SIRNode::Opcode>,
    SIR<SIRNode, SIRException>: SIROwned<SIRNode, SIRException>,
{
    #[derive(Debug, Clone)]
    struct TempBlockInfo<SIRNode, SIRException>
    where
        SIRNode: Clone + std::fmt::Debug,
        SIRException: Clone + std::fmt::Debug,
    {
        stack: InfiniteVec<SIRExpression<SIRNode, SIRException>>,
        statements: SIR<SIRNode, SIRException>,
        phi_map: Vec<(isize, AuxVar)>,
    }

    let mut temp_blocks: Vec<TempBlockInfo<SIRNode, SIRException>> = vec![];

    // Keep track of which blocks we already processed
    let mut visited_blocks: Vec<usize> = vec![];

    #[derive(Debug, Clone)]
    struct TempEdgeInfo<SIRNode, SIRException>
    where
        SIRNode: Clone + std::fmt::Debug,
        SIRException: Clone + std::fmt::Debug,
    {
        stack: InfiniteVec<SIRExpression<SIRNode, SIRException>>,
        statements: Vec<SIRStatement<SIRNode, SIRException>>,
        phi_map: Vec<(isize, AuxVar)>,
    }

    // (default, branch)
    let mut temp_edges: Vec<(
        TempEdgeInfo<SIRNode, SIRException>,
        TempEdgeInfo<SIRNode, SIRException>,
    )> = vec![];

    let mut names = HashMap::new();

    // Keeps track of the different stacks used to enter a basic block
    // TODO: We have to find a way to merge stacks
    let mut has_changed: HashMap<usize, Vec<Vec<SIRExpression<SIRNode, SIRException>>>> =
        HashMap::new();

    // Create isolated IR blocks
    for block in &cfg.blocks {
        let (statements, stack, phi_map) =
            bb_to_ir::<ExtInstruction, SIRNode, SIRException>(&block.instructions, &mut names)?;

        temp_blocks.push(TempBlockInfo {
            stack,
            statements,
            phi_map,
        });

        let mut default_stack = vec![].into();
        let mut default_phi_map = vec![];

        let default_statements = match &block.default_block {
            BlockIndexInfo::Edge(BranchEdge {
                reason: branch_reason,
                ..
            }) => {
                if let Some(opcode) = branch_reason.get_opcode() {
                    instruction_to_ir::<ExtInstruction, SIRNode, SIRException>(
                        opcode.clone(),
                        0, // The oparg doesn't matter for the stack effect in the case of a branch opcode (oparg is the jump target)
                        false, // Don't take the jump for the default block
                        &mut default_stack,
                        &mut default_phi_map,
                        &mut names,
                    )?
                } else if let Some(lasti) = branch_reason.get_lasti() {
                    exception_to_ir::<SIRNode, SIRException>(
                        lasti.clone(),
                        false,
                        &mut default_stack,
                        &mut default_phi_map,
                        &mut names,
                    )?
                } else {
                    unreachable!()
                }
            }
            _ => vec![],
        };

        let mut branch_stack = vec![].into();
        let mut branch_phi_map = vec![];

        let branch_statements = match &block.branch_block {
            BlockIndexInfo::Edge(BranchEdge {
                reason: branch_reason,
                ..
            }) => {
                if let Some(opcode) = branch_reason.get_opcode() {
                    instruction_to_ir::<ExtInstruction, SIRNode, SIRException>(
                        opcode.clone(),
                        0, // The oparg doesn't matter for the stack effect in the case of a branch opcode (oparg is the jump target)
                        true, // Take the jump for the branch block
                        &mut branch_stack,
                        &mut branch_phi_map,
                        &mut names,
                    )?
                } else if let Some(lasti) = branch_reason.get_lasti() {
                    exception_to_ir::<SIRNode, SIRException>(
                        lasti.clone(),
                        true,
                        &mut branch_stack,
                        &mut branch_phi_map,
                        &mut names,
                    )?
                } else {
                    unreachable!()
                }
            }
            _ => vec![],
        };

        temp_edges.push((
            TempEdgeInfo {
                stack: default_stack,
                statements: default_statements,
                phi_map: default_phi_map,
            },
            TempEdgeInfo {
                stack: branch_stack,
                statements: branch_statements,
                phi_map: branch_phi_map,
            },
        ));
    }

    #[derive(Debug)]
    struct BlockQueue<SIRNode, SIRException, BranchReason>
    where
        BranchReason: BranchReasonTrait,
        SIRNode: Clone + std::fmt::Debug,
        SIRException: Clone + std::fmt::Debug,
    {
        block_index: BlockIndexInfo<BranchReason>,
        /// Only `None` when `block_index` is the first block.
        /// The `bool`` shows if we need to take the `branch` edge or not
        from_block_index: Option<(usize, bool)>,
        curr_stack: Vec<SIRExpression<SIRNode, SIRException>>,
    }

    // Fill the empty phi nodes with actual values
    let mut queue: Vec<BlockQueue<SIRNode, SIRException, BranchReason>> = vec![BlockQueue {
        block_index: cfg.start_index.clone(),
        from_block_index: None,
        curr_stack: vec![],
    }];

    while let Some(mut block_element) = queue.pop() {
        // The branch opcode is used to calculate any effects the branch instruction might have
        let index = match block_element.block_index.get_block_index() {
            Some(BlockIndex::Index(index)) => *index,
            _ => continue,
        };

        let edge_info = if let Some((from_block_index, use_branch)) = block_element.from_block_index
        {
            let (default, branch) = temp_edges.get_mut(from_block_index).unwrap();

            Some(if use_branch { branch } else { default })
        } else {
            None
        };

        let block_info = temp_blocks.get_mut(index).unwrap();

        let already_analysed = visited_blocks.contains(&index);

        if let Some(edge_info) = edge_info {
            // Fill empty phi nodes in the edge statements
            let deleted_items = fill_phi_nodes(
                &mut block_element.curr_stack,
                &mut edge_info.statements,
                &edge_info.phi_map,
            )?;

            // Add the branch stack after processing the branch statements
            extend_merge_stack(
                &mut block_element.curr_stack,
                &edge_info.stack,
                deleted_items,
            )?;
        }

        if already_analysed {
            if let Some(stacks) = has_changed.get_mut(&index) {
                if stacks.contains(&block_element.curr_stack) {
                    // Already processed this block but we still processed the statements of the edge
                    continue;
                } else {
                    // Entered with a different stack
                    stacks.push(block_element.curr_stack.clone());
                }
            }
        } else {
            has_changed.insert(index, vec![block_element.curr_stack.clone()]);
        }

        dbg!(&block_element);
        dbg!(&block_info);

        let deleted_items = fill_phi_nodes(
            &mut block_element.curr_stack,
            &mut block_info.statements.0,
            &block_info.phi_map,
        )?;

        dbg!(&block_element);
        dbg!(&block_info);

        extend_merge_stack(
            &mut block_element.curr_stack,
            &block_info.stack,
            deleted_items,
        )?;

        queue.push(BlockQueue {
            block_index: cfg.blocks.index(index).default_block.clone(),
            curr_stack: block_element.curr_stack.to_vec(),
            from_block_index: Some((index, false)),
        });

        queue.push(BlockQueue {
            block_index: cfg.blocks.index(index).branch_block.clone(),
            curr_stack: block_element.curr_stack.to_vec(),
            from_block_index: Some((index, true)),
        });

        if !visited_blocks.contains(&index) {
            visited_blocks.push(index);
        }
    }

    if visited_blocks.len() != temp_blocks.len() {
        return Err(Error::NotAllBlocksProcessed);
    }

    let blocks: Vec<_> = temp_blocks
        .into_iter()
        .enumerate()
        .map(|(index, v)| SIRBlock {
            nodes: v.statements,
            branch_block: cfg.blocks[index].branch_block.into_sir(
                if temp_edges.get(index).unwrap().1.statements.is_empty() {
                    None
                } else {
                    Some(temp_edges.get(index).unwrap().1.statements.clone())
                },
            ),
            default_block: cfg.blocks[index].default_block.into_sir(
                if temp_edges.get(index).unwrap().0.statements.is_empty() {
                    None
                } else {
                    Some(temp_edges.get(index).unwrap().0.statements.clone())
                },
            ),
        })
        .collect::<Vec<_>>();

    let sir_cfg = SIRControlFlowGraph::<SIRNode, SIRException, BranchReason> {
        start_index: SIRBlockIndexInfo::Fallthrough(BlockIndex::Index(0)),
        blocks,
    };

    Ok(sir_cfg)
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use rayon::vec;

    use crate::cfg::{ControlFlowGraph, create_cfg, simple_cfg_to_ext_cfg};
    use crate::sir::{
        AuxVar, SIR, StackItem, bb_to_ir, cfg_to_ir, extend_merge_stack, fill_phi_nodes,
        instruction_to_ir, process_stack_effects,
    };
    use crate::traits::{GenericInstruction, GenericSIRException};
    use crate::utils::InfiniteVec;
    use crate::v311::ext_instructions::{ExtInstruction, ExtInstructions};
    use crate::v311::instructions::Instruction;
    use crate::v311::opcodes::sir::{SIRException, SIRNode};
    use crate::{CodeObject, v311};

    #[test]
    fn test_stack_processing() {
        let inputs = [
            StackItem {
                name: "first",
                count: 1,
                index: 3,
            },
            StackItem {
                name: "second",
                count: 1,
                index: 1,
            },
        ]
        .as_slice();

        let outputs = [
            StackItem {
                name: "res_first",
                count: 1,
                index: 0,
            },
            StackItem {
                name: "res_second",
                count: 1,
                index: 4,
            },
        ]
        .as_slice();

        let mut stack: InfiniteVec<_> = vec![].into();

        let fake_var = crate::sir::SIRExpression::AuxVar(AuxVar {
            name: "test".into(),
        });

        stack.insert(-2, fake_var.clone());

        assert_eq!(stack.negative_len(), 2);
        assert_eq!(stack.positive_len(), 0);
        assert_eq!(
            stack.iter().cloned().collect::<Vec<_>>(),
            vec![Some(fake_var), None]
        );

        let mut phi_map = vec![];
        let mut names = HashMap::new();

        process_stack_effects::<SIRNode, SIRException>(
            inputs,
            outputs,
            &mut stack,
            &mut phi_map,
            &mut names,
        )
        .unwrap();

        assert_eq!(phi_map.len(), 1);
        assert_eq!(phi_map.first().unwrap().0, -4);

        assert_eq!(stack.positive_len(), 1);
        assert_eq!(stack.negative_len(), 4);

        assert_eq!(
            stack.iter().cloned().collect::<Vec<_>>(),
            vec![
                Some(crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "res_first_0".into(),
                })),
                None,
                None,
                None,
                Some(crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "res_second_0".into(),
                })),
            ]
        );

        // New stack processing attempt

        let outputs = [
            StackItem {
                name: "const",
                count: 1,
                index: 0,
            },
            StackItem {
                name: "const",
                count: 1,
                index: 1,
            },
        ]
        .as_slice();

        let mut stack: InfiniteVec<_> = vec![].into();
        let mut phi_map = vec![];
        let mut names = HashMap::new();

        process_stack_effects::<SIRNode, SIRException>(
            &[],
            outputs,
            &mut stack,
            &mut phi_map,
            &mut names,
        )
        .unwrap();

        assert_eq!(stack.positive_len(), 2);
        assert_eq!(stack.negative_len(), 0);

        assert_eq!(
            stack.collect_pairs(),
            vec![
                (
                    0,
                    &crate::sir::SIRExpression::AuxVar(AuxVar {
                        name: "const_0".into()
                    })
                ),
                (
                    1,
                    &crate::sir::SIRExpression::AuxVar(AuxVar {
                        name: "const_1".into()
                    })
                )
            ]
        )
    }

    #[test]
    fn test_call_stack_processing() {
        let inputs = [
            StackItem {
                name: "method_or_null",
                count: 1,
                index: 2,
            },
            StackItem {
                name: "self_or_callable",
                count: 1,
                index: 1,
            },
            StackItem {
                name: "args",
                count: 1,
                index: 0,
            },
        ]
        .as_slice();

        let outputs = [StackItem {
            name: "res",
            count: 1,
            index: 0,
        }]
        .as_slice();

        let mut stack: InfiniteVec<_> = vec![].into();

        stack.push(crate::sir::SIRExpression::AuxVar(AuxVar {
            name: "null_0".into(),
        }));

        stack.push(crate::sir::SIRExpression::AuxVar(AuxVar {
            name: "value_0".into(),
        }));

        stack.push(crate::sir::SIRExpression::AuxVar(AuxVar {
            name: "value_1".into(),
        }));

        assert_eq!(stack.negative_len(), 0);
        assert_eq!(stack.positive_len(), 3);

        let mut phi_map = vec![];
        let mut names = HashMap::new();

        process_stack_effects::<SIRNode, SIRException>(
            inputs,
            outputs,
            &mut stack,
            &mut phi_map,
            &mut names,
        )
        .unwrap();

        assert_eq!(phi_map.len(), 0);

        assert_eq!(stack.positive_len(), 1);
        assert_eq!(stack.negative_len(), 0);

        assert_eq!(
            stack.iter().cloned().collect::<Vec<_>>(),
            vec![Some(crate::sir::SIRExpression::AuxVar(AuxVar {
                name: "res_0".into(),
            })),]
        );

        // POP_TOP the result

        let inputs = [StackItem {
            name: "top",
            count: 1,
            index: 0,
        }]
        .as_slice();

        process_stack_effects::<SIRNode, SIRException>(
            inputs,
            &[],
            &mut stack,
            &mut phi_map,
            &mut names,
        )
        .unwrap();

        assert_eq!(stack.positive_len(), 0);
        assert_eq!(stack.negative_len(), 0);
    }

    #[test]
    fn test_middle_stack_processing() {
        // In this test we will process stack usages below 0 that appear after the first instruction

        let inputs = [StackItem {
            name: "top",
            count: 1,
            index: 0,
        }]
        .as_slice();

        let mut stack: InfiniteVec<_> = vec![].into();

        let mut phi_map = vec![];
        let mut names = HashMap::new();

        // Pop all 3 values
        for _ in 0..3 {
            process_stack_effects::<SIRNode, SIRException>(
                inputs,
                &[],
                &mut stack,
                &mut phi_map,
                &mut names,
            )
            .unwrap();
        }

        assert_eq!(phi_map.len(), 3);

        dbg!(phi_map);

        assert_eq!(stack.positive_len(), 0);
        assert_eq!(stack.negative_len(), 0);

        // Pop all in the same input with a filled phi map

        let inputs = [
            StackItem {
                name: "first",
                count: 1,
                index: 2,
            },
            StackItem {
                name: "second",
                count: 1,
                index: 1,
            },
            StackItem {
                name: "third",
                count: 1,
                index: 0,
            },
        ]
        .as_slice();

        let mut stack: InfiniteVec<_> = vec![].into();

        let mut phi_map = vec![
            (
                -1,
                AuxVar {
                    name: "test_1".into(),
                },
            ),
            (
                -2,
                AuxVar {
                    name: "test_2".into(),
                },
            ),
            (
                -3,
                AuxVar {
                    name: "test_3".into(),
                },
            ),
        ];
        let mut names = HashMap::new();

        // Pop all 3 values
        process_stack_effects::<SIRNode, SIRException>(
            inputs,
            &[],
            &mut stack,
            &mut phi_map,
            &mut names,
        )
        .unwrap();

        let mut phi_indexes = phi_map.iter().cloned().collect::<Vec<_>>();

        phi_indexes.sort_by_key(|(i, _)| *i);

        assert_eq!(
            phi_indexes,
            vec![
                (
                    -6,
                    AuxVar {
                        name: "phi_0".into()
                    }
                ),
                (
                    -5,
                    AuxVar {
                        name: "phi_1".into()
                    }
                ),
                (
                    -4,
                    AuxVar {
                        name: "phi_2".into()
                    }
                ),
                (
                    -3,
                    AuxVar {
                        name: "test_3".into()
                    }
                ),
                (
                    -2,
                    AuxVar {
                        name: "test_2".into()
                    }
                ),
                (
                    -1,
                    AuxVar {
                        name: "test_1".into()
                    }
                )
            ]
        );

        // Pop all in the same input with an empty phi map

        let inputs = [
            StackItem {
                name: "first",
                count: 1,
                index: 2,
            },
            StackItem {
                name: "second",
                count: 1,
                index: 1,
            },
            StackItem {
                name: "third",
                count: 1,
                index: 0,
            },
        ]
        .as_slice();

        let mut stack: InfiniteVec<_> = vec![].into();

        let mut phi_map = vec![];
        let mut names = HashMap::new();

        // Pop all 3 values
        process_stack_effects::<SIRNode, SIRException>(
            inputs,
            &[],
            &mut stack,
            &mut phi_map,
            &mut names,
        )
        .unwrap();

        let mut phi_indexes = phi_map.iter().cloned().collect::<Vec<_>>();

        phi_indexes.sort_by_key(|(i, _)| *i);

        assert_eq!(
            phi_indexes,
            vec![
                (
                    -3,
                    AuxVar {
                        name: "phi_0".into()
                    }
                ),
                (
                    -2,
                    AuxVar {
                        name: "phi_1".into()
                    }
                ),
                (
                    -1,
                    AuxVar {
                        name: "phi_2".into()
                    }
                ),
            ]
        );
    }

    #[test]
    fn test_merge_stack() {
        let instructions = vec![
            crate::v311::instructions::Instruction::ForIter(0),
            crate::v311::instructions::Instruction::StoreName(0),
        ];

        let mut curr_stack: Vec<crate::sir::SIRExpression<SIRNode, SIRException>> =
            vec![crate::sir::SIRExpression::AuxVar(AuxVar {
                name: "og_iter".into(),
            })];

        let mut statements = vec![];

        let mut stack: InfiniteVec<_> = vec![].into();
        let mut phi_map = vec![];
        let mut names = HashMap::new();

        for instruction in instructions {
            // Pop all 3 values
            let stmts = instruction_to_ir::<
                crate::v311::ext_instructions::ExtInstruction,
                SIRNode,
                SIRException,
            >(
                instruction.get_opcode(),
                0,
                false,
                &mut stack,
                &mut phi_map,
                &mut names,
            )
            .unwrap();

            statements.extend(stmts);
        }

        let deleted_items = fill_phi_nodes(&mut curr_stack, &mut statements, &mut phi_map).unwrap();

        extend_merge_stack(&mut curr_stack, &mut stack, deleted_items).unwrap();

        dbg!(&curr_stack, &statements);

        assert_eq!(curr_stack.len(), 1);
        assert_eq!(
            *curr_stack.first().unwrap(),
            crate::sir::SIRExpression::AuxVar(AuxVar {
                name: "iter_0".into()
            },),
        );

        assert_eq!(
            statements,
            vec![
                crate::sir::SIRStatement::Assignment(
                    AuxVar {
                        name: "phi_0".into()
                    },
                    crate::sir::SIRExpression::PhiNode(vec![AuxVar {
                        name: "og_iter".into()
                    }]),
                ),
                crate::sir::SIRStatement::TupleAssignment(
                    vec![
                        AuxVar {
                            name: "iter_0".into()
                        },
                        AuxVar {
                            name: "next_0".into()
                        },
                    ],
                    crate::sir::SIRExpression::Call(crate::sir::Call {
                        node: SIRNode {
                            opcode: crate::v311::opcodes::Opcode::FOR_ITER,
                            oparg: 0,
                            input: vec![StackItem {
                                name: "iter",
                                count: 1,
                                index: 0,
                            },],
                            output: vec![
                                StackItem {
                                    name: "iter",
                                    count: 1,
                                    index: 0,
                                },
                                StackItem {
                                    name: "next",
                                    count: 1,
                                    index: 1,
                                },
                            ],
                        },
                        stack_inputs: vec![crate::sir::SIRExpression::AuxVar(AuxVar {
                            name: "phi_0".into()
                        },),],
                    },),
                ),
                crate::sir::SIRStatement::DisregardCall(crate::sir::Call {
                    node: SIRNode {
                        opcode: crate::v311::opcodes::Opcode::STORE_NAME,
                        oparg: 0,
                        input: vec![StackItem {
                            name: "value",
                            count: 1,
                            index: 0,
                        },],
                        output: vec![],
                    },
                    stack_inputs: vec![crate::sir::SIRExpression::AuxVar(AuxVar {
                        name: "next_0".into()
                    },),],
                },),
            ]
        );
    }

    #[test]
    fn test_merge_stack_2() {
        let instructions = vec![
            crate::v311::instructions::Instruction::StoreName(0),
            crate::v311::instructions::Instruction::LoadConst(0),
        ];

        let mut curr_stack: Vec<crate::sir::SIRExpression<SIRNode, SIRException>> = vec![
            crate::sir::SIRExpression::AuxVar(AuxVar {
                name: "iter_0".into(),
            }),
            crate::sir::SIRExpression::AuxVar(AuxVar {
                name: "next_0".into(),
            }),
        ];

        let mut statements = vec![];

        let mut stack: InfiniteVec<_> = vec![].into();
        let mut phi_map = vec![];
        let mut names = HashMap::new();

        for instruction in instructions {
            // Pop all 3 values
            let stmts = instruction_to_ir::<
                crate::v311::ext_instructions::ExtInstruction,
                SIRNode,
                SIRException,
            >(
                instruction.get_opcode(),
                0,
                false,
                &mut stack,
                &mut phi_map,
                &mut names,
            )
            .unwrap();

            statements.extend(stmts);
        }

        let deleted_items = fill_phi_nodes(&mut curr_stack, &mut statements, &mut phi_map).unwrap();

        extend_merge_stack(&mut curr_stack, &mut stack, deleted_items).unwrap();

        dbg!(&curr_stack, &statements);

        assert_eq!(curr_stack.len(), 2);
        assert_eq!(
            *curr_stack,
            vec![
                crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "iter_0".into()
                },),
                crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "value_0".into()
                },)
            ],
        );
    }

    #[test]
    fn test_stack_gap_processing() {
        let inputs = [
            StackItem {
                name: "bottom",
                count: 1,
                index: 2,
            },
            StackItem {
                name: "top",
                count: 1,
                index: 0,
            },
        ]
        .as_slice();

        let outputs = [
            StackItem {
                name: "top",
                count: 1,
                index: 0,
            },
            StackItem {
                name: "bottom",
                count: 1,
                index: 3,
            },
        ]
        .as_slice();

        let mut curr_stack: Vec<crate::sir::SIRExpression<SIRNode, SIRException>> = vec![
            crate::sir::SIRExpression::AuxVar(AuxVar {
                name: "first".into(),
            }),
            crate::sir::SIRExpression::AuxVar(AuxVar {
                name: "second".into(),
            }),
            crate::sir::SIRExpression::AuxVar(AuxVar {
                name: "third".into(),
            }),
        ];

        let mut stack: InfiniteVec<_> = vec![].into();
        let mut phi_map = vec![];
        let mut names = HashMap::new();

        let (_, _, mut statements) = process_stack_effects::<SIRNode, SIRException>(
            inputs,
            outputs,
            &mut stack,
            &mut phi_map,
            &mut names,
        )
        .unwrap();

        let deleted_items = fill_phi_nodes(&mut curr_stack, &mut statements, &mut phi_map).unwrap();

        extend_merge_stack(&mut curr_stack, &mut stack, deleted_items).unwrap();

        assert_eq!(
            curr_stack,
            vec![
                crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "top_0".into()
                },),
                crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "second".into()
                },),
                crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "bottom_0".into()
                },),
            ]
        )
    }

    #[test]
    fn test_stack_gap_processing_2() {
        let mut statements = vec![];

        let inputs = [StackItem {
            name: "bottom",
            count: 1,
            index: 2,
        }]
        .as_slice();

        let outputs = [
            StackItem {
                name: "top",
                count: 1,
                index: 0,
            },
            StackItem {
                name: "bottom",
                count: 1,
                index: 3,
            },
        ]
        .as_slice();

        let mut curr_stack: Vec<crate::sir::SIRExpression<SIRNode, SIRException>> = vec![
            crate::sir::SIRExpression::AuxVar(AuxVar {
                name: "first".into(),
            }),
            crate::sir::SIRExpression::AuxVar(AuxVar {
                name: "second".into(),
            }),
            crate::sir::SIRExpression::AuxVar(AuxVar {
                name: "third".into(),
            }),
        ];

        let mut stack: InfiniteVec<_> = vec![].into();
        let mut phi_map = vec![];
        let mut names = HashMap::new();

        let (_, _, stmts) = process_stack_effects::<SIRNode, SIRException>(
            inputs,
            outputs,
            &mut stack,
            &mut phi_map,
            &mut names,
        )
        .unwrap();

        statements.extend(stmts);

        let mut sanity_stack = curr_stack.clone();

        let deleted_items =
            fill_phi_nodes(&mut curr_stack, &mut statements.clone(), &mut phi_map).unwrap();

        extend_merge_stack(&mut sanity_stack, &mut stack, deleted_items).unwrap();

        assert_eq!(
            sanity_stack,
            vec![
                crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "top_0".into()
                },),
                crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "second".into()
                },),
                crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "third".into()
                },),
                crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "bottom_0".into()
                },),
            ]
        );

        // pop top element

        let inputs = [StackItem {
            name: "top",
            count: 1,
            index: 0,
        }]
        .as_slice();

        let (_, _, stmts) = process_stack_effects::<SIRNode, SIRException>(
            inputs,
            &[],
            &mut stack,
            &mut phi_map,
            &mut names,
        )
        .unwrap();

        statements.extend(stmts);

        let mut sanity_stack = curr_stack.clone();

        let deleted_items =
            fill_phi_nodes(&mut curr_stack, &mut statements.clone(), &mut phi_map).unwrap();

        extend_merge_stack(&mut sanity_stack, &mut stack, deleted_items).unwrap();

        assert_eq!(
            sanity_stack,
            vec![
                crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "top_0".into()
                },),
                crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "second".into()
                },),
                crate::sir::SIRExpression::AuxVar(AuxVar {
                    name: "third".into()
                },),
            ]
        );

        let inputs = [
            StackItem {
                name: "first",
                count: 1,
                index: 1,
            },
            StackItem {
                name: "second",
                count: 1,
                index: 0,
            },
        ]
        .as_slice();

        let outputs = [StackItem {
            name: "out",
            count: 1,
            index: 0,
        }]
        .as_slice();

        dbg!(&stack, &curr_stack);

        let (_, _, stmts) = process_stack_effects::<SIRNode, SIRException>(
            inputs,
            outputs,
            &mut stack,
            &mut phi_map,
            &mut names,
        )
        .unwrap();

        statements.extend(stmts);

        dbg!(&stack, &curr_stack);

        let deleted_items = fill_phi_nodes(&mut curr_stack, &mut statements, &mut phi_map).unwrap();

        extend_merge_stack(&mut curr_stack, &mut stack, deleted_items).unwrap();

        dbg!(curr_stack);
        dbg!(statements);
    }

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
            bb_to_ir::<ExtInstruction, SIRNode, SIRException>(
                &ext_instructions,
                &mut HashMap::new()
            )
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

        let cfg = create_cfg(instructions.to_vec(), None).unwrap();

        let cfg: ControlFlowGraph<ExtInstruction, v311::opcodes::BranchReason> =
            simple_cfg_to_ext_cfg::<
                Instruction,
                ExtInstruction,
                ExtInstructions,
                v311::opcodes::BranchReason,
            >(&cfg)
            .unwrap();

        let ir_cfg =
            cfg_to_ir::<ExtInstruction, SIRNode, SIRException, v311::opcodes::BranchReason>(&cfg)
                .unwrap();

        println!("{}", ir_cfg.make_dot_graph());

        insta::assert_debug_snapshot!(ir_cfg);
    }

    #[test]
    fn test_complex_cfg_to_sir() {
        let program = crate::load_code(&b"\xe3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\xf3\\\x00\x00\x00\x97\x00\x02\x00e\x00d\x00\xa6\x01\x00\x00\xab\x01\x00\x00\x00\x00\x00\x00\x00\x00D\x00]\x1fZ\x01e\x01d\x01k\x02\x00\x00\x00\x00r\x0c\x02\x00e\x02d\x02\xa6\x01\x00\x00\xab\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x8c\x14\x02\x00e\x02d\x03\xa6\x01\x00\x00\xab\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x8c d\x04S\x00)\x05\xe9\n\x00\x00\x00\xe9\t\x00\x00\x00\xda\x03yay\xda\x03nayN)\x03\xda\x05range\xda\x01x\xda\x05print\xa9\x00\xf3\x00\x00\x00\x00z\x08<string>\xfa\x08<module>r\x0b\x00\x00\x00\x01\x00\x00\x00sM\x00\x00\x00\xf0\x03\x01\x01\x01\xe0\t\x0e\x88\x15\x88r\x89\x19\x8c\x19\xf0\x00\x04\x01\x15\xf0\x00\x04\x01\x15\x80A\xd8\x07\x08\x88A\x82v\x80v\xd8\x08\r\x88\x05\x88e\x89\x0c\x8c\x0c\x88\x0c\x88\x0c\xe0\x08\r\x88\x05\x88e\x89\x0c\x8c\x0c\x88\x0c\x88\x0c\xf0\t\x04\x01\x15\xf0\x00\x04\x01\x15r\n\x00\x00\x00"[..], (3, 11).into()).unwrap();

        let instructions = match program {
            CodeObject::V311(code) => code.code,
            _ => unreachable!(),
        };

        let cfg = create_cfg(instructions.to_vec(), None).unwrap();

        let cfg = simple_cfg_to_ext_cfg::<
            Instruction,
            ExtInstruction,
            ExtInstructions,
            v311::opcodes::BranchReason,
        >(&cfg)
        .unwrap();

        let ir_cfg =
            cfg_to_ir::<ExtInstruction, SIRNode, SIRException, v311::opcodes::BranchReason>(&cfg)
                .unwrap();

        println!("{}", ir_cfg.make_dot_graph());
        insta::assert_debug_snapshot!(ir_cfg);
    }

    #[test]
    fn test_exception_block_310() {
        // print("hi")
        // try:
        //     print(1)
        // except:
        //     print(2)
        let program = crate::load_code(&b"\xe3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00@\x00\x00\x00s,\x00\x00\x00e\x00d\x00\x83\x01\x01\x00z\x07e\x00d\x01\x83\x01\x01\x00W\x00d\x03S\x00\x01\x00\x01\x00\x01\x00e\x00d\x02\x83\x01\x01\x00Y\x00d\x03S\x00)\x04Z\x02hi\xe9\x01\x00\x00\x00\xe9\x02\x00\x00\x00N)\x01\xda\x05print\xa9\x00r\x04\x00\x00\x00r\x04\x00\x00\x00z\x08<string>\xda\x08<module>\x01\x00\x00\x00s\n\x00\x00\x00\x08\x01\x02\x01\x0e\x01\x06\x01\x0e\x01"[..], (3, 10).into()).unwrap();

        let instructions = match program {
            CodeObject::V310(code) => code.code,
            _ => unreachable!(),
        };

        let cfg = create_cfg(instructions.to_vec(), None).unwrap();

        let cfg = simple_cfg_to_ext_cfg::<
            crate::v310::instructions::Instruction,
            crate::v310::ext_instructions::ExtInstruction,
            crate::v310::ext_instructions::ExtInstructions,
            crate::v310::opcodes::Opcode,
        >(&cfg)
        .unwrap();

        println!("{}", cfg.make_dot_graph());

        let ir_cfg = cfg_to_ir::<
            crate::v310::ext_instructions::ExtInstruction,
            crate::v310::opcodes::sir::SIRNode,
            crate::v310::opcodes::sir::SIRException,
            crate::v310::opcodes::Opcode,
        >(&cfg)
        .unwrap();

        println!("{}", ir_cfg.make_dot_graph());
        insta::assert_debug_snapshot!(ir_cfg);
    }

    #[test]
    fn test_exception_block_311() {
        // print("hi")
        // try:
        //     print(1)
        // except:
        //     print(2)
        let program = crate::load_code(&b"\xe3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\xf3Z\x00\x00\x00\x97\x00\x02\x00e\x00d\x00\xa6\x01\x00\x00\xab\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\t\x00\x02\x00e\x00d\x01\xa6\x01\x00\x00\xab\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00d\x03S\x00#\x00\x01\x00\x02\x00e\x00d\x02\xa6\x01\x00\x00\xab\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00Y\x00d\x03S\x00x\x03Y\x00w\x01)\x04\xda\x02hi\xe9\x01\x00\x00\x00\xe9\x02\x00\x00\x00N)\x01\xda\x05print\xa9\x00\xf3\x00\x00\x00\x00z\x08<string>\xfa\x08<module>r\x08\x00\x00\x00\x01\x00\x00\x00sD\x00\x00\x00\xf0\x03\x01\x01\x01\xe0\x00\x05\x80\x05\x80d\x81\x0b\x84\x0b\x80\x0b\xf0\x02\x03\x01\r\xd8\x04\t\x80E\x88!\x81H\x84H\x80H\x80H\x80H\xf8\xf0\x02\x01\x01\r\xd8\x04\t\x80E\x88!\x81H\x84H\x80H\x80H\x80H\x80H\xf8\xf8\xf8s\x08\x00\x00\x00\x8d\x0b\x1a\x00\x9a\r*\x03"[..], (3, 11).into()).unwrap();

        let (instructions, exception_table) = match program {
            CodeObject::V311(code) => (code.code.clone(), code.exception_table().unwrap()),
            _ => unreachable!(),
        };

        let cfg = create_cfg(instructions.to_vec(), Some(exception_table)).unwrap();

        let cfg = simple_cfg_to_ext_cfg::<
            Instruction,
            ExtInstruction,
            ExtInstructions,
            v311::opcodes::BranchReason,
        >(&cfg)
        .unwrap();

        let ir_cfg =
            cfg_to_ir::<ExtInstruction, SIRNode, SIRException, v311::opcodes::BranchReason>(&cfg)
                .unwrap();

        println!("{}", ir_cfg.make_dot_graph());
    }
}
