use std::{collections::HashMap, hash::BuildHasherDefault, ops::Index};

#[cfg(feature = "dot")]
use crate::utils::BlockKind;
use crate::{
    cfg::{BlockIndex, BlockIndexInfo, BranchEdge, ControlFlowGraph},
    sir_passes::RemoveSinglePhiNodes,
    traits::{
        BlockSliceExt, BranchReasonTrait, GenericInstruction, GenericOpcode, GenericSIRException,
        GenericSIRNode, SIRCFGPass, SIROwned,
    },
    utils::{InfiniteStack, generate_var_name},
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

impl std::fmt::Display for AuxVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self)
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct StackItem {
    pub name: &'static str,
    pub count: u32,
    /// Index of the (first) item on the stack (0 = TOS)
    pub index: isize,
}

#[derive(PartialEq, Debug, Clone)]
pub enum SIRExpression<SIRNode: GenericSIRNode> {
    Call(Call<SIRNode>),
    Exception(ExceptionCall<SIRNode>),
    PhiNode(Vec<AuxVar>),
    /// This value is used to represent the value that comes from the start of a generator
    /// In practice this should be `None` but we don't care about that
    GeneratorStart,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Call<SIRNode: GenericSIRNode> {
    pub node: SIRNode,
    pub stack_inputs: Vec<AuxVar>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct ExceptionCall<SIRNode: GenericSIRNode> {
    pub exception: SIRNode::SIRException,
    pub stack_inputs: Vec<AuxVar>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum SIRStatement<SIRNode: GenericSIRNode> {
    Assignment(AuxVar, SIRExpression<SIRNode>),
    TupleAssignment(Vec<AuxVar>, SIRExpression<SIRNode>),
    /// For when there is no output value (or it is not used)
    DisregardCall(Call<SIRNode>),
    /// This is used to "pop" variables when a stack depth was forced by an exception.
    UseVar(AuxVar),
}

#[derive(PartialEq, Debug, Clone)]
pub struct SIR<SIRNode: GenericSIRNode>(pub Vec<SIRStatement<SIRNode>>);

impl<SIRNode: GenericSIRNode> SIR<SIRNode> {
    /// Returns the indexes of the statements where the var is used
    pub fn find_var_usages(&self, var: &AuxVar) -> Vec<usize> {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(i, v)| match v {
                SIRStatement::DisregardCall(Call {
                    node: _,
                    stack_inputs,
                }) => stack_inputs.contains(&var).then(|| i),
                SIRStatement::UseVar(_) => None,
                SIRStatement::Assignment(_, expr) | SIRStatement::TupleAssignment(_, expr) => {
                    match expr {
                        SIRExpression::Call(Call {
                            node: _,
                            stack_inputs,
                        }) => stack_inputs.contains(&var).then(|| i),
                        SIRExpression::Exception(ExceptionCall {
                            exception: _,
                            stack_inputs,
                        }) => stack_inputs.contains(&var).then(|| i),
                        SIRExpression::PhiNode(vars) => vars.contains(&var).then(|| i),
                        SIRExpression::GeneratorStart => None,
                    }
                }
            })
            .collect()
    }

    pub fn is_var_used(&self, var: &AuxVar) -> bool {
        self.0.iter().any(|v| match v {
            SIRStatement::DisregardCall(Call {
                node: _,
                stack_inputs,
            }) => stack_inputs.contains(&var),
            SIRStatement::UseVar(_) => false,
            SIRStatement::Assignment(_, expr) | SIRStatement::TupleAssignment(_, expr) => {
                match expr {
                    SIRExpression::Call(Call {
                        node: _,
                        stack_inputs,
                    }) => stack_inputs.contains(&var),
                    SIRExpression::Exception(ExceptionCall {
                        exception: _,
                        stack_inputs,
                    }) => stack_inputs.contains(&var),
                    SIRExpression::PhiNode(vars) => vars.contains(&var),
                    SIRExpression::GeneratorStart => false,
                }
            }
        })
    }

    /// Finds the definition for single assignments of a var
    pub fn get_var_single_definition(&self, var: &AuxVar) -> Option<SIRExpression<SIRNode>> {
        self.0.iter().find_map(|v| match v {
            SIRStatement::DisregardCall(_) => None,
            SIRStatement::UseVar(_) => None,
            SIRStatement::TupleAssignment(_, _) => None,
            SIRStatement::Assignment(assigned_var, expr) => {
                if var == assigned_var {
                    Some(expr.clone())
                } else {
                    None
                }
            }
        })
    }

    /// Finds the definition for tuple assignments of a var
    pub fn get_var_tuple_definition(&self, var: &AuxVar) -> Option<SIRExpression<SIRNode>> {
        self.0.iter().find_map(|v| match v {
            SIRStatement::DisregardCall(_) => None,
            SIRStatement::UseVar(_) => None,
            SIRStatement::Assignment(_, _) => None,
            SIRStatement::TupleAssignment(assigned_vars, expr) => {
                for assigned_var in assigned_vars {
                    if var == assigned_var {
                        return Some(expr.clone());
                    }
                }

                None
            }
        })
    }
}

impl<SIRNode: GenericSIRNode> From<Vec<SIRStatement<SIRNode>>> for SIR<SIRNode> {
    fn from(value: Vec<SIRStatement<SIRNode>>) -> Self {
        SIR(value)
    }
}

fn process_phi_item<SIRNode>(
    phi_var: &AuxVar,
    stack_item: &StackValue,
    statements: &mut [SIRStatement<SIRNode>],
) -> bool
where
    SIRNode: GenericSIRNode,
{
    let mut found = false;

    // Add the item to the phi node
    // Loop both the branch statements and the actual basic block statements
    for statement in statements.iter_mut() {
        match (statement, stack_item) {
            (
                SIRStatement::Assignment(var, SIRExpression::PhiNode(values)),
                StackValue::AuxVar(item),
            ) => {
                if var == phi_var {
                    values.push(item.clone());
                    found = true;
                }
            }
            (SIRStatement::Assignment(var, value), StackValue::GeneratorStart) => {
                if var == phi_var {
                    *value = SIRExpression::GeneratorStart;
                    found = true;
                }
            }
            _ => {}
        }
    }

    found
}

fn fill_phi_nodes<SIRNode: GenericSIRNode>(
    curr_stack: &mut [StackValue],
    statements: &mut [SIRStatement<SIRNode>],
    phi_stack: &InfiniteStack<AuxVar>,
) -> Result<(), Error> {
    for (i, var) in phi_stack.data.iter_pairs() {
        let found;

        let item_index = (curr_stack.len() as isize + i) as usize;

        if let Some(item) = curr_stack.get(item_index) {
            found = process_phi_item(var, item, statements);
        } else {
            return Err(Error::InvalidStackAccess);
        }

        if !found {
            return Err(Error::PhiNodeNotPopulated);
        }
    }

    Ok(())
}

fn insert_replace_stack<T>(
    phi_stack: &mut InfiniteStack<T>,
    index: isize,
    value: T,
    allow_reusage: bool,
) -> Result<(), Error>
where
    T: Clone + std::fmt::Debug + PartialEq,
{
    let entry = phi_stack.data.get_mut(index);
    if let Some(Some(val)) = entry {
        if allow_reusage {
            *val = value;
        } else {
            return Err(Error::StackItemReused);
        }
    } else if let Some(val) = entry
        && val.is_none()
    {
        *val = Some(value);
    } else {
        phi_stack.data.insert(index, value);
    }

    Ok(())
}

fn process_stack_effects<SIRNode>(
    inputs: &[StackItem],
    outputs: &[StackItem],
    net_stack_delta: isize,
    stack: &mut InfiniteStack<AuxVar>,
    phi_stack: &mut InfiniteStack<AuxVar>,
    names: &mut HashMap<&'static str, u32>,
) -> Result<(Vec<AuxVar>, Vec<AuxVar>, Vec<SIRStatement<SIRNode>>), Error>
where
    SIRNode: GenericSIRNode,
{
    let mut stack_inputs = vec![];

    let mut statements = vec![];

    let mut tos = stack.get_tos_index().unwrap_or(0);

    for input in inputs {
        for count in 0..input.count {
            let index = (stack.carrot) + (input.index + count as isize);

            let value = stack.data.get_mut(index);

            if index < 0 && (value.is_none() || value == Some(&mut None)) {
                // Stack item from other basic block, create an empty phi node that we populate later.

                let phi = SIRExpression::PhiNode(vec![]);

                let var = AuxVar {
                    name: generate_var_name("phi", names),
                };

                statements.push(SIRStatement::Assignment(var.clone(), phi));
                stack_inputs.push(var.clone());

                insert_replace_stack(phi_stack, index, var, false)?;
            } else {
                match value {
                    Some(Some(value)) => {
                        stack_inputs.push(value.clone());
                    }
                    _ => {
                        unreachable!()
                    }
                }

                if index != tos {
                    // This means the value is not at the top of the stack
                    // We will place None here since it is guaranteed that in the outputs we will write a value here

                    *value.unwrap() = None;
                } else {
                    if index > 0 {
                        stack.data.remove(tos);
                    } else {
                        // Don't remove in the negative part, write None instead
                        *value.unwrap() = None;
                    }

                    tos -= 1;
                }
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

            let index = stack.carrot + (output.index as i32 + count as i32) as isize;

            let value = stack.data.get(index);

            if index < 0 && (value.is_none() || value == Some(&None)) {
                debug_assert!(phi_stack.data.get(index).is_some());
            }

            insert_replace_stack(stack, index, var, true)?;
        }
    }

    // Move carrot to the new correct position
    stack.carrot += net_stack_delta;

    Ok((stack_inputs, stack_outputs, statements))
}

fn instruction_to_ir<ExtInstruction, SIRNode>(
    opcode: ExtInstruction::Opcode,
    oparg: ExtInstruction::OpargType,
    jump: bool,
    stack: &mut InfiniteStack<AuxVar>,
    phi_stack: &mut InfiniteStack<AuxVar>,
    names: &mut HashMap<&'static str, u32>,
) -> Result<(Vec<SIRStatement<SIRNode>>, SIRStatement<SIRNode>), Error>
where
    ExtInstruction: GenericInstruction<OpargType = u32>,
    SIRNode: GenericSIRNode<Opcode = ExtInstruction::Opcode>,
{
    let node = SIRNode::new(opcode.clone(), oparg, jump);

    let (stack_inputs, stack_outputs, statements) = process_stack_effects(
        node.get_inputs(),
        node.get_outputs(),
        node.get_net_stack_delta(),
        stack,
        phi_stack,
        names,
    )?;

    let call = Call::<SIRNode> { node, stack_inputs };

    let instruction_statement = if stack_outputs.len() > 1 {
        SIRStatement::TupleAssignment(stack_outputs, SIRExpression::Call(call))
    } else if !stack_outputs.is_empty() {
        SIRStatement::Assignment(
            stack_outputs.first().unwrap().clone(),
            SIRExpression::Call(call),
        )
    } else {
        SIRStatement::DisregardCall(call)
    };

    Ok((statements, instruction_statement))
}

fn exception_to_ir<SIRNode>(
    lasti: bool,
    stack_depth: usize,
    jump: bool,
    stack: &mut InfiniteStack<AuxVar>,
    phi_stack: &mut InfiniteStack<AuxVar>,
    names: &mut HashMap<&'static str, u32>,
) -> Result<Vec<SIRStatement<SIRNode>>, Error>
where
    SIRNode: GenericSIRNode,
{
    let exception = SIRNode::SIRException::new(lasti, stack_depth, jump);

    let (stack_inputs, stack_outputs, mut statements) = process_stack_effects(
        exception.get_inputs(),
        exception.get_outputs(),
        exception.get_net_stack_delta(),
        stack,
        phi_stack,
        names,
    )?;

    let call = ExceptionCall::<SIRNode> {
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

fn force_exception_stack_depth<SIRNode: GenericSIRNode>(
    stack_depth: usize,
    stack: &mut [StackValue],
    phi_stack: &mut InfiniteStack<AuxVar>,
    names: &mut HashMap<&'static str, u32>,
) -> Result<Vec<SIRStatement<SIRNode>>, Error> {
    let mut statements: Vec<SIRStatement<SIRNode>> = vec![];

    // Exceptions can force a stack depth, so we need to pop all items until we reach this stack height
    assert!(stack.len() >= stack_depth);

    // We start counting from the TOS (on the negative side)
    let mut index = 0;

    while stack.len() - (-index) as usize > stack_depth {
        index -= 1;
        let phi = SIRExpression::PhiNode(vec![]);

        let var = AuxVar {
            name: generate_var_name("phi", names),
        };

        statements.push(SIRStatement::Assignment(var.clone(), phi));
        statements.push(SIRStatement::UseVar(var.clone()));

        insert_replace_stack(phi_stack, index, var, false)?;
    }

    Ok(statements)
}

/// Internal function that is used while converting an Ext CFG to SIR nodes.
/// This function is meant to process a single block.
fn bb_to_ir<ExtInstruction, SIRNode>(
    instructions: &[ExtInstruction],
    names: &mut HashMap<&'static str, u32>,
) -> Result<(SIR<SIRNode>, InfiniteStack<AuxVar>, InfiniteStack<AuxVar>), Error>
where
    ExtInstruction: GenericInstruction<OpargType = u32>,
    SIR<SIRNode>: SIROwned<SIRNode>,
    SIRNode: GenericSIRNode<Opcode = ExtInstruction::Opcode>,
{
    let mut phi_statements: Vec<SIRStatement<SIRNode>> = vec![];
    let mut instruction_statements: Vec<SIRStatement<SIRNode>> = vec![];

    // Every basic block starts with an empty stack.
    // When we try to access stack items below 0, we know it's accessing items from a different basic block.
    let mut stack: InfiniteStack<AuxVar> = vec![].into();

    // When we assign a phi node to a var we keep track of what stack index this phi node is representing
    let mut phi_stack: InfiniteStack<AuxVar> = vec![].into();

    for instruction in instructions {
        // In a basic block there shouldn't be any jumps. (the last jump instruction is removed in the cfg)
        debug_assert!(!instruction.is_jump());

        let (statements, instruction_statement) = instruction_to_ir::<ExtInstruction, SIRNode>(
            instruction.get_opcode(),
            instruction.get_raw_value(),
            false,
            &mut stack,
            &mut phi_stack,
            names,
        )?;

        phi_statements.extend(statements);

        instruction_statements.push(instruction_statement);
    }

    phi_statements.extend(instruction_statements);

    Ok((SIR::new(phi_statements), stack, phi_stack))
}

/// Used to represent the opcode that was used for this branch and the block index it's jumping to.
/// We do this so the value of the branch instruction cannot represent a wrong index.
/// This also shows which inputs and outputs the opcode uses.
#[derive(Debug, Clone, PartialEq)]
pub struct SIRBranchEdge<SIRNode: GenericSIRNode> {
    pub reason: <SIRNode::Opcode as GenericOpcode>::BranchReason,
    pub statements: Option<SIR<SIRNode>>,
    pub block_index: BlockIndex,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SIRBlockIndexInfo<SIRNode: GenericSIRNode> {
    Edge(SIRBranchEdge<SIRNode>),
    /// For blocks that fallthrough with no opcode (cannot be generated by Python, used by internal algorithms)
    Fallthrough(BlockIndex),
    /// For blocks without a target
    NoIndex,
}

impl<SIRNode: GenericSIRNode> SIRBlockIndexInfo<SIRNode> {
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
pub struct SIRNormalBlock<SIRNode>
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
/// Represents an exception block in the control flow graph
/// This includes a list of block indexes that belong to this exception block
pub struct SIRExceptionBlock<SIRNode: GenericSIRNode> {
    pub block_indexes: Vec<usize>,
    /// Index to the exception handler block
    pub exception_handler: SIRBranchEdge<SIRNode>,
    /// Index to default block
    pub default_block: SIRBlockIndexInfo<SIRNode>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SIRBlock<SIRNode: GenericSIRNode> {
    NormalBlock(SIRNormalBlock<SIRNode>),
    ExceptionBlock(SIRExceptionBlock<SIRNode>),
}

impl<SIRNode: GenericSIRNode> SIRBlock<SIRNode> {
    pub fn get_branch_block(&self) -> SIRBlockIndexInfo<SIRNode> {
        match self {
            SIRBlock::NormalBlock(block) => block.branch_block.clone(),
            SIRBlock::ExceptionBlock(block) => {
                SIRBlockIndexInfo::Edge(block.exception_handler.clone())
            }
        }
    }

    pub fn get_default_block(&self) -> SIRBlockIndexInfo<SIRNode> {
        match self {
            SIRBlock::NormalBlock(block) => block.default_block.clone(),
            SIRBlock::ExceptionBlock(block) => block.default_block.clone(),
        }
    }

    pub fn get_nodes_mut(&mut self) -> Option<&mut SIR<SIRNode>> {
        match self {
            SIRBlock::NormalBlock(block) => Some(&mut block.nodes),
            SIRBlock::ExceptionBlock(_) => None,
        }
    }

    pub fn get_nodes(self) -> Option<SIR<SIRNode>> {
        match self {
            SIRBlock::NormalBlock(block) => Some(block.nodes),
            SIRBlock::ExceptionBlock(_) => None,
        }
    }

    pub fn get_nodes_ref(&self) -> Option<&SIR<SIRNode>> {
        match self {
            SIRBlock::NormalBlock(block) => Some(&block.nodes),
            SIRBlock::ExceptionBlock(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SIRControlFlowGraph<SIRNode: GenericSIRNode> {
    pub blocks: Vec<SIRBlock<SIRNode>>,
    pub start_index: SIRBlockIndexInfo<SIRNode>,
}

impl<SIRNode: GenericSIRNode> BlockSliceExt<SIRNode> for [SIRBlock<SIRNode>] {
    fn find_exception_block(&self, index_to_search: usize) -> Option<usize> {
        for (i, block) in self.iter().enumerate() {
            match block {
                SIRBlock::ExceptionBlock(block) => {
                    if block.block_indexes.contains(&index_to_search) {
                        return Some(i);
                    }
                }
                _ => continue,
            }
        }

        None
    }
}

impl<SIRNode: GenericSIRNode> SIRControlFlowGraph<SIRNode> {
    pub fn find_exception_block(&self, index_to_search: usize) -> Option<usize> {
        self.blocks.find_exception_block(index_to_search)
    }
}

#[cfg(feature = "dot")]
impl<SIRNode: GenericSIRNode> SIRControlFlowGraph<SIRNode> {
    fn add_block(
        graph: &mut petgraph::Graph<(BlockKind, String), String>,
        blocks: &[SIRBlock<SIRNode>],
        block_index: Option<BlockIndex>,
        block_map: &mut HashMap<Option<BlockIndex>, NodeIndex>,
    ) -> Option<NodeIndex>
    where
        SIR<SIRNode>: SIROwned<SIRNode>,
    {
        let (block, index) = match &block_index {
            Some(BlockIndex::Index(index)) => (blocks.get(*index).unwrap(), index),
            _ => return None,
        };

        let text = match block {
            SIRBlock::NormalBlock(block) => {
                format!("{}", block.nodes)
            }
            SIRBlock::ExceptionBlock(_) => "EXCEPTION".to_string(),
        };

        let kind = match block {
            SIRBlock::NormalBlock(_) => {
                if blocks.find_exception_block(*index).is_some() {
                    BlockKind::InExceptionRange
                } else {
                    BlockKind::NormalBlock
                }
            }
            SIRBlock::ExceptionBlock(_) => BlockKind::ExceptionBlock,
        };

        let index = if let std::collections::hash_map::Entry::Vacant(e) =
            block_map.entry(block_index.clone())
        {
            let index = graph.add_node((kind.clone(), text));
            e.insert(index);

            index
        } else {
            block_map[&block_index]
        };

        let (branch_index, branch_statements) = match block.get_branch_block() {
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
            let index = Self::add_block(graph, blocks, branch_index.clone(), block_map);

            if let Some(index) = index {
                block_map.insert(branch_index, index);
                Some(index)
            } else {
                match branch_index {
                    Some(BlockIndex::InvalidIndex(invalid_index)) => Some(graph.add_node((
                        kind.clone(),
                        format!("invalid jump to index {}", invalid_index),
                    ))),
                    Some(BlockIndex::Index(_)) => unreachable!(),
                    None => None,
                }
            }
        };

        let (default_index, default_statements) = match block.get_default_block() {
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
            let index = Self::add_block(graph, blocks, default_index.clone(), block_map);

            if let Some(index) = index {
                block_map.insert(default_index, index);
                Some(index)
            } else {
                match default_index {
                    Some(BlockIndex::InvalidIndex(invalid_index)) => Some(
                        graph.add_node((kind, format!("invalid jump to index {}", invalid_index))),
                    ),
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
        SIR<SIRNode>: SIROwned<SIRNode>,
    {
        let mut graph = petgraph::Graph::<(BlockKind, String), String>::new();

        Self::add_block(
            &mut graph,
            &self.blocks,
            self.start_index.get_block_index().cloned(),
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

                    format!(
                        r#"label = "{}", color = {}, shape=rect"#,
                        e.weight().replace("\n", r"\l"),
                        color
                    )
                },
                &|_, (_, (kind, s))| {
                    let label = s.replace("\n", r"\l");

                    match kind {
                        BlockKind::NormalBlock => {
                            format!(r#"shape=rect, label="{}""#, label)
                        }
                        BlockKind::ExceptionBlock => {
                            format!(
                                r#"shape=rect, style=filled, fillcolor=orange, label="{}""#,
                                label
                            )
                        }
                        BlockKind::InExceptionRange => {
                            format!(r#"shape=rect, color=orange, label="{}""#, label)
                        }
                    }
                },
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
fn extend_merge_stack(
    vec_to_extend: &mut Vec<StackValue>,
    infinite_stack: &InfiniteStack<AuxVar>,
    phi_stack: &InfiniteStack<AuxVar>,
) -> Result<(), Error> {
    let original_len = vec_to_extend.len();

    let pairs = infinite_stack.data.iter_pairs();

    for (i, item) in pairs {
        let real_index: usize = (original_len as isize + i).try_into().unwrap();

        if i < 0 {
            vec_to_extend[real_index] = StackValue::AuxVar(item.clone());
        } else {
            debug_assert!(real_index == vec_to_extend.len());

            vec_to_extend.push(StackValue::AuxVar(item.clone()));
        }
    }

    for (i, _) in phi_stack.data.iter_pairs().rev() {
        let real_index: usize = (original_len as isize + i).try_into().unwrap();

        let value = infinite_stack.data.get(i);

        if i < 0
            && real_index == vec_to_extend.len() - 1
            && (value.is_none() || value == Some(&None))
        {
            // if we use the TOS as phi item we pop

            vec_to_extend.pop();
        }
    }

    Ok(())
}

#[derive(Debug, PartialEq, Clone)]
enum StackValue {
    AuxVar(AuxVar),
    GeneratorStart,
}

/// In certain versions a generator starts with a value on the stack so we need to account for that
pub fn cfg_to_ir<ExtInstruction, SIRNode>(
    cfg: &ControlFlowGraph<ExtInstruction>,
    add_generator_value: bool,
) -> Result<SIRControlFlowGraph<SIRNode>, Error>
where
    ExtInstruction: GenericInstruction<OpargType = u32>,
    SIRNode: GenericSIRNode<Opcode = ExtInstruction::Opcode>,
    <ExtInstruction::Opcode as GenericOpcode>::BranchReason:
        BranchReasonTrait<Opcode = ExtInstruction::Opcode>,
    SIR<SIRNode>: SIROwned<SIRNode>,
{
    #[derive(Debug, Clone)]
    struct NormalTempBlockInfo<SIRNode: GenericSIRNode>
    where
        SIRNode: Clone + std::fmt::Debug,
    {
        stack: InfiniteStack<AuxVar>,
        statements: SIR<SIRNode>,
        phi_stack: InfiniteStack<AuxVar>,
    }

    enum TempBlockInfo<SIRNode: GenericSIRNode> {
        NormalBlock(NormalTempBlockInfo<SIRNode>),
        ExceptionBlock,
    }

    let mut temp_blocks: Vec<TempBlockInfo<SIRNode>> = Vec::with_capacity(cfg.blocks.len());

    // Keep track of which blocks we already processed
    let mut visited_blocks: Vec<usize> = Vec::with_capacity(cfg.blocks.len());

    #[derive(Debug, Clone)]
    struct TempEdgeInfo<SIRNode: GenericSIRNode> {
        stack: InfiniteStack<AuxVar>,
        statements: Vec<SIRStatement<SIRNode>>,
        phi_stack: InfiniteStack<AuxVar>,
    }

    // (default, branch)
    let mut temp_edges: Vec<(TempEdgeInfo<SIRNode>, TempEdgeInfo<SIRNode>)> =
        Vec::with_capacity(cfg.blocks.len());

    let mut names = HashMap::new();

    // Keeps track of the different stacks used to enter a basic block
    // TODO: We have to find a way to merge stacks
    let mut has_changed: nohash_hasher::IntMap<usize, Vec<Vec<StackValue>>> =
        nohash_hasher::IntMap::with_capacity_and_hasher(
            cfg.blocks.len(),
            BuildHasherDefault::default(),
        );

    // Create isolated IR blocks
    for block in &cfg.blocks {
        match block {
            crate::cfg::Block::NormalBlock(block) => {
                let (statements, stack, phi_stack) =
                    bb_to_ir::<ExtInstruction, SIRNode>(&block.instructions, &mut names)?;

                temp_blocks.push(TempBlockInfo::NormalBlock(NormalTempBlockInfo {
                    stack,
                    statements,
                    phi_stack,
                }));
            }
            crate::cfg::Block::ExceptionBlock(_) => temp_blocks.push(TempBlockInfo::ExceptionBlock),
        }

        let mut default_stack = vec![].into();
        let mut default_phi_stack = vec![].into();

        let default_statements = match &block.get_default_block() {
            BlockIndexInfo::Edge(BranchEdge {
                reason: branch_reason,
                ..
            }) => {
                if let Some(opcode) = branch_reason.get_opcode() {
                    let (mut statements, instruction_statement) =
                        instruction_to_ir::<ExtInstruction, SIRNode>(
                            opcode.clone(),
                            0, // The oparg doesn't matter for the stack effect in the case of a branch opcode (oparg is the jump target)
                            false, // Don't take the jump for the default block
                            &mut default_stack,
                            &mut default_phi_stack,
                            &mut names,
                        )?;

                    statements.push(instruction_statement);

                    statements
                } else if branch_reason.is_exception() {
                    let lasti = branch_reason.get_lasti().unwrap();
                    let stack_depth = branch_reason.get_stack_depth().unwrap();

                    exception_to_ir::<SIRNode>(
                        lasti,
                        stack_depth,
                        false,
                        &mut default_stack,
                        &mut default_phi_stack,
                        &mut names,
                    )?
                } else {
                    unreachable!()
                }
            }
            _ => vec![],
        };

        let mut branch_stack = vec![].into();
        let mut branch_phi_stack = vec![].into();

        let branch_statements = match &block.get_branch_block() {
            BlockIndexInfo::Edge(BranchEdge {
                reason: branch_reason,
                ..
            }) => {
                if let Some(opcode) = branch_reason.get_opcode() {
                    let (mut statements, instruction_statement) =
                        instruction_to_ir::<ExtInstruction, SIRNode>(
                            opcode.clone(),
                            0, // The oparg doesn't matter for the stack effect in the case of a branch opcode (oparg is the jump target)
                            true, // Take the jump for the branch block
                            &mut branch_stack,
                            &mut branch_phi_stack,
                            &mut names,
                        )?;

                    statements.push(instruction_statement);

                    statements
                } else if branch_reason.is_exception() {
                    let lasti = branch_reason.get_lasti().unwrap();
                    let stack_depth = branch_reason.get_stack_depth().unwrap();

                    exception_to_ir::<SIRNode>(
                        lasti,
                        stack_depth,
                        true,
                        &mut branch_stack,
                        &mut branch_phi_stack,
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
                phi_stack: default_phi_stack,
            },
            TempEdgeInfo {
                stack: branch_stack,
                statements: branch_statements,
                phi_stack: branch_phi_stack,
            },
        ));
    }

    #[derive(Debug)]
    struct BlockQueue<SIRNode: GenericSIRNode> {
        block_index: BlockIndexInfo<<SIRNode::Opcode as GenericOpcode>::BranchReason>,
        /// Only `None` when `block_index` is the first block.
        /// The `bool`` shows if we need to take the `branch` edge or not
        from_block_index: Option<(usize, bool)>,
        curr_stack: Vec<StackValue>,
    }

    // Fill the empty phi nodes with actual values
    let mut queue: Vec<BlockQueue<SIRNode>> = vec![BlockQueue {
        block_index: cfg.start_index.clone(),
        from_block_index: None,
        curr_stack: if add_generator_value {
            vec![StackValue::GeneratorStart]
        } else {
            vec![]
        },
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
            // We need to force the stack depth specified by the exception here
            if let (BlockIndexInfo::Edge(BranchEdge { reason, .. }), Some((_, is_branch))) =
                (&block_element.block_index, &block_element.from_block_index)
                && *is_branch
                && reason.is_exception()
                && reason.get_stack_depth().unwrap() < block_element.curr_stack.len()
            {
                let mut phi_stack = vec![].into();

                let mut statements = force_exception_stack_depth(
                    reason.get_stack_depth().unwrap(),
                    &mut block_element.curr_stack,
                    &mut phi_stack,
                    &mut names,
                )?;

                // Fill empty phi nodes in the new exception statements
                fill_phi_nodes(&mut block_element.curr_stack, &mut statements, &phi_stack)?;

                statements.extend(edge_info.statements.clone());
                edge_info.statements = statements;

                // This will pop the used phi items
                extend_merge_stack(&mut block_element.curr_stack, &vec![].into(), &phi_stack)?;

                assert_eq!(
                    reason.get_stack_depth().unwrap(),
                    block_element.curr_stack.len()
                )
            }

            // Fill empty phi nodes in the edge statements
            fill_phi_nodes(
                &mut block_element.curr_stack,
                &mut edge_info.statements,
                &edge_info.phi_stack,
            )?;

            // Add the branch stack after processing the branch statements
            extend_merge_stack(
                &mut block_element.curr_stack,
                &edge_info.stack,
                &edge_info.phi_stack,
            )?;
        }

        if already_analysed {
            if let Some(stacks) = has_changed.get_mut(&index) {
                if block_element
                    .curr_stack
                    .iter()
                    .enumerate()
                    .rev()
                    .all(|(i, v)| {
                        stacks.iter().any(|l| {
                            let relative_index = block_element.curr_stack.len() - i;
                            l.len() >= relative_index
                                && l.get(l.len() - relative_index).is_some()
                                && l[l.len() - relative_index] == *v
                        })
                    })
                {
                    // Already processed this block with these stack elements but we still processed the statements of the edge
                    continue;
                } else {
                    stacks.push(block_element.curr_stack.clone());
                }
            }
        } else {
            has_changed.insert(index, vec![block_element.curr_stack.clone()]);
        }

        if let TempBlockInfo::NormalBlock(block_info) = block_info {
            fill_phi_nodes(
                &mut block_element.curr_stack,
                &mut block_info.statements.0,
                &block_info.phi_stack,
            )?;

            extend_merge_stack(
                &mut block_element.curr_stack,
                &block_info.stack,
                &block_info.phi_stack,
            )?;
        }

        queue.push(BlockQueue {
            block_index: cfg.blocks.index(index).get_default_block().clone(),
            curr_stack: block_element.curr_stack.to_vec(),
            from_block_index: Some((index, false)),
        });

        queue.push(BlockQueue {
            block_index: cfg.blocks.index(index).get_branch_block().clone(),
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
        .map(|(index, v)| match v {
            TempBlockInfo::NormalBlock(v) => SIRBlock::NormalBlock(SIRNormalBlock {
                nodes: v.statements,
                branch_block: cfg.blocks[index].get_branch_block().into_sir(
                    if temp_edges.get(index).unwrap().1.statements.is_empty() {
                        None
                    } else {
                        Some(temp_edges.get(index).unwrap().1.statements.clone())
                    },
                ),
                default_block: cfg.blocks[index].get_default_block().into_sir(
                    if temp_edges.get(index).unwrap().0.statements.is_empty() {
                        None
                    } else {
                        Some(temp_edges.get(index).unwrap().0.statements.clone())
                    },
                ),
            }),
            TempBlockInfo::ExceptionBlock => {
                let block = match &cfg.blocks[index] {
                    crate::cfg::Block::ExceptionBlock(block) => block,
                    _ => unreachable!(),
                };

                let exception_handler = BlockIndexInfo::Edge(block.exception_handler.clone())
                    .into_sir(if temp_edges.get(index).unwrap().1.statements.is_empty() {
                        None
                    } else {
                        Some(temp_edges.get(index).unwrap().1.statements.clone())
                    });

                let exception_handler = match exception_handler {
                    SIRBlockIndexInfo::Edge(edge) => edge,
                    _ => unreachable!(),
                };

                SIRBlock::ExceptionBlock(SIRExceptionBlock {
                    block_indexes: block.block_indexes.clone(),
                    exception_handler,
                    default_block: block.default_block.clone().into_sir(
                        if temp_edges.get(index).unwrap().0.statements.is_empty() {
                            None
                        } else {
                            Some(temp_edges.get(index).unwrap().0.statements.clone())
                        },
                    ),
                })
            }
        })
        .collect::<Vec<_>>();

    let mut sir_cfg = SIRControlFlowGraph::<SIRNode> {
        start_index: SIRBlockIndexInfo::Fallthrough(BlockIndex::Index(0)),
        blocks,
    };

    RemoveSinglePhiNodes::new().run_on(&mut sir_cfg);

    Ok(sir_cfg)
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use crate::cfg::{create_cfg, simple_cfg_to_ext_cfg};
    use crate::sir::{
        AuxVar, StackItem, StackValue, bb_to_ir, cfg_to_ir, extend_merge_stack, fill_phi_nodes,
        instruction_to_ir, process_stack_effects,
    };
    use crate::traits::GenericInstruction;
    use crate::utils::{ExceptionTableEntry, InfiniteStack};
    use crate::v311::ext_instructions::ExtInstruction;
    use crate::v311::instructions::Instruction;
    use crate::v311::opcodes::sir::SIRNode;
    use crate::{CodeObject, v311};

    // #[test]
    // fn test_stack_processing() {
    //     let inputs = [
    //         StackItem {
    //             name: "first",
    //             count: 1,
    //             index: -3,
    //         },
    //         StackItem {
    //             name: "second",
    //             count: 1,
    //             index: 1,
    //         },
    //     ]
    //     .as_slice();

    //     let outputs = [
    //         StackItem {
    //             name: "res_first",
    //             count: 1,
    //             index: 3,
    //         },
    //         StackItem {
    //             name: "res_second",
    //             count: 1,
    //             index: 4,
    //         },
    //     ]
    //     .as_slice();

    //     let mut stack: crate::utils::InfiniteStack<_> = vec![].into();

    //     let fake_var = StackValue::AuxVar(AuxVar {
    //         name: "test".into(),
    //     });

    //     stack.data.insert(-2, fake_var.clone());

    //     assert_eq!(stack.data.negative_len(), 2);
    //     assert_eq!(stack.data.positive_len(), 0);
    //     assert_eq!(
    //         stack.data.iter().cloned().collect::<Vec<_>>(),
    //         vec![Some(fake_var), None]
    //     );

    //     let mut phi_stack = vec![].into();
    //     let mut names = HashMap::new();

    //     process_stack_effects::<SIRNode, SIRException>(
    //         inputs,
    //         outputs,
    //         0,
    //         &mut stack,
    //         &mut phi_stack,
    //         &mut names,
    //     )
    //     .unwrap();

    //     assert_eq!(phi_stack.data.len(), 1);
    //     assert_eq!(phi_stack.data.first().unwrap().index, -4);

    //     assert_eq!(stack.data.positive_len(), 1);
    //     assert_eq!(stack.data.negative_len(), 4);

    //     assert_eq!(
    //         stack.iter().cloned().collect::<Vec<_>>(),
    //         vec![
    //             Some(
    //                 StackValue::AuxVar(AuxVar {
    //                     name: "res_first_0".into(),
    //                 })
    //                 .into()
    //             ),
    //             None,
    //             Some(crate::sir::InfiniteStackItem::Deleted),
    //             None,
    //             Some(
    //                 StackValue::AuxVar(AuxVar {
    //                     name: "res_second_0".into(),
    //                 })
    //                 .into()
    //             ),
    //         ]
    //     );

    //     // New stack processing attempt

    //     let outputs = [
    //         StackItem {
    //             name: "const",
    //             count: 1,
    //             index: 0,
    //         },
    //         StackItem {
    //             name: "const",
    //             count: 1,
    //             index: 1,
    //         },
    //     ]
    //     .as_slice();

    //     let mut stack: InfiniteVec<_> = vec![].into();
    //     let mut phi_map = vec![];
    //     let mut names = HashMap::new();

    //     process_stack_effects::<SIRNode, SIRException>(
    //         &[],
    //         outputs,
    //         &mut stack,
    //         &mut phi_map,
    //         &mut names,
    //     )
    //     .unwrap();

    //     assert_eq!(stack.positive_len(), 2);
    //     assert_eq!(stack.negative_len(), 0);

    //     assert_eq!(
    //         stack.iter_pairs().collect::<Vec<_>>(),
    //         vec![
    //             (
    //                 0,
    //                 &StackValue::AuxVar(AuxVar {
    //                     name: "const_0".into()
    //                 })
    //                 .into()
    //             ),
    //             (
    //                 1,
    //                 &StackValue::AuxVar(AuxVar {
    //                     name: "const_1".into()
    //                 })
    //                 .into()
    //             )
    //         ]
    //     )
    // }

    // #[test]
    // fn test_call_stack_processing() {
    //     let inputs = [
    //         StackItem {
    //             name: "method_or_null",
    //             count: 1,
    //             index: 2,
    //         },
    //         StackItem {
    //             name: "self_or_callable",
    //             count: 1,
    //             index: 1,
    //         },
    //         StackItem {
    //             name: "args",
    //             count: 1,
    //             index: 0,
    //         },
    //     ]
    //     .as_slice();

    //     let outputs = [StackItem {
    //         name: "res",
    //         count: 1,
    //         index: 0,
    //     }]
    //     .as_slice();

    //     let mut stack: InfiniteVec<_> = vec![].into();

    //     stack.push(
    //         StackValue::AuxVar(AuxVar {
    //             name: "null_0".into(),
    //         })
    //         .into(),
    //     );

    //     stack.push(
    //         StackValue::AuxVar(AuxVar {
    //             name: "value_0".into(),
    //         })
    //         .into(),
    //     );

    //     stack.push(
    //         StackValue::AuxVar(AuxVar {
    //             name: "value_1".into(),
    //         })
    //         .into(),
    //     );

    //     assert_eq!(stack.negative_len(), 0);
    //     assert_eq!(stack.positive_len(), 3);

    //     let mut phi_map = vec![];
    //     let mut names = HashMap::new();

    //     process_stack_effects::<SIRNode, SIRException>(
    //         inputs,
    //         outputs,
    //         &mut stack,
    //         &mut phi_map,
    //         &mut names,
    //     )
    //     .unwrap();

    //     assert_eq!(phi_map.len(), 0);

    //     assert_eq!(stack.positive_len(), 1);
    //     assert_eq!(stack.negative_len(), 0);

    //     assert_eq!(
    //         stack.iter().cloned().collect::<Vec<_>>(),
    //         vec![Some(
    //             StackValue::AuxVar(AuxVar {
    //                 name: "res_0".into(),
    //             })
    //             .into()
    //         ),]
    //     );

    //     // POP_TOP the result

    //     let inputs = [StackItem {
    //         name: "top",
    //         count: 1,
    //         index: 0,
    //     }]
    //     .as_slice();

    //     process_stack_effects::<SIRNode, SIRException>(
    //         inputs,
    //         &[],
    //         &mut stack,
    //         &mut phi_map,
    //         &mut names,
    //     )
    //     .unwrap();

    //     assert_eq!(stack.positive_len(), 0);
    //     assert_eq!(stack.negative_len(), 0);
    // }

    // #[test]
    // fn test_middle_stack_processing() {
    //     // In this test we will process stack usages below 0 that appear after the first instruction

    //     let inputs = [StackItem {
    //         name: "top",
    //         count: 1,
    //         index: 0,
    //     }]
    //     .as_slice();

    //     let mut stack: InfiniteVec<_> = vec![].into();

    //     let mut phi_map = vec![];
    //     let mut names = HashMap::new();

    //     // Pop all 3 values
    //     for _ in 0..3 {
    //         process_stack_effects::<SIRNode, SIRException>(
    //             inputs,
    //             &[],
    //             &mut stack,
    //             &mut phi_map,
    //             &mut names,
    //         )
    //         .unwrap();
    //     }

    //     assert_eq!(phi_map.len(), 3);

    //     dbg!(phi_map);

    //     assert_eq!(stack.positive_len(), 0);
    //     assert_eq!(stack.negative_len(), 0);

    //     // Pop all in the same input with a filled phi map

    //     let inputs = [
    //         StackItem {
    //             name: "first",
    //             count: 1,
    //             index: 2,
    //         },
    //         StackItem {
    //             name: "second",
    //             count: 1,
    //             index: 1,
    //         },
    //         StackItem {
    //             name: "third",
    //             count: 1,
    //             index: 0,
    //         },
    //     ]
    //     .as_slice();

    //     let mut stack: InfiniteVec<_> = vec![].into();

    //     let mut phi_map = vec![
    //         PhiEntry::new(
    //             -1,
    //             AuxVar {
    //                 name: "test_1".into(),
    //             },
    //         ),
    //         PhiEntry::new(
    //             -2,
    //             AuxVar {
    //                 name: "test_2".into(),
    //             },
    //         ),
    //         PhiEntry::new(
    //             -3,
    //             AuxVar {
    //                 name: "test_3".into(),
    //             },
    //         ),
    //     ];
    //     let mut names = HashMap::new();

    //     // Pop all 3 values
    //     process_stack_effects::<SIRNode, SIRException>(
    //         inputs,
    //         &[],
    //         &mut stack,
    //         &mut phi_map,
    //         &mut names,
    //     )
    //     .unwrap();

    //     let mut phi_indexes = phi_map.iter().cloned().collect::<Vec<_>>();

    //     phi_indexes.sort_by_key(|PhiEntry { index: i, .. }| *i);

    //     assert_eq!(
    //         phi_indexes,
    //         vec![
    //             PhiEntry::new(
    //                 -6,
    //                 AuxVar {
    //                     name: "phi_0".into()
    //                 }
    //             ),
    //             PhiEntry::new(
    //                 -5,
    //                 AuxVar {
    //                     name: "phi_1".into()
    //                 }
    //             ),
    //             PhiEntry::new(
    //                 -4,
    //                 AuxVar {
    //                     name: "phi_2".into()
    //                 }
    //             ),
    //             PhiEntry::new(
    //                 -3,
    //                 AuxVar {
    //                     name: "test_3".into()
    //                 }
    //             ),
    //             PhiEntry::new(
    //                 -2,
    //                 AuxVar {
    //                     name: "test_2".into()
    //                 }
    //             ),
    //             PhiEntry::new(
    //                 -1,
    //                 AuxVar {
    //                     name: "test_1".into()
    //                 }
    //             )
    //         ]
    //     );

    //     // Pop all in the same input with an empty phi map

    //     let inputs = [
    //         StackItem {
    //             name: "first",
    //             count: 1,
    //             index: 2,
    //         },
    //         StackItem {
    //             name: "second",
    //             count: 1,
    //             index: 1,
    //         },
    //         StackItem {
    //             name: "third",
    //             count: 1,
    //             index: 0,
    //         },
    //     ]
    //     .as_slice();

    //     let mut stack: InfiniteVec<_> = vec![].into();

    //     let mut phi_map = vec![];
    //     let mut names = HashMap::new();

    //     // Pop all 3 values
    //     process_stack_effects::<SIRNode, SIRException>(
    //         inputs,
    //         &[],
    //         &mut stack,
    //         &mut phi_map,
    //         &mut names,
    //     )
    //     .unwrap();

    //     let mut phi_indexes = phi_map.iter().cloned().collect::<Vec<_>>();

    //     phi_indexes.sort_by_key(|PhiEntry { index: i, .. }| *i);

    //     assert_eq!(
    //         phi_indexes,
    //         vec![
    //             PhiEntry::new(
    //                 -3,
    //                 AuxVar {
    //                     name: "phi_0".into()
    //                 }
    //             ),
    //             PhiEntry::new(
    //                 -2,
    //                 AuxVar {
    //                     name: "phi_1".into()
    //                 }
    //             ),
    //             PhiEntry::new(
    //                 -1,
    //                 AuxVar {
    //                     name: "phi_2".into()
    //                 }
    //             ),
    //         ]
    //     );
    // }

    #[test]
    fn test_merge_stack() {
        let instructions = vec![
            crate::v311::instructions::Instruction::ForIter(0),
            crate::v311::instructions::Instruction::StoreName(0),
        ];

        let mut curr_stack = vec![StackValue::AuxVar(AuxVar {
            name: "og_iter".into(),
        })];

        let mut statements = vec![];

        let mut stack: InfiniteStack<_> = vec![].into();
        let mut phi_stack = vec![].into();
        let mut names = HashMap::new();

        for instruction in instructions {
            // Pop all 3 values
            let (stmts, inst_stmt) =
                instruction_to_ir::<crate::v311::ext_instructions::ExtInstruction, SIRNode>(
                    instruction.get_opcode(),
                    0,
                    false,
                    &mut stack,
                    &mut phi_stack,
                    &mut names,
                )
                .unwrap();

            statements.extend(stmts);
            statements.push(inst_stmt);
        }

        fill_phi_nodes(&mut curr_stack, &mut statements, &phi_stack).unwrap();

        extend_merge_stack(&mut curr_stack, &stack, &phi_stack).unwrap();

        assert_eq!(curr_stack.len(), 1);
        assert_eq!(
            *curr_stack.first().unwrap(),
            StackValue::AuxVar(AuxVar {
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
                                index: -1,
                            },],
                            output: vec![
                                StackItem {
                                    name: "iter",
                                    count: 1,
                                    index: -1,
                                },
                                StackItem {
                                    name: "next",
                                    count: 1,
                                    index: 0,
                                },
                            ],
                            net_stack_delta: 1,
                        },
                        stack_inputs: vec![AuxVar {
                            name: "phi_0".into()
                        },],
                    },),
                ),
                crate::sir::SIRStatement::DisregardCall(crate::sir::Call {
                    node: SIRNode {
                        opcode: crate::v311::opcodes::Opcode::STORE_NAME,
                        oparg: 0,
                        input: vec![StackItem {
                            name: "value",
                            count: 1,
                            index: -1,
                        },],
                        output: vec![],
                        net_stack_delta: -1
                    },
                    stack_inputs: vec![AuxVar {
                        name: "next_0".into()
                    },],
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

        let mut curr_stack = vec![
            StackValue::AuxVar(AuxVar {
                name: "iter_0".into(),
            }),
            StackValue::AuxVar(AuxVar {
                name: "next_0".into(),
            }),
        ];

        let mut statements = vec![];

        let mut stack: InfiniteStack<_> = vec![].into();
        let mut phi_stack = vec![].into();
        let mut names = HashMap::new();

        for instruction in instructions {
            // Pop all 3 values
            let (stmts, inst_stmt) =
                instruction_to_ir::<crate::v311::ext_instructions::ExtInstruction, SIRNode>(
                    instruction.get_opcode(),
                    0,
                    false,
                    &mut stack,
                    &mut phi_stack,
                    &mut names,
                )
                .unwrap();

            statements.extend(stmts);
            statements.push(inst_stmt);
        }

        fill_phi_nodes(&mut curr_stack, &mut statements, &phi_stack).unwrap();

        extend_merge_stack(&mut curr_stack, &stack, &phi_stack).unwrap();

        assert_eq!(curr_stack.len(), 2);
        assert_eq!(
            *curr_stack,
            vec![
                StackValue::AuxVar(AuxVar {
                    name: "iter_0".into()
                },),
                StackValue::AuxVar(AuxVar {
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
                index: -3,
            },
            StackItem {
                name: "top",
                count: 1,
                index: -1,
            },
        ]
        .as_slice();

        let outputs = [
            StackItem {
                name: "top",
                count: 1,
                index: -3,
            },
            StackItem {
                name: "bottom",
                count: 1,
                index: -1,
            },
        ]
        .as_slice();

        let mut curr_stack = vec![
            StackValue::AuxVar(AuxVar {
                name: "first".into(),
            }),
            StackValue::AuxVar(AuxVar {
                name: "second".into(),
            }),
            StackValue::AuxVar(AuxVar {
                name: "third".into(),
            }),
        ];

        let mut stack: InfiniteStack<_> = vec![].into();
        let mut phi_stack = vec![].into();
        let mut names = HashMap::new();

        let (_, _, mut statements) = process_stack_effects::<SIRNode>(
            inputs,
            outputs,
            0,
            &mut stack,
            &mut phi_stack,
            &mut names,
        )
        .unwrap();

        fill_phi_nodes(&mut curr_stack, &mut statements, &phi_stack).unwrap();

        extend_merge_stack(&mut curr_stack, &stack, &phi_stack).unwrap();

        assert_eq!(
            curr_stack,
            vec![
                StackValue::AuxVar(AuxVar {
                    name: "top_0".into()
                },),
                StackValue::AuxVar(AuxVar {
                    name: "second".into()
                },),
                StackValue::AuxVar(AuxVar {
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
            index: -3,
        }]
        .as_slice();

        let outputs = [
            StackItem {
                name: "top",
                count: 1,
                index: -3,
            },
            StackItem {
                name: "bottom",
                count: 1,
                index: 0,
            },
        ]
        .as_slice();

        let mut curr_stack = vec![
            StackValue::AuxVar(AuxVar {
                name: "first".into(),
            }),
            StackValue::AuxVar(AuxVar {
                name: "second".into(),
            }),
            StackValue::AuxVar(AuxVar {
                name: "third".into(),
            }),
        ];

        let mut stack: InfiniteStack<_> = vec![].into();
        let mut phi_stack = vec![].into();
        let mut names = HashMap::new();

        let (_, _, stmts) = process_stack_effects::<SIRNode>(
            inputs,
            outputs,
            1,
            &mut stack,
            &mut phi_stack,
            &mut names,
        )
        .unwrap();

        statements.extend(stmts);

        let mut sanity_stack = curr_stack.clone();

        fill_phi_nodes(&mut curr_stack, &mut statements.clone(), &phi_stack).unwrap();

        extend_merge_stack(&mut sanity_stack, &stack, &phi_stack).unwrap();

        assert_eq!(
            sanity_stack,
            vec![
                StackValue::AuxVar(AuxVar {
                    name: "top_0".into()
                },),
                StackValue::AuxVar(AuxVar {
                    name: "second".into()
                },),
                StackValue::AuxVar(AuxVar {
                    name: "third".into()
                },),
                StackValue::AuxVar(AuxVar {
                    name: "bottom_0".into()
                },),
            ]
        );

        // pop top element

        let inputs = [StackItem {
            name: "top",
            count: 1,
            index: -1,
        }]
        .as_slice();

        let (_, _, stmts) = process_stack_effects::<SIRNode>(
            inputs,
            &[],
            -1,
            &mut stack,
            &mut phi_stack,
            &mut names,
        )
        .unwrap();

        statements.extend(stmts);

        let mut sanity_stack = curr_stack.clone();

        fill_phi_nodes(&mut curr_stack, &mut statements.clone(), &phi_stack).unwrap();

        extend_merge_stack(&mut sanity_stack, &stack, &phi_stack).unwrap();

        assert_eq!(
            sanity_stack,
            vec![
                StackValue::AuxVar(AuxVar {
                    name: "top_0".into()
                },),
                StackValue::AuxVar(AuxVar {
                    name: "second".into()
                },),
                StackValue::AuxVar(AuxVar {
                    name: "third".into()
                },),
            ]
        );

        let inputs = [
            StackItem {
                name: "first",
                count: 1,
                index: -2,
            },
            StackItem {
                name: "second",
                count: 1,
                index: -1,
            },
        ]
        .as_slice();

        let outputs = [StackItem {
            name: "out",
            count: 1,
            index: -2,
        }]
        .as_slice();

        dbg!(&stack, &curr_stack);

        let (_, _, stmts) = process_stack_effects::<SIRNode>(
            inputs,
            outputs,
            -1,
            &mut stack,
            &mut phi_stack,
            &mut names,
        )
        .unwrap();

        statements.extend(stmts);

        dbg!(&stack, &curr_stack);

        fill_phi_nodes(&mut curr_stack, &mut statements, &phi_stack).unwrap();

        extend_merge_stack(&mut curr_stack, &stack, &phi_stack).unwrap();

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

        let cfg = create_cfg(instructions.to_vec(), None).unwrap();

        let cfg = simple_cfg_to_ext_cfg(&cfg).unwrap();

        let ir_cfg = cfg_to_ir::<ExtInstruction, SIRNode>(&cfg, false).unwrap();

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

        let cfg = simple_cfg_to_ext_cfg(&cfg).unwrap();

        let ir_cfg = cfg_to_ir::<ExtInstruction, SIRNode>(&cfg, false).unwrap();

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

        let cfg = simple_cfg_to_ext_cfg(&cfg).unwrap();

        println!("{}", cfg.make_dot_graph());

        let ir_cfg = cfg_to_ir::<_, crate::v310::opcodes::sir::SIRNode>(&cfg, false).unwrap();

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

        let cfg = simple_cfg_to_ext_cfg(&cfg).unwrap();

        let ir_cfg = cfg_to_ir::<_, SIRNode>(&cfg, false).unwrap();

        println!("{}", ir_cfg.make_dot_graph());

        insta::assert_debug_snapshot!(ir_cfg);
    }

    #[test]
    fn test_lot_of_args_call() {
        let program = crate::load_code(&b"\xe3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\n\x00\x00\x00C\x00\x00\x00s>\x00\x00\x00t\x00j\x01D\x00]\x19}\x00|\x00\xa0\x02t\x00j\x03\xa1\x01\x01\x00t\x00j\x04\xa0\x05t\x00j\x06|\x00t\x00j\x07t\x00t\x00j\x03t\x00j\x08t\x00j\t\xa1\x07\x01\x00q\x03d\x00S\x00)\x01N)\n\xda\x04selfZ\x08_socketsZ\x06listenZ\x08_backlogZ\x05_loopZ\x0e_start_servingZ\x11_protocol_factoryZ\x0c_ssl_contextZ\x16_ssl_handshake_timeoutZ\x15_ssl_shutdown_timeout)\x01Z\x04sock\xa9\x00r\x02\x00\x00\x00\xfa\x07<stdin>\xda\x04test\x01\x00\x00\x00s\x10\x00\x00\x00\n\x01\x0c\x01\x06\x01\n\x01\n\x01\x04\x01\x06\xfd\x04\xfe"[..], (3, 10).into()).unwrap();

        let instructions = match program {
            CodeObject::V310(code) => code.code.clone(),
            _ => unreachable!(),
        };

        let cfg = create_cfg(instructions.to_vec(), None).unwrap();

        let cfg = simple_cfg_to_ext_cfg(&cfg).unwrap();

        let ir_cfg = cfg_to_ir::<_, crate::v310::opcodes::sir::SIRNode>(&cfg, false).unwrap();

        println!("{}", ir_cfg.make_dot_graph());

        insta::assert_debug_snapshot!(ir_cfg);
    }

    #[test]
    fn test_310_with_block() {
        use crate::v310::instructions::{Instruction, Instructions};
        use crate::v310::opcodes::sir::SIRNode;

        let ext_instructions = Instructions::new(vec![
            Instruction::LoadConst(0),
            Instruction::JumpForward(0), // We do this so the value comes from a different BB
            Instruction::PopBlock(0),
            Instruction::LoadConst(0),
            Instruction::DupTop(0),
            Instruction::DupTop(0),
            Instruction::CallFunction(3),
            Instruction::PopTop(0),
        ])
        .to_resolved()
        .unwrap();

        let cfg = create_cfg(ext_instructions.to_vec(), None).unwrap();

        let ir_cfg = cfg_to_ir::<_, SIRNode>(&cfg, false).unwrap();

        println!("{}", ir_cfg.make_dot_graph());

        insta::assert_debug_snapshot!(ir_cfg);
    }

    #[test]
    fn test_310_nested_try() {
        use crate::v310::instructions::{Instruction, Instructions};
        use crate::v310::opcodes::sir::SIRNode;

        let ext_instructions = Instructions::new(vec![
            Instruction::SetupFinally(2),
            Instruction::LoadConst(0),
            Instruction::ReturnValue(0),
            Instruction::PopTop(0), // exc, tb, type, exc, tb, type
            Instruction::PopTop(0),
            Instruction::PopTop(0),
            Instruction::LoadFast(0),
            Instruction::RotFour(0), // Accesses 3 values from a different BB
            Instruction::PopExcept(0),
            Instruction::ReturnValue(0),
        ])
        .to_resolved()
        .unwrap();

        let cfg = create_cfg(ext_instructions.to_vec(), None).unwrap();

        let ir_cfg = cfg_to_ir::<_, SIRNode>(&cfg, false).unwrap();

        println!("{}", ir_cfg.make_dot_graph());

        insta::assert_debug_snapshot!(ir_cfg);
    }

    #[test]
    fn test_force_stack_depth() {
        // An exception table entry can force the stack depth

        let ext_instructions = v311::instructions::Instructions::new(vec![
            Instruction::LoadConst(0),
            Instruction::LoadConst(0), // This value should be removed by the forced stack depth
            Instruction::PopTop(0),
            Instruction::PopTop(0),
            Instruction::LoadConst(0), // Exception entry starts here
            Instruction::ReturnValue(0),
            Instruction::PopTop(0), // Exception target
            Instruction::PopTop(0),
            Instruction::LoadConst(0),
            Instruction::ReturnValue(0),
        ])
        .to_resolved()
        .unwrap();

        let exception_table = vec![ExceptionTableEntry {
            start: 2,
            end: 6,
            target: 6,
            depth: 1,
            lasti: false,
        }];

        let cfg = create_cfg(ext_instructions.to_vec(), Some(exception_table)).unwrap();

        println!("{}", cfg.make_dot_graph());

        let ir_cfg = cfg_to_ir::<_, SIRNode>(&cfg, false).unwrap();

        println!("{}", ir_cfg.make_dot_graph());

        insta::assert_debug_snapshot!(ir_cfg);
    }
}
