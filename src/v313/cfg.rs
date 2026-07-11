use crate::{
    cfg::{CFGIndexRange, ControlFlowGraph}, traits::FinalizeCFG, v313::{ext_instructions::ExtInstruction, instructions::Instruction},
};

impl FinalizeCFG<Instruction> for ControlFlowGraph<Instruction> {
    fn finalize_cfg(&mut self, map: Option<&mut Vec<CFGIndexRange>>) -> Result<(), crate::error::Error> {
        Ok(())
    }
}

impl FinalizeCFG<ExtInstruction> for ControlFlowGraph<ExtInstruction> {
    fn finalize_cfg(&mut self, map: Option<&mut Vec<CFGIndexRange>>) -> Result<(), crate::error::Error> {
        Ok(())
    }
}
