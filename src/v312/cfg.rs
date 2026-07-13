use crate::{
    cfg::{CFGIndexRange, ControlFlowGraph},
    traits::FinalizeCFG,
    v312::{ext_instructions::ExtInstruction, instructions::Instruction},
};

impl FinalizeCFG<Instruction> for ControlFlowGraph<Instruction> {
    fn finalize_cfg(
        &mut self,
        _map: Option<&mut Vec<CFGIndexRange>>,
    ) -> Result<(), crate::error::Error> {
        Ok(())
    }
}

impl FinalizeCFG<ExtInstruction> for ControlFlowGraph<ExtInstruction> {
    fn finalize_cfg(
        &mut self,
        _map: Option<&mut Vec<CFGIndexRange>>,
    ) -> Result<(), crate::error::Error> {
        Ok(())
    }
}
