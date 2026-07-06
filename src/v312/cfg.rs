use crate::{cfg::ControlFlowGraph, traits::FinalizeCFG, v312::{ext_instructions::ExtInstruction, instructions::Instruction}};

impl FinalizeCFG<Instruction> for ControlFlowGraph<Instruction> {
    fn finalize_cfg(
        &mut self,
    ) -> Result<(), crate::error::Error> {
        Ok(())
    }
}

impl FinalizeCFG<ExtInstruction> for ControlFlowGraph<ExtInstruction> {
    fn finalize_cfg(
        &mut self,
    ) -> Result<(), crate::error::Error> {
        Ok(())
    }
}
