use super::{C220MtePipeline, C220MtePipelineError, Mte2Generator};
use crate::isa::c220::mte::smask::C220SmaskTransfer;
use crate::sim::c220::mte::read::{C220MteReadIssue, C220MteReadKind, C220MteReadTransfer};

impl C220MtePipeline {
    pub fn can_issue_external_smask(&self, transfer: C220SmaskTransfer) -> bool {
        transfer.instruction.source_mode == 0
            && (transfer.descriptor.is_empty()
                || (self.generator(C220MteReadKind::Default).can_issue()
                    && (self.selected_mte2_generator == Some(Mte2Generator::Default)
                        || self.mte2_generator_idle())))
    }

    pub fn issue_external_smask(
        &mut self,
        instruction_id: u64,
        transfer: C220SmaskTransfer,
    ) -> Result<C220MteReadIssue, C220MtePipelineError> {
        if transfer.instruction.source_mode != 0 {
            return Err(C220MtePipelineError::WrongCommandLane);
        }
        if !self.can_issue_external_smask(transfer) {
            return Err(C220MtePipelineError::CommandBusy);
        }
        if transfer.descriptor.is_empty() {
            return Ok(C220MteReadIssue {
                tick: self.events.tick(),
                instruction_id,
                request_count: 0,
                completion_ready: true,
            });
        }
        let index = C220MteReadKind::Default.index();
        let issue = self.generator_events[index].issue(
            &mut self.events,
            &mut self.generators[index],
            instruction_id,
            C220MteReadTransfer::Smask(transfer),
        )?;
        self.selected_mte2_generator = Some(Mte2Generator::Default);
        Ok(issue)
    }
}
