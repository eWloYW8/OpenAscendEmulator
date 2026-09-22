use super::{
    C220MteL1CycleInputs, C220MteL1Error, C220MteL1Interface, C220MteL1OutputSend,
    C220MteL1OutputTransfer, C220MteL1ReadPort, C220MteL1ReadRequest, C220MteL1ReadSend,
};
use crate::sim::common::event::{EventDispatcher, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220MteL1Callback {
    InputReady(C220MteL1ReadPort),
    ResponseReady,
    AcknowledgmentReady,
    RetirementReady,
    SendRequest,
    ReceiveResponse,
    SendOutput,
    Retire,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum C220MteL1EventOutcome<T> {
    Readiness,
    Request(C220MteL1ReadSend<T>),
    Response(Option<C220MteL1ReadRequest<T>>),
    Output(C220MteL1OutputSend<C220MteL1ReadRequest<T>>),
    Retired(Option<C220MteL1OutputTransfer<C220MteL1ReadRequest<T>>>),
}

/// Shared input queues wake one request sender. Responses, output forwarding,
/// and local retirement have independent consumers. The owner samples current
/// port credits and the eligible response head for each callback, then commits
/// returned transfers before requesting the next callback. A response must not
/// be removed from its transport until the interface accepts it successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MteL1Events {
    input_valid: [EventId; 3],
    response_valid: EventId,
    acknowledgment_valid: EventId,
    retirement_valid: EventId,
}

impl C220MteL1Events {
    pub fn register<T: Copy>(
        events: &mut EventDispatcher<T>,
        clock: EventId,
        tag: impl Fn(C220MteL1Callback) -> T,
    ) -> Self {
        let input_valid = std::array::from_fn(|_| events.add_event());
        let response_valid = events.add_event();
        let acknowledgment_valid = events.add_event();
        let retirement_valid = events.add_event();
        let send = events.add_process(tag(C220MteL1Callback::SendRequest), false);
        for port in [
            C220MteL1ReadPort::Port0,
            C220MteL1ReadPort::Port1,
            C220MteL1ReadPort::Port2,
        ] {
            let probe = events.add_process(tag(C220MteL1Callback::InputReady(port)), false);
            events.subscribe(clock, probe);
            events.subscribe(input_valid[port as usize], send);
        }
        for (probe, callback, valid) in [
            (
                C220MteL1Callback::ResponseReady,
                C220MteL1Callback::ReceiveResponse,
                response_valid,
            ),
            (
                C220MteL1Callback::AcknowledgmentReady,
                C220MteL1Callback::SendOutput,
                acknowledgment_valid,
            ),
            (
                C220MteL1Callback::RetirementReady,
                C220MteL1Callback::Retire,
                retirement_valid,
            ),
        ] {
            let probe = events.add_process(tag(probe), false);
            events.subscribe(clock, probe);
            let process = events.add_process(tag(callback), false);
            events.subscribe(valid, process);
        }
        Self {
            input_valid,
            response_valid,
            acknowledgment_valid,
            retirement_valid,
        }
    }

    pub fn handle<T: Copy, P: Copy>(
        &self,
        callback: C220MteL1Callback,
        events: &mut EventDispatcher<T>,
        interface: &mut C220MteL1Interface<P>,
        inputs: C220MteL1CycleInputs,
    ) -> Result<C220MteL1EventOutcome<P>, C220MteL1Error> {
        let tick = events.tick();
        let (valid, ready) = match callback {
            C220MteL1Callback::InputReady(port) => (
                self.input_valid[port as usize],
                interface.input_ready_tick(port),
            ),
            C220MteL1Callback::ResponseReady => {
                (self.response_valid, inputs.response.map(|_| tick))
            }
            C220MteL1Callback::AcknowledgmentReady => (
                self.acknowledgment_valid,
                interface.acknowledgment_ready_tick(),
            ),
            C220MteL1Callback::RetirementReady => {
                (self.retirement_valid, interface.retirement_ready_tick())
            }
            C220MteL1Callback::SendRequest => {
                return interface
                    .send_request(tick, inputs.request_ready)
                    .map(C220MteL1EventOutcome::Request);
            }
            C220MteL1Callback::ReceiveResponse => {
                return interface
                    .receive_response(tick, inputs.response)
                    .map(C220MteL1EventOutcome::Response);
            }
            C220MteL1Callback::SendOutput => {
                return interface
                    .send_output(tick, inputs.output_credits)
                    .map(C220MteL1EventOutcome::Output);
            }
            C220MteL1Callback::Retire => {
                return interface.retire(tick).map(C220MteL1EventOutcome::Retired);
            }
        };
        if ready.is_some_and(|ready| ready <= tick) {
            events.notify_at(valid, tick);
        }
        Ok(C220MteL1EventOutcome::Readiness)
    }
}
