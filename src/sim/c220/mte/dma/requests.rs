use super::{C220DmaDestinationLayout, C220DmaUopMode, C220DmaUopRequest, C220DmaUops};
use crate::sim::c220::mte::load2d::C220Load2dExternalRequests;
use crate::sim::c220::mte::uop::C220DmaUopRoute;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Requests {
    Dma(C220DmaUops),
    Load2d(C220Load2dExternalRequests),
}

impl Requests {
    pub(super) fn sid(&self) -> u8 {
        match self {
            Self::Dma(requests) => requests.sid(),
            Self::Load2d(requests) => requests.sid(),
        }
    }

    pub(super) fn metadata(&self) -> (C220DmaDestinationLayout, C220DmaUopMode, bool) {
        match self {
            Self::Dma(requests) => (
                requests.destination(),
                requests.mode(),
                requests.out_of_order(),
            ),
            Self::Load2d(requests) => requests.dma_metadata(),
        }
    }
}

impl Iterator for Requests {
    type Item = C220DmaUopRequest;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Dma(requests) => requests.next(),
            Self::Load2d(requests) => requests.next().map(|request| C220DmaUopRequest {
                route: C220DmaUopRoute::Ordinary,
                burst_index: u16::from(request.repeat_index),
                source_address: request.source_address,
                destination_address: request.destination_address,
                bytes: request.bytes,
                last_in_burst: request.last_in_block,
            }),
        }
    }
}
