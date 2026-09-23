use super::*;
use crate::sim::c220::mte::uop::{
    C220DmaDestinationLayout, C220DmaUopMode, C220DmaUopRequest, C220DmaUopRoute,
};

#[test]
fn destination_splitting_preserves_reserved_tags_and_recycles_after_response() {
    let mut commands = C220BiuWriteCommands::new(C220BiuWriteConfig {
        outstanding: NonZeroU32::new(3).unwrap(),
        weights: [1; 3],
        source_bandwidth: NonZeroU32::new(32).unwrap(),
    });
    let input = C220BiuWriteInput {
        subcore: C220BiuSubcore::Vector0,
        gather_stride: None,
        generated: C220DmaGenerated {
            instruction_id: 9,
            uop_index: 0,
            ready_tick: 0,
            request: C220DmaUopRequest {
                route: C220DmaUopRoute::Ordinary,
                burst_index: 0,
                source_address: 7,
                destination_address: 64,
                bytes: 600,
                last_in_burst: true,
            },
            destination: C220DmaDestinationLayout {
                base: 64,
                burst_bytes: 600,
                burst_stride: 600,
            },
            mode: C220DmaUopMode::Fixed128,
            out_of_order: false,
            last_in_instruction: true,
        },
    };
    commands.push(0, input).unwrap();
    for tick in 0..6 {
        commands.advance(tick).unwrap();
    }
    assert_eq!(
        commands.advance(6).unwrap().stall,
        Some(C220BiuWriteCommandStall::TransportFull)
    );
    assert_eq!(commands.reserved_tag().unwrap().get(), 3);
    assert_eq!(commands.free_tag_count(), 0);
    assert!(commands.awaiting_dbid(NonZeroU32::new(1).unwrap()).is_err());
    let first = commands.take_request(7).unwrap().unwrap().command;
    assert_eq!(
        (
            first.input.generated.request.source_address,
            first.input.generated.request.destination_address,
            first.input.generated.request.bytes
        ),
        (7, 64, 64)
    );
    commands.awaiting_dbid(first.tag).unwrap();
    commands.mark_dbid(first.tag);
    assert!(!commands.begin_source(first.tag).last_in_instruction);
    assert!(commands.awaiting_dbid(first.tag).is_err());
    assert_eq!(commands.advance(7).unwrap().sent.unwrap().tag.get(), 3);
    let second = commands.take_request(8).unwrap().unwrap().command;
    commands.mark_dbid(second.tag);
    assert!(!commands.begin_source(second.tag).last_in_instruction);
    assert_eq!(
        (second.byte_offset, second.input.generated.request.bytes),
        (64, 128)
    );
    assert_eq!(
        commands.advance(8).unwrap().stall,
        Some(C220BiuWriteCommandStall::NoTag)
    );
    commands.release_tag(first.tag).unwrap();
    assert_eq!(commands.advance(9).unwrap().sent.unwrap().tag, first.tag);
    commands.release_tag(second.tag).unwrap();
    let mut bytes = vec![
        first.input.generated.request.bytes,
        second.input.generated.request.bytes,
    ];
    let mut tail = None;
    for tick in 10..25 {
        if let Some(transfer) = commands.take_request(tick).unwrap() {
            let command = transfer.command;
            commands.mark_dbid(command.tag);
            let source = commands.begin_source(command.tag);
            bytes.push(command.input.generated.request.bytes);
            if source.last_in_instruction {
                tail = Some(bytes.len());
            }
            commands.release_tag(command.tag).unwrap();
        }
        commands.advance(tick).unwrap();
    }
    assert_eq!(bytes, [64, 128, 128, 128, 128, 24]);
    assert_eq!(tail, Some(6));
    assert!(commands.is_idle());
    assert_eq!(commands.free_tag_count(), 3);

    let mut reordered = C220BiuWriteCommands::new(commands.config);
    let mut input = input;
    input.generated.request.destination_address = 0;
    input.generated.request.bytes = 384;
    reordered.push(0, input).unwrap();
    let mut delivered = Vec::new();
    for tick in 0..10 {
        reordered.advance(tick).unwrap();
        if let Some(transfer) = reordered.take_request(tick).unwrap() {
            delivered.push(transfer.command);
        }
    }
    assert_eq!(delivered.len(), 3);
    assert!(delivered[2].input.generated.last_in_instruction);
    for (index, tail) in [(2, false), (0, false), (1, true)] {
        let tag = delivered[index].tag;
        reordered.awaiting_dbid(tag).unwrap();
        reordered.mark_dbid(tag);
        assert_eq!(reordered.begin_source(tag).last_in_instruction, tail);
        assert_eq!(
            reordered
                .release_tag(tag)
                .unwrap()
                .input
                .generated
                .last_in_instruction,
            tail
        );
    }
    assert!(reordered.is_idle());
}
