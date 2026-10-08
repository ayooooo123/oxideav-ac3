//! A packet costs a frame or two of memory, however many frames it holds:
//! the decoder decodes on demand, one frame per `receive_frame` plus the
//! one `avcodec_send_packet` decodes ahead, not the whole packet at once.
//! Each fixture frame (768 bytes, stereo) decodes to 12 KiB of PCM; a
//! 128-byte 5.1 frame, the smallest AC-3 has, decodes to 36 KiB.
//!
//! This file holds one test, so no other test allocates while it measures.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase};

/// Counts live heap bytes and their peak.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// 16 frames of 768 bytes: 192 kb/s stereo at 48 kHz.
const FIXTURE: &[u8] = include_bytes!("fixtures/sine440_stereo.ac3");
const FRAME: usize = 768;

#[test]
fn a_packet_of_many_frames_costs_a_frame_or_two_of_memory() {
    let mut reg = CodecRegistry::new();
    oxideav_ac3::register_codecs(&mut reg);
    let mut failures = Vec::new();
    for codec in ["ac3", "eac3"] {
        let mut dec = reg
            .first_decoder(&CodecParameters::audio(CodecId::new(codec)))
            .unwrap();
        // one frame first, so tables built on first use are not counted
        dec.send_packet(&Packet::new(
            0,
            TimeBase::new(1, 48_000),
            FIXTURE[..FRAME].to_vec(),
        ))
        .unwrap();
        assert!(
            matches!(dec.receive_frame(), Ok(Frame::Audio(_))),
            "{codec}: first frame"
        );

        // 128 frames in one packet: the fixture 8 times over
        let data: Vec<u8> = FIXTURE
            .iter()
            .copied()
            .cycle()
            .take(FIXTURE.len() * 8)
            .collect();
        let packet = Packet::new(0, TimeBase::new(1, 48_000), data);
        let base = LIVE.load(Ordering::Relaxed);
        PEAK.store(base, Ordering::Relaxed);

        dec.send_packet(&packet).unwrap();
        for _ in 0..3 {
            let Frame::Audio(a) = dec.receive_frame().unwrap() else {
                panic!("{codec}: not audio")
            };
            assert_eq!((a.samples, a.data.len()), (1536, 2), "{codec}");
        }
        // the packet's copy (96 KiB) and a few 12 KiB frames; decoding all
        // 128 frames up front holds 1.5 MiB
        let used = PEAK.load(Ordering::Relaxed) - base;
        if used > 512 << 10 {
            failures.push(format!(
                "{codec}: {used} bytes for 3 frames of a 128-frame packet"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
