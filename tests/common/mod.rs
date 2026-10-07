//! Shared helpers for the integration tests (and, via `#[path]`, the
//! `equal_rate_report` example).
#![allow(dead_code)]

pub mod rd;

use oxideav_core::AudioFrame;

/// A frame of the default decoder (planar float, FFmpeg's channel order)
/// interleaved.
pub fn f32_interleaved(a: &AudioFrame) -> Vec<f32> {
    let planes: Vec<Vec<f32>> = a
        .data
        .iter()
        .map(|p| {
            p.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        })
        .collect();
    let mut out = Vec::with_capacity(a.samples as usize * planes.len());
    for i in 0..a.samples as usize {
        out.extend(planes.iter().map(|p| p[i]));
    }
    out
}

/// [`f32_interleaved`] as S16LE bytes, converted as FFmpeg's `flt` → `s16`
/// does (round to nearest even, clip).
pub fn s16_interleaved(a: &AudioFrame) -> Vec<u8> {
    f32_interleaved(a)
        .into_iter()
        .flat_map(|s| {
            ((s * 32768.0).round_ties_even().clamp(-32768.0, 32767.0) as i16).to_le_bytes()
        })
        .collect()
}
