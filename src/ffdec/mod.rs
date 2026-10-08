//! FFmpeg's float AC-3 / E-AC-3 decoder (FFmpeg commit 2da55bf,
//! `libavcodec/ac3dec.c` + `eac3dec.c` with their tables and DSP), ported to
//! Rust. This is the decoder [`crate::decoder::make_decoder`] and
//! [`crate::decoder::make_eac3_decoder`] return: its output is FFmpeg's
//! `fltp` output for the same packets, frame for frame.
//!
//! What it reproduces, as FFmpeg 2da55bf does it:
//! * the zero-mantissa dither and the spectral-extension noise, drawn from
//!   an `AVLFG` seeded with 0 when the decoder opens;
//! * dynamic range compression at `drc_scale` 1 (the stream's `dynrng`);
//! * concealment: a frame cut short (or failing to decode) repeats the
//!   previous frame's last block, so a stream's cut-off last frame still
//!   yields a full frame;
//! * packets carrying several frames: decoded on demand, one frame per
//!   `receive_frame` (and one ahead in `send_packet`, as
//!   `avcodec_send_packet` does), the rest of the packet fed again, as
//!   FFmpeg's decode loop does, so a packet costs a frame or two of memory;
//! * E-AC-3 dependent substreams merged into the independent stream's
//!   channels through the custom channel map.
//!
//! Not ported: the `downmix`, `heavy_compr`, `target_level`, `drc_scale` and
//! `cons_noisegen` options (their defaults are what runs), and the
//! fixed-point decoder.
//!
//! Copyright (c) the FFmpeg developers and the authors named in each file;
//! LGPL-2.1-or-later (see LICENSE-LGPL).

mod ac3dec;
mod bits;
mod dsp;
mod header;
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
mod tables;
#[allow(clippy::unreadable_literal)]
mod vq_tables;

use std::collections::VecDeque;

use oxideav_core::{
    AudioFormat, AudioFrame, CodecId, CodecParameters, Decoder, Error, Frame, Packet, Result,
    SampleFormat,
};

use ac3dec::{Ac3Context, DecodeError};

/// One decoder output: a frame and its layout, or the error FFmpeg's decode
/// call returned at that point of the packet.
type Output = std::result::Result<(AudioFrame, AudioFormat), Error>;

/// The AC-3 / E-AC-3 decoder: FFmpeg's, frame for frame.
pub(crate) struct FfAc3Decoder {
    codec_id: CodecId,
    container_rate: u32,
    ctx: Box<Ac3Context>,
    /// Packets not yet fully decoded.
    pending: VecDeque<Pending>,
    /// The output decoded ahead, as `avcodec_send_packet` fills
    /// `buffer_frame`: at most one.
    ahead: Option<Output>,
    /// Layout of the frame `receive_frame` returned last.
    returned: Option<AudioFormat>,
    eof: bool,
}

/// A packet not yet fully decoded: its bytes from `pos` on, and the
/// timestamp of its first frame until that frame is decoded.
struct Pending {
    data: Vec<u8>,
    pos: usize,
    pts: Option<i64>,
}

impl FfAc3Decoder {
    pub(crate) fn new(params: &CodecParameters) -> Self {
        let container_rate = params.sample_rate.unwrap_or(0);
        Self {
            codec_id: params.codec_id.clone(),
            container_rate,
            ctx: Box::new(Ac3Context::new(container_rate)),
            pending: VecDeque::new(),
            ahead: None,
            returned: None,
            eof: false,
        }
    }

    /// The next output of the pending packets, as one pass of FFmpeg's
    /// `decode_simple_receive_frame`: each call consumes what
    /// `ac3_decode_frame` reports and the rest of the packet is fed again
    /// without its timestamp; an error drops the rest of the packet.
    fn decode_next(&mut self) -> Option<Output> {
        while let Some(packet) = self.pending.front_mut() {
            let rest = &packet.data[packet.pos..];
            let left = rest.len();
            let pts = packet.pts.take();
            match self.ctx.decode_frame(rest) {
                Ok((consumed, frame)) => {
                    // FFmpeg would feed a packet it consumed nothing of again
                    // forever; there is nothing more to decode from it.
                    if consumed == 0 || consumed >= left {
                        self.pending.pop_front();
                    } else {
                        packet.pos += consumed;
                    }
                    if let Some(frame) = frame {
                        let format = AudioFormat {
                            sample_format: SampleFormat::F32P,
                            sample_rate: frame.sample_rate,
                            channels: frame.planes.len() as u16,
                        };
                        let samples = frame.planes.first().map_or(0, Vec::len) as u32;
                        let data = frame
                            .planes
                            .iter()
                            .map(|plane| plane.iter().flat_map(|s| s.to_le_bytes()).collect())
                            .collect();
                        return Some(Ok((AudioFrame { samples, pts, data }, format)));
                    }
                }
                Err(e) => {
                    self.pending.pop_front();
                    return Some(Err(to_error(e)));
                }
            }
        }
        None
    }
}

fn to_error(e: DecodeError) -> Error {
    match e {
        DecodeError::InvalidData(msg) => Error::invalid(format!("AC-3: {msg}")),
        DecodeError::Unsupported(msg) => Error::unsupported(format!("AC-3: {msg}")),
    }
}

impl Decoder for FfAc3Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    /// Holds the packet and, as `avcodec_send_packet` does, decodes one
    /// frame ahead when none is waiting; `receive_frame` decodes the rest,
    /// one frame per call. A decode error drops the rest of its packet and
    /// is returned in its turn, after the frames decoded before it.
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if !packet.data.is_empty() {
            self.pending.push_back(Pending {
                data: packet.data.clone(),
                pos: 0,
                pts: packet.pts,
            });
        }
        if self.ahead.is_none() {
            self.ahead = self.decode_next();
        }
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        match self.ahead.take().or_else(|| self.decode_next()) {
            Some(Ok((frame, format))) => {
                self.returned = Some(format);
                Ok(Frame::Audio(frame))
            }
            Some(Err(e)) => Err(e),
            None if self.eof => Err(Error::Eof),
            None => Err(Error::NeedMore),
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    /// A decoder as freshly opened. (FFmpeg's `ac3_decode_flush` also
    /// zeroes the decoder's options, dropping `drc_scale` to 0; a reopened
    /// decoder, as `ffmpeg -ss` decodes from a seek point, keeps them.)
    fn reset(&mut self) -> Result<()> {
        *self.ctx = Ac3Context::new(self.container_rate);
        self.pending.clear();
        self.ahead = None;
        self.returned = None;
        self.eof = false;
        Ok(())
    }

    /// The layout of the frame `receive_frame` returned last, or before the
    /// first, of the one decoded ahead: planar float at the stream's rate,
    /// in FFmpeg's channel order.
    fn output_audio_format(&self) -> Option<AudioFormat> {
        self.returned.or(match &self.ahead {
            Some(Ok((_, format))) => Some(*format),
            _ => None,
        })
    }
}
