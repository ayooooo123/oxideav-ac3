// Port of the checked `GetBitContext` reader FFmpeg's AC-3 decoder uses
// (libavcodec/get_bits.h, FFmpeg commit 2da55bf).
// Copyright (c) the FFmpeg developers; LGPL-2.1-or-later (see LICENSE-LGPL).

/// MSB-first reader over `size` bytes of a buffer. Like FFmpeg's safe
/// reader, the position saturates 8 bits past the end; bits beyond the
/// buffer read as zero (FFmpeg reads its zeroed padding there).
pub(crate) struct BitReader<'a> {
    buf: &'a [u8],
    index: usize,
    size_in_bits_plus8: usize,
}

impl<'a> BitReader<'a> {
    /// `init_get_bits8(buf, size)`: `size` may exceed `buf.len()`; the
    /// missing bytes read as zero.
    pub(crate) fn new(buf: &'a [u8], size: usize) -> Self {
        Self {
            buf,
            index: 0,
            size_in_bits_plus8: size * 8 + 8,
        }
    }

    fn byte(&self, i: usize) -> u64 {
        u64::from(self.buf.get(i).copied().unwrap_or(0))
    }

    /// The next `n` (0..=32) bits without consuming them.
    pub(crate) fn show(&self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let first = self.index >> 3;
        let mut window = 0u64;
        for k in 0..5 {
            window = (window << 8) | self.byte(first + k);
        }
        let shift = 40 - (self.index & 7) as u32 - n;
        ((window >> shift) & ((1u64 << n) - 1)) as u32
    }

    pub(crate) fn skip(&mut self, n: usize) {
        self.index = (self.index + n).min(self.size_in_bits_plus8);
    }

    /// `get_bits` / `get_bits_long`: `n` in 0..=32.
    pub(crate) fn get(&mut self, n: u32) -> u32 {
        let v = self.show(n);
        self.skip(n as usize);
        v
    }

    pub(crate) fn get1(&mut self) -> u32 {
        self.get(1)
    }

    /// `get_sbits`: `n` in 1..=32, two's complement.
    pub(crate) fn get_s(&mut self, n: u32) -> i32 {
        let v = self.get(n);
        ((v << (32 - n)) as i32) >> (32 - n)
    }
}
