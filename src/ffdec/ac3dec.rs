// Port of FFmpeg's float AC-3 / E-AC-3 decoder (FFmpeg commit 2da55bf:
// libavcodec/ac3dec.c, eac3dec.c, ac3dec_float.c), without the optional
// downmix and consistent-noise modes.
// Copyright (c) 2007-2008 Bartlomiej Wolowiec, (c) 2007-2011 Justin
// Ruggles, (c) 2007-2008 Jacob Lindquist, Christian Pfennig, Simon Thurman
// and the FFmpeg developers; LGPL-2.1-or-later (see LICENSE-LGPL).

use std::sync::LazyLock;

use super::bits::BitReader;
use super::dsp::{
    calc_bap, calc_mask, calc_psd, kbd_window, vector_fmul_window, BitAllocParams, Dba, Imdct, Lfg,
    AC3_CRITICAL_BANDS, AC3_MAX_COEFS, DBA_NEW, DBA_NONE, DBA_RESERVED,
};
use super::header::{
    find_syncword, parse_header, HeaderInfo, ParseError, AC3_CHMODE_MONO, AC3_CHMODE_STEREO,
    EAC3_FRAME_TYPE_DEPENDENT, EAC3_FRAME_TYPE_INDEPENDENT, EAC3_FRAME_TYPE_RESERVED,
    EAC3_MAX_CHANNELS, EAC3_SR_CODE_REDUCED,
};
use super::tables::*;
use super::vq_tables::*;

const AC3_MAX_CHANNELS: usize = 7;
const AC3_BLOCK_SIZE: usize = 256;
const AC3_MAX_BLOCKS: usize = 6;
const AC3_MAX_CPL_BANDS: usize = 18;
const SPX_MAX_BANDS: usize = 17;
const CPL_CH: usize = 0;
const AC3_OUTPUT_LFEON: i32 = 8;
const AC3_FRAME_BUFFER_SIZE: usize = 32768;
const EXP_REUSE: i32 = 0;
const EXP_D45: i32 = 3;

const AC3_CHMODE_DUALMONO: i32 = 0;
const AC3_CHMODE_3F: i32 = 3;
const AC3_CHMODE_2F1R: i32 = 4;
const AC3_CHMODE_3F1R: i32 = 5;
const AC3_CHMODE_2F2R: i32 = 6;
const AC3_CHMODE_3F2R: i32 = 7;

const EAC3_GAQ_NO: i32 = 0;
const EAC3_GAQ_12: i32 = 1;
const EAC3_GAQ_14: i32 = 2;
const EAC3_GAQ_124: i32 = 3;

/// Why a decode call failed: FFmpeg's negative return values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DecodeError {
    InvalidData(&'static str),
    Unsupported(&'static str),
}

/// One decoded frame: planar float, FFmpeg's native channel order.
pub(crate) struct DecodedFrame {
    pub sample_rate: u32,
    pub planes: Vec<Vec<f32>>,
}

/// `parse_frame_header`'s failures: a header parse error code or a plain
/// error FFmpeg does not try to conceal.
enum HeaderError {
    Parse(ParseError),
    Fatal(DecodeError),
}

/// Lookup tables `ac3_float_tables_init` and `ac3_init_static` build.
struct StaticTables {
    dynamic_range: [f32; 256],
    heavy_dynamic_range: [f32; 256],
    ungroup_3_in_7_bits: [[u8; 3]; 128],
    bap1_mantissas: [[i32; 3]; 32],
    bap2_mantissas: [[i32; 3]; 128],
    bap3_mantissas: [i32; 8],
    bap4_mantissas: [[i32; 2]; 128],
    bap5_mantissas: [i32; 16],
    /// `scale_factors`: 2^-i, padded past exponent 24 (never reached by a
    /// validated exponent) so a lookup cannot go out of range.
    scale_factors: [f32; 32],
}

fn symmetric_dequant(code: i32, levels: i32) -> i32 {
    ((code - (levels >> 1)) * (1 << 24)) / levels
}

static TABLES: LazyLock<StaticTables> = LazyLock::new(build_tables);

fn tables() -> &'static StaticTables {
    &TABLES
}

fn build_tables() -> StaticTables {
    {
        let mut t = StaticTables {
            dynamic_range: [0.0; 256],
            heavy_dynamic_range: [0.0; 256],
            ungroup_3_in_7_bits: [[0; 3]; 128],
            bap1_mantissas: [[0; 3]; 32],
            bap2_mantissas: [[0; 3]; 128],
            bap3_mantissas: [0; 8],
            bap4_mantissas: [[0; 2]; 128],
            bap5_mantissas: [0; 16],
            scale_factors: [0.0; 32],
        };
        for i in 0..256i32 {
            let v = (i >> 5) - ((i >> 7) << 3) - 5;
            t.dynamic_range[i as usize] = 2f32.powi(v) * ((i & 0x1F) | 0x20) as f32;
            let v = (i >> 4) - ((i >> 7) << 4) - 4;
            t.heavy_dynamic_range[i as usize] = 2f32.powi(v) * ((i & 0xF) | 0x10) as f32;
        }
        for i in 0..128 {
            t.ungroup_3_in_7_bits[i] = [(i / 25) as u8, ((i % 25) / 5) as u8, ((i % 25) % 5) as u8];
        }
        for i in 0..32 {
            for k in 0..3 {
                t.bap1_mantissas[i][k] =
                    symmetric_dequant(i32::from(UNGROUP_3_IN_5_BITS_TAB[i][k]), 3);
            }
        }
        for i in 0..128 {
            for k in 0..3 {
                t.bap2_mantissas[i][k] =
                    symmetric_dequant(i32::from(t.ungroup_3_in_7_bits[i][k]), 5);
            }
            t.bap4_mantissas[i] = [
                symmetric_dequant(i as i32 / 11, 11),
                symmetric_dequant(i as i32 % 11, 11),
            ];
        }
        for code in 0..7 {
            t.bap3_mantissas[code as usize] = symmetric_dequant(code, 7);
        }
        for code in 0..15 {
            t.bap5_mantissas[code as usize] = symmetric_dequant(code, 15);
        }
        for (i, f) in t.scale_factors.iter_mut().enumerate() {
            *f = 1.0 / (1u64 << i) as f32;
        }
        t
    }
}

/// Grouped mantissas for 3-, 5- and 11-level quantization.
#[derive(Default)]
struct MantGroups {
    b1_mant: [i32; 2],
    b2_mant: [i32; 2],
    b4_mant: i32,
    b1: usize,
    b2: usize,
    b4: i32,
}

/// Where FFmpeg's `outptr` / `output` pointers point: the last block kept
/// for concealment (`s->output[slot]`), or a block of the frame being
/// built (`s->output_buffer[slot] + pos`).
#[derive(Clone, Copy)]
enum OutRef {
    Last(usize),
    Buf(usize, usize),
}

/// `AC3DecodeContext` for the float decoder.
pub(crate) struct Ac3Context {
    imdct_128: Imdct,
    imdct_256: Imdct,
    window: Vec<f32>,
    outptr: [OutRef; AC3_MAX_CHANNELS],
    /// `avctx->sample_rate` / `avctx->ch_layout`: what the frames report.
    avctx_sample_rate: i32,
    ch_layout: u64,

    // bit stream information
    frame_type: i32,
    substreamid: i32,
    superframe_size: i32,
    frame_size: i32,
    sample_rate: i32,
    num_blocks: i32,
    bitstream_id: i32,
    channel_mode: i32,
    lfe_on: i32,
    dialog_normalization: [i32; 2],
    compression_exists: [i32; 2],
    channel_map: i32,
    center_mix_level: i32,
    surround_mix_level: i32,
    eac3: i32,
    eac3_subsbtreamid_found: i32,

    // options
    drc_scale: f32,
    heavy_compression: bool,
    target_level: i32,
    level_gain: [f32; 2],

    // frame syntax parameters
    snr_offset_strategy: i32,
    block_switch_syntax: i32,
    dither_flag_syntax: i32,
    bit_allocation_syntax: i32,
    fast_gain_syntax: i32,
    dba_syntax: i32,
    skip_syntax: i32,

    // standard coupling
    cpl_in_use: [i32; AC3_MAX_BLOCKS],
    cpl_strategy_exists: [i32; AC3_MAX_BLOCKS],
    channel_in_cpl: [i32; AC3_MAX_CHANNELS],
    phase_flags_in_use: i32,
    phase_flags: [i32; AC3_MAX_CPL_BANDS],
    num_cpl_bands: i32,
    cpl_band_struct: [u8; AC3_MAX_CPL_BANDS],
    cpl_band_sizes: [u8; AC3_MAX_CPL_BANDS],
    first_cpl_coords: [i32; AC3_MAX_CHANNELS],
    cpl_coords: [[i32; AC3_MAX_CPL_BANDS]; AC3_MAX_CHANNELS],

    // spectral extension
    spx_in_use: i32,
    channel_uses_spx: [u8; AC3_MAX_CHANNELS],
    spx_atten_code: [i8; AC3_MAX_CHANNELS],
    spx_src_start_freq: i32,
    spx_dst_end_freq: i32,
    spx_dst_start_freq: i32,
    num_spx_bands: i32,
    spx_band_struct: [u8; SPX_MAX_BANDS],
    spx_band_sizes: [u8; SPX_MAX_BANDS],
    first_spx_coords: [u8; AC3_MAX_CHANNELS],
    spx_noise_blend: [[f32; SPX_MAX_BANDS]; AC3_MAX_CHANNELS],
    spx_signal_blend: [[f32; SPX_MAX_BANDS]; AC3_MAX_CHANNELS],

    // adaptive hybrid transform
    channel_uses_aht: [i32; AC3_MAX_CHANNELS],
    pre_mantissa: Vec<[[i32; AC3_MAX_BLOCKS]; AC3_MAX_COEFS]>,

    // channel
    fbw_channels: i32,
    channels: i32,
    lfe_ch: i32,
    downmixed: i32,
    output_mode: i32,
    prev_output_mode: i32,
    out_channels: i32,

    // dynamic range
    dynamic_range: [f32; 2],
    heavy_dynamic_range: [f32; 2],

    // bandwidth
    start_freq: [i32; AC3_MAX_CHANNELS],
    end_freq: [i32; AC3_MAX_CHANNELS],

    // rematrixing
    num_rematrixing_bands: i32,
    rematrixing_flags: [i32; 4],

    // exponents
    num_exp_groups: [i32; AC3_MAX_CHANNELS],
    dexps: [[i8; AC3_MAX_COEFS]; AC3_MAX_CHANNELS],
    exp_strategy: [[i32; AC3_MAX_CHANNELS]; AC3_MAX_BLOCKS],

    // bit allocation
    bit_alloc_params: BitAllocParams,
    first_cpl_leak: i32,
    snr_offset: [i32; AC3_MAX_CHANNELS],
    fast_gain: [i32; AC3_MAX_CHANNELS],
    bap: [[u8; AC3_MAX_COEFS]; AC3_MAX_CHANNELS],
    psd: [[i16; AC3_MAX_COEFS]; AC3_MAX_CHANNELS],
    band_psd: [[i16; AC3_CRITICAL_BANDS]; AC3_MAX_CHANNELS],
    mask: [[i16; AC3_CRITICAL_BANDS]; AC3_MAX_CHANNELS],
    dba_mode: [i32; AC3_MAX_CHANNELS],
    dba_nsegs: [i32; AC3_MAX_CHANNELS],
    dba_offsets: [[u8; 8]; AC3_MAX_CHANNELS],
    dba_lengths: [[u8; 8]; AC3_MAX_CHANNELS],
    dba_values: [[u8; 8]; AC3_MAX_CHANNELS],

    // zero-mantissa dithering
    dither_flag: [i32; AC3_MAX_CHANNELS],
    dith_state: Lfg,

    // IMDCT
    block_switch: [i32; AC3_MAX_CHANNELS],

    coeffs: [[f32; AC3_MAX_COEFS]; AC3_MAX_CHANNELS],
    transform_coeffs: [[f32; AC3_MAX_COEFS]; AC3_MAX_CHANNELS],
    delay: Vec<[f32; AC3_BLOCK_SIZE]>,
    tmp_output: [f32; AC3_BLOCK_SIZE],
    output: Vec<[f32; AC3_BLOCK_SIZE]>,
    input_buffer: Vec<u8>,
    output_buffer: Vec<[f32; AC3_BLOCK_SIZE * 6]>,
}

impl Ac3Context {
    /// `ac3_decode_init` with the decoder's default options. `sample_rate`
    /// is the container's (`avctx->sample_rate` before the first frame).
    pub(crate) fn new(sample_rate: u32) -> Self {
        Self {
            imdct_128: Imdct::new(128),
            imdct_256: Imdct::new(256),
            window: kbd_window(5.0, 256),
            outptr: [OutRef::Last(0); AC3_MAX_CHANNELS],
            avctx_sample_rate: i32::try_from(sample_rate).unwrap_or(0),
            ch_layout: 0,
            frame_type: 0,
            substreamid: 0,
            superframe_size: 0,
            frame_size: 0,
            sample_rate: 0,
            num_blocks: 0,
            bitstream_id: 0,
            channel_mode: 0,
            lfe_on: 0,
            dialog_normalization: [0; 2],
            compression_exists: [0; 2],
            channel_map: 0,
            center_mix_level: 0,
            surround_mix_level: 0,
            eac3: 0,
            eac3_subsbtreamid_found: 0,
            drc_scale: 1.0,
            heavy_compression: false,
            target_level: 0,
            level_gain: [0.0; 2],
            snr_offset_strategy: 0,
            block_switch_syntax: 0,
            dither_flag_syntax: 0,
            bit_allocation_syntax: 0,
            fast_gain_syntax: 0,
            dba_syntax: 0,
            skip_syntax: 0,
            cpl_in_use: [0; AC3_MAX_BLOCKS],
            cpl_strategy_exists: [0; AC3_MAX_BLOCKS],
            channel_in_cpl: [0; AC3_MAX_CHANNELS],
            phase_flags_in_use: 0,
            phase_flags: [0; AC3_MAX_CPL_BANDS],
            num_cpl_bands: 0,
            cpl_band_struct: [0; AC3_MAX_CPL_BANDS],
            cpl_band_sizes: [0; AC3_MAX_CPL_BANDS],
            first_cpl_coords: [0; AC3_MAX_CHANNELS],
            cpl_coords: [[0; AC3_MAX_CPL_BANDS]; AC3_MAX_CHANNELS],
            spx_in_use: 0,
            channel_uses_spx: [0; AC3_MAX_CHANNELS],
            spx_atten_code: [0; AC3_MAX_CHANNELS],
            spx_src_start_freq: 0,
            spx_dst_end_freq: 0,
            spx_dst_start_freq: 0,
            num_spx_bands: 0,
            spx_band_struct: [0; SPX_MAX_BANDS],
            spx_band_sizes: [0; SPX_MAX_BANDS],
            first_spx_coords: [0; AC3_MAX_CHANNELS],
            spx_noise_blend: [[0.0; SPX_MAX_BANDS]; AC3_MAX_CHANNELS],
            spx_signal_blend: [[0.0; SPX_MAX_BANDS]; AC3_MAX_CHANNELS],
            channel_uses_aht: [0; AC3_MAX_CHANNELS],
            pre_mantissa: vec![[[0; AC3_MAX_BLOCKS]; AC3_MAX_COEFS]; AC3_MAX_CHANNELS],
            fbw_channels: 0,
            channels: 0,
            lfe_ch: 0,
            downmixed: 1,
            output_mode: 0,
            prev_output_mode: 0,
            out_channels: 0,
            dynamic_range: [0.0; 2],
            heavy_dynamic_range: [0.0; 2],
            start_freq: [0; AC3_MAX_CHANNELS],
            end_freq: [0; AC3_MAX_CHANNELS],
            num_rematrixing_bands: 0,
            rematrixing_flags: [0; 4],
            num_exp_groups: [0; AC3_MAX_CHANNELS],
            dexps: [[0; AC3_MAX_COEFS]; AC3_MAX_CHANNELS],
            exp_strategy: [[0; AC3_MAX_CHANNELS]; AC3_MAX_BLOCKS],
            bit_alloc_params: BitAllocParams::default(),
            first_cpl_leak: 0,
            snr_offset: [0; AC3_MAX_CHANNELS],
            fast_gain: [0; AC3_MAX_CHANNELS],
            bap: [[0; AC3_MAX_COEFS]; AC3_MAX_CHANNELS],
            psd: [[0; AC3_MAX_COEFS]; AC3_MAX_CHANNELS],
            band_psd: [[0; AC3_CRITICAL_BANDS]; AC3_MAX_CHANNELS],
            mask: [[0; AC3_CRITICAL_BANDS]; AC3_MAX_CHANNELS],
            dba_mode: [0; AC3_MAX_CHANNELS],
            dba_nsegs: [0; AC3_MAX_CHANNELS],
            dba_offsets: [[0; 8]; AC3_MAX_CHANNELS],
            dba_lengths: [[0; 8]; AC3_MAX_CHANNELS],
            dba_values: [[0; 8]; AC3_MAX_CHANNELS],
            dither_flag: [0; AC3_MAX_CHANNELS],
            dith_state: Lfg::seed0(),
            block_switch: [0; AC3_MAX_CHANNELS],
            coeffs: [[0.0; AC3_MAX_COEFS]; AC3_MAX_CHANNELS],
            transform_coeffs: [[0.0; AC3_MAX_COEFS]; AC3_MAX_CHANNELS],
            delay: vec![[0.0; AC3_BLOCK_SIZE]; EAC3_MAX_CHANNELS],
            tmp_output: [0.0; AC3_BLOCK_SIZE],
            output: vec![[0.0; AC3_BLOCK_SIZE]; EAC3_MAX_CHANNELS],
            input_buffer: vec![0; AC3_FRAME_BUFFER_SIZE],
            output_buffer: vec![[0.0; AC3_BLOCK_SIZE * 6]; EAC3_MAX_CHANNELS],
        }
    }

    /// `parse_frame_header`.
    fn parse_frame_header(&mut self, gbc: &mut BitReader) -> Result<(), HeaderError> {
        let hdr = parse_header(gbc).map_err(HeaderError::Parse)?;

        self.bit_alloc_params.sr_code = hdr.sr_code;
        self.bitstream_id = hdr.bitstream_id;
        self.channel_mode = hdr.channel_mode;
        self.lfe_on = hdr.lfe_on;
        self.bit_alloc_params.sr_shift = hdr.sr_shift;
        self.sample_rate = hdr.sample_rate;
        self.channels = hdr.channels;
        self.fbw_channels = self.channels - self.lfe_on;
        self.lfe_ch = self.fbw_channels + 1;
        self.frame_size = hdr.frame_size;
        self.superframe_size += hdr.frame_size;
        if hdr.bitstream_id <= 10 {
            self.center_mix_level = hdr.center_mix_level;
            self.surround_mix_level = hdr.surround_mix_level;
        }
        self.num_blocks = hdr.num_blocks;
        self.frame_type = hdr.frame_type;
        self.substreamid = hdr.substreamid;

        if self.lfe_on != 0 {
            let lfe = self.lfe_ch as usize;
            self.start_freq[lfe] = 0;
            self.end_freq[lfe] = 7;
            self.num_exp_groups[lfe] = 2;
            self.channel_in_cpl[lfe] = 0;
        }

        if self.bitstream_id <= 10 {
            self.eac3 = 0;
            self.snr_offset_strategy = 2;
            self.block_switch_syntax = 1;
            self.dither_flag_syntax = 1;
            self.bit_allocation_syntax = 1;
            self.fast_gain_syntax = 0;
            self.first_cpl_leak = 0;
            self.dba_syntax = 1;
            self.skip_syntax = 1;
            self.channel_uses_aht = [0; AC3_MAX_CHANNELS];
            for i in 0..(if self.channel_mode != 0 { 1 } else { 2 }) {
                self.dialog_normalization[i] = hdr.dialog_normalization[i];
                if self.dialog_normalization[i] == 0 {
                    self.dialog_normalization[i] = -31;
                }
                if self.target_level != 0 {
                    self.level_gain[i] =
                        2f32.powf((self.target_level - self.dialog_normalization[i]) as f32 / 6.0);
                }
                self.compression_exists[i] = hdr.compression_exists[i];
                if self.compression_exists[i] != 0 {
                    self.heavy_dynamic_range[i] =
                        tables().heavy_dynamic_range[hdr.heavy_dynamic_range[i] as usize];
                }
            }
            Ok(())
        } else {
            self.eac3 = 1;
            self.eac3_parse_header(gbc, &hdr)
        }
    }

    /// `ff_eac3_parse_header`.
    fn eac3_parse_header(
        &mut self,
        gbc: &mut BitReader,
        hdr: &HeaderInfo,
    ) -> Result<(), HeaderError> {
        if self.frame_type == EAC3_FRAME_TYPE_RESERVED {
            return Err(HeaderError::Parse(ParseError::FrameType));
        }
        if self.substreamid != 0 {
            self.eac3_subsbtreamid_found = 1;
            return Err(HeaderError::Parse(ParseError::FrameType));
        }
        if self.bit_alloc_params.sr_code == EAC3_SR_CODE_REDUCED {
            return Err(HeaderError::Fatal(DecodeError::Unsupported(
                "E-AC-3 reduced sampling rate",
            )));
        }

        for i in 0..(if self.channel_mode != 0 { 1 } else { 2 }) {
            self.dialog_normalization[i] = hdr.dialog_normalization[i];
            if self.dialog_normalization[i] == 0 {
                self.dialog_normalization[i] = -31;
            }
            if self.target_level != 0 {
                self.level_gain[i] =
                    2f32.powf((self.target_level - self.dialog_normalization[i]) as f32 / 6.0);
            }
            if hdr.compression_exists[i] != 0 {
                self.heavy_dynamic_range[i] =
                    tables().heavy_dynamic_range[hdr.heavy_dynamic_range[i] as usize];
            }
        }

        self.channel_map = hdr.channel_map;
        self.center_mix_level = hdr.center_mix_level;
        self.surround_mix_level = hdr.surround_mix_level;

        let fbw = self.fbw_channels as usize;
        let channels = self.channels as usize;
        let num_blocks = self.num_blocks as usize;

        let (ac3_exponent_strategy, parse_aht_info) = if num_blocks == 6 {
            (gbc.get1(), gbc.get1())
        } else {
            (1, 0)
        };

        self.snr_offset_strategy = gbc.get(2) as i32;
        let parse_transient_proc_info = gbc.get1();

        self.block_switch_syntax = gbc.get1() as i32;
        if self.block_switch_syntax == 0 {
            self.block_switch = [0; AC3_MAX_CHANNELS];
        }

        self.dither_flag_syntax = gbc.get1() as i32;
        if self.dither_flag_syntax == 0 {
            for ch in 1..=fbw {
                self.dither_flag[ch] = 1;
            }
        }
        self.dither_flag[CPL_CH] = 0;
        self.dither_flag[self.lfe_ch as usize] = 0;

        self.bit_allocation_syntax = gbc.get1() as i32;
        if self.bit_allocation_syntax == 0 {
            self.bit_alloc_params.slow_decay = SLOW_DECAY_TAB[2];
            self.bit_alloc_params.fast_decay = FAST_DECAY_TAB[1];
            self.bit_alloc_params.slow_gain = SLOW_GAIN_TAB[1];
            self.bit_alloc_params.db_per_bit = DB_PER_BIT_TAB[2];
            self.bit_alloc_params.floor = FLOOR_TAB[7];
        }

        self.fast_gain_syntax = gbc.get1() as i32;
        self.dba_syntax = gbc.get1() as i32;
        self.skip_syntax = gbc.get1() as i32;
        let parse_spx_atten_data = gbc.get1();

        // coupling strategy occurrence and coupling use per block
        let mut num_cpl_blocks = 0;
        if self.channel_mode > 1 {
            for blk in 0..num_blocks {
                self.cpl_strategy_exists[blk] = i32::from(blk == 0 || gbc.get1() != 0);
                if self.cpl_strategy_exists[blk] != 0 {
                    self.cpl_in_use[blk] = gbc.get1() as i32;
                } else {
                    self.cpl_in_use[blk] = self.cpl_in_use[blk - 1];
                }
                num_cpl_blocks += self.cpl_in_use[blk];
            }
        } else {
            self.cpl_in_use = [0; AC3_MAX_BLOCKS];
        }

        // exponent strategy data
        if ac3_exponent_strategy != 0 {
            for blk in 0..num_blocks {
                for ch in usize::from(self.cpl_in_use[blk] == 0)..=fbw {
                    self.exp_strategy[blk][ch] = gbc.get(2) as i32;
                }
            }
        } else {
            let first = usize::from(!(self.channel_mode > 1 && num_cpl_blocks != 0));
            for ch in first..=fbw {
                let frmchexpstr = gbc.get(5) as usize;
                for blk in 0..6 {
                    self.exp_strategy[blk][ch] = i32::from(EAC3_FRM_EXPSTR[frmchexpstr][blk]);
                }
            }
        }
        // LFE exponent strategy
        if self.lfe_on != 0 {
            for blk in 0..num_blocks {
                self.exp_strategy[blk][self.lfe_ch as usize] = gbc.get1() as i32;
            }
        }
        // original exponent strategies if this stream was converted from AC-3
        if self.frame_type == EAC3_FRAME_TYPE_INDEPENDENT && (num_blocks == 6 || gbc.get1() != 0) {
            gbc.skip(5 * fbw);
        }

        // determine which channels use AHT
        if parse_aht_info != 0 {
            self.channel_uses_aht[CPL_CH] = 0;
            for ch in usize::from(num_cpl_blocks != 6)..=channels {
                let mut use_aht = true;
                for blk in 1..6 {
                    if self.exp_strategy[blk][ch] != EXP_REUSE
                        || (ch == 0 && self.cpl_strategy_exists[blk] != 0)
                    {
                        use_aht = false;
                        break;
                    }
                }
                self.channel_uses_aht[ch] = i32::from(use_aht && gbc.get1() != 0);
            }
        } else {
            self.channel_uses_aht = [0; AC3_MAX_CHANNELS];
        }

        // per-frame SNR offset
        if self.snr_offset_strategy == 0 {
            let csnroffst = (gbc.get(6) as i32 - 15) << 4;
            let snroffst = (csnroffst + gbc.get(4) as i32) << 2;
            for ch in 0..=channels {
                self.snr_offset[ch] = snroffst;
            }
        }

        // transient pre-noise processing data
        if parse_transient_proc_info != 0 {
            for _ in 1..=fbw {
                if gbc.get1() != 0 {
                    gbc.skip(10); // transient processing location
                    gbc.skip(8); // transient processing length
                }
            }
        }

        // spectral extension attenuation data
        for ch in 1..=fbw {
            self.spx_atten_code[ch] = if parse_spx_atten_data != 0 && gbc.get1() != 0 {
                gbc.get(5) as i8
            } else {
                -1
            };
        }

        // block start information
        if num_blocks > 1 && gbc.get1() != 0 {
            let log2 = 31 - ((self.frame_size - 2).max(1) as u32).leading_zeros() as i32;
            let block_start_bits = (self.num_blocks - 1) * (4 + log2);
            gbc.skip(block_start_bits.max(0) as usize);
        }

        // syntax state initialization
        for ch in 1..=fbw {
            self.first_spx_coords[ch] = 1;
            self.first_cpl_coords[ch] = 1;
        }
        self.first_cpl_leak = 1;

        Ok(())
    }

    /// `decode_exponents`.
    fn decode_exponents(
        gbc: &mut BitReader,
        exp_strategy: i32,
        ngrps: i32,
        absexp: u8,
        dexps: &mut [i8],
    ) -> Result<(), ()> {
        let ungroup = &tables().ungroup_3_in_7_bits;
        let ngrps = ngrps.max(0) as usize;
        let group_size = exp_strategy + i32::from(exp_strategy == EXP_D45);
        if ngrps * 3 * group_size.max(1) as usize > dexps.len() {
            return Err(());
        }
        let mut dexp = [0i32; 256 * 3];
        if ngrps * 3 > dexp.len() {
            return Err(());
        }
        for grp in 0..ngrps {
            let expacc = gbc.get(7) as usize;
            if expacc >= 125 {
                return Err(());
            }
            for k in 0..3 {
                dexp[grp * 3 + k] = i32::from(ungroup[expacc][k]);
            }
        }

        let mut prevexp = i32::from(absexp);
        let mut j = 0;
        for &d in &dexp[..ngrps * 3] {
            prevexp += d - 2;
            if !(0..=24).contains(&prevexp) {
                return Err(());
            }
            let n = match group_size {
                4 => 4,
                2 => 2,
                1 => 1,
                _ => 0,
            };
            for _ in 0..n {
                dexps[j] = prevexp as i8;
                j += 1;
            }
        }
        Ok(())
    }

    /// `calc_transform_coeffs_cpl`.
    fn calc_transform_coeffs_cpl(&mut self) {
        let mut bin = self.start_freq[CPL_CH] as usize;
        for band in 0..self.num_cpl_bands as usize {
            let band_start = bin.min(AC3_MAX_COEFS);
            let band_end = (bin + usize::from(self.cpl_band_sizes[band])).min(AC3_MAX_COEFS);
            for ch in 1..=self.fbw_channels as usize {
                if self.channel_in_cpl[ch] != 0 {
                    let cpl_coord = self.cpl_coords[ch][band] as f32 * (1.0 / (1 << 23) as f32);
                    for b in band_start..band_end {
                        self.coeffs[ch][b] = self.coeffs[CPL_CH][b] * cpl_coord;
                    }
                    if ch == 2 && self.phase_flags[band] != 0 {
                        for b in band_start..band_end {
                            self.coeffs[2][b] = -self.coeffs[2][b];
                        }
                    }
                }
            }
            bin = band_end;
        }
    }

    /// `ac3_decode_transform_coeffs_ch`.
    fn ac3_decode_transform_coeffs_ch(
        &mut self,
        gbc: &mut BitReader,
        ch_index: usize,
        m: &mut MantGroups,
    ) {
        let t = tables();
        let start_freq = self.start_freq[ch_index].max(0) as usize;
        let end_freq = (self.end_freq[ch_index].max(0) as usize).min(AC3_MAX_COEFS);
        let dither = ch_index == CPL_CH || self.dither_flag[ch_index] != 0;

        for freq in start_freq..end_freq {
            let bap = self.bap[ch_index][freq];
            let mantissa: i32 = match bap {
                0 => {
                    if dither {
                        (((self.dith_state.get() >> 8) * 181) >> 8).wrapping_sub(5_931_008) as i32
                    } else {
                        0
                    }
                }
                1 => {
                    if m.b1 != 0 {
                        m.b1 -= 1;
                        m.b1_mant[m.b1]
                    } else {
                        let bits = gbc.get(5) as usize;
                        m.b1_mant[1] = t.bap1_mantissas[bits][1];
                        m.b1_mant[0] = t.bap1_mantissas[bits][2];
                        m.b1 = 2;
                        t.bap1_mantissas[bits][0]
                    }
                }
                2 => {
                    if m.b2 != 0 {
                        m.b2 -= 1;
                        m.b2_mant[m.b2]
                    } else {
                        let bits = gbc.get(7) as usize;
                        m.b2_mant[1] = t.bap2_mantissas[bits][1];
                        m.b2_mant[0] = t.bap2_mantissas[bits][2];
                        m.b2 = 2;
                        t.bap2_mantissas[bits][0]
                    }
                }
                3 => t.bap3_mantissas[gbc.get(3) as usize],
                4 => {
                    if m.b4 != 0 {
                        m.b4 = 0;
                        m.b4_mant
                    } else {
                        let bits = gbc.get(7) as usize;
                        m.b4_mant = t.bap4_mantissas[bits][1];
                        m.b4 = 1;
                        t.bap4_mantissas[bits][0]
                    }
                }
                5 => t.bap5_mantissas[gbc.get(4) as usize],
                _ => {
                    // 6 to 15: shift the mantissa and sign-extend it
                    let bap = usize::from(bap.min(15));
                    let bits = u32::from(QUANTIZATION_TAB[bap]);
                    ((gbc.get_s(bits) as u32) << (24 - bits)) as i32
                }
            };
            let exp = (self.dexps[ch_index][freq] as u8 & 31) as usize;
            self.coeffs[ch_index][freq] = mantissa as f32 * t.scale_factors[exp];
        }
    }

    /// `remove_dithering`.
    fn remove_dithering(&mut self) {
        for ch in 1..=self.fbw_channels as usize {
            if self.dither_flag[ch] == 0 && self.channel_in_cpl[ch] != 0 {
                let start = self.start_freq[CPL_CH].max(0) as usize;
                let end = (self.end_freq[CPL_CH].max(0) as usize).min(AC3_MAX_COEFS);
                for i in start..end {
                    if self.bap[CPL_CH][i] == 0 {
                        self.coeffs[ch][i] = 0.0;
                    }
                }
            }
        }
    }

    /// `decode_transform_coeffs_ch`.
    fn decode_transform_coeffs_ch(
        &mut self,
        gbc: &mut BitReader,
        blk: usize,
        ch: usize,
        m: &mut MantGroups,
    ) {
        if self.channel_uses_aht[ch] == 0 {
            self.ac3_decode_transform_coeffs_ch(gbc, ch, m);
        } else {
            // with AHT, the mantissas of all blocks are coded in block 0
            if blk == 0 {
                self.eac3_decode_transform_coeffs_aht_ch(gbc, ch);
            }
            let t = tables();
            let start = self.start_freq[ch].max(0) as usize;
            let end = (self.end_freq[ch].max(0) as usize).min(AC3_MAX_COEFS);
            for bin in start..end {
                let exp = (self.dexps[ch][bin] as u8 & 31) as usize;
                self.coeffs[ch][bin] =
                    self.pre_mantissa[ch][bin][blk] as f32 * t.scale_factors[exp];
            }
        }
    }

    /// `decode_transform_coeffs`.
    fn decode_transform_coeffs(&mut self, gbc: &mut BitReader, blk: usize) {
        let mut m = MantGroups::default();
        let mut got_cplchan = false;

        for ch in 1..=self.channels as usize {
            self.decode_transform_coeffs_ch(gbc, blk, ch, &mut m);
            // the coupling channel's coefficients follow those of the first
            // coupled channel
            let end = if self.channel_in_cpl[ch] != 0 {
                if !got_cplchan {
                    self.decode_transform_coeffs_ch(gbc, blk, CPL_CH, &mut m);
                    self.calc_transform_coeffs_cpl();
                    got_cplchan = true;
                }
                self.end_freq[CPL_CH]
            } else {
                self.end_freq[ch]
            };
            for c in (end.max(0) as usize).min(AC3_MAX_COEFS)..AC3_MAX_COEFS {
                self.coeffs[ch][c] = 0.0;
            }
        }

        self.remove_dithering();
    }

    /// `do_rematrixing`.
    fn do_rematrixing(&mut self) {
        let end = self.end_freq[1].min(self.end_freq[2]).max(0) as usize;
        for bnd in 0..self.num_rematrixing_bands.clamp(0, 4) as usize {
            if self.rematrixing_flags[bnd] != 0 {
                let bndend = end
                    .min(usize::from(REMATRIX_BAND_TAB[bnd + 1]))
                    .min(AC3_MAX_COEFS);
                for i in usize::from(REMATRIX_BAND_TAB[bnd])..bndend {
                    let tmp0 = self.coeffs[1][i];
                    self.coeffs[1][i] += self.coeffs[2][i];
                    self.coeffs[2][i] = tmp0 - self.coeffs[2][i];
                }
            }
        }
    }

    /// `do_imdct`.
    fn do_imdct(&mut self, channels: usize, offset: usize) {
        for ch in 1..=channels {
            let delay = ch - 1 + offset;
            let mut out = [0f32; AC3_BLOCK_SIZE];
            if self.block_switch[ch] != 0 {
                let mut x = [0f32; 128];
                for i in 0..128 {
                    x[i] = self.transform_coeffs[ch][2 * i];
                }
                self.imdct_128.inverse(&mut self.tmp_output[..128], &x);
                vector_fmul_window(
                    &mut out,
                    &self.delay[delay][..128],
                    &self.tmp_output[..128],
                    &self.window,
                    128,
                );
                for i in 0..128 {
                    x[i] = self.transform_coeffs[ch][2 * i + 1];
                }
                self.imdct_128.inverse(&mut self.delay[delay][..128], &x);
            } else {
                self.imdct_256
                    .inverse(&mut self.tmp_output, &self.transform_coeffs[ch]);
                vector_fmul_window(
                    &mut out,
                    &self.delay[delay][..128],
                    &self.tmp_output[..128],
                    &self.window,
                    128,
                );
                let (head, _) = self.delay[delay].split_at_mut(128);
                head.copy_from_slice(&self.tmp_output[128..256]);
            }
            self.write_out(self.outptr[ch - 1], &out);
        }
    }

    fn write_out(&mut self, dst: OutRef, block: &[f32; AC3_BLOCK_SIZE]) {
        match dst {
            OutRef::Last(slot) => self.output[slot] = *block,
            OutRef::Buf(slot, pos) => {
                self.output_buffer[slot][pos..pos + AC3_BLOCK_SIZE].copy_from_slice(block)
            }
        }
    }

    fn read_out(&self, src: OutRef) -> [f32; AC3_BLOCK_SIZE] {
        match src {
            OutRef::Last(slot) => self.output[slot],
            OutRef::Buf(slot, pos) => {
                let mut b = [0f32; AC3_BLOCK_SIZE];
                b.copy_from_slice(&self.output_buffer[slot][pos..pos + AC3_BLOCK_SIZE]);
                b
            }
        }
    }

    /// `ac3_upmix_delay`: FFmpeg runs it once, at the first block whose
    /// channels mix long and short transforms, even without a downmix.
    fn upmix_delay(&mut self) {
        let zero = [0f32; AC3_BLOCK_SIZE];
        match self.channel_mode {
            AC3_CHMODE_DUALMONO | AC3_CHMODE_STEREO => self.delay[1] = self.delay[0],
            AC3_CHMODE_2F2R => {
                self.delay[3] = zero;
                self.delay[2] = zero;
            }
            AC3_CHMODE_2F1R => self.delay[2] = zero,
            AC3_CHMODE_3F2R => {
                self.delay[4] = zero;
                self.delay[3] = zero;
                self.delay[2] = self.delay[1];
                self.delay[1] = zero;
            }
            AC3_CHMODE_3F1R => {
                self.delay[3] = zero;
                self.delay[2] = self.delay[1];
                self.delay[1] = zero;
            }
            AC3_CHMODE_3F => {
                self.delay[2] = self.delay[1];
                self.delay[1] = zero;
            }
            _ => {}
        }
    }

    /// `decode_band_structure`.
    #[allow(clippy::too_many_arguments)]
    fn decode_band_structure(
        gbc: &mut BitReader,
        blk: usize,
        eac3: bool,
        start_subband: usize,
        end_subband: usize,
        default_band_struct: &[u8],
        num_bands: &mut i32,
        band_sizes: &mut [u8],
        band_struct: &mut [u8],
    ) {
        let n_subbands = end_subband - start_subband;
        if blk == 0 {
            band_struct.copy_from_slice(default_band_struct);
        }
        let bs = &mut band_struct[start_subband + 1..];
        if !eac3 || gbc.get1() != 0 {
            for subbnd in 0..n_subbands - 1 {
                bs[subbnd] = gbc.get1() as u8;
            }
        }

        // number of bands and band sizes from the band structure
        let mut n_bands = n_subbands;
        let mut bnd_sz = [0u8; 22];
        bnd_sz[0] = 12;
        let mut bnd = 0;
        for subbnd in 1..n_subbands {
            if bs[subbnd - 1] != 0 {
                n_bands -= 1;
                bnd_sz[bnd] += 12;
            } else {
                bnd += 1;
                bnd_sz[bnd] = 12;
            }
        }
        *num_bands = n_bands as i32;
        band_sizes[..n_bands].copy_from_slice(&bnd_sz[..n_bands]);
    }

    /// `spx_strategy`.
    fn spx_strategy(&mut self, gbc: &mut BitReader, blk: usize) -> Result<(), ()> {
        let fbw = self.fbw_channels as usize;
        if self.channel_mode == AC3_CHMODE_MONO {
            self.channel_uses_spx[1] = 1;
        } else {
            let mut channel_uses_spx = gbc.get(fbw as u32);
            for ch in (1..=fbw).rev() {
                self.channel_uses_spx[ch] = (channel_uses_spx & 1) as u8;
                channel_uses_spx >>= 1;
            }
        }

        let dst_start_freq = gbc.get(2) as i32;
        let mut start_subband = gbc.get(3) as i32 + 2;
        if start_subband > 7 {
            start_subband += start_subband - 7;
        }
        let mut end_subband = gbc.get(3) as i32 + 5;
        if end_subband > 7 {
            end_subband += end_subband - 7;
        }
        let dst_start_freq = dst_start_freq * 12 + 25;
        let src_start_freq = start_subband * 12 + 25;
        let dst_end_freq = end_subband * 12 + 25;

        if start_subband >= end_subband || dst_start_freq >= src_start_freq {
            return Err(());
        }

        self.spx_dst_start_freq = dst_start_freq;
        self.spx_src_start_freq = src_start_freq;
        self.spx_dst_end_freq = dst_end_freq;

        Self::decode_band_structure(
            gbc,
            blk,
            self.eac3 != 0,
            start_subband as usize,
            end_subband as usize,
            &EAC3_DEFAULT_SPX_BAND_STRUCT,
            &mut self.num_spx_bands,
            &mut self.spx_band_sizes,
            &mut self.spx_band_struct,
        );
        Ok(())
    }

    /// `spx_coordinates`. (`av_clipf` is `FFMIN(FFMAX(a, lo), hi)`, which
    /// `max` / `min` reproduce for every input; `clamp` differs on NaN.)
    #[allow(clippy::manual_clamp)]
    fn spx_coordinates(&mut self, gbc: &mut BitReader) {
        for ch in 1..=self.fbw_channels as usize {
            if self.channel_uses_spx[ch] != 0 {
                if self.first_spx_coords[ch] != 0 || gbc.get1() != 0 {
                    self.first_spx_coords[ch] = 0;
                    let spx_blend = gbc.get(5) as f32 * (1.0 / 32.0);
                    let master_spx_coord = gbc.get(2) as i32 * 3;

                    let mut bin = self.spx_src_start_freq;
                    for bnd in 0..self.num_spx_bands as usize {
                        let bandsize = i32::from(self.spx_band_sizes[bnd]);

                        // blending factors
                        let mut nratio = (bin + (bandsize >> 1)) as f32
                            / self.spx_dst_end_freq as f32
                            - spx_blend;
                        nratio = nratio.max(0.0).min(1.0);
                        let nblend = (3.0 * nratio).sqrt(); // noise scaled by sqrt(3) for unity variance
                        let sblend = (1.0 - nratio).sqrt();
                        bin += bandsize;

                        // spx coordinates
                        let spx_coord_exp = gbc.get(4) as i32;
                        let mut spx_coord_mant = gbc.get(2) as i32;
                        if spx_coord_exp == 15 {
                            spx_coord_mant <<= 1;
                        } else {
                            spx_coord_mant += 4;
                        }
                        spx_coord_mant <<= 25 - spx_coord_exp - master_spx_coord;

                        let spx_coord = spx_coord_mant as f32 * (1.0 / (1 << 23) as f32);
                        self.spx_noise_blend[ch][bnd] = nblend * spx_coord;
                        self.spx_signal_blend[ch][bnd] = sblend * spx_coord;
                    }
                }
            } else {
                self.first_spx_coords[ch] = 1;
            }
        }
    }

    /// `coupling_strategy`.
    fn coupling_strategy(
        &mut self,
        gbc: &mut BitReader,
        blk: usize,
        bit_alloc_stages: &mut [u8; AC3_MAX_CHANNELS],
    ) -> Result<(), ()> {
        let fbw = self.fbw_channels as usize;
        *bit_alloc_stages = [3; AC3_MAX_CHANNELS];
        if self.eac3 == 0 {
            self.cpl_in_use[blk] = gbc.get1() as i32;
        }
        if self.cpl_in_use[blk] != 0 {
            if self.channel_mode < AC3_CHMODE_STEREO {
                return Err(()); // coupling not allowed in mono or dual-mono
            }
            if self.eac3 != 0 && gbc.get1() != 0 {
                return Err(()); // enhanced coupling (FFmpeg: AVERROR_PATCHWELCOME)
            }

            if self.eac3 != 0 && self.channel_mode == AC3_CHMODE_STEREO {
                self.channel_in_cpl[1] = 1;
                self.channel_in_cpl[2] = 1;
            } else {
                for ch in 1..=fbw {
                    self.channel_in_cpl[ch] = gbc.get1() as i32;
                }
            }

            if self.channel_mode == AC3_CHMODE_STEREO {
                self.phase_flags_in_use = gbc.get1() as i32;
            }

            let cpl_start_subband = gbc.get(4) as i32;
            let cpl_end_subband = if self.spx_in_use != 0 {
                (self.spx_src_start_freq - 37) / 12
            } else {
                gbc.get(4) as i32 + 3
            };
            if cpl_start_subband >= cpl_end_subband {
                return Err(());
            }
            self.start_freq[CPL_CH] = cpl_start_subband * 12 + 37;
            self.end_freq[CPL_CH] = cpl_end_subband * 12 + 37;

            Self::decode_band_structure(
                gbc,
                blk,
                self.eac3 != 0,
                cpl_start_subband as usize,
                cpl_end_subband as usize,
                &EAC3_DEFAULT_CPL_BAND_STRUCT,
                &mut self.num_cpl_bands,
                &mut self.cpl_band_sizes,
                &mut self.cpl_band_struct,
            );
        } else {
            for ch in 1..=fbw {
                self.channel_in_cpl[ch] = 0;
                self.first_cpl_coords[ch] = 1;
            }
            self.first_cpl_leak = self.eac3;
            self.phase_flags_in_use = 0;
        }
        Ok(())
    }

    /// `coupling_coordinates`.
    fn coupling_coordinates(&mut self, gbc: &mut BitReader, blk: usize) -> Result<(), ()> {
        let mut cpl_coords_exist = false;
        for ch in 1..=self.fbw_channels as usize {
            if self.channel_in_cpl[ch] != 0 {
                if (self.eac3 != 0 && self.first_cpl_coords[ch] != 0) || gbc.get1() != 0 {
                    self.first_cpl_coords[ch] = 0;
                    cpl_coords_exist = true;
                    let master_cpl_coord = 3 * gbc.get(2) as i32;
                    for bnd in 0..self.num_cpl_bands as usize {
                        let cpl_coord_exp = gbc.get(4) as i32;
                        let cpl_coord_mant = gbc.get(4) as i32;
                        let mut c = if cpl_coord_exp == 15 {
                            cpl_coord_mant << 22
                        } else {
                            (cpl_coord_mant + 16) << 21
                        };
                        c >>= cpl_coord_exp + master_cpl_coord;
                        self.cpl_coords[ch][bnd] = c;
                    }
                } else if blk == 0 {
                    return Err(()); // new coupling coordinates must be present in block 0
                }
            } else {
                self.first_cpl_coords[ch] = 1;
            }
        }
        if self.channel_mode == AC3_CHMODE_STEREO && cpl_coords_exist {
            for bnd in 0..self.num_cpl_bands as usize {
                self.phase_flags[bnd] = if self.phase_flags_in_use != 0 {
                    gbc.get1() as i32
                } else {
                    0
                };
            }
        }
        Ok(())
    }

    /// `decode_audio_block`.
    fn decode_audio_block(
        &mut self,
        gbc: &mut BitReader,
        blk: usize,
        offset: usize,
    ) -> Result<(), ()> {
        let fbw = self.fbw_channels as usize;
        let channels = self.channels as usize;
        let lfe_ch = self.lfe_ch as usize;
        let channel_mode = self.channel_mode;
        let mut bit_alloc_stages = [0u8; AC3_MAX_CHANNELS];

        // block switch flags
        let mut different_transforms = false;
        if self.block_switch_syntax != 0 {
            for ch in 1..=fbw {
                self.block_switch[ch] = gbc.get1() as i32;
                if ch > 1 && self.block_switch[ch] != self.block_switch[1] {
                    different_transforms = true;
                }
            }
        }

        // dithering flags
        if self.dither_flag_syntax != 0 {
            for ch in 1..=fbw {
                self.dither_flag[ch] = gbc.get1() as i32;
            }
        }

        // dynamic range
        let mut i = usize::from(self.channel_mode == 0);
        loop {
            if gbc.get1() != 0 {
                // drc_scale > 1 applies DRC asymmetrically, enhancing quiet sounds
                let range_bits = gbc.get(8) as usize;
                let range = tables().dynamic_range[range_bits];
                self.dynamic_range[i] = if range_bits <= 127 || self.drc_scale <= 1.0 {
                    range.powf(self.drc_scale)
                } else {
                    range
                };
            } else if blk == 0 {
                self.dynamic_range[i] = 1.0;
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }

        // spectral extension strategy
        if self.eac3 != 0 && (blk == 0 || gbc.get1() != 0) {
            self.spx_in_use = gbc.get1() as i32;
            if self.spx_in_use != 0 {
                self.spx_strategy(gbc, blk)?;
            }
        }
        if self.eac3 == 0 || self.spx_in_use == 0 {
            self.spx_in_use = 0;
            for ch in 1..=fbw {
                self.channel_uses_spx[ch] = 0;
                self.first_spx_coords[ch] = 1;
            }
        }

        // spectral extension coordinates
        if self.spx_in_use != 0 {
            self.spx_coordinates(gbc);
        }

        // coupling strategy
        let new_cpl_strategy = if self.eac3 != 0 {
            self.cpl_strategy_exists[blk] != 0
        } else {
            gbc.get1() != 0
        };
        if new_cpl_strategy {
            self.coupling_strategy(gbc, blk, &mut bit_alloc_stages)?;
        } else if self.eac3 == 0 {
            if blk == 0 {
                return Err(()); // new coupling strategy must be present in block 0
            }
            self.cpl_in_use[blk] = self.cpl_in_use[blk - 1];
        }
        let cpl_in_use = self.cpl_in_use[blk] != 0;
        let first_ch = usize::from(!cpl_in_use);

        // coupling coordinates
        if cpl_in_use {
            self.coupling_coordinates(gbc, blk)?;
        }

        // stereo rematrixing strategy and band structure
        if channel_mode == AC3_CHMODE_STEREO {
            if (self.eac3 != 0 && blk == 0) || gbc.get1() != 0 {
                self.num_rematrixing_bands = 4;
                if cpl_in_use && self.start_freq[CPL_CH] <= 61 {
                    self.num_rematrixing_bands -= 1 + i32::from(self.start_freq[CPL_CH] == 37);
                } else if self.spx_in_use != 0 && self.spx_src_start_freq <= 61 {
                    self.num_rematrixing_bands -= 1;
                }
                for bnd in 0..self.num_rematrixing_bands as usize {
                    self.rematrixing_flags[bnd] = gbc.get1() as i32;
                }
            } else if blk == 0 {
                // new rematrixing strategy not present in block 0
                self.num_rematrixing_bands = 0;
            }
        }

        // exponent strategies for each channel
        for ch in first_ch..=channels {
            if self.eac3 == 0 {
                self.exp_strategy[blk][ch] = gbc.get(2 - u32::from(ch == lfe_ch)) as i32;
            }
            if self.exp_strategy[blk][ch] != EXP_REUSE {
                bit_alloc_stages[ch] = 3;
            }
        }

        // channel bandwidth
        for ch in 1..=fbw {
            self.start_freq[ch] = 0;
            if self.exp_strategy[blk][ch] != EXP_REUSE {
                let prev = self.end_freq[ch];
                if self.channel_in_cpl[ch] != 0 {
                    self.end_freq[ch] = self.start_freq[CPL_CH];
                } else if self.channel_uses_spx[ch] != 0 {
                    self.end_freq[ch] = self.spx_src_start_freq;
                } else {
                    let bandwidth_code = gbc.get(6) as i32;
                    if bandwidth_code > 60 {
                        return Err(());
                    }
                    self.end_freq[ch] = bandwidth_code * 3 + 73;
                }
                let group_size = 3 << (self.exp_strategy[blk][ch] - 1);
                self.num_exp_groups[ch] = (self.end_freq[ch] + group_size - 4) / group_size;
                if blk > 0 && self.end_freq[ch] != prev {
                    bit_alloc_stages = [3; AC3_MAX_CHANNELS];
                }
            }
        }
        if cpl_in_use && self.exp_strategy[blk][CPL_CH] != EXP_REUSE {
            self.num_exp_groups[CPL_CH] = (self.end_freq[CPL_CH] - self.start_freq[CPL_CH])
                / (3 << (self.exp_strategy[blk][CPL_CH] - 1));
        }

        // decode exponents for each channel
        for ch in first_ch..=channels {
            if self.exp_strategy[blk][ch] != EXP_REUSE {
                let absexp = (gbc.get(4) << u32::from(ch == 0)) as i8;
                self.dexps[ch][0] = absexp;
                let start = (self.start_freq[ch] + i32::from(ch != 0)).max(0) as usize;
                if start > AC3_MAX_COEFS {
                    return Err(());
                }
                let (strategy, ngrps) = (self.exp_strategy[blk][ch], self.num_exp_groups[ch]);
                Self::decode_exponents(
                    gbc,
                    strategy,
                    ngrps,
                    absexp as u8,
                    &mut self.dexps[ch][start..],
                )?;
                if ch != CPL_CH && ch != lfe_ch {
                    gbc.skip(2); // gainrng
                }
            }
        }

        // bit allocation information
        if self.bit_allocation_syntax != 0 {
            if gbc.get1() != 0 {
                let sr_shift = self.bit_alloc_params.sr_shift;
                self.bit_alloc_params.slow_decay = SLOW_DECAY_TAB[gbc.get(2) as usize] >> sr_shift;
                self.bit_alloc_params.fast_decay = FAST_DECAY_TAB[gbc.get(2) as usize] >> sr_shift;
                self.bit_alloc_params.slow_gain = SLOW_GAIN_TAB[gbc.get(2) as usize];
                self.bit_alloc_params.db_per_bit = DB_PER_BIT_TAB[gbc.get(2) as usize];
                self.bit_alloc_params.floor = FLOOR_TAB[gbc.get(3) as usize];
                for ch in first_ch..=channels {
                    bit_alloc_stages[ch] = bit_alloc_stages[ch].max(2);
                }
            } else if blk == 0 {
                return Err(()); // new bit allocation info must be present in block 0
            }
        }

        // signal-to-noise ratio offsets and fast gains (signal-to-mask ratios)
        if self.eac3 == 0 || blk == 0 {
            if self.snr_offset_strategy != 0 && gbc.get1() != 0 {
                let mut snr = 0;
                let csnr = (gbc.get(6) as i32 - 15) << 4;
                for ch in first_ch..=channels {
                    if ch == first_ch || self.snr_offset_strategy == 2 {
                        snr = (csnr + gbc.get(4) as i32) << 2;
                    }
                    // run at least the last bit allocation stage if the snr offset changes
                    if blk != 0 && self.snr_offset[ch] != snr {
                        bit_alloc_stages[ch] = bit_alloc_stages[ch].max(1);
                    }
                    self.snr_offset[ch] = snr;

                    // fast gain (normal AC-3 only)
                    if self.eac3 == 0 {
                        let prev = self.fast_gain[ch];
                        self.fast_gain[ch] = FAST_GAIN_TAB[gbc.get(3) as usize];
                        if blk != 0 && prev != self.fast_gain[ch] {
                            bit_alloc_stages[ch] = bit_alloc_stages[ch].max(2);
                        }
                    }
                }
            } else if self.eac3 == 0 && blk == 0 {
                return Err(()); // new snr offsets must be present in block 0
            }
        }

        // fast gain (E-AC-3 only)
        if self.fast_gain_syntax != 0 && gbc.get1() != 0 {
            for ch in first_ch..=channels {
                let prev = self.fast_gain[ch];
                self.fast_gain[ch] = FAST_GAIN_TAB[gbc.get(3) as usize];
                if blk != 0 && prev != self.fast_gain[ch] {
                    bit_alloc_stages[ch] = bit_alloc_stages[ch].max(2);
                }
            }
        } else if self.eac3 != 0 && blk == 0 {
            for ch in first_ch..=channels {
                self.fast_gain[ch] = FAST_GAIN_TAB[4];
            }
        }

        // E-AC-3 to AC-3 converter SNR offset
        if self.frame_type == EAC3_FRAME_TYPE_INDEPENDENT && gbc.get1() != 0 {
            gbc.skip(10);
        }

        // coupling leak information
        if cpl_in_use {
            if self.first_cpl_leak != 0 || gbc.get1() != 0 {
                let fl = gbc.get(3) as i32;
                let sl = gbc.get(3) as i32;
                // run the last 2 bit allocation stages for the coupling channel if the leak changes
                if blk != 0
                    && (fl != self.bit_alloc_params.cpl_fast_leak
                        || sl != self.bit_alloc_params.cpl_slow_leak)
                {
                    bit_alloc_stages[CPL_CH] = bit_alloc_stages[CPL_CH].max(2);
                }
                self.bit_alloc_params.cpl_fast_leak = fl;
                self.bit_alloc_params.cpl_slow_leak = sl;
            } else if self.eac3 == 0 && blk == 0 {
                return Err(()); // new coupling leak info must be present in block 0
            }
            self.first_cpl_leak = 0;
        }

        // delta bit allocation information
        if self.dba_syntax != 0 && gbc.get1() != 0 {
            for ch in first_ch..=fbw {
                self.dba_mode[ch] = gbc.get(2) as i32;
                if self.dba_mode[ch] == DBA_RESERVED {
                    return Err(());
                }
                bit_alloc_stages[ch] = bit_alloc_stages[ch].max(2);
            }
            for ch in first_ch..=fbw {
                if self.dba_mode[ch] == DBA_NEW {
                    self.dba_nsegs[ch] = gbc.get(3) as i32 + 1;
                    for seg in 0..self.dba_nsegs[ch] as usize {
                        self.dba_offsets[ch][seg] = gbc.get(5) as u8;
                        self.dba_lengths[ch][seg] = gbc.get(4) as u8;
                        self.dba_values[ch][seg] = gbc.get(3) as u8;
                    }
                    bit_alloc_stages[ch] = bit_alloc_stages[ch].max(2);
                }
            }
        } else if blk == 0 {
            for ch in 0..=channels {
                self.dba_mode[ch] = DBA_NONE;
            }
        }

        // bit allocation
        for ch in first_ch..=channels {
            let start = self.start_freq[ch].max(0) as usize;
            let end = (self.end_freq[ch].max(0) as usize).min(253);
            if bit_alloc_stages[ch] > 2 {
                calc_psd(
                    &self.dexps[ch],
                    start,
                    end,
                    &mut self.psd[ch],
                    &mut self.band_psd[ch],
                );
            }
            if bit_alloc_stages[ch] > 1 {
                let dba = Dba {
                    mode: self.dba_mode[ch],
                    nsegs: self.dba_nsegs[ch],
                    offsets: &self.dba_offsets[ch],
                    lengths: &self.dba_lengths[ch],
                    values: &self.dba_values[ch],
                };
                calc_mask(
                    &self.bit_alloc_params,
                    &self.band_psd[ch],
                    start.min(252),
                    end,
                    self.fast_gain[ch],
                    ch == lfe_ch,
                    dba,
                    &mut self.mask[ch],
                )?;
            }
            if bit_alloc_stages[ch] > 0 {
                let bap_tab = if self.channel_uses_aht[ch] != 0 {
                    &EAC3_HEBAP_TAB
                } else {
                    &BAP_TAB
                };
                calc_bap(
                    &self.mask[ch],
                    &self.psd[ch],
                    start,
                    end,
                    self.snr_offset[ch],
                    self.bit_alloc_params.floor,
                    bap_tab,
                    &mut self.bap[ch],
                );
            }
        }

        // unused dummy data
        if self.skip_syntax != 0 && gbc.get1() != 0 {
            let skipl = gbc.get(9) as usize;
            gbc.skip(8 * skipl);
        }

        // unpack the transform coefficients; this also uncouples channels
        self.decode_transform_coeffs(gbc, blk);

        // recover coefficients if rematrixing is in use
        if self.channel_mode == AC3_CHMODE_STEREO {
            self.do_rematrixing();
        }

        // apply scaling to coefficients (headroom, dynrng)
        for ch in 1..=channels {
            let audio_channel = if self.channel_mode == AC3_CHMODE_DUALMONO && ch <= 2 {
                2 - ch
            } else {
                0
            };
            let mut gain = if self.heavy_compression && self.compression_exists[audio_channel] != 0
            {
                self.heavy_dynamic_range[audio_channel]
            } else {
                self.dynamic_range[audio_channel]
            };
            if self.target_level != 0 {
                gain *= self.level_gain[audio_channel];
            }
            gain *= 1.0 / 4_194_304.0;
            for k in 0..AC3_MAX_COEFS {
                self.transform_coeffs[ch][k] = self.coeffs[ch][k] * gain;
            }
        }

        // apply spectral extension to high frequency bins
        if self.spx_in_use != 0 {
            self.apply_spectral_extension();
        }

        // MDCT. The downmix this port leaves out would mix before or after
        // it; the delay upmix FFmpeg runs on the first block mixing long and
        // short transforms happens either way.
        if different_transforms {
            if self.downmixed != 0 {
                self.downmixed = 0;
                self.upmix_delay();
            }
            self.do_imdct(channels, offset);
        } else {
            self.do_imdct(self.out_channels.max(0) as usize, offset);
        }

        Ok(())
    }

    /// `ff_eac3_apply_spectral_extension`.
    fn apply_spectral_extension(&mut self) {
        let mut wrapflag = [0u8; SPX_MAX_BANDS];
        wrapflag[0] = 1;
        let mut copy_sizes: Vec<i32> = Vec::with_capacity(2 * SPX_MAX_BANDS);
        let mut rms_energy = [0f32; SPX_MAX_BANDS];
        let src_start = self.spx_src_start_freq;
        let dst_start = self.spx_dst_start_freq;
        let num_bands = (self.num_spx_bands.max(0) as usize).min(SPX_MAX_BANDS);

        // copy index mapping; wrap flags mark where the notch filter goes
        let mut bin = dst_start;
        for bnd in 0..num_bands {
            let bandsize = i32::from(self.spx_band_sizes[bnd]);
            if bin + bandsize > src_start {
                copy_sizes.push(bin - dst_start);
                bin = dst_start;
                wrapflag[bnd] = 1;
            }
            let mut i = 0;
            while i < bandsize {
                if bin == src_start {
                    copy_sizes.push(bin - dst_start);
                    bin = dst_start;
                }
                let copysize = (bandsize - i).min(src_start - bin);
                if copysize <= 0 {
                    break;
                }
                bin += copysize;
                i += copysize;
            }
        }
        copy_sizes.push(bin - dst_start);

        for ch in 1..=self.fbw_channels as usize {
            if self.channel_uses_spx[ch] == 0 {
                continue;
            }
            let coeffs = &mut self.transform_coeffs[ch];

            // copy coeffs from normal bands to extension bands
            let mut bin = src_start as usize;
            for &n in &copy_sizes {
                let n = n.max(0) as usize;
                let src = dst_start as usize;
                if bin + n > AC3_MAX_COEFS || src + n > AC3_MAX_COEFS {
                    break;
                }
                coeffs.copy_within(src..src + n, bin);
                bin += n;
            }

            // RMS energy of each SPX band
            let mut bin = src_start as usize;
            for bnd in 0..num_bands {
                let bandsize = usize::from(self.spx_band_sizes[bnd]);
                let mut accum = 0f32;
                for _ in 0..bandsize {
                    let c = coeffs.get(bin).copied().unwrap_or(0.0);
                    bin += 1;
                    accum += c * c;
                }
                rms_energy[bnd] = (accum / bandsize as f32).sqrt();
            }

            // notch filter at transitions between normal and extension bands
            // and at all wrap points
            if self.spx_atten_code[ch] >= 0 {
                let atten = EAC3_SPX_ATTEN_TAB[(self.spx_atten_code[ch] as usize).min(31)];
                let mut bin = (src_start - 2).max(0) as usize;
                for bnd in 0..num_bands {
                    if wrapflag[bnd] != 0 && bin + 5 <= AC3_MAX_COEFS {
                        let c = &mut coeffs[bin..bin + 5];
                        c[0] *= atten[0];
                        c[1] *= atten[1];
                        c[2] *= atten[2];
                        c[3] *= atten[1];
                        c[4] *= atten[0];
                    }
                    bin += usize::from(self.spx_band_sizes[bnd]);
                }
            }

            // noise-blended scaling from the RMS energy, the blending factors
            // and the SPX coordinates of each band
            let mut bin = src_start as usize;
            for bnd in 0..num_bands {
                let nscale =
                    self.spx_noise_blend[ch][bnd] * rms_energy[bnd] * (1.0 / i32::MIN as f32);
                let sscale = self.spx_signal_blend[ch][bnd];
                for _ in 0..self.spx_band_sizes[bnd] {
                    let noise = nscale * (self.dith_state.get() as i32) as f32;
                    if let Some(c) = coeffs.get_mut(bin) {
                        *c *= sscale;
                        *c += noise;
                    }
                    bin += 1;
                }
            }
        }
    }

    /// `ff_eac3_decode_transform_coeffs_aht_ch`.
    fn eac3_decode_transform_coeffs_aht_ch(&mut self, gbc: &mut BitReader, ch: usize) {
        let start = self.start_freq[ch].max(0) as usize;
        let end = (self.end_freq[ch].max(0) as usize).min(AC3_MAX_COEFS);
        let mut gaq_gain = [0i32; AC3_MAX_COEFS + 3];

        let gaq_mode = gbc.get(2) as i32;
        let end_bap = if gaq_mode < 2 { 12 } else { 17 };

        // GAQ gain codes for bins with hebap between 8 and end_bap
        let mut gs = 0;
        if gaq_mode == EAC3_GAQ_12 || gaq_mode == EAC3_GAQ_14 {
            // 1-bit GAQ gain codes
            for bin in start..end {
                let b = self.bap[ch][bin];
                if b > 7 && b < end_bap {
                    gaq_gain[gs] = (gbc.get1() << (gaq_mode - 1)) as i32;
                    gs += 1;
                }
            }
        } else if gaq_mode == EAC3_GAQ_124 {
            // 1.67-bit GAQ gain codes (3 codes in 5 bits)
            let mut gc = 2;
            for bin in start..end {
                let b = self.bap[ch][bin];
                if b > 7 && b < 17 {
                    let take = gc == 2;
                    gc += 1;
                    if take {
                        let group_code = (gbc.get(5) as usize).min(26);
                        for k in 0..3 {
                            gaq_gain[gs] = i32::from(UNGROUP_3_IN_5_BITS_TAB[group_code][k]);
                            gs += 1;
                        }
                        gc = 0;
                    }
                }
            }
        }

        let mut gs = 0;
        for bin in start..end {
            let hebap = usize::from(self.bap[ch][bin]).min(19);
            let bits = u32::from(EAC3_BITS_VS_HEBAP[hebap]);
            let pre = &mut self.pre_mantissa[ch][bin];
            if hebap == 0 {
                // zero-mantissa dithering
                for blk in 0..6 {
                    pre[blk] = (self.dith_state.get() & 0x7F_FFFF) as i32 - 0x40_0000;
                }
            } else if hebap < 8 {
                // vector quantization
                let v = gbc.get(bits) as usize;
                let row: [i16; 6] = match hebap {
                    1 => VQ_HEBAP1[v],
                    2 => VQ_HEBAP2[v],
                    3 => VQ_HEBAP3[v],
                    4 => VQ_HEBAP4[v],
                    5 => VQ_HEBAP5[v],
                    6 => VQ_HEBAP6[v],
                    _ => VQ_HEBAP7[v],
                };
                for blk in 0..6 {
                    pre[blk] = i32::from(row[blk]) * (1 << 8);
                }
            } else {
                // gain adaptive quantization
                let log_gain = if gaq_mode != EAC3_GAQ_NO && (hebap as u8) < end_bap {
                    let g = gaq_gain[gs.min(gaq_gain.len() - 1)];
                    gs += 1;
                    g
                } else {
                    0
                };
                let gbits = (bits as i32 - log_gain).max(1) as u32;

                for blk in 0..6 {
                    let mut mant = gbc.get_s(gbits);
                    if log_gain != 0 && mant == -(1 << (gbits - 1)) {
                        // large mantissa
                        let mbits = (bits as i32 - (2 - log_gain)).max(1) as u32;
                        mant = gbc.get_s(mbits);
                        mant = ((mant as u32) << (23 - (mbits - 1))) as i32;
                        // remap the mantissa to correct for asymmetric quantization
                        let b = if mant >= 0 {
                            1 << (23 - log_gain)
                        } else {
                            EAC3_GAQ_REMAP_2_4_B[hebap - 8][(log_gain - 1) as usize] * (1 << 8)
                        };
                        let a = i64::from(EAC3_GAQ_REMAP_2_4_A[hebap - 8][(log_gain - 1) as usize]);
                        mant = mant
                            .wrapping_add(((a * i64::from(mant)) >> 15) as i32)
                            .wrapping_add(b);
                    } else {
                        // small mantissa, no GAQ, or Gk=1
                        mant = mant.wrapping_mul(1 << (24 - bits));
                        if log_gain == 0 {
                            // remap for no GAQ or Gk=1
                            let r = i64::from(EAC3_GAQ_REMAP_1[hebap - 8]);
                            mant = mant.wrapping_add(((r * i64::from(mant)) >> 15) as i32);
                        }
                    }
                    pre[blk] = mant;
                }
            }
            idct6(pre);
        }
    }

    /// `ac3_decode_frame`: bytes of `pkt` consumed and, when FFmpeg sets
    /// `got_frame`, the frame.
    pub(crate) fn decode_frame(
        &mut self,
        pkt: &[u8],
    ) -> Result<(usize, Option<DecodedFrame>), DecodeError> {
        let mut input = std::mem::take(&mut self.input_buffer);
        let result = self.decode_frame_from(pkt, &mut input);
        self.input_buffer = input;
        result
    }

    fn decode_frame_from(
        &mut self,
        pkt: &[u8],
        input: &mut [u8],
    ) -> Result<(usize, Option<DecodedFrame>), DecodeError> {
        let full_buf_size = pkt.len();
        self.superframe_size = 0;

        let Some(sync) = find_syncword(pkt) else {
            return Err(DecodeError::InvalidData("no AC-3 sync word"));
        };
        if sync > 10 {
            return Ok((sync, None));
        }
        let src = &pkt[sync..];
        let mut buf_size = src.len();

        // copy the input to the context so a damaged stream cannot make the
        // reader run past its end; byte-swapped AC-3 is swapped back
        let n = buf_size.min(AC3_FRAME_BUFFER_SIZE);
        if buf_size >= 2 && src[0] == 0x77 && src[1] == 0x0B {
            for k in 0..n >> 1 {
                input[2 * k] = src[2 * k + 1];
                input[2 * k + 1] = src[2 * k];
            }
        } else {
            input[..n].copy_from_slice(&src[..n]);
        }

        let mut buf_off = 0usize;
        let mut skip = 0usize;
        let mut got_independent_frame = false;
        let mut err;
        let mut channel_map;
        let mut offset;

        loop {
            // dependent_frame:
            let mut gbc = BitReader::new(input.get(buf_off..).unwrap_or(&[]), buf_size);

            err = match self.parse_frame_header(&mut gbc) {
                Ok(()) => i32::from(self.frame_size as usize > buf_size), // incomplete frame
                Err(HeaderError::Parse(ParseError::Sync)) => {
                    return Err(DecodeError::InvalidData("frame sync error"));
                }
                Err(HeaderError::Parse(ParseError::FrameType)) => {
                    // skip an unsupported substream, conceal anything else
                    if self.substreamid != 0 {
                        return Ok((buf_size, None));
                    }
                    1
                }
                Err(HeaderError::Parse(ParseError::ChannelMap)) => {
                    return Err(DecodeError::InvalidData("invalid channel map"));
                }
                Err(HeaderError::Parse(_)) => 1,
                Err(HeaderError::Fatal(e)) => return Err(e),
            };

            if self.frame_type == EAC3_FRAME_TYPE_DEPENDENT && !got_independent_frame {
                // a dependent frame without its independent frame
                return Ok((full_buf_size.min(self.frame_size.max(0) as usize), None));
            }

            // channel config
            if err == 0 || (self.channels != 0 && self.out_channels != self.channels) {
                self.out_channels = self.channels;
                self.output_mode = self.channel_mode;
                if self.lfe_on != 0 {
                    self.output_mode |= AC3_OUTPUT_LFEON;
                }
            } else if self.channels == 0 {
                return Err(DecodeError::InvalidData("unable to determine channel mode"));
            }

            let mode = (self.output_mode & !AC3_OUTPUT_LFEON) as usize & 7;
            self.ch_layout = CHANNEL_LAYOUT_TAB[mode];
            if self.output_mode & AC3_OUTPUT_LFEON != 0 {
                self.ch_layout |= CH_LOW_FREQUENCY;
            }

            // decode the audio blocks
            channel_map = DEC_CHANNEL_MAP[mode][(self.lfe_on & 1) as usize];
            offset = if self.frame_type == EAC3_FRAME_TYPE_DEPENDENT {
                AC3_MAX_CHANNELS
            } else {
                0
            };
            let out_channels = (self.out_channels.max(0) as usize).min(6);
            let mut output = [OutRef::Last(0); AC3_MAX_CHANNELS];
            for ch in 0..AC3_MAX_CHANNELS {
                output[ch] = OutRef::Last(ch + offset);
                self.outptr[ch] = OutRef::Last(ch + offset);
            }
            for ch in 0..(self.channels.max(0) as usize).min(out_channels) {
                self.outptr[usize::from(channel_map[ch])] = OutRef::Buf(ch + offset, 0);
            }
            for blk in 0..(self.num_blocks.max(0) as usize).min(AC3_MAX_BLOCKS) {
                if err == 0 && self.decode_audio_block(&mut gbc, blk, offset).is_err() {
                    err = 1;
                }
                if err != 0 {
                    for ch in 0..out_channels {
                        let block = self.read_out(output[ch]);
                        self.output_buffer[ch + offset]
                            [AC3_BLOCK_SIZE * blk..AC3_BLOCK_SIZE * (blk + 1)]
                            .copy_from_slice(&block);
                    }
                }
                for ch in 0..out_channels {
                    output[ch] = self.outptr[usize::from(channel_map[ch])];
                }
                for ch in 0..out_channels {
                    if ch == 0 || channel_map[ch] != 0 {
                        let p = &mut self.outptr[usize::from(channel_map[ch])];
                        if let OutRef::Buf(slot, pos) = *p {
                            *p = OutRef::Buf(slot, (pos + AC3_BLOCK_SIZE).min(AC3_BLOCK_SIZE * 5));
                        }
                    }
                }
            }

            // keep the last block for error concealment in the next frame
            for ch in 0..out_channels {
                self.output[ch + offset] = self.read_out(output[ch]);
            }

            // check for a dependent frame
            let frame_size = self.frame_size.max(0) as usize;
            if buf_size > frame_size {
                if buf_size - frame_size <= 16 {
                    skip = buf_size - frame_size;
                    break;
                }
                let next = input.get(buf_off + frame_size..).unwrap_or(&[]);
                let mut g2 = BitReader::new(next, buf_size - frame_size);
                let hdr = parse_header(&mut g2)
                    .map_err(|_| DecodeError::InvalidData("invalid header after the frame"))?;
                if hdr.frame_type == EAC3_FRAME_TYPE_DEPENDENT
                    && hdr.num_blocks == self.num_blocks
                    && self.sample_rate == hdr.sample_rate
                {
                    buf_off += frame_size;
                    buf_size -= frame_size;
                    self.prev_output_mode = self.output_mode;
                    got_independent_frame = true;
                    continue;
                }
            }
            break;
        }

        // skip:
        if err == 0 {
            self.avctx_sample_rate = self.sample_rate;
        }
        if self.avctx_sample_rate == 0 {
            return Err(DecodeError::InvalidData(
                "could not determine the sample rate",
            ));
        }

        let mut extended_channel_map = [0usize; EAC3_MAX_CHANNELS];
        for (ch, m) in extended_channel_map.iter_mut().enumerate() {
            *m = ch;
        }

        if self.frame_type == EAC3_FRAME_TYPE_DEPENDENT {
            let mut ich_layout =
                CHANNEL_LAYOUT_TAB[(self.prev_output_mode & !AC3_OUTPUT_LFEON) as usize & 7];
            let channel_map_size =
                usize::from(CHANNELS_TAB[(self.output_mode & !AC3_OUTPUT_LFEON) as usize & 7])
                    + (self.lfe_on & 1) as usize;
            let mut extend = 0usize;

            if self.prev_output_mode & AC3_OUTPUT_LFEON != 0 {
                ich_layout |= CH_LOW_FREQUENCY;
            }
            let mut channel_layout = ich_layout;
            for (ch, &(_, bits)) in EAC3_CUSTOM_CHANNEL_MAP_LOCATIONS.iter().enumerate() {
                if self.channel_map & (1 << (EAC3_MAX_CHANNELS - ch - 1)) != 0 {
                    channel_layout |= bits;
                }
            }
            if channel_layout.count_ones() as usize > EAC3_MAX_CHANNELS {
                return Err(DecodeError::InvalidData("too many channels coded"));
            }
            self.ch_layout = channel_layout;

            let index_of = |chan: u32| -> Option<usize> {
                if channel_layout & (1u64 << chan) == 0 {
                    return None;
                }
                Some((channel_layout & ((1u64 << chan) - 1)).count_ones() as usize)
            };
            'map: for (ch, &(single, bits)) in EAC3_CUSTOM_CHANNEL_MAP_LOCATIONS.iter().enumerate()
            {
                if self.channel_map & (1 << (EAC3_MAX_CHANNELS - ch - 1)) == 0 {
                    continue;
                }
                if single {
                    let index = index_of(bits.trailing_zeros())
                        .ok_or(DecodeError::InvalidData("channel map"))?;
                    if extend >= channel_map_size {
                        break 'map;
                    }
                    extended_channel_map[index] = offset + usize::from(channel_map[extend]);
                    extend += 1;
                } else {
                    for i in 0..64u32 {
                        if (1u64 << i) & bits != 0 {
                            let index =
                                index_of(i).ok_or(DecodeError::InvalidData("channel map"))?;
                            if extend >= channel_map_size {
                                break;
                            }
                            extended_channel_map[index] = offset + usize::from(channel_map[extend]);
                            extend += 1;
                        }
                    }
                }
            }
        }

        // the output frame
        let nb_samples = (self.num_blocks.max(0) as usize).min(AC3_MAX_BLOCKS) * AC3_BLOCK_SIZE;
        let channels = self.ch_layout.count_ones() as usize;
        let planes = (0..channels)
            .map(|ch| self.output_buffer[extended_channel_map[ch]][..nb_samples].to_vec())
            .collect();

        let consumed = if self.superframe_size == 0 {
            full_buf_size.min(self.frame_size.max(0) as usize + skip)
        } else {
            full_buf_size.min(self.superframe_size.max(0) as usize + skip)
        };
        Ok((
            consumed,
            Some(DecodedFrame {
                sample_rate: self.avctx_sample_rate as u32,
                planes,
            }),
        ))
    }
}

/// `idct6`: 6-point IDCT of the pre-mantissas, 24-bit fixed point.
fn idct6(pre_mant: &mut [i32; 6]) {
    const COEFF_0: i64 = 10_273_905; // lrint(M_SQRT2*cos(2*M_PI/12)*(1<<23))
    const COEFF_1: i64 = 11_863_283; // lrint(M_SQRT2*(1<<23))
    const COEFF_2: i64 = 3_070_444; // lrint(M_SQRT2*cos(5*M_PI/12)*(1<<23))
    let p = *pre_mant;

    let odd1 = p[1].wrapping_sub(p[3]).wrapping_sub(p[5]);

    let mut even2 = ((i64::from(p[2]) * COEFF_0) >> 23) as i32;
    let tmp = ((i64::from(p[4]) * COEFF_1) >> 23) as i32;
    let mut odd0 = ((i64::from(p[1].wrapping_add(p[5])) * COEFF_2) >> 23) as i32;

    let mut even0 = p[0].wrapping_add(tmp >> 1);
    let even1 = p[0].wrapping_sub(tmp);

    let tmp = even0;
    even0 = tmp.wrapping_add(even2);
    even2 = tmp.wrapping_sub(even2);

    let tmp = odd0;
    odd0 = tmp.wrapping_add(p[1]).wrapping_add(p[3]);
    let odd2 = tmp.wrapping_add(p[5]).wrapping_sub(p[3]);

    pre_mant[0] = even0.wrapping_add(odd0);
    pre_mant[1] = even1.wrapping_add(odd1);
    pre_mant[2] = even2.wrapping_add(odd2);
    pre_mant[3] = even2.wrapping_sub(odd2);
    pre_mant[4] = even1.wrapping_sub(odd1);
    pre_mant[5] = even0.wrapping_sub(odd0);
}
