// Port of FFmpeg's AC-3 / E-AC-3 header parser (libavcodec/ac3_parser.c,
// FFmpeg commit 2da55bf).
// Copyright (c) 2003 Fabrice Bellard, (c) 2003 Michael Niedermayer,
// (c) 2006 Justin Ruggles and the FFmpeg developers; LGPL-2.1-or-later
// (see LICENSE-LGPL).

use super::bits::BitReader;
use super::tables::{
    BITRATE_TAB, CHANNELS_TAB, EAC3_CUSTOM_CHANNEL_MAP_LOCATIONS, FRAME_SIZE_TAB, SAMPLE_RATE_TAB,
};

pub(crate) const EAC3_FRAME_TYPE_INDEPENDENT: i32 = 0;
pub(crate) const EAC3_FRAME_TYPE_DEPENDENT: i32 = 1;
pub(crate) const EAC3_FRAME_TYPE_AC3_CONVERT: i32 = 2;
pub(crate) const EAC3_FRAME_TYPE_RESERVED: i32 = 3;
pub(crate) const EAC3_SR_CODE_REDUCED: i32 = 3;
pub(crate) const AC3_HEADER_SIZE: i32 = 7;
pub(crate) const EAC3_MAX_CHANNELS: usize = 16;

pub(crate) const AC3_CHMODE_MONO: i32 = 1;
pub(crate) const AC3_CHMODE_STEREO: i32 = 2;
pub(crate) const AC3_CHMODE_2F2R: i32 = 6;
pub(crate) const AC3_DSURMOD_NOTINDICATED: i32 = 0;

/// `AC3ParseError`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParseError {
    Sync,
    Bsid,
    SampleRate,
    FrameSize,
    FrameType,
    ChannelMap,
}

/// `AC3HeaderInfo`.
#[derive(Clone, Debug, Default)]
pub(crate) struct HeaderInfo {
    pub sr_code: i32,
    pub bitstream_id: i32,
    pub bitstream_mode: i32,
    pub channel_mode: i32,
    pub lfe_on: i32,
    pub frame_type: i32,
    pub substreamid: i32,
    pub center_mix_level: i32,
    pub surround_mix_level: i32,
    pub channel_map: i32,
    pub num_blocks: i32,
    pub dolby_surround_mode: i32,
    pub sr_shift: i32,
    pub sample_rate: i32,
    pub bit_rate: i32,
    pub channels: i32,
    pub frame_size: i32,
    pub dialog_normalization: [i32; 2],
    pub compression_exists: [i32; 2],
    pub heavy_dynamic_range: [i32; 2],
    pub center_mix_level_ltrt: i32,
    pub surround_mix_level_ltrt: i32,
    pub dolby_headphone_mode: i32,
    pub dolby_surround_ex_mode: i32,
    pub lfe_mix_level_exists: i32,
    pub lfe_mix_level: i32,
    pub preferred_downmix: i32,
    pub eac3_extension_type_a: i32,
}

const EAC3_BLOCKS: [i32; 4] = [1, 2, 3, 6];
const CENTER_LEVELS: [i32; 4] = [4, 5, 6, 5];
const SURROUND_LEVELS: [i32; 4] = [4, 6, 7, 6];

/// `ff_ac3_find_syncword`: offset of the first sync word (either byte
/// order), or `None`.
pub(crate) fn find_syncword(buf: &[u8]) -> Option<usize> {
    let size = buf.len();
    let mut i = 1;
    while i < size {
        if buf[i] == 0x77 || buf[i] == 0x0B {
            if (buf[i] ^ buf[i - 1]) == (0x77 ^ 0x0B) {
                return Some(i - 1);
            } else if i + 1 < size && (buf[i] ^ buf[i + 1]) == (0x77 ^ 0x0B) {
                return Some(i);
            }
        }
        i += 2;
    }
    None
}

fn clip(v: i32, lo: i32, hi: i32) -> i32 {
    v.clamp(lo, hi)
}

fn ac3_parse_header(gbc: &mut BitReader, hdr: &mut HeaderInfo) {
    for i in 0..(if hdr.channel_mode != 0 { 1 } else { 2 }) {
        hdr.dialog_normalization[i] = -(gbc.get(5) as i32);
        hdr.compression_exists[i] = gbc.get1() as i32;
        if hdr.compression_exists[i] != 0 {
            hdr.heavy_dynamic_range[i] = gbc.get(8) as i32;
        }
        if gbc.get1() != 0 {
            gbc.skip(8); // language code
        }
        if gbc.get1() != 0 {
            gbc.skip(7); // audio production information
        }
    }

    gbc.skip(2); // copyright bit and original bitstream bit

    if hdr.bitstream_id != 6 {
        if gbc.get1() != 0 {
            gbc.skip(14); // timecode1
        }
        if gbc.get1() != 0 {
            gbc.skip(14); // timecode2
        }
    } else {
        if gbc.get1() != 0 {
            hdr.preferred_downmix = gbc.get(2) as i32;
            hdr.center_mix_level_ltrt = gbc.get(3) as i32;
            hdr.surround_mix_level_ltrt = clip(gbc.get(3) as i32, 3, 7);
            hdr.center_mix_level = gbc.get(3) as i32;
            hdr.surround_mix_level = clip(gbc.get(3) as i32, 3, 7);
        }
        if gbc.get1() != 0 {
            hdr.dolby_surround_ex_mode = gbc.get(2) as i32;
            hdr.dolby_headphone_mode = gbc.get(2) as i32;
            gbc.skip(10); // adconvtyp (1), xbsi2 (8), encinfo (1)
        }
    }

    // additional bitstream info
    if gbc.get1() != 0 {
        let mut i = gbc.get(6) as i32;
        loop {
            gbc.skip(8);
            if i == 0 {
                break;
            }
            i -= 1;
        }
    }
}

fn eac3_parse_header(gbc: &mut BitReader, hdr: &mut HeaderInfo) -> Result<(), ParseError> {
    if hdr.frame_type == EAC3_FRAME_TYPE_RESERVED {
        return Err(ParseError::FrameType);
    }
    if hdr.substreamid != 0 {
        return Err(ParseError::FrameType);
    }

    gbc.skip(5); // bitstream id

    for i in 0..(if hdr.channel_mode != 0 { 1 } else { 2 }) {
        hdr.dialog_normalization[i] = -(gbc.get(5) as i32);
        hdr.compression_exists[i] = gbc.get1() as i32;
        if hdr.compression_exists[i] != 0 {
            hdr.heavy_dynamic_range[i] = gbc.get(8) as i32;
        }
    }

    // dependent stream channel map
    if hdr.frame_type == EAC3_FRAME_TYPE_DEPENDENT && gbc.get1() != 0 {
        let channel_map = gbc.get(16) as i32;
        let mut channel_layout = 0u64;
        for (i, &(_, bits)) in EAC3_CUSTOM_CHANNEL_MAP_LOCATIONS.iter().enumerate() {
            if channel_map & (1 << (EAC3_MAX_CHANNELS - i - 1)) != 0 {
                channel_layout |= bits;
            }
        }
        if channel_layout.count_ones() as usize > EAC3_MAX_CHANNELS {
            return Err(ParseError::ChannelMap);
        }
        hdr.channel_map = channel_map;
    }

    // mixing metadata
    if gbc.get1() != 0 {
        if hdr.channel_mode > AC3_CHMODE_STEREO {
            hdr.preferred_downmix = gbc.get(2) as i32;
            if hdr.channel_mode & 1 != 0 {
                hdr.center_mix_level_ltrt = gbc.get(3) as i32;
                hdr.center_mix_level = gbc.get(3) as i32;
            }
            if hdr.channel_mode & 4 != 0 {
                hdr.surround_mix_level_ltrt = clip(gbc.get(3) as i32, 3, 7);
                hdr.surround_mix_level = clip(gbc.get(3) as i32, 3, 7);
            }
        }

        if hdr.lfe_on != 0 {
            hdr.lfe_mix_level_exists = gbc.get1() as i32;
            if hdr.lfe_mix_level_exists != 0 {
                hdr.lfe_mix_level = gbc.get(5) as i32;
            }
        }

        if hdr.frame_type == EAC3_FRAME_TYPE_INDEPENDENT {
            for _ in 0..(if hdr.channel_mode != 0 { 1 } else { 2 }) {
                if gbc.get1() != 0 {
                    gbc.skip(6); // program scale factor
                }
            }
            if gbc.get1() != 0 {
                gbc.skip(6); // external program scale factor
            }
            match gbc.get(2) {
                1 => gbc.skip(5),
                2 => gbc.skip(12),
                3 => {
                    let mix_data_size = ((gbc.get(5) + 2) << 3) as usize;
                    gbc.skip(mix_data_size);
                }
                _ => {}
            }
            if hdr.channel_mode < AC3_CHMODE_STEREO {
                for _ in 0..(if hdr.channel_mode != 0 { 1 } else { 2 }) {
                    if gbc.get1() != 0 {
                        gbc.skip(8); // pan mean direction index
                        gbc.skip(6); // reserved paninfo bits
                    }
                }
            }
            if gbc.get1() != 0 {
                for _ in 0..hdr.num_blocks {
                    if hdr.num_blocks == 1 || gbc.get1() != 0 {
                        gbc.skip(5);
                    }
                }
            }
        }
    }

    // informational metadata
    if gbc.get1() != 0 {
        hdr.bitstream_mode = gbc.get(3) as i32;
        gbc.skip(2); // copyright bit and original bitstream bit
        if hdr.channel_mode == AC3_CHMODE_STEREO {
            hdr.dolby_surround_mode = gbc.get(2) as i32;
            hdr.dolby_headphone_mode = gbc.get(2) as i32;
        }
        if hdr.channel_mode >= AC3_CHMODE_2F2R {
            hdr.dolby_surround_ex_mode = gbc.get(2) as i32;
        }
        for _ in 0..(if hdr.channel_mode != 0 { 1 } else { 2 }) {
            if gbc.get1() != 0 {
                gbc.skip(8); // mix level, room type, A/D converter type
            }
        }
        if hdr.sr_code != EAC3_SR_CODE_REDUCED {
            gbc.skip(1); // source sample rate code
        }
    }

    // converter synchronization flag
    if hdr.frame_type == EAC3_FRAME_TYPE_INDEPENDENT && hdr.num_blocks != 6 {
        gbc.skip(1);
    }

    // original frame size code if this stream was converted from AC-3
    if hdr.frame_type == EAC3_FRAME_TYPE_AC3_CONVERT && (hdr.num_blocks == 6 || gbc.get1() != 0) {
        gbc.skip(6);
    }

    // additional bitstream info
    if gbc.get1() != 0 {
        let addbsil = gbc.get(6) as i32;
        let mut i = 0;
        while i < addbsil + 1 {
            if i == 0 {
                gbc.skip(7);
                hdr.eac3_extension_type_a = gbc.get1() as i32;
                if hdr.eac3_extension_type_a != 0 {
                    gbc.skip(8); // complexity_index_type_a
                    i += 1;
                }
            } else {
                gbc.skip(8);
            }
            i += 1;
        }
    }

    Ok(())
}

/// `ff_ac3_parse_header`.
pub(crate) fn parse_header(gbc: &mut BitReader) -> Result<HeaderInfo, ParseError> {
    let mut hdr = HeaderInfo::default();

    if gbc.get(16) != 0x0B77 {
        return Err(ParseError::Sync);
    }

    // read ahead to bsid to distinguish between AC-3 and E-AC-3
    hdr.bitstream_id = (gbc.show(29) & 0x1F) as i32;
    if hdr.bitstream_id > 16 {
        return Err(ParseError::Bsid);
    }

    hdr.num_blocks = 6;
    hdr.center_mix_level = 5; // -4.5dB
    hdr.surround_mix_level = 6; // -6.0dB
    hdr.dolby_surround_mode = AC3_DSURMOD_NOTINDICATED;

    if hdr.bitstream_id <= 10 {
        gbc.skip(16); // crc1
        hdr.sr_code = gbc.get(2) as i32;
        if hdr.sr_code == 3 {
            return Err(ParseError::SampleRate);
        }

        let frame_size_code = gbc.get(6) as usize;
        if frame_size_code > 37 {
            return Err(ParseError::FrameSize);
        }
        let ac3_bit_rate_code = frame_size_code >> 1;

        gbc.skip(5); // bsid, already read

        hdr.bitstream_mode = gbc.get(3) as i32;
        hdr.channel_mode = gbc.get(3) as i32;

        if hdr.channel_mode == AC3_CHMODE_STEREO {
            hdr.dolby_surround_mode = gbc.get(2) as i32;
        } else {
            if hdr.channel_mode & 1 != 0 && hdr.channel_mode != AC3_CHMODE_MONO {
                hdr.center_mix_level = CENTER_LEVELS[gbc.get(2) as usize];
            }
            if hdr.channel_mode & 4 != 0 {
                hdr.surround_mix_level = SURROUND_LEVELS[gbc.get(2) as usize];
            }
        }
        hdr.lfe_on = gbc.get1() as i32;

        hdr.sr_shift = hdr.bitstream_id.max(8) - 8;
        hdr.sample_rate = SAMPLE_RATE_TAB[hdr.sr_code as usize] >> hdr.sr_shift;
        hdr.bit_rate = (i32::from(BITRATE_TAB[ac3_bit_rate_code]) * 1000) >> hdr.sr_shift;
        hdr.channels = i32::from(CHANNELS_TAB[hdr.channel_mode as usize]) + hdr.lfe_on;
        hdr.frame_size = i32::from(FRAME_SIZE_TAB[frame_size_code][hdr.sr_code as usize]) * 2;
        hdr.frame_type = EAC3_FRAME_TYPE_AC3_CONVERT;
        hdr.substreamid = 0;

        ac3_parse_header(gbc, &mut hdr);
    } else {
        hdr.frame_type = gbc.get(2) as i32;
        if hdr.frame_type == EAC3_FRAME_TYPE_RESERVED {
            return Err(ParseError::FrameType);
        }

        hdr.substreamid = gbc.get(3) as i32;

        hdr.frame_size = ((gbc.get(11) + 1) << 1) as i32;
        if hdr.frame_size < AC3_HEADER_SIZE {
            return Err(ParseError::FrameSize);
        }

        hdr.sr_code = gbc.get(2) as i32;
        if hdr.sr_code == 3 {
            let sr_code2 = gbc.get(2) as usize;
            if sr_code2 == 3 {
                return Err(ParseError::SampleRate);
            }
            hdr.sample_rate = SAMPLE_RATE_TAB[sr_code2] / 2;
            hdr.sr_shift = 1;
        } else {
            hdr.num_blocks = EAC3_BLOCKS[gbc.get(2) as usize];
            hdr.sample_rate = SAMPLE_RATE_TAB[hdr.sr_code as usize];
            hdr.sr_shift = 0;
        }

        hdr.channel_mode = gbc.get(3) as i32;
        hdr.lfe_on = gbc.get1() as i32;

        hdr.bit_rate = (8i64 * i64::from(hdr.frame_size) * i64::from(hdr.sample_rate)
            / (i64::from(hdr.num_blocks) * 256)) as i32;
        hdr.channels = i32::from(CHANNELS_TAB[hdr.channel_mode as usize]) + hdr.lfe_on;

        eac3_parse_header(gbc, &mut hdr)?;
    }

    Ok(hdr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syncword_is_found_in_either_byte_order_on_any_alignment() {
        assert_eq!(find_syncword(&[0x0B, 0x77, 0, 0]), Some(0));
        assert_eq!(find_syncword(&[0, 0x0B, 0x77, 0]), Some(1));
        assert_eq!(find_syncword(&[0x77, 0x0B]), Some(0));
        assert_eq!(find_syncword(&[0, 0, 0x77, 0x0B]), Some(2));
        assert_eq!(find_syncword(&[0x0B]), None);
        assert_eq!(find_syncword(&[0, 0x0B]), None);
    }
}
