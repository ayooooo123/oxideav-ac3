// DSP pieces of FFmpeg's float AC-3 decoder (FFmpeg commit 2da55bf):
// libavutil/lfg.h (AVLFG), libavutil/mathematics.c (av_bessel_i0),
// libavcodec/kbdwin.c, libavutil/float_dsp.c (vector_fmul_window_c),
// libavutil/tx_template.c (the inverse MDCT's definition), libavcodec/ac3.c
// and ac3dsp.c (bit allocation).
// Copyright (c) the FFmpeg developers; LGPL-2.1-or-later (see LICENSE-LGPL).

use super::tables::{
    BAND_START_TAB, BIN_TO_BAND_TAB, HEARING_THRESHOLD_TAB, LFG_SEED0_STATE, LOG_ADD_TAB,
};

pub(crate) const AC3_CRITICAL_BANDS: usize = 50;
pub(crate) const AC3_MAX_COEFS: usize = 256;

pub(crate) const DBA_REUSE: i32 = 0;
pub(crate) const DBA_NEW: i32 = 1;
pub(crate) const DBA_NONE: i32 = 2;
pub(crate) const DBA_RESERVED: i32 = 3;

/// `AVLFG`, the lagged Fibonacci generator the decoder dithers with.
#[derive(Clone)]
pub(crate) struct Lfg {
    state: [u32; 64],
    index: u32,
}

impl Lfg {
    /// `av_lfg_init(&lfg, 0)`.
    pub(crate) fn seed0() -> Self {
        Self {
            state: LFG_SEED0_STATE,
            index: 0,
        }
    }

    /// `av_lfg_get`.
    pub(crate) fn get(&mut self) -> u32 {
        let i = self.index;
        let a = self.state[(i.wrapping_sub(24) & 63) as usize]
            .wrapping_add(self.state[(i.wrapping_sub(55) & 63) as usize]);
        self.state[(i & 63) as usize] = a;
        self.index = i.wrapping_add(1);
        a
    }
}

fn eval_poly(coeff: &[f64], x: f64) -> f64 {
    let mut sum = coeff[coeff.len() - 1];
    for &c in coeff[..coeff.len() - 1].iter().rev() {
        sum *= x;
        sum += c;
    }
    sum
}

/// `av_bessel_i0`: modified Bessel function of the first kind, order zero
/// (Blair and Edwards' minimax rational approximations). The coefficients
/// are FFmpeg's, digit for digit.
#[allow(clippy::excessive_precision)]
fn bessel_i0(x: f64) -> f64 {
    const P1: [f64; 15] = [
        -2.233_558_263_947_437_524_9e15,
        -5.505_036_967_301_842_775_3e14,
        -3.294_008_762_740_774_916_6e13,
        -8.492_510_124_711_415_749_9e11,
        -1.191_274_610_498_523_719_2e10,
        -1.031_306_670_873_798_074_7e8,
        -5.954_562_601_984_789_822_1e5,
        -2.412_519_587_604_189_677_5e3,
        -7.093_534_744_921_054_919_0e0,
        -1.545_397_779_178_685_104_1e-2,
        -2.517_264_467_068_897_505_1e-5,
        -3.051_722_645_045_106_744_6e-8,
        -2.684_344_857_346_848_327_8e-11,
        -1.598_222_667_565_318_464_6e-14,
        -5.248_786_662_794_569_980_0e-18,
    ];
    const Q1: [f64; 6] = [
        -2.233_558_263_947_437_524_5e15,
        7.885_869_256_675_100_298_8e12,
        -1.220_706_739_780_897_984_6e10,
        1.037_708_105_806_216_614_4e7,
        -4.852_756_017_996_277_304_5e3,
        1.0,
    ];
    const P2: [f64; 7] = [
        -2.221_026_223_330_657_329_6e-4,
        1.306_739_203_810_692_405_5e-2,
        -4.470_080_572_117_445_392_3e-1,
        5.567_451_837_124_076_139_7e0,
        -2.351_794_567_923_948_162_1e1,
        3.161_132_281_870_113_120_7e1,
        -9.609_002_196_865_618_000_0e0,
    ];
    const Q2: [f64; 8] = [
        -5.519_433_023_100_548_022_8e-4,
        3.254_769_759_481_961_506_2e-2,
        -1.115_175_918_874_131_264_5e0,
        1.398_259_535_389_285_154_2e1,
        -6.022_800_206_674_334_058_3e1,
        8.553_956_325_801_292_960_0e1,
        -3.144_669_027_513_549_150_0e1,
        1.0,
    ];
    if x == 0.0 {
        return 1.0;
    }
    let x = x.abs();
    if x <= 15.0 {
        let y = x * x;
        eval_poly(&P1, y) / eval_poly(&Q1, y)
    } else {
        let y = 1.0 / x - 1.0 / 15.0;
        let r = eval_poly(&P2, y) / eval_poly(&Q2, y);
        let factor = x.exp() / x.sqrt();
        factor * r
    }
}

/// `ff_kbd_window_init(window, alpha, n)`.
pub(crate) fn kbd_window(alpha: f32, n: usize) -> Vec<f32> {
    let mut temp = vec![0f64; n / 2 + 1];
    let a = f64::from(alpha) * std::f64::consts::PI / n as f64;
    let alpha2 = 4.0 * a * a;
    let mut scale = 0.0f64;
    for (i, t) in temp.iter_mut().enumerate() {
        let tmp = (i * (n - i)) as f64 * alpha2;
        *t = bessel_i0(tmp.sqrt());
        scale += *t * if i != 0 && i < n / 2 { 2.0 } else { 1.0 };
    }
    let scale = 1.0 / (scale + 1.0);
    let mut window = vec![0f32; n];
    let mut sum = 0.0f64;
    for i in 0..n {
        sum += if i <= n / 2 { temp[i] } else { temp[n - i] };
        window[i] = (sum * scale).sqrt() as f32;
    }
    window
}

/// `vector_fmul_window_c`: overlap-add of `src0` (the previous half block)
/// and `src1` under the symmetric window `win` (2 * `len` values), writing
/// 2 * `len` samples.
pub(crate) fn vector_fmul_window(
    dst: &mut [f32],
    src0: &[f32],
    src1: &[f32],
    win: &[f32],
    len: usize,
) {
    for k in 0..len {
        // FFmpeg's i = k - len (negative index from the midpoint), j = len - 1 - k.
        let i = k;
        let j = 2 * len - 1 - k;
        let s0 = src0[i];
        let s1 = src1[len - 1 - k];
        let wi = win[i];
        let wj = win[j];
        dst[i] = s0 * wj - s1 * wi;
        dst[j] = s0 * wi + s1 * wj;
    }
}

/// The inverse MDCT of `av_tx` (`AV_TX_FLOAT_MDCT`, inverse, scale 1.0) for
/// `n` coefficients: the half-length output, `n` samples, which is the
/// DCT-IV of the input in reverse order (`ff_tx_mdct_naive_inv`). Computed
/// through an `n / 2`-point complex FFT in double precision.
pub(crate) struct Imdct {
    n: usize,
    pre: Vec<(f64, f64)>,
    post: Vec<(f64, f64)>,
    fft_twiddle: Vec<(f64, f64)>,
    bitrev: Vec<usize>,
}

impl Imdct {
    pub(crate) fn new(n: usize) -> Self {
        let m = n / 2;
        let pi = std::f64::consts::PI;
        let pre = (0..m)
            .map(|j| {
                let a = -pi * (j as f64 + 0.25) / n as f64;
                (a.cos(), a.sin())
            })
            .collect();
        let post = (0..m)
            .map(|k| {
                let a = -pi * k as f64 / n as f64;
                (a.cos(), a.sin())
            })
            .collect();
        let fft_twiddle = (0..m / 2)
            .map(|k| {
                let a = -2.0 * pi * k as f64 / m as f64;
                (a.cos(), a.sin())
            })
            .collect();
        let bits = m.trailing_zeros();
        let bitrev = (0..m)
            .map(|i| i.reverse_bits() >> (usize::BITS - bits))
            .collect();
        Self {
            n,
            pre,
            post,
            fft_twiddle,
            bitrev,
        }
    }

    /// `input` holds `n` coefficients, `output` receives `n` samples.
    pub(crate) fn inverse(&self, output: &mut [f32], input: &[f32]) {
        let n = self.n;
        let m = n / 2;
        let mut z = vec![(0f64, 0f64); m];
        for j in 0..m {
            let re = f64::from(input[2 * j]);
            let im = f64::from(input[n - 1 - 2 * j]);
            let (c, s) = self.pre[j];
            z[self.bitrev[j]] = (re * c - im * s, re * s + im * c);
        }
        // iterative radix-2 FFT (forward, e^{-2 pi i nk / m})
        let mut size = 2;
        while size <= m {
            let half = size / 2;
            let step = m / size;
            for start in (0..m).step_by(size) {
                for k in 0..half {
                    let (wr, wi) = self.fft_twiddle[k * step];
                    let (ar, ai) = z[start + k];
                    let (br, bi) = z[start + k + half];
                    let tr = br * wr - bi * wi;
                    let ti = br * wi + bi * wr;
                    z[start + k] = (ar + tr, ai + ti);
                    z[start + k + half] = (ar - tr, ai - ti);
                }
            }
            size *= 2;
        }
        for k in 0..m {
            let (zr, zi) = z[k];
            let (c, s) = self.post[k];
            let ur = zr * c - zi * s;
            let ui = zr * s + zi * c;
            // u[2k] = Re, u[n-1-2k] = -Im; output[i] = u[n-1-i]
            output[n - 1 - 2 * k] = ur as f32;
            output[2 * k] = (-ui) as f32;
        }
    }
}

/// `AC3BitAllocParameters`.
#[derive(Clone, Copy, Default)]
pub(crate) struct BitAllocParams {
    pub sr_code: i32,
    pub sr_shift: i32,
    pub slow_gain: i32,
    pub slow_decay: i32,
    pub fast_decay: i32,
    pub db_per_bit: i32,
    pub floor: i32,
    pub cpl_fast_leak: i32,
    pub cpl_slow_leak: i32,
}

/// `ff_ac3_bit_alloc_calc_psd`.
pub(crate) fn calc_psd(
    exp: &[i8; AC3_MAX_COEFS],
    start: usize,
    end: usize,
    psd: &mut [i16; AC3_MAX_COEFS],
    band_psd: &mut [i16; AC3_CRITICAL_BANDS],
) {
    for bin in start..end {
        psd[bin] = (3072 - (i32::from(exp[bin]) << 7)) as i16;
    }

    let mut bin = start;
    while bin < end.min(28) {
        band_psd[bin] = psd[bin];
        bin += 1;
    }
    if bin >= end {
        return;
    }

    let mut band = usize::from(BIN_TO_BAND_TAB[bin]);
    loop {
        let mut v = i32::from(psd[bin]);
        bin += 1;
        let band_end = usize::from(BAND_START_TAB[band + 1]).min(end);
        while bin < band_end {
            let p = i32::from(psd[bin]);
            let max = v.max(p);
            let adr = (max - ((v + p + 1) >> 1)).min(255);
            v = max + i32::from(LOG_ADD_TAB[adr as usize]);
            bin += 1;
        }
        band_psd[band] = v as i16;
        band += 1;
        if end <= usize::from(BAND_START_TAB[band]) {
            break;
        }
    }
}

fn calc_lowcomp1(a: i32, b0: i32, b1: i32, c: i32) -> i32 {
    if b0 + 256 == b1 {
        c
    } else if b0 > b1 {
        (a - 64).max(0)
    } else {
        a
    }
}

fn calc_lowcomp(a: i32, b0: i32, b1: i32, bin: usize) -> i32 {
    if bin < 7 {
        calc_lowcomp1(a, b0, b1, 384)
    } else if bin < 20 {
        calc_lowcomp1(a, b0, b1, 320)
    } else {
        (a - 128).max(0)
    }
}

/// Delta bit allocation of one channel.
pub(crate) struct Dba<'a> {
    pub mode: i32,
    pub nsegs: i32,
    pub offsets: &'a [u8; 8],
    pub lengths: &'a [u8; 8],
    pub values: &'a [u8; 8],
}

/// `ff_ac3_bit_alloc_calc_mask`; `Err` where FFmpeg returns nonzero.
#[allow(clippy::too_many_arguments)]
pub(crate) fn calc_mask(
    s: &BitAllocParams,
    band_psd: &[i16; AC3_CRITICAL_BANDS],
    start: usize,
    end: usize,
    fast_gain: i32,
    is_lfe: bool,
    dba: Dba,
    mask: &mut [i16; AC3_CRITICAL_BANDS],
) -> Result<(), ()> {
    let mut excite = [0i16; AC3_CRITICAL_BANDS];
    if end == 0 {
        return Err(());
    }
    let bp = |b: usize| i32::from(band_psd[b]);

    let band_start = usize::from(BIN_TO_BAND_TAB[start]);
    let band_end = usize::from(BIN_TO_BAND_TAB[end - 1]) + 1;

    let begin;
    let mut fastleak;
    let mut slowleak;
    if band_start == 0 {
        let mut lowcomp = 0;
        lowcomp = calc_lowcomp1(lowcomp, bp(0), bp(1), 384);
        excite[0] = (bp(0) - fast_gain - lowcomp) as i16;
        lowcomp = calc_lowcomp1(lowcomp, bp(1), bp(2), 384);
        excite[1] = (bp(1) - fast_gain - lowcomp) as i16;
        let mut b = 7;
        fastleak = 0;
        slowleak = 0;
        for band in 2..7 {
            if !(is_lfe && band == 6) {
                lowcomp = calc_lowcomp1(lowcomp, bp(band), bp(band + 1), 384);
            }
            fastleak = bp(band) - fast_gain;
            slowleak = bp(band) - s.slow_gain;
            excite[band] = (fastleak - lowcomp) as i16;
            if !(is_lfe && band == 6) && bp(band) <= bp(band + 1) {
                b = band + 1;
                break;
            }
        }

        let end1 = band_end.min(22);
        for band in b..end1 {
            if !(is_lfe && band == 6) {
                lowcomp = calc_lowcomp(lowcomp, bp(band), bp(band + 1), band);
            }
            fastleak = (fastleak - s.fast_decay).max(bp(band) - fast_gain);
            slowleak = (slowleak - s.slow_decay).max(bp(band) - s.slow_gain);
            excite[band] = (fastleak - lowcomp).max(slowleak) as i16;
        }
        begin = 22;
    } else {
        // coupling channel
        begin = band_start;
        fastleak = (s.cpl_fast_leak << 8) + 768;
        slowleak = (s.cpl_slow_leak << 8) + 768;
    }

    for band in begin..band_end {
        fastleak = (fastleak - s.fast_decay).max(bp(band) - fast_gain);
        slowleak = (slowleak - s.slow_decay).max(bp(band) - s.slow_gain);
        excite[band] = fastleak.max(slowleak) as i16;
    }

    for band in band_start..band_end {
        let tmp = s.db_per_bit - bp(band);
        if tmp > 0 {
            excite[band] = (i32::from(excite[band]) + (tmp >> 2)) as i16;
        }
        let threshold = HEARING_THRESHOLD_TAB[band >> s.sr_shift][s.sr_code as usize];
        mask[band] = threshold.max(i32::from(excite[band])) as i16;
    }

    if dba.mode == DBA_REUSE || dba.mode == DBA_NEW {
        if dba.nsegs > 8 {
            return Err(());
        }
        let mut band = band_start;
        for seg in 0..dba.nsegs as usize {
            band += usize::from(dba.offsets[seg]);
            let len = usize::from(dba.lengths[seg]);
            if band >= AC3_CRITICAL_BANDS || len > AC3_CRITICAL_BANDS - band {
                return Err(());
            }
            let value = i32::from(dba.values[seg]);
            let delta = if value >= 4 {
                (value - 3) * 128
            } else {
                (value - 4) * 128
            };
            for _ in 0..len {
                mask[band] = (i32::from(mask[band]) + delta) as i16;
                band += 1;
            }
        }
    }
    Ok(())
}

/// `ac3_bit_alloc_calc_bap_c`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn calc_bap(
    mask: &[i16; AC3_CRITICAL_BANDS],
    psd: &[i16; AC3_MAX_COEFS],
    start: usize,
    end: usize,
    snr_offset: i32,
    floor: i32,
    bap_tab: &[u8; 64],
    bap: &mut [u8; AC3_MAX_COEFS],
) {
    if snr_offset == -960 {
        bap.fill(0);
        return;
    }
    let clip6 = |v: i32| v.clamp(0, 63) as usize;

    let mut bin = start;
    while bin < end.min(28) {
        let m = ((i32::from(mask[bin]) - snr_offset - floor).max(0) & 0x1FE0) + floor;
        let address = clip6((i32::from(psd[bin]) - m) >> 5);
        bap[bin] = bap_tab[address];
        bin += 1;
    }
    if bin >= end {
        return;
    }

    let mut band = usize::from(BIN_TO_BAND_TAB[bin]);
    loop {
        let m = ((i32::from(mask[band]) - snr_offset - floor).max(0) & 0x1FE0) + floor;
        band += 1;
        let band_end = usize::from(BAND_START_TAB[band]).min(end);
        while bin < band_end {
            let address = clip6((i32::from(psd[bin]) - m) >> 5);
            bap[bin] = bap_tab[address];
            bin += 1;
        }
        if end <= band_end {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ff_tx_mdct_naive_inv`, the definition the fast transform must meet.
    fn naive_inverse(input: &[f32]) -> Vec<f64> {
        let len2 = input.len();
        let len = len2 / 2;
        let phase = std::f64::consts::PI / (4.0 * len2 as f64);
        let mut out = vec![0f64; len2];
        for i in 0..len {
            let i_d = phase * (4 * len - 2 * i - 1) as f64;
            let i_u = phase * (3 * len2 + 2 * i + 1) as f64;
            let (mut sum_d, mut sum_u) = (0f64, 0f64);
            for (j, &x) in input.iter().enumerate() {
                let a = (2 * j + 1) as f64;
                sum_d += (a * i_d).cos() * f64::from(x);
                sum_u += (a * i_u).cos() * f64::from(x);
            }
            out[i] = sum_d;
            out[i + len] = -sum_u;
        }
        out
    }

    #[test]
    fn imdct_matches_av_tx_definition_for_both_block_lengths() {
        let mut lfg = Lfg::seed0();
        for n in [128usize, 256] {
            let input: Vec<f32> = (0..n)
                .map(|_| (lfg.get() as i32) as f32 / 2_147_483_648.0)
                .collect();
            let mut fast = vec![0f32; n];
            Imdct::new(n).inverse(&mut fast, &input);
            let naive = naive_inverse(&input);
            for (k, (&f, &r)) in fast.iter().zip(&naive).enumerate() {
                assert!(
                    (f64::from(f) - r).abs() < 1e-5,
                    "n {n} sample {k}: {f} vs {r}"
                );
            }
        }
    }

    #[test]
    fn kbd_window_is_power_complementary() {
        let w = kbd_window(5.0, 256);
        for i in 0..128 {
            let s = f64::from(w[i]).powi(2) + f64::from(w[255 - i]).powi(2);
            assert!((s - 1.0).abs() < 1e-6, "{i}: {s}");
        }
        assert!(w.windows(2).all(|p| p[0] <= p[1]));
    }

    #[test]
    fn lfg_follows_its_recurrence_from_the_seeded_state() {
        let mut lfg = Lfg::seed0();
        let first = lfg.get();
        assert_eq!(first, LFG_SEED0_STATE[40].wrapping_add(LFG_SEED0_STATE[9]));
        let second = lfg.get();
        assert_eq!(
            second,
            LFG_SEED0_STATE[41].wrapping_add(LFG_SEED0_STATE[10])
        );
    }
}
