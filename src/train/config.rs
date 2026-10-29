//! Architectures (upstream `cfg/dec/intra/*.cfg` and `utils/parsecli.py`).

use alloc::vec::Vec;

use crate::header::{CoolChicHeader, FinalUpsampling, PerModule, SynLayer};

/// Upstream's decoder configurations for intra coding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arch {
    /// `vhop.cfg`
    VeryHigh,
    /// `hop.cfg` (upstream default)
    High,
    /// `mop.cfg`
    Medium,
    /// `lop.cfg`
    Low,
}

fn layer(out_ft: u32, k_size: u32, residual: bool, relu: bool) -> SynLayer {
    SynLayer {
        out_ft,
        k_size,
        residual,
        relu,
    }
}

/// Upstream's inter-frame residue header: `header()` with `change_n_out_synth`
/// applied — every `X` layer (all but the first in the preset strings) takes
/// `n_out` output features (4 for a P residue, 5 for a B residue).
#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
pub(crate) fn header_inter(arch: Arch, h: u32, w: u32, n_out: u32) -> CoolChicHeader {
    let mut hdr = header(arch, h, w);
    for l in hdr.layers_synthesis.iter_mut().skip(1) {
        l.out_ft = n_out;
    }
    hdr
}

/// Upstream's default motion Cool-chic (`--arm_motion 6,2/stabiliser`,
/// `--layers_synthesis_motion 16-1-linear-relu,X-1-linear-none/stabiliser`,
/// `--latent_resolution_motion 2-6`, `--hyperlatent_resolution_motion no`,
/// `--ifce_resolution_motion 2-2`, `--output_feature_ifce_motion 6`,
/// `--ups_k_size_motion 8`, `--ups_preconcat_k_size_motion 7`): a fixed latent
/// range of 2-6, no hyperlatent and **nearest** final upsampling (blocks of
/// constant flow). `n_out` is 2 for a P frame, 4 for a B frame.
#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
pub(crate) fn header_motion(h: u32, w: u32, n_out: u32) -> CoolChicHeader {
    CoolChicHeader {
        linear_stabiliser_synth: true,
        ups_k_size: 8,
        ups_preconcat_k_size: 7,
        output_feature_ifce: 6,
        spatial_context_arm: 6,
        linear_stabiliser_arm: true,
        n_hidden_layers_arm: 2,
        img_size: [h, w],
        latent_resolution: [2, 6],
        n_latent_grids: 5,
        flag_common_randomness: false,
        final_upsampling: FinalUpsampling::Nearest,
        nn_q_step: PerModule {
            arm: [1.0; 2],
            ifce: [1.0; 2],
            upsampling: [1.0; 2],
            synthesis: [1.0; 2],
        },
        nn_q_shift: PerModule {
            arm: [0; 2],
            ifce: [0; 2],
            upsampling: [0; 2],
            synthesis: [0; 2],
        },
        nn_expgol_cnt: PerModule {
            arm: [0; 2],
            ifce: [0; 2],
            upsampling: [0; 2],
            synthesis: [0; 2],
        },
        nn_n_bytes: 0,
        nn_n_bit_pad: 0,
        n_bytes_latent: 0,
        ifce_resolution: Some([2, 2]),
        hyperlatent_resolution: None,
        layers_synthesis: alloc::vec![layer(16, 1, false, true), layer(n_out, 1, false, false)],
    }
}

/// Header for an RGB picture of `h × w` with architecture `arch` ("auto" resolutions).
pub(crate) fn header(arch: Arch, h: u32, w: u32) -> CoolChicHeader {
    let (ctx, hidden, ifce, first) = match arch {
        Arch::VeryHigh => (20, 2, 6, 64),
        Arch::High => (14, 2, 6, 48),
        Arch::Medium => (10, 2, 4, 16),
        Arch::Low => (6, 2, 2, 8),
    };
    let mut layers: Vec<SynLayer> =
        alloc::vec![layer(first, 1, false, true), layer(3, 1, false, false)];
    if arch != Arch::Low {
        layers.push(layer(3, 3, true, true));
    }
    layers.push(layer(3, 3, true, false));
    let n_pixels = h as u64 * w as u64;
    let v = if n_pixels < 1_000_000 {
        6
    } else if n_pixels < 3_000_000 {
        7
    } else {
        8
    };
    let lat = [0, v];
    let hyper = [4, v];
    let n_latent_grids = (lat[1] - lat[0] + 1) + (hyper[1] - hyper[0] + 1);
    CoolChicHeader {
        linear_stabiliser_synth: true,
        ups_k_size: 8,
        ups_preconcat_k_size: 7,
        output_feature_ifce: ifce,
        spatial_context_arm: ctx,
        linear_stabiliser_arm: true,
        n_hidden_layers_arm: hidden,
        img_size: [h, w],
        latent_resolution: lat,
        n_latent_grids,
        flag_common_randomness: false,
        final_upsampling: FinalUpsampling::Bicubic,
        nn_q_step: PerModule {
            arm: [1.0; 2],
            ifce: [1.0; 2],
            upsampling: [1.0; 2],
            synthesis: [1.0; 2],
        },
        nn_q_shift: PerModule {
            arm: [0; 2],
            ifce: [0; 2],
            upsampling: [0; 2],
            synthesis: [0; 2],
        },
        nn_expgol_cnt: PerModule {
            arm: [0; 2],
            ifce: [0; 2],
            upsampling: [0; 2],
            synthesis: [0; 2],
        },
        nn_n_bytes: 0,
        nn_n_bit_pad: 0,
        n_bytes_latent: 0,
        ifce_resolution: Some([0, 2]),
        hyperlatent_resolution: Some(hyper),
        layers_synthesis: layers,
    }
}
