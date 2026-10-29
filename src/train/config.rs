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

/// Upstream's "auto" lowest latent level (`parsecli.py`): 1/64 below 1
/// megapixel, 1/128 below 3, 1/256 beyond.
fn auto_hi(n_pixels: u64) -> u32 {
    if n_pixels < 1_000_000 {
        6
    } else if n_pixels < 3_000_000 {
        7
    } else {
        8
    }
}

/// `"lo-hi"` into a grid range, or `auto`/`no` per `parsecli`.
fn resolution(s: &str, auto: Option<[u32; 2]>) -> Result<Option<[u32; 2]>, crate::Error> {
    match s {
        "no" => Ok(None),
        "auto" => Ok(auto),
        _ => {
            let (a, b) = s.split_once('-').ok_or(crate::Error::InvalidArgument(
                "resolution must be \"no\", \"auto\" or \"lo-hi\"",
            ))?;
            let (a, b): (u32, u32) = (
                a.parse().map_err(|_| {
                    crate::Error::InvalidArgument("resolution bounds must be integers")
                })?,
                b.parse().map_err(|_| {
                    crate::Error::InvalidArgument("resolution bounds must be integers")
                })?,
            );
            if a > b || b > 15 {
                return Err(crate::Error::InvalidArgument(
                    "resolution must satisfy lo <= hi <= 15",
                ));
            }
            Ok(Some([a, b]))
        }
    }
}

/// Apply [`crate::unstable::ModuleArch`] overrides on top of a preset header,
/// mirroring `parsecli.py::get_coolchic_param_from_args` (including the
/// derived rules: `ifce_resolution=no` forces `output_feature_ifce` to 0 and
/// `n_latent_grids` is recomputed from the final ranges). `n_out` is the
/// module's synthesis output feature count substituted for `X`; `n_pixels`
/// drives the `"auto"` resolutions. Returns `InvalidArgument` on malformed
/// strings or values the header cannot serialize.
#[cfg(feature = "encoder")]
pub(crate) fn apply_module_arch(
    hdr: &mut CoolChicHeader,
    ov: &crate::unstable::ModuleArch,
    n_out: u32,
    n_pixels: u64,
) -> Result<(), crate::Error> {
    use crate::Error::InvalidArgument as Bad;
    if let Some(ls) = ov.layers_synthesis.as_deref().filter(|s| !s.is_empty()) {
        hdr.linear_stabiliser_synth = ls.ends_with("/stabiliser");
        let x = alloc::format!("{n_out}");
        let body = ls.replace("/stabiliser", "");
        let mut layers = Vec::new();
        for tok in body.split(',') {
            if tok.is_empty() {
                continue;
            }
            let tok = tok.replace('X', &x);
            let parts: Vec<&str> = tok.split('-').collect();
            let [of, k, mode, nl] = parts.as_slice() else {
                return Err(Bad("layers_synthesis tokens are out_ft-k-mode-act"));
            };
            let layer = SynLayer {
                out_ft: of
                    .parse()
                    .map_err(|_| Bad("layer out_ft must be an integer"))?,
                k_size: k
                    .parse()
                    .map_err(|_| Bad("layer kernel must be an integer"))?,
                residual: match *mode {
                    "linear" => false,
                    "residual" => true,
                    _ => return Err(Bad("layer mode must be linear or residual")),
                },
                relu: match *nl {
                    "none" => false,
                    "relu" => true,
                    _ => return Err(Bad("layer non-linearity must be none or relu")),
                },
            };
            if layer.out_ft > 127 || layer.k_size > 15 {
                return Err(Bad("layer out_ft <= 127 and kernel <= 15 required"));
            }
            layers.push(layer);
        }
        if layers.is_empty() || layers.len() > 7 {
            return Err(Bad("layers_synthesis needs 1..=7 layers"));
        }
        hdr.layers_synthesis = layers;
    }
    if let Some(arm) = ov.arm.as_deref().filter(|s| !s.is_empty()) {
        hdr.linear_stabiliser_arm = arm.ends_with("/stabiliser");
        let body = arm.replace("/stabiliser", "");
        let pair: Vec<&str> = body.split(',').collect();
        let [ctx, hidden] = pair.as_slice() else {
            return Err(Bad("arm must be \"context,hidden\""));
        };
        let (ctx, hidden): (u32, u32) = (
            ctx.parse()
                .map_err(|_| Bad("arm values must be integers"))?,
            hidden
                .parse()
                .map_err(|_| Bad("arm values must be integers"))?,
        );
        if ctx > 63 || hidden > 7 {
            return Err(Bad("arm requires context <= 63 and hidden <= 7"));
        }
        hdr.spatial_context_arm = ctx;
        hdr.n_hidden_layers_arm = hidden;
    }
    if let Some(v) = ov.output_feature_ifce {
        if v > 31 {
            return Err(Bad("output_feature_ifce <= 31 required"));
        }
        hdr.output_feature_ifce = v;
    }
    if let Some(s) = ov.ifce_resolution.as_deref().filter(|s| !s.is_empty()) {
        hdr.ifce_resolution = resolution(s, None)?;
    }
    if let Some(s) = ov.latent_resolution.as_deref().filter(|s| !s.is_empty()) {
        hdr.latent_resolution = resolution(s, Some([0, auto_hi(n_pixels)]))?
            .ok_or(Bad("latent_resolution cannot be \"no\""))?;
    }
    if let Some(s) = ov
        .hyperlatent_resolution
        .as_deref()
        .filter(|s| !s.is_empty())
    {
        hdr.hyperlatent_resolution = resolution(s, Some([4, auto_hi(n_pixels)]))?;
    }
    if let Some(v) = ov.ups_k_size {
        if v > 15 {
            return Err(Bad("ups_k_size <= 15 required"));
        }
        hdr.ups_k_size = v;
    }
    if let Some(v) = ov.ups_preconcat_k_size {
        if v > 15 {
            return Err(Bad("ups_preconcat_k_size <= 15 required"));
        }
        hdr.ups_preconcat_k_size = v;
    }
    // Upstream: `ifce_resolution=no` forces the IFCE features to 0; a range
    // with zero features is unrepresentable (the field only serializes when
    // the count is non-zero).
    if hdr.ifce_resolution.is_none() {
        hdr.output_feature_ifce = 0;
    } else if hdr.output_feature_ifce == 0 {
        return Err(Bad("ifce_resolution needs output_feature_ifce > 0"));
    }
    hdr.n_latent_grids = (hdr.latent_resolution[1] - hdr.latent_resolution[0] + 1)
        + hdr.hyperlatent_resolution.map_or(0, |[a, b]| b - a + 1);
    if hdr.n_latent_grids > 31 {
        return Err(Bad("latent + hyperlatent ranges exceed 31 grids"));
    }
    crate::decode::arch(hdr).map(|_| ())
}

/// Upstream's inter-frame residue header (`cfg/dec/residue/*.cfg`, **not** the
/// intra cfgs — different ARM width, IFCE and synthesis): `X` layers take
/// `n_out` output features (4 for a P residue, 5 for a B residue). Latent and
/// hyperlatent resolutions are "auto" like the intra header.
#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
pub(crate) fn header_inter(arch: Arch, h: u32, w: u32, n_out: u32) -> CoolChicHeader {
    // (ctx, hidden, ifce, first) per cfg/dec/residue/{hop,mop,lop,vlop}.cfg.
    // There is no vhop residue: VeryHigh maps to hop, the biggest offered.
    let (ctx, hidden, ifce, first) = match arch {
        Arch::VeryHigh | Arch::High => (10, 2, 4, 48),
        Arch::Medium => (10, 2, 2, 16),
        Arch::Low => (6, 2, 0, 16),
    };
    let mut hdr = header(arch, h, w);
    hdr.spatial_context_arm = ctx;
    hdr.n_hidden_layers_arm = hidden;
    hdr.output_feature_ifce = ifce;
    hdr.ifce_resolution = if ifce == 0 { None } else { Some([0, 2]) };
    hdr.layers_synthesis = alloc::vec![
        layer(first, 1, false, true),
        layer(n_out, 1, false, false),
        layer(n_out, 3, true, false),
    ];
    hdr
}

/// Upstream's motion Cool-chic (`cfg/dec/motion/{mop,lop}.cfg`): a fixed latent
/// range of 2-6, no hyperlatent and **nearest** final upsampling (blocks of
/// constant flow). `n_out` is 2 for a P frame, 4 for a B frame. `Low` uses the
/// lop motion cfg (`arm 6,1`, no IFCE, `8-1-linear-relu` stem); the other archs
/// use mop (`arm 6,2`, IFCE 2 at `2-2`, `16-1-linear-relu` stem).
#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
pub(crate) fn header_motion(arch: Arch, h: u32, w: u32, n_out: u32) -> CoolChicHeader {
    let (hidden, first, ifce) = match arch {
        // lop.cfg: `arm_motion 6,1`, `ifce_resolution_motion no` (→ ifce out 0).
        Arch::Low => (1, 8, 0),
        // mop.cfg: `arm_motion 6,2`, `output_feature_ifce_motion 2` at `2-2`.
        _ => (2, 16, 2),
    };
    CoolChicHeader {
        linear_stabiliser_synth: true,
        ups_k_size: 8,
        ups_preconcat_k_size: 7,
        output_feature_ifce: ifce,
        spatial_context_arm: 6,
        linear_stabiliser_arm: true,
        n_hidden_layers_arm: hidden,
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
        ifce_resolution: if ifce == 0 { None } else { Some([2, 2]) },
        hyperlatent_resolution: None,
        layers_synthesis: alloc::vec![layer(first, 1, false, true), layer(n_out, 1, false, false)],
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
