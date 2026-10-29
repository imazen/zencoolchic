//! Gradient check against `torch.autograd` (dumps from `scripts/oracle/dump_grad.py`,
//! directory `ZENCOOLCHIC_GRAD_DIR`).

use std::string::String;
use std::vec::Vec;

use super::*;
use crate::header::{CoolChicHeader, FinalUpsampling, PerModule, SynLayer};

fn header_from(cfg: &serde_json::Value) -> CoolChicHeader {
    let u = |k: &str| cfg[k].as_u64().unwrap() as u32;
    let pair = |k: &str| {
        [
            cfg[k][0].as_u64().unwrap() as u32,
            cfg[k][1].as_u64().unwrap() as u32,
        ]
    };
    let layers = cfg["layers_synthesis"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| {
            let s: Vec<&str> = l.as_str().unwrap().split('-').collect();
            SynLayer {
                out_ft: s[0].parse().unwrap(),
                k_size: s[1].parse().unwrap(),
                residual: s[2] == "residual",
                relu: s[3] == "relu",
            }
        })
        .collect::<Vec<_>>();
    let mut h = CoolChicHeader {
        linear_stabiliser_synth: true,
        ups_k_size: 8,
        ups_preconcat_k_size: 7,
        output_feature_ifce: u("output_feature_ifce"),
        spatial_context_arm: u("spatial_context_arm"),
        linear_stabiliser_arm: true,
        n_hidden_layers_arm: u("n_hidden_layers_arm"),
        img_size: pair("img_size"),
        latent_resolution: pair("latent_resolution"),
        n_latent_grids: 0,
        flag_common_randomness: cfg["flag_common_randomness"].as_bool().unwrap_or(false),
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
        ifce_resolution: Some(pair("ifce_resolution")),
        hyperlatent_resolution: Some(pair("hyperlatent_resolution")),
        layers_synthesis: layers,
    };
    // n_latent_grids is derived from the resolutions.
    let [l0, l1] = h.latent_resolution;
    let [h0, h1] = h.hyperlatent_resolution.unwrap();
    h.n_latent_grids = (l1 - l0 + 1) + (h1 - h0 + 1);
    h
}

/// Points a parameter name of upstream's state dict at the matching tensor.
fn slot<'a>(p: &'a mut Params, name: &str, ifce_grids: &[usize]) -> Option<&'a mut Vec<f32>> {
    let parts: Vec<&str> = name.split('.').collect();
    let wb = |l: &'a mut Linear, k: &str| if k == "weight" { &mut l.w } else { &mut l.b };
    let wbc = |l: &'a mut ConvLayer, k: &str| if k == "weight" { &mut l.w } else { &mut l.b };
    match parts.as_slice() {
        ["latent_grids", i] => Some(&mut p.latents[i.parse::<usize>().unwrap()]),
        ["arm", "mlp", i, k] => Some(wb(&mut p.arm.layers[i.parse::<usize>().unwrap() / 2], k)),
        ["arm", "stabiliser_branch", k] => Some(wb(p.arm.stabiliser.as_mut().unwrap(), k)),
        ["ifce", "arms", j, "mlp", "0", k] => {
            let g = ifce_grids[j.parse::<usize>().unwrap()];
            Some(wb(&mut p.ifce[g].as_mut().unwrap().layers[0], k))
        }
        ["synthesis", "output_transform", k] => Some(wbc(&mut p.out_transform, k)),
        ["synthesis", "stabiliser_branch", k] => Some(wbc(p.syn_stab.as_mut().unwrap(), k)),
        ["synthesis", "main_branch", i, k] => {
            Some(wbc(&mut p.syn[i.parse::<usize>().unwrap() / 2], k))
        }
        [
            "upsampling",
            "conv_transpose2ds",
            i,
            "parametrizations",
            "weight",
            "original",
        ] => Some(&mut p.ups_t[i.parse::<usize>().unwrap()]),
        [
            "upsampling",
            "conv2ds",
            i,
            "parametrizations",
            "weight",
            "original",
        ] => Some(&mut p.ups_c[i.parse::<usize>().unwrap()]),
        ["upsampling", _, _, "bias"] => None, // unused by the forward pass
        _ => panic!("unmapped parameter {name}"),
    }
}

fn l2(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn gradients_match_torch_autograd() {
    let mut found = 0;
    for name in crate::oracle_fixtures::cases("gradients") {
        found += 1;
        let cfg: serde_json::Value = serde_json::from_slice(&crate::oracle_fixtures::read(
            "gradients",
            &std::format!("{name}.json"),
        ))
        .unwrap();
        let bin = crate::oracle_fixtures::read("gradients", &std::format!("{name}.bin"));
        assert!(bin.len().is_multiple_of(4), "{name}: unaligned F32 dump");
        let floats: Vec<f32> = bin
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let mut tensors: Vec<(String, Vec<f32>)> = Vec::new();
        let mut off = 0;
        for t in cfg["tensors"].as_array().unwrap() {
            let n: usize = t["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .product();
            tensors.push((
                t["name"].as_str().unwrap().into(),
                floats[off..off + n].to_vec(),
            ));
            off += n;
        }
        assert_eq!(off, floats.len(), "{name}: complete tensor payload");
        let h = header_from(&cfg);
        let a = crate::decode::arch(&h).unwrap();
        let shape = Shape::new(&h, &a).unwrap();
        let mut p = shape.zero_params(&h, &a);
        let ifce_grids: Vec<usize> = (0..p.ifce.len()).filter(|&g| p.ifce[g].is_some()).collect();
        let mut target = Vec::new();
        let mut uniform = Vec::new();
        for (name, v) in &tensors {
            if name == "target" {
                target = v.clone();
            } else if name == "uniform_noise" {
                uniform = v.clone();
            } else if name == "vgg"
                || name == "decoded"
                || name == "grad_dec"
                || name.starts_with("tap")
                || name.starts_with("gtap")
            {
                // "vgg": the miniature feature stack of `miniwass` dumps, read below.
                // "decoded"/"grad_dec"/"tap{i}"/"gtap{i}": the clamped output picture,
                // its wasserstein gradients, and its VGG taps — all compared below.
            } else if let Some(n) = name.strip_prefix("grad:") {
                let _ = n;
            } else if let Some(s) = slot(&mut p, name, &ifce_grids) {
                assert_eq!(s.len(), v.len(), "{name}");
                s.copy_from_slice(v);
            }
        }
        let mut g = p.zeros_like();
        let noise = kumaraswamy_noise(&uniform, cfg["noise_parameter"].as_f64().unwrap() as f32);
        let dist_weight = |k: &str| cfg["dist_weight"][k].as_f64().unwrap_or(0.0) as f32;
        let step = StepCfg {
            quantizer: Quantizer::Softround {
                temperature: cfg["softround_temperature"].as_f64().unwrap() as f32,
            },
            gain: 16.0,
            lambda: cfg["lmbda"].as_f64().unwrap() as f32,
            eval_bitdepth: None,
            dist: if cfg["dist_weight"].is_object() {
                Dist {
                    mse: dist_weight("mse"),
                    wass: dist_weight("wasserstein"),
                }
            } else {
                Dist::MSE
            },
            frame_data_type: match cfg["frame_data_type"].as_str() {
                None | Some("rgb") => crate::header::FrameDataType::Rgb,
                Some("yuv420") => crate::header::FrameDataType::Yuv420,
                Some("yuv444") => crate::header::FrameDataType::Yuv444,
                Some("flow") => crate::header::FrameDataType::Flow,
                Some(other) => panic!("{name}: unknown frame_data_type {other}"),
            },
        };
        // Wasserstein dumps need the feature weights: `miniwass` dumps embed a tiny
        // stack as the "vgg" tensor with dims in `vgg_dims`; real-VGG dumps point at
        // `dump_vgg.py`'s flat f32 file through `ZENCOOLCHIC_VGG16`.
        let wasserstein = (step.dist.wass != 0.0).then(|| {
            let vgg = if cfg["vgg_dims"].is_array() {
                let dims: Vec<(usize, usize)> = cfg["vgg_dims"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| {
                        (
                            d[0].as_u64().unwrap() as usize,
                            d[1].as_u64().unwrap() as usize,
                        )
                    })
                    .collect();
                let weights = &tensors
                    .iter()
                    .find(|(n, _)| n == "vgg")
                    .unwrap_or_else(|| panic!("{name}: missing vgg tensor"))
                    .1;
                wasserstein::Vgg16::from_dims(&dims, weights).unwrap()
            } else {
                let path = std::env::var("ZENCOOLCHIC_VGG16")
                    .expect("wasserstein fixture without embedded weights or ZENCOOLCHIC_VGG16");
                let bytes = std::fs::read(&path)
                    .unwrap_or_else(|e| panic!("{name}: cannot read {path}: {e}"));
                wasserstein::Vgg16::from_le_bytes(&bytes).unwrap()
            };
            wasserstein::Wasserstein::new(
                std::sync::Arc::new(vgg),
                &target,
                h.img_size[0] as usize,
                h.img_size[1] as usize,
            )
        });
        let terms = forward_backward(
            &shape,
            &p,
            &target,
            &noise,
            step,
            wasserstein.as_ref(),
            &mut g,
        );
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-30);
        std::eprintln!(
            "{}: loss {:.7} vs {:.7} (rel {:.1e}); mse rel {:.1e}; rate {:.2} vs {:.2} (rel {:.1e})",
            name,
            terms.loss,
            cfg["loss"].as_f64().unwrap(),
            rel(terms.loss, cfg["loss"].as_f64().unwrap()),
            rel(terms.mse, cfg["mse"].as_f64().unwrap()),
            terms.rate_bits,
            cfg["rate_bits"].as_f64().unwrap(),
            rel(terms.rate_bits, cfg["rate_bits"].as_f64().unwrap()),
        );
        if let Some(w) = cfg["wasserstein"].as_f64() {
            std::eprintln!(
                "  wasserstein {:.9} vs {:.9} (rel {:.1e})",
                terms.wass,
                w,
                rel(terms.wass, w)
            );
            assert!(rel(terms.wass, w) < 1e-5, "wasserstein mismatch");
        }
        // Per-tap distances on the dumped `decoded` picture isolate which VGG stage
        // differs from the oracle.
        if let (Some(wm), Some((_, dec))) = (
            wasserstein.as_ref(),
            tensors.iter().find(|(n, _)| n == "decoded"),
        ) {
            for (i, f) in wm
                .tap_features(dec, shape.img.0, shape.img.1)
                .iter()
                .enumerate()
            {
                let tname: String = std::format!("tap{i}");
                if let Some((_, theirs)) = tensors.iter().find(|(n, _)| n == &tname) {
                    let diff: Vec<f32> = f.iter().zip(theirs).map(|(a, b)| a - b).collect();
                    std::eprintln!(
                        "  tap{i} feat rel err {:.2e} (|theirs| {:.3e})",
                        l2(&diff) / l2(theirs).max(1e-30),
                        l2(theirs)
                    );
                }
            }
            let mine = wm.tap_dists(dec, shape.img.0, shape.img.1);
            if let Some(theirs) = cfg["wasserstein_per_tap"].as_array() {
                for (i, (m, t)) in mine.iter().zip(theirs).enumerate() {
                    let t = t.as_f64().unwrap();
                    std::eprintln!("  tap{i} dist {m:.9} vs {t:.9} (rel {:.1e})", rel(*m, t));
                }
            }
            // Per-tap gradient oracle: isolates the stats backward from the VGG chain.
            for (i, f) in wm
                .tap_grads(dec, shape.img.0, shape.img.1)
                .iter()
                .enumerate()
            {
                let tname: String = std::format!("gtap{i}");
                if let Some((_, theirs)) = tensors.iter().find(|(n, _)| n == &tname) {
                    let diff: Vec<f32> = f.iter().zip(theirs).map(|(a, b)| a - b).collect();
                    std::eprintln!(
                        "  tap{i} grad rel err {:.2e} (|theirs| {:.3e})",
                        l2(&diff) / l2(theirs).max(1e-30),
                        l2(theirs)
                    );
                }
            }
            // Full backward oracle: `d wasserstein / d decoded` from torch autograd.
            let (_d0, tape) = wm.forward(dec, shape.img.0, shape.img.1);
            let g = wm.backward(&tape);
            if let Some((_, theirs)) = tensors.iter().find(|(n, _)| n == "grad_dec") {
                let diff: Vec<f32> = g.iter().zip(theirs).map(|(a, b)| a - b).collect();
                let r = l2(&diff) / l2(theirs).max(1e-30);
                std::eprintln!(
                    "  d_wass/d_dec rel err {r:.2e} (|theirs| {:.3e}, max|diff| {:.3e})",
                    l2(theirs),
                    diff.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
                );
                assert!(r < 1e-3, "wasserstein backward mismatch");
            }
        }
        let mut worst = 0.0f64;
        for (name, v) in &tensors {
            let Some(pname) = name.strip_prefix("grad:") else {
                continue;
            };
            if pname == "dec" {
                continue; // `grad_dec`: compared against `wm.backward` above
            }
            if pname.contains("output_transform") {
                continue; // frozen upstream: torch gives no gradient for it
            }
            let Some(mine) = slot(&mut g, pname, &ifce_grids) else {
                continue;
            };
            let diff: Vec<f32> = mine.iter().zip(v).map(|(a, b)| a - b).collect();
            let r = l2(&diff) / l2(v).max(1e-30);
            assert!(r.is_finite(), "{pname}: nonfinite gradient comparison");
            worst = worst.max(r);
            std::eprintln!("  {pname:<70} |g| {:.3e}  rel err {r:.2e}", l2(v));
        }
        std::eprintln!("  worst relative gradient error {worst:.2e}");
        assert!(
            rel(terms.loss, cfg["loss"].as_f64().unwrap()) < 1e-5,
            "loss mismatch"
        );
        assert!(worst < 1e-3, "gradient mismatch");
    }
    assert!(found > 0, "no gradient dumps");
}
