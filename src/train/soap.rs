//! SOAP optimizer (upstream `coolchic/training/soap.py`, after Vyas et al. 2024: Adam in the
//! eigenbasis of Shampoo's Kronecker factors).
//!
//! Per tensor of shape `(d0, d1, …)`:
//! - every axis with `d <= max_precond_dim` keeps `GG_axis = EMA(G_(axis) G_(axis)ᵀ)` and an
//!   orthonormal basis `Q_axis`; other axes are left as they are;
//! - the first step only initializes `GG` and `Q = eigh(GG)` (eigenvalues descending) and
//!   does not move the parameter;
//! - later steps run Adam on `G' = G ×_axis Q_axisᵀ` (bias correction folded into the step
//!   size, `eps` added after the square root), rotate the update back, apply it, then
//!   decoupled weight decay `p -= lr · wd · p`;
//! - after each step `GG` is updated with the raw gradient and, every
//!   `precondition_frequency` steps, `Q` is refreshed by one QR power iteration (the
//!   second-moment estimate is re-ordered with the new eigenvalue order).
//!
//! 1-D tensors (upstream `precondition_1d=False`) and `max_precond_dim = 0` (the latents)
//! reduce to this Adam variant.

use alloc::vec;
use alloc::vec::Vec;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SoapConfig {
    pub(crate) lr: f32,
    pub(crate) betas: (f32, f32),
    pub(crate) eps: f32,
    pub(crate) weight_decay: f32,
    pub(crate) precondition_frequency: usize,
    pub(crate) max_precond_dim: usize,
}

/// Optimizer state of one tensor.
#[derive(Clone, Debug)]
pub(crate) struct SoapState {
    shape: Vec<usize>,
    step: usize,
    initialized: bool,
    exp_avg: Vec<f32>,
    exp_avg_sq: Vec<f32>,
    /// Per axis: `Some(GG)` (d × d, row-major) when the axis is preconditioned.
    gg: Vec<Option<Vec<f32>>>,
    /// Per axis: basis, columns are eigenvectors.
    q: Vec<Option<Vec<f32>>>,
}

impl SoapState {
    pub(crate) fn new(shape: &[usize], max_precond_dim: usize) -> Self {
        let n: usize = shape.iter().product();
        let pre = |d: usize| shape.len() > 1 && d <= max_precond_dim;
        Self {
            shape: shape.to_vec(),
            step: 0,
            initialized: false,
            exp_avg: vec![0.0; n],
            exp_avg_sq: vec![0.0; n],
            gg: shape
                .iter()
                .map(|&d| pre(d).then(|| vec![0.0; d * d]))
                .collect(),
            q: shape.iter().map(|_| None).collect(),
        }
    }

    /// One optimizer step for parameter `p` with gradient `g`.
    pub(crate) fn step(&mut self, cfg: &SoapConfig, p: &mut [f32], g: &[f32]) {
        let (b1, b2) = cfg.betas;
        if !self.initialized {
            self.initialized = true;
            self.update_preconditioner(cfg, g);
            return;
        }
        let gp = self.project(g);
        self.step += 1;
        for i in 0..gp.len() {
            self.exp_avg[i] = self.exp_avg[i] * b1 + gp[i] * (1.0 - b1);
            self.exp_avg_sq[i] = self.exp_avg_sq[i] * b2 + gp[i] * gp[i] * (1.0 - b2);
        }
        let t = self.step as i32;
        let bc1 = 1.0 - libm::powf(b1, t as f32);
        let bc2 = 1.0 - libm::powf(b2, t as f32);
        let step_size = cfg.lr * libm::sqrtf(bc2) / bc1;
        let ratio: Vec<f32> = self
            .exp_avg
            .iter()
            .zip(&self.exp_avg_sq)
            .map(|(&m, &v)| m / (libm::sqrtf(v) + cfg.eps))
            .collect();
        let upd = self.project_back(&ratio);
        for (pi, u) in p.iter_mut().zip(&upd) {
            *pi += -step_size * u;
        }
        if cfg.weight_decay > 0.0 {
            let k = -cfg.lr * cfg.weight_decay;
            for pi in p.iter_mut() {
                *pi += *pi * k;
            }
        }
        self.update_preconditioner(cfg, g);
    }

    fn any_q(&self) -> bool {
        self.q.iter().any(|q| q.is_some())
    }

    fn update_preconditioner(&mut self, cfg: &SoapConfig, g: &[f32]) {
        if self.any_q() {
            let m = core::mem::take(&mut self.exp_avg);
            self.exp_avg = self.project_back(&m);
        }
        let beta = cfg.betas.1; // shampoo_beta = betas[1]
        for axis in 0..self.shape.len() {
            if let Some(gg) = &mut self.gg[axis] {
                let outer = mode_gram(g, &self.shape, axis);
                for (a, o) in gg.iter_mut().zip(&outer) {
                    *a += (1.0 - beta) * (*o - *a);
                }
            }
        }
        if !self.any_q() && self.step == 0 {
            for axis in 0..self.shape.len() {
                if let Some(gg) = &self.gg[axis] {
                    self.q[axis] = Some(eigh_descending(gg, self.shape[axis]));
                }
            }
        }
        if self.step > 0 && self.step.is_multiple_of(cfg.precondition_frequency) {
            self.refresh_qr();
        }
        if self.step > 0 && self.any_q() {
            let m = core::mem::take(&mut self.exp_avg);
            self.exp_avg = self.project(&m);
        }
    }

    /// `get_orthogonal_matrix_QR`.
    fn refresh_qr(&mut self) {
        for axis in 0..self.shape.len() {
            let (Some(m), Some(o)) = (&self.gg[axis], &self.q[axis]) else {
                continue;
            };
            let d = self.shape[axis];
            // est_eig = diag(oᵀ m o)
            let mut est: Vec<(f64, usize)> = (0..d)
                .map(|c| {
                    let mut s = 0.0f64;
                    for i in 0..d {
                        let mut mo = 0.0f64;
                        for j in 0..d {
                            mo += m[i * d + j] as f64 * o[j * d + c] as f64;
                        }
                        s += o[i * d + c] as f64 * mo;
                    }
                    (s, c)
                })
                .collect();
            // Descending, stable (torch.argsort is not stable, but ties are measure-zero).
            est.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
            let idx: Vec<usize> = est.iter().map(|e| e.1).collect();
            self.exp_avg_sq = index_select(&self.exp_avg_sq, &self.shape, axis, &idx);
            let mut o_sorted = vec![0.0f32; d * d];
            for i in 0..d {
                for (c, &src) in idx.iter().enumerate() {
                    o_sorted[i * d + c] = o[i * d + src];
                }
            }
            // power iteration + QR
            let mut pw = vec![0.0f64; d * d];
            for i in 0..d {
                for c in 0..d {
                    let mut s = 0.0f64;
                    for j in 0..d {
                        s += m[i * d + j] as f64 * o_sorted[j * d + c] as f64;
                    }
                    pw[i * d + c] = s;
                }
            }
            self.q[axis] = Some(qr_q(&pw, d).into_iter().map(|v| v as f32).collect());
        }
    }

    /// `G ×_axis Qᵀ` for every preconditioned axis.
    fn project(&self, x: &[f32]) -> Vec<f32> {
        let mut cur = x.to_vec();
        for axis in 0..self.shape.len() {
            if let Some(q) = &self.q[axis] {
                cur = mode_product(&cur, &self.shape, axis, q, true);
            }
        }
        cur
    }

    /// `G ×_axis Q` for every preconditioned axis.
    fn project_back(&self, x: &[f32]) -> Vec<f32> {
        let mut cur = x.to_vec();
        for axis in 0..self.shape.len() {
            if let Some(q) = &self.q[axis] {
                cur = mode_product(&cur, &self.shape, axis, q, false);
            }
        }
        cur
    }
}

/// Strides of a row-major tensor around `axis`: (outer count, axis length, inner count).
fn split(shape: &[usize], axis: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    (outer, shape[axis], inner)
}

/// Mode-`axis` product: `y[.., a, ..] = Σ_i x[.., i, ..] · (transpose ? Q[i][a] : Q[a][i])`.
fn mode_product(x: &[f32], shape: &[usize], axis: usize, q: &[f32], transpose: bool) -> Vec<f32> {
    let (outer, d, inner) = split(shape, axis);
    let mut y = vec![0.0f32; x.len()];
    for o in 0..outer {
        for a in 0..d {
            for i in 0..d {
                let w = if transpose {
                    q[i * d + a]
                } else {
                    q[a * d + i]
                };
                if w == 0.0 {
                    continue;
                }
                let src = &x[(o * d + i) * inner..(o * d + i + 1) * inner];
                let dst = &mut y[(o * d + a) * inner..(o * d + a + 1) * inner];
                for (dv, &sv) in dst.iter_mut().zip(src) {
                    *dv += sv * w;
                }
            }
        }
    }
    y
}

/// `G_(axis) G_(axis)ᵀ`: contraction of `g` with itself over every axis but `axis`.
fn mode_gram(g: &[f32], shape: &[usize], axis: usize) -> Vec<f32> {
    let (outer, d, inner) = split(shape, axis);
    let mut out = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0f32;
            for o in 0..outer {
                let a = &g[(o * d + i) * inner..(o * d + i + 1) * inner];
                let b = &g[(o * d + j) * inner..(o * d + j + 1) * inner];
                for (x, y) in a.iter().zip(b) {
                    s += x * y;
                }
            }
            out[i * d + j] = s;
        }
    }
    out
}

fn index_select(x: &[f32], shape: &[usize], axis: usize, idx: &[usize]) -> Vec<f32> {
    let (outer, d, inner) = split(shape, axis);
    let mut y = vec![0.0f32; x.len()];
    for o in 0..outer {
        for (a, &src) in idx.iter().enumerate() {
            y[(o * d + a) * inner..(o * d + a + 1) * inner]
                .copy_from_slice(&x[(o * d + src) * inner..(o * d + src + 1) * inner]);
        }
    }
    y
}

/// Eigenvectors of a symmetric matrix (cyclic Jacobi, f64), columns ordered by descending
/// eigenvalue (upstream: `torch.linalg.eigh` then `flip`).
fn eigh_descending(m: &[f32], d: usize) -> Vec<f32> {
    let mut a: Vec<f64> = m.iter().map(|&v| v as f64).collect();
    let mut v = vec![0.0f64; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0f64;
        for i in 0..d {
            for j in i + 1..d {
                off += a[i * d + j] * a[i * d + j];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..d {
            for q in p + 1..d {
                let apq = a[p * d + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + libm::sqrt(theta * theta + 1.0));
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / libm::sqrt(t * t + 1.0);
                let s = t * c;
                for k in 0..d {
                    let akp = a[k * d + p];
                    let akq = a[k * d + q];
                    a[k * d + p] = c * akp - s * akq;
                    a[k * d + q] = s * akp + c * akq;
                }
                for k in 0..d {
                    let apk = a[p * d + k];
                    let aqk = a[q * d + k];
                    a[p * d + k] = c * apk - s * aqk;
                    a[q * d + k] = s * apk + c * aqk;
                }
                for k in 0..d {
                    let vkp = v[k * d + p];
                    let vkq = v[k * d + q];
                    v[k * d + p] = c * vkp - s * vkq;
                    v[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&x, &y| {
        a[y * d + y]
            .partial_cmp(&a[x * d + x])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut out = vec![0.0f32; d * d];
    for i in 0..d {
        for (c, &src) in order.iter().enumerate() {
            out[i * d + c] = v[i * d + src] as f32;
        }
    }
    out
}

/// Q factor of a square matrix by Householder reflections (f64), as LAPACK's `geqrf` +
/// `orgqr` behind `torch.linalg.qr`: orthonormal to rounding however rank-deficient the input
/// (SOAP's Kronecker factors usually are). Column signs may differ from LAPACK's; SOAP is
/// invariant to them.
fn qr_q(a: &[f64], d: usize) -> Vec<f64> {
    let mut r = a.to_vec();
    // Householder vectors, one per column.
    let mut vs: Vec<Vec<f64>> = Vec::with_capacity(d);
    for c in 0..d {
        let norm = libm::sqrt((c..d).map(|i| r[i * d + c] * r[i * d + c]).sum::<f64>());
        let mut v = vec![0.0f64; d];
        if norm > 0.0 {
            let alpha = if r[c * d + c] > 0.0 { -norm } else { norm };
            for i in c..d {
                v[i] = r[i * d + c];
            }
            v[c] -= alpha;
            let vn = libm::sqrt(v.iter().map(|x| x * x).sum::<f64>());
            if vn > 0.0 {
                for x in v.iter_mut() {
                    *x /= vn;
                }
                // R = (I - 2 v vᵀ) R
                for j in 0..d {
                    let dotv: f64 = (c..d).map(|i| v[i] * r[i * d + j]).sum();
                    for i in c..d {
                        r[i * d + j] -= 2.0 * v[i] * dotv;
                    }
                }
            } else {
                v.fill(0.0);
            }
        }
        vs.push(v);
    }
    // Q = H_0 H_1 … H_{d-1} applied to the identity.
    let mut q = vec![0.0f64; d * d];
    for i in 0..d {
        q[i * d + i] = 1.0;
    }
    for v in vs.iter().rev() {
        for j in 0..d {
            let dotv: f64 = (0..d).map(|i| v[i] * q[i * d + j]).sum();
            if dotv != 0.0 {
                for i in 0..d {
                    q[i * d + j] -= 2.0 * v[i] * dotv;
                }
            }
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigh_diagonalizes() {
        let d = 5;
        let mut seed = 7u32;
        let mut r = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        let b: Vec<f32> = (0..d * d).map(|_| r()).collect();
        let mut m = vec![0.0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                m[i * d + j] = (0..d).map(|k| b[i * d + k] * b[j * d + k]).sum();
            }
        }
        let q = eigh_descending(&m, d);
        // qᵀ m q is diagonal with descending entries.
        let mut prev = f64::INFINITY;
        for a in 0..d {
            for c in 0..d {
                let mut s = 0.0f64;
                for i in 0..d {
                    for j in 0..d {
                        s += q[i * d + a] as f64 * m[i * d + j] as f64 * q[j * d + c] as f64;
                    }
                }
                if a == c {
                    assert!(s <= prev + 1e-6);
                    prev = s;
                } else {
                    assert!(s.abs() < 1e-5, "off-diagonal {s}");
                }
            }
        }
    }

    #[test]
    fn qr_is_orthonormal_on_rank_deficient_input() {
        // Power iteration of a rank-3 48×48 Gram matrix (SOAP's synthesis 3×48 weight).
        let d = 48;
        let mut seed = 11u32;
        let mut r = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f64 / (1 << 24) as f64 - 0.5
        };
        let b: Vec<f64> = (0..3 * d).map(|_| r()).collect();
        let mut m = vec![0.0f64; d * d];
        for i in 0..d {
            for j in 0..d {
                m[i * d + j] = (0..3).map(|k| b[k * d + i] * b[k * d + j]).sum();
            }
        }
        let q = qr_q(&m, d);
        for a in 0..d {
            for c in 0..d {
                let s: f64 = (0..d).map(|i| q[i * d + a] * q[i * d + c]).sum();
                let want = if a == c { 1.0 } else { 0.0 };
                assert!((s - want).abs() < 1e-12, "qᵀq[{a}][{c}] = {s}");
            }
        }
    }

    #[test]
    fn adam_mode_matches_closed_form() {
        // max_precond_dim = 0: plain Adam variant; first step is skipped.
        let cfg = SoapConfig {
            lr: 0.01,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            precondition_frequency: 1,
            max_precond_dim: 0,
        };
        let mut st = SoapState::new(&[1, 1, 2, 2], 0);
        let mut p = vec![1.0f32, -1.0, 0.5, 0.0];
        let g = [0.1f32, -0.2, 0.3, 0.0];
        st.step(&cfg, &mut p, &g);
        assert_eq!(p, [1.0, -1.0, 0.5, 0.0]);
        st.step(&cfg, &mut p, &g);
        // m = 0.1 g, v = 0.001 g², step 1: step_size = lr · sqrt(0.001) / 0.1
        let ss = 0.01 * libm::sqrtf(1.0 - 0.999) / (1.0 - 0.9);
        for (i, &gi) in g.iter().enumerate() {
            let m = 0.1 * gi;
            let v = 0.001 * gi * gi;
            let want = [1.0f32, -1.0, 0.5, 0.0][i] - ss * m / (libm::sqrtf(v) + 1e-8);
            assert!((p[i] - want).abs() < 1e-6, "{i}: {} vs {want}", p[i]);
        }
    }
}

#[cfg(all(test, feature = "oracle"))]
mod oracle {
    use super::*;

    #[test]
    fn matches_upstream_soap() {
        let dir = std::env::var("ZENCOOLCHIC_SOAP_DIR")
            .expect("ZENCOOLCHIC_SOAP_DIR (scripts/oracle/dump_soap.py)");
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(std::format!("{dir}/soap.json")).unwrap())
                .unwrap();
        let bin = std::fs::read(std::format!("{dir}/soap.bin")).unwrap();
        let f: Vec<f32> = bin
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let mut off = 0;
        let mut take = |n: usize| {
            let v = f[off..off + n].to_vec();
            off += n;
            v
        };
        for c in meta["cases"].as_array().unwrap() {
            let shape: Vec<usize> = c["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let n: usize = shape.iter().product();
            let cfg = SoapConfig {
                lr: c["lr"].as_f64().unwrap() as f32,
                betas: (
                    c["betas"][0].as_f64().unwrap() as f32,
                    c["betas"][1].as_f64().unwrap() as f32,
                ),
                eps: 1e-8,
                weight_decay: c["weight_decay"].as_f64().unwrap() as f32,
                precondition_frequency: c["precondition_frequency"].as_u64().unwrap() as usize,
                max_precond_dim: c["max_precond_dim"].as_u64().unwrap() as usize,
            };
            let mut p = take(n);
            let p0 = p.clone();
            let mut st = SoapState::new(&shape, cfg.max_precond_dim);
            for _ in 0..c["steps"].as_u64().unwrap() {
                let g = take(n);
                st.step(&cfg, &mut p, &g);
            }
            let want = take(n);
            let moved: f64 = want
                .iter()
                .zip(&p0)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let err: f64 = want
                .iter()
                .zip(&p)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            std::eprintln!(
                "{}: |Δp| {moved:.4e}, |p - torch| {err:.3e} (rel to movement {:.2e})",
                c["name"],
                err / moved
            );
            // A rank-deficient first G·Gᵀ (any non-square matrix) leaves eigh's null-space
            // basis arbitrary, and the trajectory depends on it: upstream run against itself
            // with that basis rotated differs by 1.6% (scripts/oracle/soap_null_space_probe.py).
            // Only full-rank cases are held to trajectory parity.
            let full_rank = shape.len() == 1
                || shape
                    .iter()
                    .filter(|&&d| d > 1)
                    .all(|&d| d == shape[shape.len() - 1]);
            if full_rank {
                assert!(err / moved < 1e-3, "{}", c["name"]);
            }
        }
    }
}
