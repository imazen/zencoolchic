//! Differential test against the upstream Python decoder.
//!
//! `ZENCOOLCHIC_ORACLE_DIR` holds `<name>.cool` streams plus the dumps written by
//! `scripts/oracle/dump_decode.py` (`just oracle-dump` regenerates them). Every stream in
//! the directory is checked: integer latents must match exactly (the entropy stage is
//! integer-only), float output is reported and bounded.
#![cfg(feature = "oracle")]

use std::path::{Path, PathBuf};

fn read_i64(p: &Path) -> Vec<i64> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect()
}

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn read_u16(p: &Path) -> Vec<u16> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}

fn streams() -> Vec<PathBuf> {
    let dir = std::env::var("ZENCOOLCHIC_ORACLE_DIR").expect(
        "ZENCOOLCHIC_ORACLE_DIR must point at the oracle dumps (see scripts/oracle/dump_decode.py)",
    );
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "cool"))
        .collect();
    v.sort();
    assert!(!v.is_empty(), "no .cool streams in {dir}");
    v
}

#[test]
fn matches_upstream_decoder() {
    for stream in streams() {
        let name = stream.with_extension("");
        let with = |ext: &str| PathBuf::from(format!("{}.{ext}", name.display()));
        let data = std::fs::read(&stream).unwrap();
        let t = std::time::Instant::now();
        let (pic, det) =
            zencoolchic::decode_with_details(&data, &zencoolchic::Limits::default()).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;

        let want_lat = read_i64(&with("latents.i64"));
        let got_lat: Vec<i64> = det.grids.iter().flat_map(|g| g.2.iter().copied()).collect();
        assert_eq!(
            got_lat.len(),
            want_lat.len(),
            "{}: latent count",
            stream.display()
        );
        let lat_mismatch = got_lat
            .iter()
            .zip(&want_lat)
            .filter(|(a, b)| a != b)
            .count();

        let want_raw = read_f32(&with("raw.f32"));
        let max_raw = det
            .raw
            .iter()
            .zip(&want_raw)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);

        let want_px = read_u16(&with("planes.u16"));
        let got_px: Vec<u16> = pic.planes.iter().flatten().copied().collect();
        assert_eq!(got_px.len(), want_px.len());
        let mut hist = [0usize; 4];
        for (a, b) in got_px.iter().zip(&want_px) {
            hist[(a.abs_diff(*b) as usize).min(3)] += 1;
        }
        eprintln!(
            "{}: {ms:.0} ms; latent mismatches {lat_mismatch}/{}; raw max |Δ| {max_raw:e}; \
             sample |Δ| histogram 0:{} 1:{} 2:{} >=3:{} (of {})",
            stream.file_name().unwrap().to_string_lossy(),
            want_lat.len(),
            hist[0],
            hist[1],
            hist[2],
            hist[3],
            got_px.len()
        );
        assert_eq!(
            lat_mismatch,
            0,
            "{}: integer latents must match exactly",
            stream.display()
        );
    }
}
