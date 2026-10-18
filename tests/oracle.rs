//! Differential test against the upstream Python decoder.
//!
//! Frozen references are checked in; a shared helper decompresses and verifies them
//! during the test. `ZENCOOLCHIC_ORACLE_DIR` explicitly selects raw regenerated dumps.
#![cfg(feature = "oracle")]

#[path = "support/oracle_fixtures.rs"]
mod oracle_fixtures;

fn read_i64(data: &[u8]) -> Vec<i64> {
    assert!(data.len().is_multiple_of(8), "unaligned I64 reference");
    data.as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect()
}

fn read_f32(data: &[u8]) -> Vec<f32> {
    assert!(data.len().is_multiple_of(4), "unaligned F32 reference");
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn read_u16(data: &[u8]) -> Vec<u16> {
    assert!(data.len().is_multiple_of(2), "unaligned U16 reference");
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}

#[test]
fn matches_upstream_decoder() {
    for name in oracle_fixtures::cases("decoder") {
        let with = |ext: &str| oracle_fixtures::read("decoder", &format!("{name}.{ext}"));
        let data = with("cool");
        let meta: serde_json::Value = serde_json::from_slice(&with("json")).unwrap();
        let (pic, det) =
            zencoolchic::decode_with_details(&data, &zencoolchic::Limits::default()).unwrap();

        let want_lat = read_i64(&with("latents.i64"));
        let got_lat: Vec<i64> = det.grids.iter().flat_map(|g| g.2.iter().copied()).collect();
        assert_eq!(got_lat.len(), want_lat.len(), "{name}: latent count");
        let shapes = meta["latent_shapes"].as_array().unwrap();
        assert_eq!(det.grids.len(), shapes.len(), "{name}: latent grids");
        for ((h, w, samples), shape) in det.grids.iter().zip(shapes) {
            assert_eq!(*h as u64, shape[0].as_u64().unwrap(), "{name}: grid height");
            assert_eq!(*w as u64, shape[1].as_u64().unwrap(), "{name}: grid width");
            assert_eq!(samples.len(), h * w, "{name}: grid samples");
        }
        let lat_mismatch = got_lat
            .iter()
            .zip(&want_lat)
            .filter(|(a, b)| a != b)
            .count();

        let want_raw = read_f32(&with("raw.f32"));
        assert_eq!(det.raw.len(), want_raw.len(), "{name}: raw sample count");
        let raw_shape = meta["raw_shape"].as_array().unwrap();
        assert_eq!(det.raw_channels as u64, raw_shape[0].as_u64().unwrap());
        assert_eq!(pic.height as u64, raw_shape[1].as_u64().unwrap());
        assert_eq!(pic.width as u64, raw_shape[2].as_u64().unwrap());
        let max_raw = det
            .raw
            .iter()
            .zip(&want_raw)
            .map(|(a, b)| {
                assert!(
                    a.is_finite() && b.is_finite(),
                    "{name}: nonfinite reconstruction"
                );
                (a - b).abs()
            })
            .fold(0f32, f32::max);

        let want_px = read_u16(&with("planes.u16"));
        let got_px: Vec<u16> = pic.planes.iter().flatten().copied().collect();
        assert_eq!(got_px.len(), want_px.len(), "{name}: pixel count");
        assert_eq!(pic.bitdepth as u64, meta["bitdepth"].as_u64().unwrap());
        let shapes = meta["plane_shapes"].as_array().unwrap();
        assert_eq!(pic.planes.len(), shapes.len(), "{name}: plane count");
        for ((plane, &(w, h)), shape) in pic.planes.iter().zip(&pic.plane_dims).zip(shapes) {
            assert_eq!(h as u64, shape[0].as_u64().unwrap(), "{name}: plane height");
            assert_eq!(w as u64, shape[1].as_u64().unwrap(), "{name}: plane width");
            assert_eq!(plane.len(), h as usize * w as usize);
        }
        let mut hist = [0usize; 4];
        for (a, b) in got_px.iter().zip(&want_px) {
            hist[(a.abs_diff(*b) as usize).min(3)] += 1;
        }
        eprintln!(
            "{name}: latent mismatches {lat_mismatch}/{}; raw max |Δ| {max_raw:e}; \
             sample |Δ| histogram 0:{} 1:{} 2:{} >=3:{} (of {})",
            want_lat.len(),
            hist[0],
            hist[1],
            hist[2],
            hist[3],
            got_px.len()
        );
        assert_eq!(
            lat_mismatch, 0,
            "{name}: integer latents must match exactly"
        );
        assert!(
            max_raw <= 9.3e-7,
            "{name}: raw error {max_raw:e} exceeds recorded bound"
        );
        assert_eq!(
            hist[2] + hist[3],
            0,
            "{name}: sample difference exceeds one step"
        );
    }
}
