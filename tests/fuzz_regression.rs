//! Fuzz seed/crash regression suite.
//!
//! Replays every file under `fuzz/seeds/` and `fuzz/regression/` through the
//! public decode entry points — the same calls the `decode_still` and
//! `decode_video` fuzz targets make. Seeds are known-valid streams and must
//! decode; regression files are minimized past crashes and must at least not
//! panic (a clean error is the fix working).
//!
//! To add a new seed: drop the (preferably minimized) crash file into
//! `fuzz/regression/` with a `crash-<sha>` name, no other action required.

use std::fs;
use std::path::PathBuf;
use zencoolchic::{FrameDataType, Limits, Picture, decode, decode_video};

const LIMITS: Limits = Limits {
    max_pixels: 1 << 20,
    max_frames: 8,
};

fn check_picture(pic: &Picture) {
    assert_eq!(pic.planes.len(), pic.plane_dims.len());
    for (plane, &(w, h)) in pic.planes.iter().zip(&pic.plane_dims) {
        assert_eq!(plane.len(), w as usize * h as usize);
    }
    if pic.data_type == FrameDataType::Rgb && pic.bitdepth == 8 {
        assert_eq!(
            pic.to_rgb8().unwrap().len(),
            3 * pic.width as usize * pic.height as usize
        );
    } else {
        assert!(pic.to_rgb8().is_none());
    }
}

/// What each fuzz target does with the input, minus libFuzzer.
fn exercise(input: &[u8]) {
    if let Ok(pic) = decode(input, &LIMITS) {
        check_picture(&pic);
    }
    if let Ok(frames) = decode_video(input, &LIMITS) {
        assert!(frames.len() <= LIMITS.max_frames as usize);
        for pic in &frames {
            check_picture(pic);
        }
    }
}

fn replay_dir(dir: &str, must_decode: bool) -> usize {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz")
        .join(dir);
    let mut n = 0;
    let Ok(entries) = fs::read_dir(&path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let data = fs::read(entry.path()).expect("read seed");
        exercise(&data);
        if must_decode {
            assert!(
                decode_video(&data, &LIMITS).is_ok(),
                "seed {} no longer decodes",
                entry.path().display()
            );
        }
        n += 1;
    }
    n
}

#[test]
fn fuzz_seeds_and_regressions_replay() {
    let seeds = replay_dir("seeds", true);
    assert!(seeds >= 2, "fuzz/seeds lost its corpus ({seeds} files)");
    let _ = replay_dir("regression", false);
}
