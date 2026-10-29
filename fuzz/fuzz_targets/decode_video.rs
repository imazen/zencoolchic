#![no_main]

use libfuzzer_sys::fuzz_target;
use zencoolchic::{Limits, Picture, decode_video};

/// Eight frames covers every frame-type/reference combination while keeping a
/// well-formed input fast; the pixel cap is the same one `decode_still` uses.
const LIMITS: Limits = Limits {
    max_pixels: 1 << 20,
    max_frames: 8,
};

fn check_picture(pic: &Picture) {
    assert_eq!(pic.planes.len(), pic.plane_dims.len());
    for (plane, &(w, h)) in pic.planes.iter().zip(&pic.plane_dims) {
        assert_eq!(plane.len(), w as usize * h as usize);
    }
}

fuzz_target!(|data: &[u8]| {
    if let Ok(frames) = decode_video(data, &LIMITS) {
        assert!(frames.len() <= LIMITS.max_frames as usize);
        for pic in &frames {
            check_picture(pic);
        }
    }
});
