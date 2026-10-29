#![no_main]

use libfuzzer_sys::fuzz_target;
use zencoolchic::{FrameDataType, Limits, Picture, decode, decode_with_details};

/// Fuzz-tight bounds: small well-formed streams still reach the entropy and
/// synthesis stages; anything claiming more work is refused at the limits
/// check before picture-sized allocation.
const LIMITS: Limits = Limits {
    max_pixels: 1 << 20,
    max_frames: 1,
};

fn check_picture(pic: &Picture) {
    assert_eq!(pic.planes.len(), pic.plane_dims.len());
    for (plane, &(w, h)) in pic.planes.iter().zip(&pic.plane_dims) {
        assert_eq!(plane.len(), w as usize * h as usize);
    }
    if pic.data_type == FrameDataType::Rgb && pic.bitdepth == 8 {
        let rgb = pic.to_rgb8().expect("rgb8 conversion refused");
        assert_eq!(rgb.len(), 3 * pic.width as usize * pic.height as usize);
    } else {
        assert!(pic.to_rgb8().is_none());
    }
}

fuzz_target!(|data: &[u8]| {
    if let Ok(pic) = decode(data, &LIMITS) {
        check_picture(&pic);
        // A successful parse must also succeed through the details path: both
        // share `decode_inner`, so any divergence is a bug.
        decode_with_details(data, &LIMITS).expect("decode_with_details diverged");
    }
});
