//! Cool-chic 5 (Orange's overfitted neural image codec) in safe Rust.
//!
//! Port of the [Cool-chic](https://github.com/Orange-OpenSource/Cool-Chic) decoder at
//! upstream commit `a6fe38a` (v5.0.1). See `PORTING.md` for the module-by-module status
//! and the parity evidence.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod arm;
mod bitio;
mod decode;
mod error;
mod expgolomb;
pub mod header;
mod latent;
mod nn;
mod rangecoder;
mod scale_table;

pub use decode::{DecodeDetails, Limits, Picture, decode, decode_with_details};
pub use error::{Error, Result};
pub use header::FrameDataType;
