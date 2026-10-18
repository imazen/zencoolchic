//! Cool-chic 5 (Orange's overfitted neural image codec) in safe Rust.
//!
//! Port of the [Cool-chic](https://github.com/Orange-OpenSource/Cool-Chic) decoder at
//! upstream commit `a6fe38a` (v5.0.1). See `PORTING.md` for the module-by-module status
//! and the parity evidence.

#![no_std]
#![forbid(unsafe_code)]
// The encoder lands in pieces; until its public entry point exists the writer is only
// reached from tests. Remove with the first public encode API.
#![cfg_attr(feature = "encoder", allow(dead_code))]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

#[cfg(all(test, feature = "oracle"))]
#[path = "../tests/support/oracle_fixtures.rs"]
mod oracle_fixtures;

mod arm;
mod bitio;
#[cfg(feature = "zencodec")]
pub mod codec;
mod decode;
mod error;
mod expgolomb;
pub mod header;
mod latent;
mod nn;
mod rangecoder;
mod scale_table;
mod simd;
#[cfg(feature = "encoder")]
mod train;
#[cfg(feature = "encoder")]
#[doc(hidden)]
pub mod unstable;
#[cfg(feature = "encoder")]
mod write;

pub use decode::{DecodeDetails, Limits, Picture, decode, decode_with_details};
pub use error::{Error, Result};
pub use header::FrameDataType;
