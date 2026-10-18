//! `zencodec` still-image traits over the Cool-chic decoder and (with `encoder`) trainer.
//!
//! # Design mismatch
//!
//! Encoding trains a network for each image: even minimum effort is 2000 iterations.
//! Quality is **uncalibrated**: `lambda = 10^(-1 - 4*q/100)`, clamping `q` to 0–100;
//! higher quality decreases the rate penalty. There is no measured mapping to SSIM2 or
//! zensim. Resource estimates remain unknown; one image's benchmark is not a cost model.
//!
//! Cool-chic has **no magic number**. Registration detects the eight-byte single-image
//! video header, a heuristic which can collide with unrelated data. Always validate headers.
//! Only one RGB I-frame is accepted here. YUV and video are rejected, never labelled RGB.
//! RGB streams have no color carrier; output assumes full-range sRGB, as the RGB8 encoder
//! requires. RGB 9–16-bit samples are scaled to full-range RGB16 with nearest integer
//! rounding (source precision is recorded in `ImageInfo`). No HDR transfer is inferred.
//!
//! The codec needs the whole image. Pull/row/animation executors are unsupported; push
//! decode copies the one-shot output to the sink. Metadata has no carrier; the encoder bakes its orientation into pixels and drops the rest.
//! Decode cancellation is checked between latent grids and network stages; an individual
//! grid or convolution is not interruptible. Encode checks each training iteration,
//! quantization candidate and RDOQ candidate. It uses the ambient rayon pool.
//!
//! Dimension, pixel, frame, input and output caps are enforced. Working-memory accounting
//! is unavailable: a finite `max_memory_bytes` is refused before work rather than ignored.
//! Likewise sequential encode requests are refused: use an ambient single-thread pool.

use crate::Error;
use crate::header::{CoolChicHeader, FrameDataType, FrameHeader, FrameType, VideoHeader};
use alloc::borrow::Cow;
use alloc::vec::Vec;
use enough::Stop;
use whereat::{At, ErrorAtExt};
use zencodec::decode::{
    Decode, DecodeCapabilities, DecodeJob, DecodeOutput, DecodeRowSink, DecoderConfig, OutputInfo,
};
use zencodec::{
    CategorizedError, CodecError, ErrorCategory, ImageError, ImageFormat, ImageFormatDefinition,
    ImageInfo, InvalidKind, LimitKind, RequestError, ResourceError, ResourceLimits, Unsupported,
    UnsupportedImageKind, UnsupportedOperation,
};
use zenpixels::{PixelBuffer, PixelDescriptor};

impl CategorizedError for Error {
    fn codec_name(&self) -> Option<&'static str> {
        Some("zencoolchic")
    }
    fn category(&self) -> ErrorCategory {
        match self {
            Self::Truncated(_) => ImageError::UnexpectedEof.into(),
            Self::Invalid(_) => ImageError::Malformed.into(),
            Self::InvalidArgument(_) => RequestError::Invalid(InvalidKind::Parameters).into(),
            Self::Unsupported(_) => ImageError::Unsupported(UnsupportedImageKind::Feature).into(),
            Self::Cancelled(r) => ErrorCategory::Stopped(*r),
            Self::LimitExceeded(what) => ResourceError::Limits(match *what {
                "frames" => LimitKind::Frames,
                "memory" | "network parameter count" => LimitKind::Memory,
                _ => LimitKind::Pixels,
            })
            .into(),
        }
    }
}

impl From<Error> for At<CodecError> {
    #[track_caller]
    fn from(e: Error) -> Self {
        CodecError::of(e.start_at())
    }
}

fn unsupported(op: UnsupportedOperation) -> At<CodecError> {
    CodecError::from_parts(
        Some("zencoolchic"),
        RequestError::Unsupported(op).into(),
        alloc::boxed::Box::new(op),
    )
    .start_at()
}

fn limit(e: zencodec::LimitExceeded) -> At<CodecError> {
    CodecError::of(e.start_at())
}

fn detect(data: &[u8]) -> bool {
    data.starts_with(&[0x00, 0x10, 0x01, 0x00, 0x00, 0x00, 0x80, 0x00])
}

/// Single-image registration. Detection is heuristic: these are header fields, not magic.
pub static COOLCHIC_FORMAT_DEFINITION: ImageFormatDefinition = ImageFormatDefinition::new(
    "coolchic",
    None,
    "Cool-chic",
    "cool",
    &["cool"],
    "image/x-coolchic",
    &["image/x-coolchic"],
    false,
    false,
    false,
    true,
    8,
    detect,
);

/// The custom format identifier for still-image Cool-chic streams.
pub fn coolchic_format() -> ImageFormat {
    ImageFormat::Custom(&COOLCHIC_FORMAT_DEFINITION)
}

/// Reusable RGB still-image decoder configuration. No external model files are needed.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct CoolChicDecoderConfig;

/// Per-image decode job, with operation limits and an owned cooperative stop token.
#[derive(Clone, Default)]
pub struct CoolChicDecodeJob {
    limits: ResourceLimits,
    stop: Option<zencodec::StopToken>,
}

static DECODE_DESCRIPTORS: &[PixelDescriptor] =
    &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB16_SRGB];
static DECODE_CAPS: DecodeCapabilities = DecodeCapabilities::new()
    .with_cheap_probe(true)
    .with_stop(true)
    .with_native_16bit(true)
    .with_enforces_max_pixels(true);

impl DecoderConfig for CoolChicDecoderConfig {
    type Error = At<CodecError>;
    type Job<'a> = CoolChicDecodeJob;
    fn formats() -> &'static [ImageFormat] {
        static FORMATS: [ImageFormat; 1] = [ImageFormat::Custom(&COOLCHIC_FORMAT_DEFINITION)];
        &FORMATS
    }
    fn supported_descriptors() -> &'static [PixelDescriptor] {
        DECODE_DESCRIPTORS
    }
    fn capabilities() -> &'static DecodeCapabilities {
        &DECODE_CAPS
    }
    fn job<'a>(self) -> Self::Job<'a> {
        CoolChicDecodeJob::default()
    }
}

fn check_budget_support(limits: &ResourceLimits) -> Result<(), At<CodecError>> {
    if limits.max_memory_bytes.is_some() {
        return Err(
            Error::InvalidArgument("finite working-memory budgets are not supported").into(),
        );
    }
    Ok(())
}

impl CoolChicDecodeJob {
    fn headers(&self, data: &[u8]) -> Result<(FrameHeader, CoolChicHeader), At<CodecError>> {
        self.stop.check().map_err(Error::from)?;
        self.limits
            .check_input_size(data.len() as u64)
            .map_err(limit)?;
        let (v, rest) = VideoHeader::parse(data)?;
        self.limits.check_frames(v.n_frames).map_err(limit)?;
        if v.n_frames != 1 {
            return Err(Error::Unsupported("video").into());
        }
        let (f, rest) = FrameHeader::parse(rest)?;
        if f.frame_type != FrameType::I || f.frame_data_type != FrameDataType::Rgb {
            return Err(Error::Unsupported("only RGB I-frames are supported by zencodec").into());
        }
        let (h, _) = CoolChicHeader::parse(rest)?;
        let [height, width] = h.img_size;
        if width == 0 || height == 0 {
            return Err(Error::Invalid("empty picture").into());
        }
        self.limits.check_dimensions(width, height).map_err(limit)?;
        self.limits
            .check_total_pixels(u64::from(width) * u64::from(height))
            .map_err(limit)?;
        // Keep the native decoder's default pixel ceiling too.
        if u64::from(width) * u64::from(height) > crate::Limits::default().max_pixels {
            return Err(Error::LimitExceeded("pixels").into());
        }
        Ok((f, h))
    }
}

fn info(f: &FrameHeader, h: &CoolChicHeader) -> ImageInfo {
    ImageInfo::new(h.img_size[1], h.img_size[0], coolchic_format())
        .with_bit_depth(f.bitdepth as u8)
        .with_channel_count(3)
}

fn native_descriptor(bitdepth: u32) -> PixelDescriptor {
    if bitdepth == 8 {
        PixelDescriptor::RGB8_SRGB
    } else {
        PixelDescriptor::RGB16_SRGB
    }
}

impl<'a> DecodeJob<'a> for CoolChicDecodeJob {
    type Error = At<CodecError>;
    type Dec = CoolChicDecoder<'a>;
    type StreamDec = Unsupported<At<CodecError>>;
    type AnimationFrameDec = Unsupported<At<CodecError>>;
    fn with_stop(mut self, stop: zencodec::StopToken) -> Self {
        self.stop = Some(stop);
        self
    }
    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }
    fn probe(&self, data: &[u8]) -> Result<ImageInfo, Self::Error> {
        let (f, h) = self.headers(data)?;
        Ok(info(&f, &h))
    }
    fn output_info(&self, data: &[u8]) -> Result<OutputInfo, Self::Error> {
        let (f, h) = self.headers(data)?;
        Ok(OutputInfo::full_decode(
            h.img_size[1],
            h.img_size[0],
            native_descriptor(f.bitdepth),
        ))
    }
    fn decoder(
        self,
        data: Cow<'a, [u8]>,
        preferred: &[PixelDescriptor],
    ) -> Result<Self::Dec, Self::Error> {
        check_budget_support(&self.limits)?;
        let (f, h) = self.headers(&data)?;
        let native = native_descriptor(f.bitdepth);
        let available: &[PixelDescriptor] = if f.bitdepth == 8 {
            DECODE_DESCRIPTORS
        } else {
            &[native]
        };
        let descriptor =
            zencodec::decode::negotiate_pixel_format(preferred, available).unwrap_or(native);
        let bytes = u64::from(h.img_size[0])
            * u64::from(h.img_size[1])
            * if descriptor == PixelDescriptor::RGB8_SRGB {
                3
            } else {
                6
            };
        self.limits.check_output_size(bytes).map_err(limit)?;
        Ok(CoolChicDecoder {
            job: self,
            data,
            descriptor,
            info: info(&f, &h),
        })
    }
    fn push_decoder(
        self,
        data: Cow<'a, [u8]>,
        sink: &mut dyn DecodeRowSink,
        preferred: &[PixelDescriptor],
    ) -> Result<OutputInfo, Self::Error> {
        zencodec::helpers::copy_decode_to_sink(self, data, sink, preferred, |e| {
            CodecError::from_parts(
                Some("zencoolchic"),
                ErrorCategory::Io(zencodec::CodecIoKind::opaque()),
                e,
            )
            .start_at()
        })
    }
    fn streaming_decoder(
        self,
        _: Cow<'a, [u8]>,
        _: &[PixelDescriptor],
    ) -> Result<Self::StreamDec, Self::Error> {
        Err(unsupported(UnsupportedOperation::RowLevelDecode))
    }
    fn animation_frame_decoder(
        self,
        _: Cow<'a, [u8]>,
        _: &[PixelDescriptor],
    ) -> Result<Self::AnimationFrameDec, Self::Error> {
        Err(unsupported(UnsupportedOperation::AnimationDecode))
    }
}

/// One-shot RGB decoder bound to a validated header and selected output descriptor.
pub struct CoolChicDecoder<'a> {
    job: CoolChicDecodeJob,
    data: Cow<'a, [u8]>,
    descriptor: PixelDescriptor,
    info: ImageInfo,
}

impl Decode for CoolChicDecoder<'_> {
    type Error = At<CodecError>;
    fn decode(self) -> Result<DecodeOutput, Self::Error> {
        let stop = self
            .job
            .stop
            .as_ref()
            .map(|s| s as &dyn Stop)
            .unwrap_or(&enough::Unstoppable);
        let picture = crate::decode::decode_checked(
            &self.data,
            &crate::Limits {
                max_pixels: self
                    .job
                    .limits
                    .max_pixels
                    .unwrap_or(crate::Limits::default().max_pixels)
                    .min(crate::Limits::default().max_pixels),
                max_frames: 1,
            },
            stop,
        )?;
        let n = picture.width as usize * picture.height as usize;
        let wide = self.descriptor == PixelDescriptor::RGB16_SRGB;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(n * if wide { 6 } else { 3 })
            .map_err(|_| {
                CodecError::new(Some("zencoolchic"), ResourceError::OutOfMemory.into()).start_at()
            })?;
        let max = (1u32 << picture.bitdepth) - 1;
        for i in 0..n {
            for plane in &picture.planes {
                if wide {
                    let sample = ((u64::from(plane[i]) * 65535 + u64::from(max) / 2)
                        / u64::from(max)) as u16;
                    bytes.extend_from_slice(&sample.to_ne_bytes());
                } else {
                    bytes.push(plane[i] as u8);
                }
            }
        }
        stop.check().map_err(Error::from)?;
        let buf = PixelBuffer::from_vec(bytes, picture.width, picture.height, self.descriptor)
            .map_err(|e| {
                CodecError::from_parts(
                    Some("zencoolchic"),
                    ErrorCategory::Internal(zencodec::InternalKind::Bug),
                    alloc::boxed::Box::new(e),
                )
                .start_at()
            })?;
        Ok(DecodeOutput::new(buf, self.info))
    }
}

#[cfg(feature = "encoder")]
mod encode;
#[cfg(feature = "encoder")]
pub use encode::{Architecture, CoolChicEncodeJob, CoolChicEncoder, CoolChicEncoderConfig};

#[cfg(all(test, feature = "encoder"))]
mod tests;
