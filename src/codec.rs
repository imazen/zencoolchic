//! `zencodec` still-image traits over the Cool-chic decoder and (with `encoder`) trainer.
//!
//! # Design mismatch
//!
//! Encoding trains a network for each image: even minimum effort is 2000 iterations.
//! `lambda = 10^(-1 - 4*q/100)`, clamping `q` to 0–100; higher quality decreases
//! the rate penalty. Measured calibration (6 CID22-512 validation images at
//! 256x256, hop, 10000 iterations, seed 0; REMAINING-WORK.md has the table):
//! q10 ~24 dB at ~0.23 bpp, q30 ~28 dB / 0.41, q50 ~33 dB / 0.89, q70 ~38 dB /
//! 1.80, q90 ~41 dB / 2.90, mean butteraugli 15.8/8.3/4.0/2.1/1.6. A second seed
//! moved results ~0.05-0.4 dB. It is a codec-specific dial, not a perceptual
//! target; content spread within one q is +-2-3 dB.
//!
//! Cool-chic has **no magic number**. Registration detects the eight-byte single-image
//! video header, a heuristic which can collide with unrelated data. Always validate headers.
//! Only one RGB I-frame is accepted here. YUV and video are rejected, never labelled RGB.
//! RGB streams have no color carrier; output assumes full-range sRGB, as the RGB8 encoder
//! requires. RGB 9–16-bit samples are scaled to full-range RGB16 with nearest integer
//! rounding (source precision is recorded in `ImageInfo`). No HDR transfer is inferred.
//!
//! The codec needs the whole image at once: the pull/streaming executor decodes fully at
//! construction and yields row batches of the result, and push decode copies the one-shot
//! output to the sink. Animation decode is lazy in coding order and yields display-order
//! frames (only RGB — `PixelDescriptor` cannot carry subsampled YUV); decoded frames stay
//! resident as P/B references, so long videos keep video-decoder memory.
//! Animation encode buffers `push_frame` RGB8 planes and trains the whole
//! video on `finish` with upstream's `intra_pos 0`, `p_pos <last>` GOP;
//! durations are dropped (the stream has no timing field). Metadata has no
//! carrier; the encoder bakes its orientation into pixels and drops the rest.
//! Decode cancellation is checked per ARM raster diagonal inside each latent grid,
//! per upsampling step and per synthesis output channel. Encode checks each
//! training iteration, quantization candidate and RDOQ candidate, on
//! problem-sized rayon pools.
//!
//! Dimension, pixel, frame, input and output caps are enforced. Decode honors
//! `max_memory_bytes` via a pre-flight working-set estimate (params, latents and
//! contexts, conv buffers, stored reference frames, output); encode estimates the
//! trainer peak the same way (plus gradients, SOAP state and backward activations)
//! once frame dimensions are known. A sequential threading policy runs encode on
//! one thread (identical bytes; only speed changes).

use crate::Error;
use crate::header::{CoolChicHeader, FrameDataType, FrameHeader, FrameType, VideoHeader};
use alloc::borrow::Cow;
use alloc::vec::Vec;
use enough::Stop;
use whereat::{At, ErrorAtExt};
#[cfg(feature = "encoder")]
use zencodec::UnsupportedOperation;
use zencodec::decode::{
    Decode, DecodeCapabilities, DecodeJob, DecodeOutput, DecodeRowSink, DecoderConfig, OutputInfo,
};
use zencodec::{
    CategorizedError, CodecError, ErrorCategory, ImageError, ImageFormat, ImageFormatDefinition,
    ImageInfo, InvalidKind, LimitKind, RequestError, ResourceError, ResourceLimits,
    UnsupportedImageKind,
};
use zenpixels::{PixelBuffer, PixelDescriptor, PixelSlice};

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

#[cfg(feature = "encoder")]
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
    .with_animation(true)
    .with_streaming(true)
    .with_native_16bit(true)
    .with_enforces_max_pixels(true)
    .with_enforces_max_memory(true);

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

fn check_decode_budget(
    limits: &ResourceLimits,
    h: &CoolChicHeader,
    inter: bool,
    n_frames: u64,
    descriptor: PixelDescriptor,
) -> Result<(), At<CodecError>> {
    if limits.max_memory_bytes.is_some() {
        let bpp = if descriptor == PixelDescriptor::RGB16_SRGB {
            6
        } else {
            3
        };
        let est = crate::decode::estimate_decode_bytes(h, inter, n_frames, bpp)?;
        limits.check_memory(est).map_err(limit)?;
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
    type StreamDec = CoolChicStreamDecoder;
    type AnimationFrameDec = CoolChicAnimationDecoder;
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
        let (f, h) = self.headers(&data)?;
        let native = native_descriptor(f.bitdepth);
        let available: &[PixelDescriptor] = if f.bitdepth == 8 {
            DECODE_DESCRIPTORS
        } else {
            &[native]
        };
        let descriptor =
            zencodec::decode::negotiate_pixel_format(preferred, available).unwrap_or(native);
        check_decode_budget(&self.limits, &h, false, 1, descriptor)?;
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
        data: Cow<'a, [u8]>,
        preferred: &[PixelDescriptor],
    ) -> Result<Self::StreamDec, Self::Error> {
        // Cool-chic decodes whole pictures; the pull API is served by decoding
        // once at construction and yielding row batches of the result.
        let decoder = self.decoder(data, preferred)?;
        let output = decoder.decode()?;
        let info = output.info().clone();
        Ok(CoolChicStreamDecoder {
            buf: output.into_buffer(),
            info,
            next_row: 0,
        })
    }
    fn animation_frame_decoder(
        self,
        data: Cow<'a, [u8]>,
        preferred: &[PixelDescriptor],
    ) -> Result<Self::AnimationFrameDec, Self::Error> {
        self.stop.check().map_err(Error::from)?;
        self.limits
            .check_input_size(data.len() as u64)
            .map_err(limit)?;
        let (v, rest) = VideoHeader::parse(&data)?;
        self.limits.check_frames(v.n_frames).map_err(limit)?;
        if v.n_frames == 0 {
            return Err(Error::Invalid("no frames").into());
        }
        let (f, rest) = FrameHeader::parse(rest)?;
        if f.frame_data_type != FrameDataType::Rgb {
            return Err(Error::Unsupported("only RGB frames are supported by zencodec").into());
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
        if u64::from(width) * u64::from(height) > crate::Limits::default().max_pixels {
            return Err(Error::LimitExceeded("pixels").into());
        }
        let native = native_descriptor(f.bitdepth);
        let available: &[PixelDescriptor] = if f.bitdepth == 8 {
            DECODE_DESCRIPTORS
        } else {
            &[native]
        };
        let descriptor =
            zencodec::decode::negotiate_pixel_format(preferred, available).unwrap_or(native);
        let inter = v
            .coding_structure()?
            .iter()
            .any(|sf| !sf.index_references.is_empty());
        check_decode_budget(&self.limits, &h, inter, u64::from(v.n_frames), descriptor)?;
        let dec = crate::decode::VideoDecoder::new(
            data.into_owned(),
            &crate::Limits {
                max_pixels: self
                    .limits
                    .max_pixels
                    .unwrap_or(crate::Limits::default().max_pixels)
                    .min(crate::Limits::default().max_pixels),
                max_frames: v.n_frames,
            },
            false,
            false,
        )?;
        Ok(CoolChicAnimationDecoder {
            dec,
            job: self,
            canvas: Vec::new(),
            yield_pos: 0,
            descriptor,
            info: info(&f, &h),
        })
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
        // Sequential threading policy → wavefront evaluation on this thread
        // only (identical output; the range coder is serial regardless).
        #[cfg(feature = "std")]
        let _seq = if self.job.limits.threading.is_parallel() {
            None
        } else {
            Some(crate::latent::SequentialThreads::install())
        };
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
        let wide = self.descriptor == PixelDescriptor::RGB16_SRGB;
        let bytes = rgb_bytes(&picture, wide)?;
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

/// Interleaved RGB8 (`wide = false`) or full-range RGB16 native-endian bytes of
/// an RGB picture's planes, scaling sub-16-bit samples with nearest integer
/// rounding.
fn rgb_bytes(picture: &crate::decode::Picture, wide: bool) -> Result<Vec<u8>, At<CodecError>> {
    let n = picture.width as usize * picture.height as usize;
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
                let sample =
                    ((u64::from(plane[i]) * 65535 + u64::from(max) / 2) / u64::from(max)) as u16;
                bytes.extend_from_slice(&sample.to_ne_bytes());
            } else {
                bytes.push(plane[i] as u8);
            }
        }
    }
    Ok(bytes)
}

fn sink_error(e: zencodec::decode::SinkError) -> At<CodecError> {
    CodecError::from_parts(
        Some("zencoolchic"),
        ErrorCategory::Io(zencodec::CodecIoKind::opaque()),
        e,
    )
    .start_at()
}

/// Pull-decoder yielding row batches of a one-shot decode: Cool-chic pictures
/// are only defined whole, so construction decodes fully and `next_batch`
/// hands out ≤64-row strips of the result.
pub struct CoolChicStreamDecoder {
    buf: PixelBuffer,
    info: ImageInfo,
    next_row: u32,
}

impl zencodec::decode::StreamingDecode for CoolChicStreamDecoder {
    type Error = At<CodecError>;
    fn next_batch(&mut self) -> Result<Option<(u32, PixelSlice<'_>)>, Self::Error> {
        if self.next_row >= self.buf.height() {
            return Ok(None);
        }
        let y = self.next_row;
        let rows = (self.buf.height() - y).min(64);
        self.next_row = y + rows;
        Ok(Some((y, self.buf.rows(y, rows).erase())))
    }
    fn info(&self) -> &ImageInfo {
        &self.info
    }
}

/// Animation decoder over a Cool-chic video stream. Frames are decoded lazily
/// in coding order and yielded in display order; decoded frames stay resident
/// as P/B references. YUV frames are refused when reached — `PixelDescriptor`
/// has no subsampled-YUV carrier — and frame sizes may differ.
pub struct CoolChicAnimationDecoder {
    dec: crate::decode::VideoDecoder<Vec<u8>>,
    job: CoolChicDecodeJob,
    canvas: Vec<u8>,
    yield_pos: usize,
    descriptor: PixelDescriptor,
    info: ImageInfo,
}

impl zencodec::decode::AnimationFrameDecoder for CoolChicAnimationDecoder {
    type Error = At<CodecError>;
    fn wrap_sink_error(err: zencodec::decode::SinkError) -> Self::Error {
        sink_error(err)
    }
    fn info(&self) -> &ImageInfo {
        &self.info
    }
    fn frame_count(&self) -> Option<u32> {
        Some(self.dec.n_frames() as u32)
    }
    fn render_next_frame(
        &mut self,
        stop: Option<&dyn Stop>,
    ) -> Result<Option<zencodec::decode::AnimationFrame<'_>>, Self::Error> {
        let i = self.yield_pos;
        if i >= self.dec.n_frames() {
            return Ok(None);
        }
        let mut check = || -> crate::Result<()> {
            self.job.stop.check().map_err(Error::from)?;
            if let Some(s) = stop {
                s.check().map_err(Error::from)?;
            }
            Ok(())
        };
        #[cfg(feature = "std")]
        let _seq = if self.job.limits.threading.is_parallel() {
            None
        } else {
            Some(crate::latent::SequentialThreads::install())
        };
        self.dec.pull_display(i, &mut check)?;
        let frame = self
            .dec
            .frame(i)
            .ok_or(Error::Invalid("uncoded display position"))?;
        let picture = frame.to_picture();
        if picture.data_type != FrameDataType::Rgb {
            return Err(Error::Unsupported("only RGB frames are supported by zencodec").into());
        }
        let wide = self.descriptor == PixelDescriptor::RGB16_SRGB;
        let bytes = u64::from(picture.width) * u64::from(picture.height) * if wide { 6 } else { 3 };
        self.job.limits.check_output_size(bytes).map_err(limit)?;
        self.canvas = rgb_bytes(&picture, wide)?;
        self.yield_pos += 1;
        let row_bytes = picture.width as usize * if wide { 6 } else { 3 };
        let slice = PixelSlice::new(
            &self.canvas,
            picture.width,
            picture.height,
            row_bytes,
            self.descriptor,
        )
        .map_err(|e| {
            CodecError::from_parts(
                Some("zencoolchic"),
                ErrorCategory::Internal(zencodec::InternalKind::Bug),
                alloc::boxed::Box::new(e),
            )
            .start_at()
        })?;
        Ok(Some(zencodec::decode::AnimationFrame::new(
            slice, 0, i as u32,
        )))
    }
    fn render_next_frame_to_sink(
        &mut self,
        stop: Option<&dyn Stop>,
        sink: &mut dyn DecodeRowSink,
    ) -> Result<Option<OutputInfo>, Self::Error> {
        zencodec::helpers::copy_frame_to_sink(self, stop, sink)
    }
}

#[cfg(feature = "encoder")]
mod encode;
#[cfg(feature = "encoder")]
pub use encode::{Architecture, CoolChicEncodeJob, CoolChicEncoder, CoolChicEncoderConfig};

#[cfg(all(test, feature = "encoder"))]
mod tests;
