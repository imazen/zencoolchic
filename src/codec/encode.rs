use super::*;
use zencodec::encode::{
    AnimationFrameEncoder, EncodeCapabilities, EncodeJob, EncodeOutput, Encoder, EncoderConfig,
};
use zenpixels::{Cicp, PixelSlice};

/// Upstream intra decoder architecture, ordered from most to least compute.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Architecture {
    /// `vhop`: ARM context 20, synthesis width 64.
    VeryHigh,
    /// `hop` (default): ARM context 14, synthesis width 48.
    #[default]
    High,
    /// `mop`: ARM context 10, synthesis width 16.
    Medium,
    /// `lop`: ARM context 6, synthesis width 8.
    Low,
}

/// Reusable RGB8 sRGB encoder settings, wrapping the per-image trainer.
///
/// Defaults match upstream: lambda 0.001, 10,000 iterations, `hop`, seed 0 and RDOQ.
/// Invalid numeric settings are rejected by `encoder()` before training or allocation.
/// See the [module documentation](super) for the uncalibrated quality mapping and costs.
#[derive(Clone, Debug, Default)]
pub struct CoolChicEncoderConfig {
    options: crate::unstable::TrainOptions,
    quality: Option<f32>,
    effort: Option<i32>,
}

impl CoolChicEncoderConfig {
    /// Set the native rate penalty in `MSE + lambda * bits/pixel` (finite, `0 < lambda <= 1`).
    /// This clears the generic quality value. Larger lambda favors smaller streams.
    #[must_use]
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.options.lambda = lambda;
        self.quality = None;
        self
    }
    /// The native rate penalty currently configured.
    pub fn lambda(&self) -> f32 {
        self.options.lambda
    }
    /// Set the total training budget, including warm-up; must be at least 2000.
    /// This clears generic effort. Quantization and RDOQ are additional work.
    #[must_use]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.options.n_itr = iterations;
        self.effort = None;
        self
    }
    /// Total training budget, including warm-up and hard-round phases.
    pub fn iterations(&self) -> usize {
        self.options.n_itr
    }
    /// Set the initial learning rate (finite and positive); default 0.01.
    #[must_use]
    pub fn with_learning_rate(mut self, rate: f32) -> Self {
        self.options.start_lr = rate;
        self
    }
    /// Select the coded decoder architecture; default `hop`.
    #[must_use]
    pub fn with_architecture(mut self, architecture: Architecture) -> Self {
        self.options.arch = match architecture {
            Architecture::VeryHigh => 0,
            Architecture::High => 1,
            Architecture::Medium => 2,
            Architecture::Low => 3,
        };
        self
    }
    /// Set the reproducible initialization/noise seed; default 0 (not torch's RNG).
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.options.seed = seed;
        self
    }
    /// Enable or disable greedy RDOQ of the quantized networks; enabled by default.
    #[must_use]
    pub fn with_rdoq(mut self, enabled: bool) -> Self {
        self.options.rdoq = enabled;
        self
    }
    fn validate(&self) -> Result<(), At<CodecError>> {
        if !self.options.lambda.is_finite()
            || self.options.lambda <= 0.0
            || self.options.lambda > 1.0
            || !self.options.start_lr.is_finite()
            || self.options.start_lr <= 0.0
            || self.options.n_itr < 2000
            || self.quality.is_some_and(|q| !q.is_finite())
        {
            return Err(Error::InvalidArgument("lambda in (0,1], positive finite lr, finite quality and >=2000 iterations required").into());
        }
        Ok(())
    }
}

static ENCODE_CAPS: EncodeCapabilities = EncodeCapabilities::new()
    .with_lossy(true)
    .with_stop(true)
    .with_animation(true)
    .with_enforces_max_pixels(true)
    .with_quality_range(0.0, 100.0)
    .with_effort_range(0, 10);

impl EncoderConfig for CoolChicEncoderConfig {
    type Error = At<CodecError>;
    type Job = CoolChicEncodeJob;
    fn format() -> ImageFormat {
        coolchic_format()
    }
    fn supported_descriptors() -> &'static [PixelDescriptor] {
        &[PixelDescriptor::RGB8_SRGB]
    }
    fn capabilities() -> &'static EncodeCapabilities {
        &ENCODE_CAPS
    }
    /// Uncalibrated monotone mapping: `lambda = 10^(-1 - 4*q/100)`, with q clamped to
    /// 0–100. This is a codec-specific dial, not a measured perceptual-quality target.
    /// Nonfinite inputs are rejected at encoder construction.
    fn with_generic_quality(mut self, q: f32) -> Self {
        self.quality = Some(q.clamp(0.0, 100.0));
        if !q.is_finite() {
            self.quality = Some(q);
        }
        self.options.lambda = libm::powf(10.0, -1.0 - 4.0 * q.clamp(0.0, 100.0) / 100.0);
        self
    }
    fn generic_quality(&self) -> Option<f32> {
        self.quality
    }
    /// Clamp effort to 0–10, mapping to `2000 + 2000*effort` iterations. Uncalibrated.
    fn with_generic_effort(mut self, effort: i32) -> Self {
        let e = effort.clamp(0, 10);
        self.effort = Some(e);
        self.options.n_itr = 2000 + 2000 * e as usize;
        self
    }
    fn generic_effort(&self) -> Option<i32> {
        self.effort
    }
    fn job(self) -> Self::Job {
        CoolChicEncodeJob {
            config: self,
            limits: ResourceLimits::none(),
            stop: None,
            orientation: zencodec::Orientation::Identity,
        }
    }
}

/// Per-image encode job carrying limits and cancellation. Metadata has no carrier.
pub struct CoolChicEncodeJob {
    config: CoolChicEncoderConfig,
    limits: ResourceLimits,
    stop: Option<zencodec::StopToken>,
    orientation: zencodec::Orientation,
}

impl EncodeJob for CoolChicEncodeJob {
    type Error = At<CodecError>;
    type Enc = CoolChicEncoder;
    type AnimationFrameEnc = CoolChicAnimationEncoder;
    fn with_stop(mut self, stop: zencodec::StopToken) -> Self {
        self.stop = Some(stop);
        self
    }
    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }
    fn with_metadata(mut self, meta: zencodec::Metadata) -> Self {
        self.orientation = meta.orientation;
        self
    }
    fn encoder(self) -> Result<Self::Enc, Self::Error> {
        self.config.validate()?;
        self.stop.check().map_err(Error::from)?;
        Ok(CoolChicEncoder { job: self })
    }
    fn animation_frame_encoder(self) -> Result<Self::AnimationFrameEnc, Self::Error> {
        self.config.validate()?;
        self.stop.check().map_err(Error::from)?;
        Ok(CoolChicAnimationEncoder {
            job: self,
            frames: Vec::new(),
            dims: None,
            input_bytes: 0,
        })
    }
}

/// One-shot RGB8 sRGB encoder. Reads strided rows without including padding in the image.
pub struct CoolChicEncoder {
    job: CoolChicEncodeJob,
}

impl Encoder for CoolChicEncoder {
    type Error = At<CodecError>;
    fn reject(op: UnsupportedOperation) -> Self::Error {
        unsupported(op)
    }
    fn encode(self, pixels: PixelSlice<'_>) -> Result<EncodeOutput, Self::Error> {
        let l = &self.job.limits;
        self.job.stop.check().map_err(Error::from)?;
        if pixels.descriptor() != PixelDescriptor::RGB8_SRGB
            || pixels.color_context().is_some_and(|c| {
                c.icc.is_some()
                    || c.cicp.is_some_and(|v| v != Cicp::SRGB)
                    || c.diffuse_white.is_some()
            })
        {
            return Err(unsupported(UnsupportedOperation::PixelFormat));
        }
        let (input_width, input_height) = (pixels.width(), pixels.rows());
        let (width, height) = self
            .job
            .orientation
            .output_dimensions(input_width, input_height);
        if width == 0 || height == 0 || width > 16383 || height > 16383 {
            return Err(Error::InvalidArgument("dimensions must be 1..=16383").into());
        }
        l.check_dimensions(width, height).map_err(limit)?;
        l.check_frames(1).map_err(limit)?;
        l.check_total_pixels(u64::from(width) * u64::from(height))
            .map_err(limit)?;
        l.check_input_size(pixels.as_strided_bytes().len() as u64)
            .map_err(limit)?;
        if l.max_output_bytes == Some(0) {
            l.check_output_size(1).map_err(limit)?;
        }
        check_encode_budget(l, self.job.config.options.arch, width, height, 1)?;
        // Borrow tightly packed inputs; gather only logical rows for strided input.
        let oriented;
        let pixels = if self.job.orientation.is_identity() {
            pixels
        } else {
            oriented = zenpixels_convert::orient::apply_orientation(pixels, self.job.orientation);
            oriented.as_slice()
        };
        let rgb = pixels.contiguous_bytes();
        let stop = self
            .job
            .stop
            .as_ref()
            .map(|s| s as &dyn Stop)
            .unwrap_or(&enough::Unstoppable);
        // A sequential threading policy runs the encode on one thread: the
        // scoped override makes every training pool single-threaded (output
        // bytes are identical — decomposition is thread-count independent).
        let _seq = if self.job.limits.threading.is_parallel() {
            None
        } else {
            Some(crate::train::SequentialPool::install())
        };
        let report = crate::unstable::encode_rgb8_with_stop(
            &rgb,
            width,
            height,
            &self.job.config.options,
            &mut |_| {},
            stop,
        )?;
        l.check_output_size(report.bitstream.len() as u64)
            .map_err(limit)?;
        stop.check().map_err(Error::from)?;
        Ok(EncodeOutput::new(report.bitstream, coolchic_format()))
    }
}

/// Pre-flight trainer working-set check, run once dimensions are known:
/// `estimate_encode_bytes` bounds params + gradients + SOAP state +
/// activations + retained references for the architecture the encode
/// will train (`config::header`, or `header_inter` for video).
fn check_encode_budget(
    limits: &ResourceLimits,
    arch_idx: u8,
    w: u32,
    h: u32,
    n_frames: u32,
) -> Result<(), At<CodecError>> {
    if limits.max_memory_bytes.is_none() {
        return Ok(());
    }
    let arch = match arch_idx {
        0 => crate::train::config::Arch::VeryHigh,
        1 => crate::train::config::Arch::High,
        2 => crate::train::config::Arch::Medium,
        _ => crate::train::config::Arch::Low,
    };
    let inter = n_frames > 1;
    // Inter residue headers carry 3 + n_refs output channels; B frames use
    // at most two references, so 5 bounds every inter payload's arch.
    let hdr = if inter {
        crate::train::config::header_inter(arch, h, w, 5)
    } else {
        crate::train::config::header(arch, h, w)
    };
    let est = crate::decode::estimate_encode_bytes(&hdr, inter, u64::from(n_frames), 3)?;
    limits.check_memory(est).map_err(limit)
}

/// Combines the job-level stop token with `push_frame`/`finish`'s per-call one.
struct Or<'a>(&'a dyn Stop, &'a dyn Stop);
impl Stop for Or<'_> {
    fn check(&self) -> Result<(), enough::StopReason> {
        self.0.check().and_then(|()| self.1.check())
    }
    fn may_stop(&self) -> bool {
        self.0.may_stop() || self.1.may_stop()
    }
}

/// Full-canvas RGB8 animation encoder: frames are unpacked as they arrive and
/// the whole video trains on `finish` — Cool-chic is a whole-video codec, so
/// there is no incremental streaming. The coding structure is upstream's
/// `intra_pos 0`, `p_pos <last>` GOP (one intra frame plus B frames, closed by
/// a P frame; a single frame is all-intra). The stream carries no timing:
/// per-frame durations are accepted and dropped.
pub struct CoolChicAnimationEncoder {
    job: CoolChicEncodeJob,
    frames: Vec<Vec<Vec<u16>>>,
    dims: Option<(u32, u32)>,
    input_bytes: u64,
}

impl AnimationFrameEncoder for CoolChicAnimationEncoder {
    type Error = At<CodecError>;
    fn reject(op: UnsupportedOperation) -> Self::Error {
        unsupported(op)
    }
    fn push_frame(
        &mut self,
        pixels: PixelSlice<'_>,
        _duration_ms: u32,
        stop: Option<&dyn Stop>,
    ) -> Result<(), Self::Error> {
        self.job.stop.check().map_err(Error::from)?;
        if let Some(s) = stop {
            s.check().map_err(Error::from)?;
        }
        if pixels.descriptor() != PixelDescriptor::RGB8_SRGB
            || pixels.color_context().is_some_and(|c| {
                c.icc.is_some()
                    || c.cicp.is_some_and(|v| v != Cicp::SRGB)
                    || c.diffuse_white.is_some()
            })
        {
            return Err(unsupported(UnsupportedOperation::PixelFormat));
        }
        let (width, height) = self
            .job
            .orientation
            .output_dimensions(pixels.width(), pixels.rows());
        if width == 0 || height == 0 || width > 16383 || height > 16383 {
            return Err(Error::InvalidArgument("dimensions must be 1..=16383").into());
        }
        match self.dims {
            Some(dims) if dims != (width, height) => {
                return Err(
                    Error::InvalidArgument("animation frames must share dimensions").into(),
                );
            }
            Some(_) => {}
            None => self.dims = Some((width, height)),
        }
        let l = &self.job.limits;
        l.check_dimensions(width, height).map_err(limit)?;
        l.check_frames(self.frames.len() as u32 + 1)
            .map_err(limit)?;
        l.check_total_pixels(u64::from(width) * u64::from(height) * (self.frames.len() as u64 + 1))
            .map_err(limit)?;
        self.input_bytes += pixels.as_strided_bytes().len() as u64;
        l.check_input_size(self.input_bytes).map_err(limit)?;
        let oriented;
        let pixels = if self.job.orientation.is_identity() {
            pixels
        } else {
            oriented = zenpixels_convert::orient::apply_orientation(pixels, self.job.orientation);
            oriented.as_slice()
        };
        let rgb = pixels.contiguous_bytes();
        let npix = width as usize * height as usize;
        self.frames.push(
            (0..3)
                .map(|c| (0..npix).map(|i| rgb[3 * i + c] as u16).collect())
                .collect(),
        );
        Ok(())
    }
    fn finish(self, stop: Option<&dyn Stop>) -> Result<EncodeOutput, Self::Error> {
        let n = self.frames.len() as u32;
        if n == 0 {
            return Err(Error::InvalidArgument("animation encode needs at least one frame").into());
        }
        let job_stop = self
            .job
            .stop
            .as_ref()
            .map(|s| s as &dyn Stop)
            .unwrap_or(&enough::Unstoppable);
        let call_stop = stop.unwrap_or(&enough::Unstoppable);
        let stop = Or(job_stop, call_stop);
        // A sequential threading policy runs the encode on one thread: the
        // scoped override makes every training pool single-threaded (output
        // bytes are identical — decomposition is thread-count independent).
        let _seq = if self.job.limits.threading.is_parallel() {
            None
        } else {
            Some(crate::train::SequentialPool::install())
        };
        let (w, h) = self.dims.unwrap_or((0, 0));
        check_encode_budget(&self.job.limits, self.job.config.options.arch, w, h, n)?;
        let planes: Vec<Vec<crate::unstable::Plane>> = self
            .frames
            .iter()
            .map(|f| {
                f.iter()
                    .map(|samples| crate::unstable::Plane {
                        width: w,
                        height: h,
                        samples,
                    })
                    .collect()
            })
            .collect();
        let views: Vec<&[crate::unstable::Plane]> = planes.iter().map(|f| f.as_slice()).collect();
        // Upstream's default GOP: intra at 0, P at the last display index.
        let intra_pos = [0u32];
        let p_pos: &[u32] = if n > 1 { &[n - 1] } else { &[] };
        let report = crate::unstable::encode_video_with_stop(
            crate::FrameDataType::Rgb,
            8,
            &views,
            &intra_pos,
            p_pos,
            &self.job.config.options,
            &mut crate::unstable::VideoOptions::default(),
            &mut |_| {},
            &stop,
        )?;
        self.job
            .limits
            .check_output_size(report.bitstream.len() as u64)
            .map_err(limit)?;
        stop.check().map_err(Error::from)?;
        Ok(EncodeOutput::new(report.bitstream, coolchic_format()))
    }
}
