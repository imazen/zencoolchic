use super::*;
use alloc::{sync::Arc, vec};
use core::sync::atomic::{AtomicUsize, Ordering};
use zencodec::decode::{AnimationFrameDecoder, StreamingDecode};
use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
use zenpixels::PixelSlice;

fn fixture(kind: FrameDataType, depth: u32) -> Vec<u8> {
    fixture_at(kind, depth, 3, 4)
}

fn fixture_at(kind: FrameDataType, depth: u32, h: u32, w: u32) -> Vec<u8> {
    let mut h = crate::train::config::header(crate::train::config::Arch::Low, h, w);
    let a = crate::decode::arch(&h).unwrap();
    let shape = crate::train::Shape::new(&h, &a).unwrap();
    let mut p = shape.zero_params(&h, &a);
    p.out_transform.b.copy_from_slice(&[0.125, 0.333, 0.722]);
    let mut q = crate::train::quantize::NnQuant {
        shift: [Some([0, 0]); 4],
        ..Default::default()
    };
    q.shift[3] = Some([-10, -10]);
    crate::train::quantize::apply_to_header(&mut h, &q);
    let (values, latents) = crate::train::quantize::to_stream(&p, &q, 16.0);
    let grids = a
        .size_per_latent
        .iter()
        .zip(latents)
        .map(|(&(h, w), data)| crate::latent::Grid { h, w, data })
        .collect::<Vec<_>>();
    crate::write::write_image(&h, kind, depth, &values, &grids).unwrap()
}

/// A three-frame RGB video (I0, B1, P2 — coding order I,P,B) through
/// `encode_video`, so the animation path exercises coding ≠ display order.
fn video_fixture() -> Vec<u8> {
    use crate::unstable::{Plane, TrainOptions, VideoOptions, encode_video};
    let (w, h) = (16usize, 16usize);
    let data: Vec<Vec<u16>> = (0..3)
        .map(|shift| {
            (0..3 * w * h)
                .map(|i| ((i * 5 + shift * 41) % 256) as u16)
                .collect()
        })
        .collect();
    let frames: Vec<Vec<Plane>> = data
        .iter()
        .map(|d| {
            (0..3)
                .map(|c| Plane {
                    width: w as u32,
                    height: h as u32,
                    samples: &d[c * w * h..(c + 1) * w * h],
                })
                .collect()
        })
        .collect();
    let views: Vec<&[Plane]> = frames.iter().map(|f| f.as_slice()).collect();
    let opt = TrainOptions {
        arch: 3,
        n_itr: 2000,
        rdoq: false,
        ..Default::default()
    };
    encode_video(
        FrameDataType::Rgb,
        8,
        &views,
        &[0],
        &[2],
        &opt,
        &mut VideoOptions::default(),
        &mut |_| {},
    )
    .unwrap()
    .bitstream
}

fn fast_config() -> CoolChicEncoderConfig {
    CoolChicEncoderConfig::default()
        .with_iterations(2000)
        .with_architecture(Architecture::Low)
        .with_rdoq(false)
}

fn category(e: At<CodecError>) -> ErrorCategory {
    e.error().category()
}

#[test]
fn registration_uses_the_serialized_header() {
    let bytes = fixture(FrameDataType::Rgb, 8);
    let registry = zencodec::ImageFormatRegistry::from_vec(vec![&COOLCHIC_FORMAT_DEFINITION]);
    assert_eq!(registry.detect(&bytes), Some(coolchic_format()));
    for n in 0..8 {
        assert_eq!(registry.detect(&bytes[..n]), None);
    }
    assert_eq!(registry.detect(&[0; 32]), None);
    assert_eq!(registry.detect(&[0, 0x10, 1, 0, 0, 8, 0, 0]), None);
}

#[test]
fn error_categories_and_erased_envelope() {
    assert_eq!(
        Error::Truncated("x").category(),
        ImageError::UnexpectedEof.into()
    );
    assert_eq!(Error::Invalid("x").category(), ImageError::Malformed.into());
    assert_eq!(
        Error::Unsupported("x").category(),
        ImageError::Unsupported(UnsupportedImageKind::Feature).into()
    );
    assert_eq!(
        Error::InvalidArgument("x").category(),
        RequestError::Invalid(InvalidKind::Parameters).into()
    );
    assert_eq!(
        Error::LimitExceeded("frames").category(),
        ResourceError::Limits(LimitKind::Frames).into()
    );
    assert_eq!(
        Error::Cancelled(enough::StopReason::TimedOut).category(),
        ErrorCategory::Stopped(enough::StopReason::TimedOut)
    );
    zencodec_testkit::assert_uses_codec_error_envelope::<
        CoolChicEncoderConfig,
        CoolChicDecoderConfig,
    >();
    zencodec_testkit::check_decode_error_envelope(CoolChicDecoderConfig, &[0]).unwrap();
}

#[test]
fn probe_needs_headers_only_and_refuses_yuv() {
    let bytes = fixture(FrameDataType::Rgb, 8);
    let (_, r) = VideoHeader::parse(&bytes).unwrap();
    let (_, r) = FrameHeader::parse(r).unwrap();
    let (_, r) = CoolChicHeader::parse(r).unwrap();
    let header_len = bytes.len() - r.len();
    let job = CoolChicDecoderConfig.job();
    let i = job.probe(&bytes[..header_len]).unwrap();
    assert_eq!((i.width, i.height), (4, 3));
    assert_eq!(i.source_color.bit_depth, Some(8));
    assert!(
        job.clone()
            .decoder(Cow::Borrowed(&bytes[..header_len]), &[])
            .unwrap()
            .decode()
            .is_err()
    );
    for kind in [FrameDataType::Yuv420, FrameDataType::Yuv444] {
        assert_eq!(
            category(job.probe(&fixture(kind, 8)).unwrap_err()),
            ImageError::Unsupported(UnsupportedImageKind::Feature).into()
        );
    }
}

#[test]
fn trait_decode_matches_native_and_u16_negotiation() {
    for depth in [8, 10, 16] {
        let bytes = fixture(FrameDataType::Rgb, depth);
        let native = crate::decode(&bytes, &crate::Limits::default()).unwrap();
        let out = CoolChicDecoderConfig
            .job()
            .decoder(Cow::Borrowed(&bytes), &[])
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!((out.width(), out.height()), (4, 3));
        if depth == 8 {
            assert_eq!(
                out.pixels().contiguous_bytes().as_ref(),
                native.to_rgb8().unwrap()
            );
            let wide = CoolChicDecoderConfig
                .job()
                .decoder(Cow::Borrowed(&bytes), &[PixelDescriptor::RGB16_SRGB])
                .unwrap()
                .decode()
                .unwrap();
            let want: Vec<u8> = native
                .to_rgb8()
                .unwrap()
                .iter()
                .flat_map(|&v| (u16::from(v) * 257).to_ne_bytes())
                .collect();
            assert_eq!(wide.pixels().contiguous_bytes().as_ref(), want);
        } else {
            assert_eq!(out.pixels().descriptor(), PixelDescriptor::RGB16_SRGB);
            let max = (1u64 << depth) - 1;
            let expected: Vec<u8> = (0..12)
                .flat_map(|i| {
                    native.planes.iter().flat_map(move |p| {
                        (((u64::from(p[i]) * 65535 + max / 2) / max) as u16).to_ne_bytes()
                    })
                })
                .collect();
            assert_eq!(out.pixels().contiguous_bytes().as_ref(), expected);
            assert!(expected.iter().any(|&b| b != 0));
        }
        zencodec_testkit::check_decode_truncation_series(CoolChicDecoderConfig, &bytes).unwrap();
    }
}

#[test]
fn decoder_caps_reject_before_pixel_work() {
    let bytes = fixture(FrameDataType::Rgb, 8);
    for (l, kind) in [
        (ResourceLimits::none().with_max_width(3), LimitKind::Width),
        (ResourceLimits::none().with_max_height(2), LimitKind::Height),
        (
            ResourceLimits::none().with_max_pixels(11),
            LimitKind::Pixels,
        ),
        (ResourceLimits::none().with_max_frames(0), LimitKind::Frames),
        (
            ResourceLimits::none().with_max_input_bytes(1),
            LimitKind::InputSize,
        ),
        (
            ResourceLimits::none().with_max_total_pixels(11),
            LimitKind::TotalPixels,
        ),
    ] {
        let e = CoolChicDecoderConfig
            .job()
            .with_limits(l)
            .probe(&bytes)
            .unwrap_err();
        assert_eq!(category(e), ResourceError::Limits(kind).into());
    }
    let job = CoolChicDecoderConfig
        .job()
        .with_limits(ResourceLimits::none().with_max_output(35));
    let e = job.decoder(Cow::Borrowed(&bytes), &[]).err().unwrap();
    assert_eq!(
        category(e),
        ResourceError::Limits(LimitKind::OutputSize).into()
    );
    assert!(
        CoolChicDecoderConfig
            .job()
            .with_limits(ResourceLimits::none().with_max_memory(1))
            .decoder(Cow::Borrowed(&bytes), &[])
            .is_err()
    );
}

#[derive(Clone)]
struct CancelAfter {
    polls: Arc<AtomicUsize>,
    max: usize,
}
impl Stop for CancelAfter {
    fn check(&self) -> Result<(), enough::StopReason> {
        if self.polls.fetch_add(1, Ordering::SeqCst) >= self.max {
            Err(enough::StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_reaches_the_training_loop_and_decode_grid_loop() {
    let polls = Arc::new(AtomicUsize::new(0));
    let stop = zencodec::StopToken::new(CancelAfter {
        polls: polls.clone(),
        max: 24,
    });
    let data = [128u8; 36];
    let pixels = PixelSlice::new(&data, 4, 3, 12, PixelDescriptor::RGB8_SRGB).unwrap();
    let e = fast_config()
        .job()
        .with_stop(stop)
        .encoder()
        .unwrap()
        .encode(pixels)
        .unwrap_err();
    assert_eq!(
        category(e),
        ErrorCategory::Stopped(enough::StopReason::Cancelled)
    );
    assert_eq!(polls.load(Ordering::SeqCst), 25);
    let bytes = fixture(FrameDataType::Rgb, 8);
    let polls = Arc::new(AtomicUsize::new(0));
    let stop = zencodec::StopToken::new(CancelAfter {
        polls: polls.clone(),
        max: 5,
    });
    let e = CoolChicDecoderConfig
        .job()
        .with_stop(stop)
        .decoder(Cow::Borrowed(&bytes), &[])
        .unwrap()
        .decode()
        .unwrap_err();
    assert_eq!(
        category(e),
        ErrorCategory::Stopped(enough::StopReason::Cancelled)
    );
    assert_eq!(polls.load(Ordering::SeqCst), 6);
}

#[test]
fn quality_effort_and_invalid_config() {
    let mut prev = f32::INFINITY;
    for q in 0..=100 {
        let c = fast_config().with_generic_quality(q as f32);
        assert!(c.lambda() < prev);
        prev = c.lambda();
        assert_eq!(c.generic_quality(), Some(q as f32));
    }
    assert_eq!(fast_config().with_generic_effort(0).iterations(), 2000);
    assert_eq!(fast_config().with_generic_effort(10).iterations(), 22000);
    for c in [
        fast_config().with_lambda(0.0),
        fast_config().with_lambda(f32::NAN),
        fast_config().with_iterations(1999),
        fast_config().with_generic_quality(f32::INFINITY),
        fast_config().with_learning_rate(-1.0),
    ] {
        assert_eq!(
            category(c.job().encoder().err().unwrap()),
            RequestError::Invalid(InvalidKind::Parameters).into()
        );
    }
}

#[test]
fn encoder_input_and_limits_are_checked() {
    for d in [
        PixelDescriptor::RGB16_SRGB,
        PixelDescriptor::RGB8,
        PixelDescriptor::RGBF32_LINEAR,
        PixelDescriptor::RGBA8_SRGB,
    ] {
        let buf = PixelBuffer::new(4, 3, d);
        let e = fast_config()
            .job()
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap_err();
        assert_eq!(
            category(e),
            RequestError::Unsupported(UnsupportedOperation::PixelFormat).into()
        );
    }
    let buf = PixelBuffer::new(4, 3, PixelDescriptor::RGB8_SRGB);
    for (l, k) in [
        (
            ResourceLimits::none().with_max_pixels(11),
            LimitKind::Pixels,
        ),
        (
            ResourceLimits::none().with_max_output(0),
            LimitKind::OutputSize,
        ),
        (
            ResourceLimits::none().with_max_input_bytes(1),
            LimitKind::InputSize,
        ),
    ] {
        let e = fast_config()
            .job()
            .with_limits(l)
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap_err();
        assert_eq!(category(e), ResourceError::Limits(k).into());
    }
}

#[test]
fn tiny_encode_round_trip_and_stride_determinism() {
    let packed: Vec<u8> = (0..36).map(|i| (i * 7) as u8).collect();
    let mut strided = vec![0xff; 45];
    for y in 0..3 {
        strided[y * 15..y * 15 + 12].copy_from_slice(&packed[y * 12..y * 12 + 12]);
    }
    let run = |pixels| {
        fast_config()
            .job()
            .encoder()
            .unwrap()
            .encode(pixels)
            .unwrap()
    };
    let a = run(PixelSlice::new(&packed, 4, 3, 12, PixelDescriptor::RGB8_SRGB).unwrap());
    let b = run(PixelSlice::new(&strided, 4, 3, 15, PixelDescriptor::RGB8_SRGB).unwrap());
    assert_eq!(a.data(), b.data());
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let parallel = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let one = single
        .install(|| run(PixelSlice::new(&packed, 4, 3, 12, PixelDescriptor::RGB8_SRGB).unwrap()));
    let four = parallel
        .install(|| run(PixelSlice::new(&packed, 4, 3, 12, PixelDescriptor::RGB8_SRGB).unwrap()));
    assert_eq!(one.data(), four.data());
    let native = crate::decode(a.data(), &crate::Limits::default()).unwrap();
    let out = CoolChicDecoderConfig
        .job()
        .decoder(Cow::Borrowed(a.data()), &[])
        .unwrap()
        .decode()
        .unwrap();
    assert_eq!(
        out.pixels().contiguous_bytes().as_ref(),
        native.to_rgb8().unwrap()
    );
    assert_eq!((out.width(), out.height()), (4, 3));
    // A finite nonzero output cap must still be checked after writing.
    let e = fast_config()
        .job()
        .with_limits(ResourceLimits::none().with_max_output(1))
        .encoder()
        .unwrap()
        .encode(PixelSlice::new(&packed, 4, 3, 12, PixelDescriptor::RGB8_SRGB).unwrap())
        .unwrap_err();
    assert_eq!(
        category(e),
        ResourceError::Limits(LimitKind::OutputSize).into()
    );
}

#[test]
fn applicable_testkit_contracts() {
    let img = zencodec_testkit::TestImage::rgb8_gradient(4, 3);
    zencodec_testkit::check_capability_honesty(fast_config(), CoolChicDecoderConfig, &img).unwrap();
    zencodec_testkit::check_metadata_no_leak(fast_config(), CoolChicDecoderConfig, &img).unwrap();
    // The published orientation check, like pixel_roundtrip, requires source pixel
    // identity. The lossy codec uses the exact oriented/preoriented stream comparison
    // below instead (user approved 2026-10-17).
    // Published testkit 0.1.0's pixel_roundtrip/cross_path require lossless input identity;
    // those checks do not apply to an overfitted lossy encoder. Native/trait parity above
    // tests decoded identity instead, including the push-sink path separately.
}

#[test]
fn push_sink_preserves_pixels_with_strided_destination() {
    struct Sink {
        bytes: Vec<u8>,
        stride: usize,
        finished: bool,
    }
    impl DecodeRowSink for Sink {
        fn provide_next_buffer(
            &mut self,
            y: u32,
            height: u32,
            width: u32,
            desc: PixelDescriptor,
        ) -> Result<zenpixels::PixelSliceMut<'_>, zencodec::decode::SinkError> {
            let start = y as usize * self.stride;
            Ok(zenpixels::PixelSliceMut::new(
                &mut self.bytes[start..],
                width,
                height,
                self.stride,
                desc,
            )
            .unwrap())
        }
        fn finish(&mut self) -> Result<(), zencodec::decode::SinkError> {
            self.finished = true;
            Ok(())
        }
    }
    let bytes = fixture(FrameDataType::Rgb, 8);
    let one = CoolChicDecoderConfig
        .job()
        .decoder(Cow::Borrowed(&bytes), &[])
        .unwrap()
        .decode()
        .unwrap();
    let mut sink = Sink {
        bytes: vec![0xaa; 45],
        stride: 15,
        finished: false,
    };
    CoolChicDecoderConfig
        .job()
        .push_decoder(Cow::Borrowed(&bytes), &mut sink, &[])
        .unwrap();
    assert!(sink.finished);
    for y in 0..3 {
        assert_eq!(&sink.bytes[y * 15..y * 15 + 12], one.pixels().row(y as u32));
        assert_eq!(&sink.bytes[y * 15 + 12..(y + 1) * 15], &[0xaa; 3]);
    }
}

#[test]
fn animation_decoder_yields_display_order_frames() {
    let stream = video_fixture();
    let native = crate::decode_video(
        &stream,
        &crate::Limits {
            max_frames: 8,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(native.len(), 3);
    let mut dec = CoolChicDecoderConfig
        .job()
        .animation_frame_decoder(Cow::Borrowed(&stream[..]), &[])
        .unwrap();
    assert_eq!(dec.frame_count(), Some(3));
    assert_eq!(dec.loop_count(), None);
    assert_eq!((dec.info().width, dec.info().height), (16, 16));
    for (i, want) in native.iter().enumerate() {
        let f = dec.render_next_frame(None).unwrap().unwrap();
        assert_eq!(f.frame_index(), i as u32);
        assert_eq!(f.duration_ms(), 0);
        let got = f.pixels();
        assert_eq!((got.width(), got.rows()), (16, 16));
        assert_eq!(got.descriptor(), PixelDescriptor::RGB8_SRGB);
        assert_eq!(
            got.contiguous_bytes().as_ref(),
            want.to_rgb8().unwrap().as_slice()
        );
    }
    assert!(dec.render_next_frame(None).unwrap().is_none());
}

#[test]
fn animation_decoder_rejects_yuv_and_honors_limits() {
    // A single-frame still is a degenerate animation: it yields once.
    let still = fixture(FrameDataType::Rgb, 8);
    let mut dec = CoolChicDecoderConfig
        .job()
        .animation_frame_decoder(Cow::Borrowed(&still[..]), &[])
        .unwrap();
    assert_eq!(dec.frame_count(), Some(1));
    let f = dec.render_next_frame(None).unwrap().unwrap();
    assert_eq!((f.pixels().width(), f.pixels().rows()), (4, 3));
    assert!(dec.render_next_frame(None).unwrap().is_none());
    for kind in [FrameDataType::Yuv420, FrameDataType::Yuv444] {
        let bytes = fixture(kind, 8);
        let e = CoolChicDecoderConfig
            .job()
            .animation_frame_decoder(Cow::Borrowed(&bytes[..]), &[])
            .err()
            .unwrap();
        assert_eq!(
            category(e),
            ImageError::Unsupported(UnsupportedImageKind::Feature).into()
        );
    }
    let stream = video_fixture();
    let e = CoolChicDecoderConfig
        .job()
        .with_limits(ResourceLimits::none().with_max_frames(2))
        .animation_frame_decoder(Cow::Borrowed(&stream[..]), &[])
        .err()
        .unwrap();
    assert_eq!(category(e), ResourceError::Limits(LimitKind::Frames).into());
}

#[test]
fn streaming_batches_cover_the_one_shot_decode() {
    // 70 rows → two batches (64 + 6), exercising the strip boundary.
    let bytes = fixture_at(FrameDataType::Rgb, 8, 70, 4);
    let mut s = CoolChicDecoderConfig
        .job()
        .streaming_decoder(Cow::Borrowed(&bytes[..]), &[])
        .unwrap();
    assert_eq!((s.info().width, s.info().height), (4, 70));
    let mut buf = Vec::new();
    let mut y = 0u32;
    while let Some((off, slice)) = s.next_batch().unwrap() {
        assert_eq!(off, y);
        y += slice.rows();
        buf.extend_from_slice(slice.contiguous_bytes().as_ref());
    }
    assert_eq!(y, 70);
    let one = CoolChicDecoderConfig
        .job()
        .decoder(Cow::Borrowed(&bytes[..]), &[])
        .unwrap()
        .decode()
        .unwrap();
    assert_eq!(buf, one.pixels().contiguous_bytes().as_ref());
}

#[test]
fn sequential_threading_encodes_identical_bytes() {
    let packed: Vec<u8> = (0..36).map(|i| (i * 7) as u8).collect();
    let pixels = || PixelSlice::new(&packed, 4, 3, 12, PixelDescriptor::RGB8_SRGB).unwrap();
    let parallel = fast_config()
        .job()
        .encoder()
        .unwrap()
        .encode(pixels())
        .unwrap();
    let sequential = fast_config()
        .job()
        .with_limits(ResourceLimits::none().with_threading(zencodec::ThreadingPolicy::Sequential))
        .encoder()
        .unwrap()
        .encode(pixels())
        .unwrap();
    assert_eq!(sequential.data(), parallel.data());
}

#[test]
fn orientation_is_baked_before_training_without_a_second_transform() {
    use zencodec::{Metadata, MetadataPolicy, Orientation};
    let packed: Vec<u8> = (0..36).map(|i| (i * 7) as u8).collect();
    let pixels = PixelSlice::new(&packed, 4, 3, 12, PixelDescriptor::RGB8_SRGB).unwrap();
    for ori in Orientation::ALL {
        let oriented = zenpixels_convert::orient::apply_orientation(pixels.clone(), ori);
        let expected = fast_config()
            .job()
            .encoder()
            .unwrap()
            .encode(oriented.as_slice())
            .unwrap();
        for policy in [
            MetadataPolicy::Web,
            MetadataPolicy::ColorAndRotation,
            MetadataPolicy::PreserveExact,
        ] {
            let actual = fast_config()
                .job()
                .with_metadata_policy(Metadata::none().with_orientation(ori), policy)
                .encoder()
                .unwrap()
                .encode(pixels.clone())
                .unwrap();
            assert_eq!(actual.data(), expected.data(), "{ori:?}");
            let picture = crate::decode(actual.data(), &crate::Limits::default()).unwrap();
            assert_eq!(
                (picture.width, picture.height),
                (oriented.width(), oriented.height())
            );
        }
    }
}
