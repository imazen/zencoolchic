//! Measure exact lossless storage of every decoder reference array.
//! Usage: fixture-compression DECODER_FIXTURE_DIR OUTPUT_DIR
#![forbid(unsafe_code)]

use std::path::Path;

fn measure(
    output: &Path,
    name: &str,
    kind: &str,
    part: usize,
    size: (usize, usize),
    bytes: &[u8],
    sample: imij::SampleType,
) {
    let (width, height) = size;
    let sample_bytes = sample.bytes();
    assert_eq!(bytes.len(), width * height * sample_bytes);
    let info = imij::ImageInfo::new(
        width.try_into().unwrap(),
        height.try_into().unwrap(),
        imij::Channels::Gray,
        sample,
    );
    let mut config = imij::EncodeConfig::default();
    config.threads = 1;
    let imi = imij::encode(bytes, info.row_bytes(), &info, &Default::default(), &config)
        .expect("IMI encoding");
    let (back_info, back) = imij::decode(&imi).expect("IMI decoding");
    assert_eq!(back_info, info);
    assert_eq!(back, bytes, "{name} {kind} {part}: IMI byte identity");

    let config = zenpng::EncodeConfig::default()
        .with_compression(zenpng::Compression::Balanced)
        .with_parallel(false)
        .with_downcast(zenpng::DowncastFlags::none());
    let stop = enough::Unstoppable;
    let png = if matches!(sample, imij::SampleType::U16) {
        let pixels: Vec<_> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| rgb::Gray(u16::from_le_bytes(*v)))
            .collect();
        zenpng::encode_gray16(
            imgref::Img::new(pixels, width, height).as_ref(),
            None,
            &config,
            &stop,
            &stop,
        )
        .unwrap()
    } else {
        // Floats and signed integers use byte-plane packing, not conversion to
        // display values. Row geometry is retained within each byte plane.
        let pixels: Vec<_> = (0..sample_bytes)
            .flat_map(|b| {
                bytes
                    .chunks_exact(sample_bytes)
                    .map(move |v| rgb::Gray(v[b]))
            })
            .collect();
        zenpng::encode_gray8(
            imgref::Img::new(pixels, width, height * sample_bytes).as_ref(),
            None,
            &config,
            &stop,
            &stop,
        )
        .unwrap()
    };
    let decoded = zenpng::decode(&png, &zenpng::PngDecodeConfig::strict(), &stop).unwrap();
    let packed = decoded.pixels.copy_to_contiguous_bytes();
    let back = if matches!(sample, imij::SampleType::U16) {
        packed
    } else {
        assert_eq!(packed.len(), bytes.len());
        let mut back = vec![0; bytes.len()];
        for (i, v) in back.chunks_exact_mut(sample_bytes).enumerate() {
            for (b, x) in v.iter_mut().enumerate() {
                *x = packed[b * width * height + i];
            }
        }
        back
    };
    assert_eq!(back, bytes, "{name} {kind} {part}: PNG byte identity");
    let stem = format!("{name}.{kind}.{part}");
    std::fs::write(output.join(format!("{stem}.imi")), &imi).unwrap();
    std::fs::write(output.join(format!("{stem}.png")), &png).unwrap();
    println!(
        "{name},{kind},{part},{width},{height},{},{},{}",
        bytes.len(),
        png.len(),
        imi.len()
    );
}

fn dimensions(v: &serde_json::Value) -> (usize, usize) {
    let d = v.as_array().unwrap();
    assert_eq!(d.len(), 2);
    (
        d[1].as_u64().unwrap() as usize,
        d[0].as_u64().unwrap() as usize,
    )
}

fn main() {
    const { assert!(cfg!(target_endian = "little"), "fixtures are little endian") };
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "usage: fixture-compression INPUT OUTPUT");
    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    assert!(
        !output.exists(),
        "refuse to replace existing experiment data"
    );
    std::fs::create_dir_all(output).unwrap();
    let mut metadata: Vec<_> = std::fs::read_dir(input)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    metadata.sort();
    assert_eq!(metadata.len(), 12, "all decoder cases are required");
    println!("case,kind,part,width,height,raw_bytes,png_bytes,imij_bytes");
    for path in metadata {
        let name = path.file_stem().unwrap().to_str().unwrap();
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for (kind, field, sample) in [
            ("latents.i64", "latent_shapes", imij::SampleType::I64),
            ("planes.u16", "plane_shapes", imij::SampleType::U16),
        ] {
            let data = std::fs::read(input.join(format!("{name}.{kind}"))).unwrap();
            let mut offset = 0;
            for (part, shape) in meta[field].as_array().unwrap().iter().enumerate() {
                let (width, height) = dimensions(shape);
                let len = width * height * sample.bytes();
                measure(
                    output,
                    name,
                    kind,
                    part,
                    (width, height),
                    &data[offset..offset + len],
                    sample,
                );
                offset += len;
            }
            assert_eq!(offset, data.len());
        }
        let shape = meta["raw_shape"].as_array().unwrap();
        assert_eq!(shape.len(), 3);
        let width = shape[2].as_u64().unwrap() as usize;
        let height = shape[1].as_u64().unwrap() as usize;
        let data = std::fs::read(input.join(format!("{name}.raw.f32"))).unwrap();
        let len = width * height * 4;
        assert_eq!(data.len(), shape[0].as_u64().unwrap() as usize * len);
        for (part, bytes) in data.chunks_exact(len).enumerate() {
            measure(
                output,
                name,
                "raw.f32",
                part,
                (width, height),
                bytes,
                imij::SampleType::F32,
            );
        }
    }
    eprintln!("fixture-compression: all inputs verified");
}
