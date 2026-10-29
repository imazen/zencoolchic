//! Encode a GOP of binary PPM (P6, 8-bit) frames to a `.cool` video bitstream:
//! display index 0 is intra, the last frame is a P frame, everything in
//! between is a B frame (upstream `--intra_pos 0 --p_pos <last>`).
//! `cargo run --release --example encvideo --features encoder -- out.cool in0.ppm in1.ppm ... [n_itr]`

fn read_ppm(data: &[u8]) -> (u32, u32, &[u8]) {
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while data[i].is_ascii_whitespace() {
            i += 1;
        }
        let s = i;
        while !data[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(String::from_utf8_lossy(&data[s..i]).to_string());
    }
    assert_eq!(fields[0], "P6");
    let (w, h): (u32, u32) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    (w, h, &data[i + 1..i + 1 + 3 * (w * h) as usize])
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_path = &args[0];
    let (paths, n_itr) = match args.last().unwrap().parse::<usize>() {
        Ok(n) => (&args[1..args.len() - 1], n),
        Err(_) => (&args[1..], 2_000),
    };
    if paths.len() < 2 {
        eprintln!("usage: encvideo out.cool in0.ppm in1.ppm [...] [n_itr]");
        std::process::exit(2);
    }
    let mut raws = Vec::new();
    let mut dims = (0u32, 0u32);
    for p in paths.iter() {
        let data = std::fs::read(p).unwrap();
        let (w, h, rgb) = read_ppm(&data);
        if dims == (0, 0) {
            dims = (w, h);
        }
        assert_eq!((w, h), dims, "all frames must share dimensions");
        raws.push(rgb.to_vec());
    }
    let (w, h) = dims;
    let npx = w as usize * h as usize;
    let plane_storage: Vec<Vec<Vec<u16>>> = raws
        .iter()
        .map(|rgb| {
            (0..3)
                .map(|c| {
                    (0..npx)
                        .map(|i| rgb[3 * i + c] as u16)
                        .collect::<Vec<u16>>()
                })
                .collect()
        })
        .collect();
    let frames: Vec<Vec<zencoolchic::unstable::Plane>> = plane_storage
        .iter()
        .map(|planes| {
            planes
                .iter()
                .map(|s| zencoolchic::unstable::Plane {
                    width: w,
                    height: h,
                    samples: s,
                })
                .collect()
        })
        .collect();
    let views: Vec<&[zencoolchic::unstable::Plane]> = frames.iter().map(|f| f.as_slice()).collect();
    let n = frames.len() as u32;
    let opt = zencoolchic::unstable::TrainOptions {
        n_itr,
        ..Default::default()
    };
    // WARP_FILTER=2|4|8|12 overrides upstream's default warp filter size.
    let warp = std::env::var("WARP_FILTER")
        .ok()
        .map(|v| v.parse().unwrap());
    let mut vopt = zencoolchic::unstable::VideoOptions {
        warp_filter_size: warp,
        ..Default::default()
    };
    let r = zencoolchic::unstable::encode_video(
        zencoolchic::FrameDataType::Rgb,
        8,
        &views,
        &[0],
        &[n - 1],
        &opt,
        &mut vopt,
        &mut |s| eprintln!("{s}"),
    )
    .unwrap();
    std::fs::write(out_path, &r.bitstream).unwrap();
    for f in &r.frames {
        println!(
            "{:?}{} psnr {:.2} dB ({} bytes cum.)",
            f.frame_type,
            f.display_index,
            f.psnr_db,
            r.bitstream.len()
        );
    }
}
