//! Decode a `.cool` file; optionally write the picture as a binary PPM/PGM.
//! `cargo run --release --example decode --features std -- in.cool [out.ppm] [--repeat N]`

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let input = &args[0];
    let output = args.get(1).filter(|a| !a.starts_with("--"));
    let repeat: usize = args
        .iter()
        .position(|a| a == "--repeat")
        .map_or(1, |i| args[i + 1].parse().unwrap());
    let data = std::fs::read(input).unwrap();
    let mut best = f64::MAX;
    let mut pic = None;
    for _ in 0..repeat {
        let t = std::time::Instant::now();
        pic = Some(zencoolchic::decode(&data, &zencoolchic::Limits::default()).unwrap());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let pic = pic.unwrap();
    eprintln!(
        "{}x{} {:?} {}-bit: best of {repeat}: {best:.1} ms",
        pic.width, pic.height, pic.data_type, pic.bitdepth
    );
    if let Some(out) = output {
        let rgb = pic.to_rgb8().expect("PPM output needs an 8-bit RGB stream");
        let mut f = format!("P6\n{} {}\n255\n", pic.width, pic.height).into_bytes();
        f.extend_from_slice(&rgb);
        std::fs::write(out, f).unwrap();
    }
}
