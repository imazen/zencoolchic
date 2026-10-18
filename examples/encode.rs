//! Encode a binary PPM (P6, 8-bit) to a `.cool` bitstream, then decode it back and report
//! PSNR and size. `cargo run --release --example encode --features encoder -- in.ppm out.cool [n_itr] [lambda] [arch]`

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
    let data = std::fs::read(&args[0]).unwrap();
    let (w, h, rgb) = read_ppm(&data);
    let mut opt = zencoolchic::unstable::TrainOptions::default();
    if let Some(n) = args.get(2) {
        opt.n_itr = n.parse().unwrap();
    }
    if let Some(l) = args.get(3) {
        opt.lambda = l.parse().unwrap();
    }
    if let Some(a) = args.get(4) {
        opt.arch = a.parse().unwrap();
    }
    let t = std::time::Instant::now();
    let r = zencoolchic::unstable::encode_rgb8(rgb, w, h, &opt, &mut |s| eprintln!("{s}")).unwrap();
    let secs = t.elapsed().as_secs_f64();
    std::fs::write(&args[1], &r.bitstream).unwrap();
    let pic = zencoolchic::decode(&r.bitstream, &zencoolchic::Limits::default()).unwrap();
    let out = pic.to_rgb8().unwrap();
    let mse: f64 = out
        .iter()
        .zip(rgb)
        .map(|(&a, &b)| ((a as f64 - b as f64) / 255.0).powi(2))
        .sum::<f64>()
        / out.len() as f64;
    let bpp = r.bitstream.len() as f64 * 8.0 / (w * h) as f64;
    println!(
        "{}x{} n_itr {} lambda {}: trained {:.3} dB / {:.4} bpp latent; written {} bytes = {:.4} bpp; decoded psnr {:.3} dB; {:.1} s",
        w,
        h,
        opt.n_itr,
        opt.lambda,
        r.trained.psnr_db,
        r.trained.rate_latent_bpp,
        r.bitstream.len(),
        bpp,
        -10.0 * mse.log10(),
        secs
    );
}
