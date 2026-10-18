//! Train a Cool-chic on a binary PPM (P6, 8-bit) and print validation PSNR / bpp.
//! `cargo run --release --example train --features encoder -- in.ppm [n_itr] [lambda] [arch]`

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data = std::fs::read(&args[0]).unwrap();
    // Minimal P6 parser (no comments).
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
    let rgb = &data[i + 1..i + 1 + 3 * (w * h) as usize];
    let mut opt = zencoolchic::unstable::TrainOptions::default();
    if let Some(n) = args.get(1) {
        opt.n_itr = n.parse().unwrap();
    }
    if let Some(l) = args.get(2) {
        opt.lambda = l.parse().unwrap();
    }
    if let Some(a) = args.get(3) {
        opt.arch = a.parse().unwrap();
    }
    let t = std::time::Instant::now();
    let r = zencoolchic::unstable::train_rgb8(rgb, w, h, &opt, &mut |s| eprintln!("{s}")).unwrap();
    println!(
        "{}x{} n_itr {} lambda {}: psnr {:.3} dB, latent rate {:.4} bpp, loss {:.6}, {:.1} s",
        w,
        h,
        opt.n_itr,
        opt.lambda,
        r.psnr_db,
        r.rate_latent_bpp,
        r.loss,
        t.elapsed().as_secs_f64()
    );
}
