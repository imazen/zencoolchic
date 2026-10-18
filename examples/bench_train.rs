//! Milliseconds per training step on a PPM: `cargo run --release --example bench_train --features encoder -- in.ppm [n] [arch]`
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data = std::fs::read(&args[0]).unwrap();
    let text_end = data.windows(4).position(|w| w == b"255\n").unwrap() + 4;
    let hdr = String::from_utf8_lossy(&data[..text_end]).to_string();
    let f: Vec<&str> = hdr.split_whitespace().collect();
    let (w, h): (u32, u32) = (f[1].parse().unwrap(), f[2].parse().unwrap());
    let n: usize = args.get(1).map_or(20, |s| s.parse().unwrap());
    let arch: u8 = args.get(2).map_or(1, |s| s.parse().unwrap());
    let ms = zencoolchic::unstable::bench_train_steps(&data[text_end..], w, h, arch, n).unwrap();
    println!("{w}x{h} arch {arch}: {ms:.1} ms per training step ({n} steps)");
}
