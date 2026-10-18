//! Lossless fixture-size measurements; C zstd is an independent test oracle.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

fn shuffle(data: &[u8], word: usize) -> Vec<u8> {
    assert_eq!(data.len() % word, 0);
    let count = data.len() / word;
    let mut out = vec![0; data.len()];
    for (i, sample) in data.chunks_exact(word).enumerate() {
        for (b, value) in sample.iter().enumerate() {
            out[b * count + i] = *value;
        }
    }
    out
}

fn unshuffle(data: &[u8], word: usize) -> Vec<u8> {
    assert_eq!(data.len() % word, 0);
    let count = data.len() / word;
    let mut out = vec![0; data.len()];
    for (i, sample) in out.chunks_exact_mut(word).enumerate() {
        for (b, value) in sample.iter_mut().enumerate() {
            *value = data[b * count + i];
        }
    }
    out
}

fn category(name: &str) -> (&str, usize) {
    if name.ends_with(".latents.i64") {
        ("latents.i64", 8)
    } else if name.ends_with(".raw.f32") {
        ("raw.f32", 4)
    } else if name.ends_with(".planes.u16") {
        ("planes.u16", 2)
    } else if name.ends_with(".bin") {
        ("gradient-soap", 4)
    } else {
        ("streams-metadata", 1)
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "zstd_fixtures FIXTURE_ROOT NEW_OUTPUT");
    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    assert!(!output.exists(), "refuse to replace experiment data");
    std::fs::create_dir_all(output).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(input.join("manifest.json")).unwrap()).unwrap();
    let files = manifest["files"].as_array().unwrap();
    assert_eq!(files.len(), 68);
    // encoder, level, representation, category -> (original, compressed)
    let mut totals: BTreeMap<(String, i32, &str, &str), (usize, usize)> = BTreeMap::new();
    let mut detail = String::from("encoder,level,layout,path,original_bytes,zstd_bytes\n");
    for (encoder, levels) in [("zenzstd", &[3, 9, 15][..]), ("C-zstd", &[19][..])] {
        for &level in levels {
            for layout in ["plain", "byte-planes"] {
                for entry in files {
                    let name = entry["path"].as_str().unwrap();
                    let source = input.join(name);
                    let data = std::fs::read(&source).unwrap();
                    assert_eq!(data.len(), entry["bytes"].as_u64().unwrap() as usize);
                    let (kind, word) = category(name);
                    let width = if layout == "plain" { 1 } else { word };
                    let packed = shuffle(&data, width);
                    assert_eq!(unshuffle(&packed, width), data);
                    let frame_path = output.join(format!("{encoder}/{level}/{layout}/{name}.zst"));
                    std::fs::create_dir_all(frame_path.parent().unwrap()).unwrap();
                    let frame = if encoder == "zenzstd" {
                        let frame = zenzstd::encoding::compress_to_vec(
                            packed.as_slice(),
                            zenzstd::encoding::CompressionLevel::Level(level),
                        );
                        std::fs::write(&frame_path, &frame).unwrap();
                        let oracle = Command::new("zstd")
                            .args(["-q", "-d", "-c"])
                            .arg(&frame_path)
                            .output()
                            .expect("independent zstd CLI is required");
                        assert!(
                            oracle.status.success(),
                            "C decoder: {}",
                            String::from_utf8_lossy(&oracle.stderr)
                        );
                        assert_eq!(oracle.stdout, packed, "C zstd: {name}");
                        frame
                    } else {
                        let packed_path = output.join(format!("packed/{layout}/{name}"));
                        std::fs::create_dir_all(packed_path.parent().unwrap()).unwrap();
                        std::fs::write(&packed_path, &packed).unwrap();
                        let encoded = Command::new("zstd")
                            .args(["-q", "-19", "-T1", "-c"])
                            .arg(packed_path)
                            .output()
                            .expect("zstd CLI is required");
                        assert!(encoded.status.success());
                        std::fs::write(&frame_path, &encoded.stdout).unwrap();
                        encoded.stdout
                    };
                    let back =
                        zenzstd::stream::decode_all_with_max(frame.as_slice(), Some(data.len()))
                            .expect("bounded pure-Rust decompression");
                    assert_eq!(unshuffle(&back, width), data, "Rust zstd: {name}");
                    let total = totals
                        .entry((encoder.to_string(), level, layout, kind))
                        .or_default();
                    total.0 += data.len();
                    total.1 += frame.len();
                    detail.push_str(&format!(
                        "{encoder},{level},{layout},{name},{},{}\n",
                        data.len(),
                        frame.len()
                    ));
                }
                eprintln!(
                    "verified {encoder} level {level} {layout}: {} files",
                    files.len()
                );
            }
        }
    }
    std::fs::write(output.join("per-file.csv"), detail).unwrap();
    let mut csv = String::from("encoder,level,layout,category,original_bytes,zstd_bytes\n");
    for ((encoder, level, layout, kind), (raw, compressed)) in totals {
        csv.push_str(&format!(
            "{encoder},{level},{layout},{kind},{raw},{compressed}\n"
        ));
    }
    std::fs::write(output.join("summary.csv"), &csv).unwrap();
    print!("{csv}");
}
