//! Shared by unit and integration tests; never part of the library API.
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::format;
use std::path::PathBuf;
use std::string::String;
use std::sync::OnceLock;
use std::vec;
use std::vec::Vec;

const DECODER: &[&str] = &[
    "kodim14",
    "mid_lat1_debug",
    "mid_noifce_nohyper",
    "mid_nostab_debug",
    "mid_wasserstein_debug",
    "narrow7x40_debug",
    "odd37x23_debug",
    "small64_hop_2k",
    "small64_lop_2k",
    "small64_vhop_2k",
    "yuv420_8b_debug",
    "yuv444_10b_debug",
];
const GRADIENTS: &[&str] = &["mid_hop", "small64_hop", "small64_small"];

fn manifest() -> &'static serde_json::Value {
    static MANIFEST: OnceLock<serde_json::Value> = OnceLock::new();
    MANIFEST.get_or_init(|| {
        let m: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/upstream/manifest.json")).unwrap();
        assert_eq!(m["schema_version"], 1);
        let mut expected = BTreeSet::new();
        for (key, names, group, extensions) in [
            (
                "decoder_cases",
                DECODER,
                "decoder",
                &["cool", "json", "latents.i64", "raw.f32", "planes.u16"][..],
            ),
            (
                "gradient_cases",
                GRADIENTS,
                "gradients",
                &["json", "bin"][..],
            ),
            ("soap_cases", &["soap"][..], "soap", &["json", "bin"][..]),
        ] {
            let cases: Vec<&str> = m[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            assert_eq!(cases, names, "{key}: required fixture inventory");
            for name in names {
                for ext in extensions {
                    expected.insert(format!("{group}/{name}.{ext}"));
                }
            }
        }
        let entries = m["files"].as_array().unwrap();
        let actual: BTreeSet<_> = entries
            .iter()
            .map(|v| String::from(v["path"].as_str().unwrap()))
            .collect();
        assert_eq!(actual.len(), entries.len(), "duplicate fixture entries");
        assert_eq!(actual, expected, "complete fixture payload inventory");
        m
    })
}

fn override_dir(group: &str) -> Option<PathBuf> {
    let variable = match group {
        "decoder" => "ZENCOOLCHIC_ORACLE_DIR",
        "gradients" => "ZENCOOLCHIC_GRAD_DIR",
        "soap" => "ZENCOOLCHIC_SOAP_DIR",
        _ => panic!("unknown fixture group {group}"),
    };
    std::env::var_os(variable).map(PathBuf::from)
}

pub(crate) fn cases(group: &str) -> Vec<String> {
    let (key, ext) = match group {
        "decoder" => ("decoder_cases", "cool"),
        "gradients" => ("gradient_cases", "json"),
        "soap" => ("soap_cases", "json"),
        _ => panic!("unknown fixture group {group}"),
    };
    let required: Vec<String> = manifest()[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| String::from(v.as_str().unwrap()))
        .collect();
    if let Some(dir) = override_dir(group) {
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("explicit fixture override {}: {e}", dir.display()))
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|v| v == ext))
            .map(|p| String::from(p.file_stem().unwrap().to_str().unwrap()))
            .collect();
        names.sort();
        for name in &required {
            assert!(
                names.contains(name),
                "{}: missing required case {name}",
                dir.display()
            );
        }
        names
    } else {
        required
    }
}

fn hex_sha256(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn decode_checked(entry: &serde_json::Value, frame: &[u8]) -> Result<Vec<u8>, String> {
    let expected: usize = entry["bytes"].as_u64().unwrap().try_into().unwrap();
    if frame.len() as u64 != entry["compressed_bytes"].as_u64().unwrap()
        || hex_sha256(frame) != entry["compressed_sha256"].as_str().unwrap()
    {
        return Err(String::from("compressed length/SHA-256 mismatch"));
    }
    let packed = zenzstd::stream::decode_all_with_max(frame, Some(expected))
        .map_err(|e| format!("zstd decompression: {e}"))?;
    if packed.len() != expected {
        return Err(String::from("decoded length mismatch"));
    }
    let word = entry["shuffle_word_bytes"].as_u64().unwrap() as usize;
    if ![1, 2, 4, 8].contains(&word) || !expected.is_multiple_of(word) {
        return Err(String::from("invalid byte-plane layout"));
    }
    let mut data = vec![0; expected];
    let count = expected / word;
    for (i, sample) in data.chunks_exact_mut(word).enumerate() {
        for (b, value) in sample.iter_mut().enumerate() {
            *value = packed[b * count + i];
        }
    }
    if hex_sha256(&data) != entry["sha256"].as_str().unwrap() {
        return Err(String::from("original SHA-256 mismatch"));
    }
    Ok(data)
}

fn read_packaged(path: &str) -> Vec<u8> {
    let entry = manifest()["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["path"].as_str() == Some(path))
        .unwrap_or_else(|| panic!("fixture missing from manifest: {path}"));
    let file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/upstream")
        .join(format!("{path}.zst"));
    let frame = std::fs::read(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    decode_checked(entry, &frame).unwrap_or_else(|e| panic!("fixture {path}: {e}"))
}

pub(crate) fn read(group: &str, name: &str) -> Vec<u8> {
    if let Some(dir) = override_dir(group) {
        let file = dir.join(name);
        std::fs::read(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()))
    } else {
        read_packaged(&format!("{group}/{name}"))
    }
}

#[test]
fn all_frozen_payloads_are_intact() {
    for entry in manifest()["files"].as_array().unwrap() {
        read_packaged(entry["path"].as_str().unwrap());
    }
}

#[test]
fn corrupted_payload_and_wrong_output_length_are_rejected() {
    let original = &manifest()["files"][0];
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/upstream")
        .join(format!("{}.zst", original["path"].as_str().unwrap()));
    let mut frame = std::fs::read(path).unwrap();
    frame[0] ^= 1;
    assert!(
        decode_checked(original, &frame)
            .unwrap_err()
            .contains("SHA-256")
    );
    frame[0] ^= 1;
    let mut wrong_length = original.clone();
    wrong_length["bytes"] = (original["bytes"].as_u64().unwrap() - 1).into();
    assert!(decode_checked(&wrong_length, &frame).is_err());
}
