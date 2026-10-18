# Upstream Cool-chic checkout with a venv at .venv (see PORTING.md "Reference environment").
coolchic := env_var_or_default("COOLCHIC_UPSTREAM", env_var("HOME") + "/tmp/coolchic-src/Cool-Chic")
oracle_dir := env_var_or_default("ZENCOOLCHIC_ORACLE_DIR", env_var("HOME") + "/tmp/coolchic-ref")

test:
    cargo test --all-targets

# Dump upstream decoder intermediates for every .cool in the oracle dir.
oracle-dump:
    for f in {{oracle_dir}}/*.cool; do {{coolchic}}/.venv/bin/python -I scripts/oracle/dump_decode.py {{coolchic}} "$f" "${f%.cool}"; done

# Differential test against the upstream decoder dumps.
oracle:
    ZENCOOLCHIC_ORACLE_DIR={{oracle_dir}} cargo test --release --features oracle --test oracle -- --nocapture

clippy:
    cargo clippy --all-targets --features oracle

grad_dir := env_var_or_default("ZENCOOLCHIC_GRAD_DIR", env_var("HOME") + "/tmp/coolchic-grad64")
sources := env_var_or_default("ZENCOOLCHIC_SOURCES", env_var("HOME") + "/tmp/coolchic-gen")

# float64 gradient dumps from upstream (ground truth for the backward pass).
grad-dump:
    mkdir -p {{grad_dir}}
    {{coolchic}}/.venv/bin/python -I scripts/oracle/dump_grad.py {{coolchic}} {{sources}}/small64.png {{grad_dir}}/small64_small small f64
    {{coolchic}}/.venv/bin/python -I scripts/oracle/dump_grad.py {{coolchic}} {{sources}}/small64.png {{grad_dir}}/small64_hop hop f64
    {{coolchic}}/.venv/bin/python -I scripts/oracle/dump_grad.py {{coolchic}} {{sources}}/mid130x97.png {{grad_dir}}/mid_hop hop f64

# Training forward/backward vs torch.autograd (float64).
grad:
    ZENCOOLCHIC_GRAD_DIR={{grad_dir}} cargo test --release --features oracle --lib train::tests -- --nocapture

soap_dir := env_var_or_default("ZENCOOLCHIC_SOAP_DIR", env_var("HOME") + "/tmp/coolchic-soap")

soap-dump:
    mkdir -p {{soap_dir}}
    {{coolchic}}/.venv/bin/python -I scripts/oracle/dump_soap.py {{coolchic}} {{soap_dir}}/soap

# SOAP optimizer vs upstream.
soap:
    ZENCOOLCHIC_SOAP_DIR={{soap_dir}} cargo test --release --features oracle --lib soap -- --nocapture

# Every upstream-oracle test (decoder, re-encode, gradients, SOAP).
oracle-all:
    ZENCOOLCHIC_ORACLE_DIR={{oracle_dir}} ZENCOOLCHIC_GRAD_DIR={{grad_dir}} ZENCOOLCHIC_SOAP_DIR={{soap_dir}} cargo test --release --features oracle

# Gate before pushing: fmt, clippy on every feature set, tests.
check:
    cargo fmt --check
    cargo clippy --all-targets -- -D warnings
    cargo clippy --all-targets --features std -- -D warnings
    cargo clippy --all-targets --features encoder -- -D warnings
    cargo clippy --all-targets --features oracle -- -D warnings
    cargo clippy --all-targets --features zencodec -- -D warnings
    cargo clippy --all-targets --features zencodec,encoder -- -D warnings
    cargo test --release --lib --features encoder
    cargo test --release --features zencodec,encoder
    cargo test --doc --features zencodec,encoder

# Compile the decode-only bridge (including its no_std path).
codec-check:
    cargo check --features zencodec
    cargo check --features zencodec,encoder

# Test the common codec contracts and tiny encoder.
codec-test:
    cargo test --release --features zencodec,encoder

# Public trait example; raw packed RGB8 sRGB input, minimum effort (2000 iterations).
codec-encode input width height output:
    cargo run --release --example codec_encode --features encoder,zencodec -- {{input}} {{width}} {{height}} {{output}}

# Developer measurement only: requires sibling zqi and zenpng checkouts.
# Run outside the parent workspace's local Cargo patches.
fixture-compression input output:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$HOME/tmp"
    cd "$HOME/tmp"
    cargo run --release --manifest-path '{{justfile_directory()}}/tools/fixture-compression/Cargo.toml' -- '{{input}}' '{{output}}'

fixture-compression-build:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$HOME/tmp"
    cd "$HOME/tmp"
    cargo build --release --manifest-path '{{justfile_directory()}}/tools/fixture-compression/Cargo.toml'

fixture-compression-check:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "$HOME/tmp"
    cargo fmt --manifest-path '{{justfile_directory()}}/tools/fixture-compression/Cargo.toml' -p zencoolchic-fixture-compression -- --check
    cargo clippy --manifest-path '{{justfile_directory()}}/tools/fixture-compression/Cargo.toml' -- -D warnings

fixture-compression-fmt:
    cargo fmt --manifest-path '{{justfile_directory()}}/tools/fixture-compression/Cargo.toml' -p zencoolchic-fixture-compression

fixture-compression-summary input output log csv python="python3":
    '{{python}}' tools/fixture-compression/summarize.py '{{input}}' '{{output}}' '{{log}}' '{{csv}}'
