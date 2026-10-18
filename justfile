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
    ZENCOOLCHIC_GRAD_DIR={{grad_dir}} cargo test --release --features oracle --lib train -- --nocapture
