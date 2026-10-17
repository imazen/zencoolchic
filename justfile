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
