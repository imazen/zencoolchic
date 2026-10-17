#!/usr/bin/env bash
# Encode the oracle test streams with upstream Cool-chic.
# Usage: scripts/oracle/make_streams.sh <Cool-Chic checkout> <source dir> <out dir>
# <source dir> holds the PNG / YUV crops (scripts/oracle/make_sources.py writes them).
set -euo pipefail
cc=$1; src=$2; out=$3
py="$cc/.venv/bin/python"
mkdir -p "$out"
enc() { # name input extra-args...
    local name=$1 input=$2; shift 2
    mkdir -p "$out/wd_$name"
    (cd "$cc" && "$py" cc_encode.py -i "$input" -o "$out/$name.cool" --workdir "$out/wd_$name" "$@") \
        > "$out/wd_$name/encode.log" 2>&1
    echo "$name $(wc -c < "$out/$name.cool") bytes"
}
enc odd37x23_debug        "$src/odd37x23.png" --debug
enc narrow7x40_debug      "$src/narrow7x40.png" --debug
enc small64_hop_2k        "$src/small64.png" --n_itr 2000 --lmbda 0.001
enc small64_lop_2k        "$src/small64.png" --n_itr 2000 --lmbda 0.004 --dec_cfg_residue cfg/dec/intra/lop.cfg
enc small64_vhop_2k       "$src/small64.png" --n_itr 2000 --lmbda 0.0005 --dec_cfg_residue cfg/dec/intra/vhop.cfg
enc mid_lat1_debug        "$src/mid130x97.png" --debug --latent_resolution_residue 1-6
enc mid_noifce_nohyper    "$src/mid130x97.png" --debug --ifce_resolution_residue no --hyperlatent_resolution_residue no
enc mid_nostab_debug      "$src/mid130x97.png" --debug --arm_residue 8,1 --layers_synthesis_residue 16-1-linear-relu,X-1-linear-none,X-3-residual-none
enc mid_wasserstein_debug "$src/mid130x97.png" --debug --tune wasserstein
enc yuv420_8b_debug       "$src/yuv_48x32_1p_yuv420_8b.yuv" --debug
enc yuv444_10b_debug      "$src/yuv_40x24_1p_yuv444_10b.yuv" --debug
