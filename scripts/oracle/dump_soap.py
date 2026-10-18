"""SOAP oracle: run upstream's SOAP for N steps on fixed gradient sequences.

Usage: python -I scripts/oracle/dump_soap.py <Cool-Chic checkout> <out_prefix>
Writes <out_prefix>.json (cases, configs) and <out_prefix>.bin (per case: initial
parameter, every step's gradient, final parameter; little-endian f32).
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import torch  # noqa: E402

from coolchic.training.soap import SOAP  # noqa: E402

torch.manual_seed(7)
N = 25
cases = [
    # name, shape, group config (as train.py builds it)
    ("w20x20", [20, 20], dict(lr=0.01, betas=(0.95, 0.95), precondition_frequency=10, max_precond_dim=256, weight_decay=0.01)),
    ("w7x5_rank_deficient", [7, 5], dict(lr=0.01, betas=(0.95, 0.95), precondition_frequency=3, max_precond_dim=256, weight_decay=0.01)),
    ("bias20", [20], dict(lr=0.01, betas=(0.95, 0.95), precondition_frequency=10, max_precond_dim=256, weight_decay=0.01)),
    ("latent", [1, 1, 16, 16], dict(lr=0.01, betas=(0.9, 0.999), precondition_frequency=1, max_precond_dim=0, weight_decay=0.0)),
]
meta = []
with open(sys.argv[2] + ".bin", "wb") as f:
    for name, shape, cfg in cases:
        p = torch.nn.Parameter(torch.randn(shape))
        # Full-rank gradients: each step's gradient is a fresh dense random tensor.
        grads = [torch.randn(shape) for _ in range(N)]
        opt = SOAP([{"params": [p], "merge_dims": False, "precondition_1d": False, **cfg}])
        f.write(p.detach().numpy().astype("<f4").tobytes())
        for g in grads:
            p.grad = g.clone()
            opt.step()
            f.write(g.numpy().astype("<f4").tobytes())
        f.write(p.detach().numpy().astype("<f4").tobytes())
        meta.append({"name": name, "shape": shape, "steps": N, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}})
json.dump({"cases": meta, "torch": torch.__version__}, open(sys.argv[2] + ".json", "w"), indent=1)
print("ok")
