"""Shows that SOAP trajectories with rank-deficient Kronecker factors depend on the arbitrary
null-space basis of the first eigh: upstream SOAP run twice, the second time with the null
space eigenvectors rotated. Usage: python -I scripts/oracle/soap_null_space_probe.py <Cool-Chic>
(2026-10-17, torch 2.14.1: 1.6% of the parameter movement after 25 steps on a 7x5 tensor.)"""
import sys
sys.path.insert(0, sys.argv[1])
import torch
from coolchic.training import soap as S
torch.manual_seed(7)
# regenerate the same 7x5 case as dump_soap.py (3rd tensor drawn after w20x20)
_ = torch.randn([20, 20]); _ = [torch.randn([20, 20]) for _ in range(25)]
p0 = torch.randn([7, 5]); grads = [torch.randn([7, 5]) for _ in range(25)]
def run(rotate):
    orig = S.SOAP.get_orthogonal_matrix
    def patched(self, mat):
        out = orig(self, mat)
        if rotate:
            res = []
            for m, q in zip(mat, out):
                if len(m) == 0:
                    res.append(q); continue
                ev = torch.linalg.eigvalsh(m).flip(0)
                null = (ev.abs() < 1e-6 * ev.abs().max()).nonzero().flatten()
                if len(null) > 1:
                    g = torch.Generator().manual_seed(3)
                    r, _ = torch.linalg.qr(torch.randn(len(null), len(null), generator=g))
                    q = q.clone(); q[:, null] = q[:, null] @ r
                res.append(q)
            return res
        return out
    S.SOAP.get_orthogonal_matrix = patched
    p = torch.nn.Parameter(p0.clone())
    opt = S.SOAP([{"params": [p], "lr": 0.01, "betas": (0.95, 0.95), "precondition_frequency": 3, "max_precond_dim": 256, "weight_decay": 0.01, "merge_dims": False, "precondition_1d": False}])
    for g in grads:
        p.grad = g.clone(); opt.step()
    S.SOAP.get_orthogonal_matrix = orig
    return p.detach()
a, b = run(False), run(True)
print("torch vs torch(rotated null space): rel to movement", ((a - b).norm() / (a - p0).norm()).item())
