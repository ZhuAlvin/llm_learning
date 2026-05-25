# NumPy from scratch: Low-Rank Decomposition (LoRA math intuition)
import numpy as np
np.random.seed(42)
d, r = 64, 4

W = np.random.randn(d, d)
Delta_W = np.random.randn(d, d) * 0.01

U, S, Vt = np.linalg.svd(Delta_W, full_matrices=False)
Delta_W_lr = (U[:, :r] * S[:r]) @ Vt[:r, :]
err = np.linalg.norm(Delta_W - Delta_W_lr) / np.linalg.norm(Delta_W)

print(f"Original rank: {np.linalg.matrix_rank(Delta_W)}")
print(f"Low-rank (r={r}) reconstruction error: {err:.6f}")
print(f"Top {r} singular values explain: {sum(S[:r]**2)/sum(S**2):.2%}")
print(f"LoRA params: 2*d*r = {2*d*r} vs d*d = {d*d} = {2*r/d:.1%}")
print("Core insight: Fine-tuning delta_W is low-rank. LoRA approximates it with tiny params.")
