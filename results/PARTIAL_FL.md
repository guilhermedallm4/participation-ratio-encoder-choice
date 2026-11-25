# Partial correlation: conditional inference by Freedman-Lane

The final review is right. The earlier null permuted ratio and width
**together** against the probe, which also destroys the width-probe
association. The correct conditional null preserves that association
and shuffles only the part of the probe that width does not explain.
That is what Freedman-Lane does, and it is what follows.

Permutation at the encoder level, the same one across all four domains, 20000 draws.

| set | k | partial rho | **p Freedman-Lane** | p of the old null | b1 | p F-L of b1 |
|---|---|---|---|---|---|---|
| all 17 | 17 | **+0.392** | **0.010** | 0.012 | +0.392 | 0.009 |
| without MAE | 16 | **+0.243** | **0.113** | 0.113 | +0.244 | 0.108 |
| without MAE and the two CLIP | 14 | **+0.173** | **0.320** | 0.327 | +0.173 | 0.310 |
| 9 architectures | 9 | **+0.049** | **0.847** | 0.848 | +0.047 | 0.842 |
| 9 pre-trainings | 9 | - | - | - | - | width constant (768 channels): nothing to control for |

The point estimate of the partial does not move: it never depended on
the null. What moves is the $p$, and the $p$ is what the review
questioned.

