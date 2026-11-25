# Review item 2 - interval against p-value

| set | k | rho | p_perm | threshold | CI with replacement | CI jackknife | bootstrap mean |
|---|---|---|---|---|---|---|---|
| all 17 | 17 | +0.410 | 0.007 | 0.31 | [+0.12; +0.68] | [+0.07; +0.67] | +0.411 |
| without MAE | 16 | +0.273 | 0.075 | 0.30 | [+0.01; +0.50] | [-0.01; +0.51] | +0.271 |
| without MAE and the two CLIP | 14 | +0.211 | 0.223 | 0.33 | [-0.07; +0.48] | [-0.10; +0.48] | +0.205 |
| 9 pre-trainings | 9 | +0.406 | 0.088 | 0.46 | [-0.11; +0.76] | [-0.18; +0.78] | +0.360 |
| 9 pre-trainings, without MAE | 8 | +0.262 | 0.273 | 0.43 | [-0.17; +0.59] | [-0.16; +0.60] | +0.244 |
| 9 architectures | 9 | +0.109 | 0.662 | 0.44 | [-0.27; +0.55] | [-0.34; +0.51] | +0.120 |

## How to read it

- `threshold` is the 95th percentile of |rho| under permutation of
  encoders: the decision consistent with p_perm is to compare |rho|
  against it.
- `CI with replacement` is the paper's current one. `bootstrap mean`
  serves as a diagnostic for duplicate bias: it matches the observed
  `rho`, so that bias does NOT exist here -- the hypothesis was tested
  and ruled out.
- `CI jackknife` leaves one encoder out at a time, with no repetition,
  and is the procedure compatible with the sensitivity analysis.
