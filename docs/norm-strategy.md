# Normalization Strategy (`post` Default, `keel` Upgrade Path)

See also: [model config](model.md), [objective](objective.md), [runtime](fsdp2.md).

## Implemented Norm Architectures

`model.norm_arch` supports:

- `post` (default)
- `keel`

Both apply only to the repo's `rope` backbone.

### `post`
Per sub-layer (attention and FFN), the block uses Post-style residual normalization:
`x_{l+1} = RMSNorm(x_l + F(x_l))`

### `keel`
KEEL-style update follows the paper form (with RMSNorm in this codebase):
`x_{l+1} = RMSNorm(alpha * x_l + F(RMSNorm(x_l)))`

Repo parameterization:

- `L` means total residual sublayers (attention + FFN), so `L = 2 * num_hidden_layers`
- paper-faithful default: `keel_alpha_init` is `1/sqrt(L)` (that is, `1/sqrt(2 * num_hidden_layers)`) when unset, with
  `keel_alpha_learnable=false` so alpha is fixed
- implementation detail: attention and FFN each have their own alpha slot (`alpha1`, `alpha2`) initialized
  to the same value; this is equivalent to paper behavior when `keel_alpha_learnable=false`
- optional extension: `keel_alpha_learnable=true` makes each sub-layer alpha trainable

Primary paper: *Post-LayerNorm Is Back: Stable, ExpressivE, and Deep* (Chen & Wei, 2026), [arXiv:2601.19895](https://arxiv.org/abs/2601.19895).

## Why `post` Is the Default Here

- Shipped training regimes in this repo are moderate-depth encoders (roughly 12-28 layers), where Post-style training is typically stable at the configured learning-rate ranges.
- We prioritize depth utilization and stronger inter-layer coupling over adopting Pre-LN by convention.
- The repo's objective is encoder pretraining quality, not matching decoder-only LLM architecture defaults.
- Pre-LN became the field default primarily to stabilize very deep, high-LR decoder training. At encoder depths used by this repo, switching to Pre-LN by default would trade away Post-style coupling without solving an active baseline stability problem.

## Why KEEL Is Opt-In (Not Default)

KEEL is kept as an upgrade path for deeper or more aggressive runs, not as a mandatory replacement for default training.

- Strength: it targets Post-style depth stability directly via residual scaling + inner normalization.
- Cost: adds architectural complexity and extra normalization work per sub-layer.
- Evidence scope: strongest published gains are decoder-only; encoder transfer is plausible but still an empirical question for each setup.

Repo policy: `post` for standard runs, `keel` when pushing depth/optimization, then promote only after empirical validation on the target regime.

## Practical Selection Guide

- Start with `model.norm_arch=post` for baseline and parity experiments.
- Move to `model.norm_arch=keel` when increasing depth or when optimization becomes fragile.
- If enabling KEEL, keep `keel_alpha_init` at default first; tune only if a specific depth/width regime needs it.
