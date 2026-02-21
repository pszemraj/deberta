# KEEL: Technical Notes

This file captures paper-specific KEEL notation used by this repo.

Paper: *Post-LayerNorm Is Back: Stable, ExpressivE, and Deep* (Chen & Wei, 2026) - [arXiv:2601.19895](https://arxiv.org/abs/2601.19895)

For project-level configuration and decision guidance (`post` vs `keel`), use [`docs/model.md`](model.md).

## Paper Form and Notation

KEEL residual form in paper notation:

`x_{l+1} = LN(alpha * x_l + F(LN(x_l)))`

Notation clarification used in this repo:

- `L` denotes total residual sublayers (attention + FFN), not transformer block count.
- default `alpha` follows the paper's `alpha = L`.
- for an encoder with `N` transformer blocks, `L = 2N`, so default `alpha = 2 * num_hidden_layers`.

## Repo Mapping

KEEL path is enabled with:

- `model.norm_arch: keel`

Relevant config knobs:

- `model.keel_alpha_init`
- `model.keel_alpha_learnable`

Scope: this applies only to the repo's `rope` backbone, not `backbone_type=hf_deberta_v2`.
