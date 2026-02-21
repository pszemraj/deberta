# KEEL: Technical Notes

This document summarizes the KEEL residual topology used optionally by the `rope` backbone.

Paper: *Post-LayerNorm Is Back: Stable, ExpressivE, and Deep* (Chen & Wei, 2026) - [arXiv:2601.19895](https://arxiv.org/abs/2601.19895)

For project-level model configuration, see [`docs/model.md`](model.md).

## What KEEL Changes

Compared with standard Post-LN residual blocks, KEEL uses a highway-style scaled skip path and an inner normalization before the transform branch.

Reference form (paper):

`x_{l+1} = LN(alpha * x_l + F(LN(x_l)))`

Main intent:

- keep Post-LN depth expressivity
- stabilize gradient flow at larger depth

## Repo Mapping

In this repo KEEL is enabled by:

- `model.norm_arch: keel`

Related KEEL configuration options (`model.keel_alpha_init`, `model.keel_alpha_learnable`) are documented in [`docs/model.md`](model.md).

Default path remains standard Post-norm (`model.norm_arch: post`).

## When to Use

For typical encoder depths, `post` is the default and usually sufficient.

Try KEEL when you are increasing depth aggressively and want additional stability margin while keeping a Post-LN style architecture.

## Scope and Limits

KEEL support here applies to the repo's `rope` backbone implementation.

It does not apply to `backbone_type=hf_deberta_v2` mode.
