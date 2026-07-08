# GPU support

FlashDeBERTa routing and kernel tuning are keyed by CUDA compute capability. The shipped
measurements come from one GPU class, `sm_120`; every other CUDA GPU still runs flash end to end
with conservative defaults and user knobs to opt into the faster routes. This page covers what to
expect on your hardware and how to tune it.

Check your capability:

```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
```

`(12, 0)` means `sm_120` - the key format used throughout the tuning table.

## Capability table

| Key | Architecture | Example GPUs |
|---|---|---|
| `sm_120` | Blackwell (consumer/workstation) | GeForce RTX 50-series, RTX PRO Blackwell |
| `sm_100` | Blackwell (datacenter) | B100, B200, GB200 |
| `sm_90` | Hopper | H100, H200, GH200 |
| `sm_89` | Ada Lovelace | GeForce RTX 40-series, L4, L40/L40S, RTX 6000 Ada |
| `sm_86` / `sm_80` | Ampere | RTX 30-series, A40, A6000 / A100 |

Practical floor: the flash path assumes bf16-capable hardware, so Ampere (`sm_80`) or newer.
Pre-Ampere GPUs are untested and unsupported for `attention_impl=flash`; use `eager` there.

## What you get out of the box

On `sm_120`, everything is measured: dense `docblock_bias` for packed doc-block batches, the
small-batch `local_bias` route, and tuned Triton tiles for every kernel family.

On any other CUDA GPU, `model.hf.attention_impl=flash` still runs flash attention - **there is no
eager fallback based on GPU type**. Eager fallbacks happen only for unsupported regimes (p2p
attention terms, dropout enabled, unexpected mask shapes), never because your capability key is
missing from the table. What changes off `sm_120`:

- **Padded routing is identical**: `fixed` below `2048`, `varlen` at `2048+`.
- **Packed doc-block batches default to the ragged `docblock` route** instead of dense
  `docblock_bias`. The dense route's speed advantage came from `sm_120`-gated backward
  specializations; without them it degrades to a PyTorch positional-gradient reduction and saves
  a dense `(B,H,S,S)` bias for backward. Ragged is the memory-light, specialization-free default.
- **The `local_bias` route stays off** (its only shipped policy row is `sm_120`-scoped); plain
  dense batches use the `fixed` route.
- **Kernel launch configs come from upstream FlashDeBERTa heuristics** (the repo-local dense-bias
  builder uses a conservative built-in tile), because the shipped tile entries are all
  `sm_120`-scoped and Triton tiles do not transfer across GPU generations - a tile sized for one
  chip's shared memory can fail to launch or run slowly on another.

At startup, the first flash batch on a GPU without measured rows logs a one-time warning naming
the detected capability and pointing here.

## Opting into the dense routes on other hardware

The measured routes work on any supported GPU; they are just not the default until measured. Two
ways to enable them:

Per-run knobs (no table needed):

```yaml
model:
  hf:
    attention_impl: flash
    flash:
      # Force dense doc-block at your packed length:
      docblock_bias_seq_len: 1024
      # Enable the small-batch local-bias route (both knobs required off-table):
      local_bias_seq_len: 1024
      local_bias_max_batch_size: 4
```

Or a capability-scoped override table, selected with `model.hf.flash.kernel_overrides_path`.
Override rows append to the shipped table, and an exact-capability row outranks the wildcard
defaults, so promoting one route for your GPU takes one row per bucket:

```json
{
  "route_policies": {
    "docblock": [
      {"seq_bucket": "1024_exact", "choice": "docblock_bias", "compute_capability": "sm_90"},
      {"seq_bucket": "4096_plus", "choice": "docblock_bias", "compute_capability": "sm_90"}
    ]
  },
  "kernels": [
    {
      "compute_capability": "sm_90",
      "route": "dense_bias",
      "kind": "fwd",
      "seq_bucket": "1024_exact",
      "head_dim": "*",
      "block_m": 64,
      "block_n": 64,
      "num_stages": 2,
      "num_warps": 4
    }
  ]
}
```

Before promoting a route on new hardware:

1. Run `tools/flashdeberta_parity_test.py` on the target GPU - correctness is
   hardware-independent in design, but verify it.
2. Compare arms with `tools/flashdeberta_rtd_profile.py --mode eager` versus `--mode flash` on
   your real config; keep whichever route wins.
3. Mind memory: dense `docblock_bias` saves the `(B,H,S,S)` bias for backward. On cards with
   less VRAM than the 32 GiB benchmark GPU, try the ragged default first at `4096`.

Kernel tile tuning (`tools/flashdeberta_varlen_tune.py`, `tools/flashdeberta_bias_tune.py`) is
optional on top of route promotion; the workflow is described in
[Advanced / FlashDeBERTa attention](flash-attention.md#retuning-for-new-hardware-or-kernel-changes).
Durable results are welcome as additional capability-scoped rows in the shipped table.

## Why kernel entries are capability-scoped

Triton launch configs (`BLOCK_M`, `BLOCK_N`, stages, warps) are tuned against one chip's shared
memory, scheduler, and SM count. Applying `sm_120` tiles globally could crash (out-of-resource at
launch) or crawl on smaller GPUs. Scoping them by capability means unknown hardware gets each
kernel's generic heuristics instead - slower than tuned, but always runnable. The same logic
applies to the `bias_docblock_specialized` backward kernels: they engage only where a matching
capability row exists.
