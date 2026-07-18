# GPU support

FlashDeBERTa routing and kernel tuning are keyed by CUDA compute capability. The shipped measurements come from one GPU class, `sm_120`; every other supported CUDA GPU runs Flash attention with conservative defaults and user knobs for measured routes.

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

On `sm_120`, the shipped table enables measured dense `docblock_bias` and small-batch `local_bias` policies plus tuned rows for promoted hot paths. Kernel families without a matching row still use the deterministic fallbacks documented in [FlashDeBERTa attention](flash-attention.md#tuning-table).

On any other supported CUDA GPU, `model.hf.attention_impl=flash` still runs Flash attention: a missing capability row does not trigger eager attention. Per-call input contracts such as attention-output requests or unsupported mask layouts can still select eager attention. What changes off `sm_120`:

- **Padded batches use the capability-independent [fixed/varlen policy](flash-attention.md#route-families).**
- **Packed doc-block batches default to the ragged `docblock` route** instead of dense `docblock_bias`. The dense route's measured speed advantage depends on `sm_120`-gated backward specializations; see [route families](flash-attention.md#route-families) for the memory tradeoff.
- **The `local_bias` route stays off** (its only shipped policy row is `sm_120`-scoped); plain
  dense batches use the `fixed` route.
- **Unmatched kernel launch configs use repo-owned deterministic fallbacks**, not upstream FlashDeBERTa heuristics. Capability-specific tuning is still recommended because a conservative tile may run slowly on another architecture.

When the first Flash batch is prepared, a GPU without measured rows logs a one-time warning naming the detected capability and pointing here.

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
defaults, so promoting one route for your GPU takes one row per bucket. Buckets like
`4096_plus` are open-ended, so bound dense rows with `max_seq_len` at the longest length you
actually measured - the dense route's saved `(B,H,S,S)` bias grows quadratically, and a row
whose bounds exclude the batch length resolves to the ragged fallback instead:

```json
{
  "route_policies": {
    "docblock": [
      {"seq_bucket": "1024_exact", "choice": "docblock_bias", "compute_capability": "sm_90"},
      {"seq_bucket": "4096_plus", "choice": "docblock_bias", "compute_capability": "sm_90",
       "max_seq_len": 4096}
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

Use `max_seq_len` and `max_batch_size` bounds that reflect the target GPU's VRAM rather than copying the shipped 32 GiB limits. A same-capability card with more VRAM can raise those bounds with a same-key override row, which outranks the shipped row. Batch preparation warns when a bound keeps a measured capability on the ragged route.

Validate route changes and kernel tiles with the [retuning workflow](flash-attention.md#retuning). Triton launch configurations depend on shared memory, scheduling, and SM count; applying one GPU's tiles globally can fail at launch or regress throughput on another architecture.
