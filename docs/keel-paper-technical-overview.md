# KEEL: Technical Overview & Commentary

**Paper**: *Post-LayerNorm Is Back: Stable, ExpressivE, and Deep* — Chen & Wei (ByteDance Seed), Jan 2026. [arXiv:2601.19895](https://arxiv.org/abs/2601.19895)

---

## 1. Core Claim

Depth scaling is a superior axis for improving Transformer expressivity compared to width or context length, but current architectures (universally Pre-LN since GPT-3/LLaMA) can't exploit it because gradient signal vanishes in deep networks. The root cause isn't LayerNorm placement *per se*—it's the **ResNet-style residual path** interacting with Post-LN. Fix the residual path, and Post-LN becomes trainable at 1000+ layers with better expressivity than Pre-LN at every depth.

## 2. The Problem with Pre-LN (and Why Post-LN Died)

**Pre-LN** (the modern default):

$$x_{l+1} = x_l + F(\text{LN}(x_l))$$

The residual path is a clean identity: $\frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F}{\partial x_l}$. Gradients flow easily → stable training. But the identity path dominates, so deeper layers contribute diminishingly to the update signal. This is the "curse of depth" (Sun et al., 2025)—Pre-LN models have **massive layer redundancy** in middle-to-deep layers. You can literally delete layers from a 80-layer LLaMA-3.3-70B and barely move perplexity.

**Post-LN** (the original Transformer):

$$x_{l+1} = \text{LN}(x_l + F(x_l))$$

Gradients pass through the Jacobian of LayerNorm applied to the *sum* of residual and transform. The cumulative gradient magnitude across $L$ layers scales as $O\left(\frac{1}{2^{L/2}}\right)$—exponential decay. This is why Post-LN diverges at scale. It was abandoned not because it's fundamentally worse, but because no one solved the gradient path problem.

## 3. KEEL Architecture

KEEL replaces the ResNet residual with a Highway-style connection:

$$x_{l+1} = \text{LN}\left(\alpha \cdot x_l + F_l(\text{LN}(x_l))\right)$$

Two modifications over vanilla Post-LN:

1. **Highway-style residual scaling**: $\alpha = L$ (total sub-layer count). The skip connection is amplified relative to the transform branch. The Post-LN at the output normalizes the magnitude, so no explicit variance constraint is needed.

2. **Inner normalization on the transform branch**: An additional LN before $F_l$. This isn't redundant with the outer Post-LN because the learnable affine $\gamma$ parameters make successive LN operations non-collapsible.

**Gradient behavior**: The cumulative gradient product across $L$ layers converges to 1 as $L \to \infty$:

$$\lim_{L \to \infty} \prod_{l=1}^{L} J^*_{\text{LN}_{l,1}}(z_l) = \lim_{L \to \infty} \left[\frac{\alpha}{\sqrt{\alpha^2 + 1}}\right]^L = 1$$

This is a clean theoretical result. No specialized initialization (unlike DeepNorm), no hybrid normalization hacks (unlike Mix-LN / HybridNorm).

**Implementation detail**: The first attention and FFN layers drop the Post-LN and $\alpha$, degenerating to standard Pre-LN blocks for stable signal initialization from the embedding layer.

## 4. How KEEL Differs from DeepNorm

DeepNorm (Wang et al., 2022) also tried to save Post-LN with $x_{l+1} = \text{LN}(\alpha x_l + F(x_l))$ and specialized weight init ($\beta = L^{-0.25}$). The paper argues DeepNorm fails for decoder-only LLMs because:

- DeepNorm bounds *forward output magnitude* (variance control), not *backward gradient flow*. Bounded forward pass ≠ healthy gradients.
- DeepNorm's stabilization is **initialization-dependent**. After trillions of tokens of pretraining, weights drift far from init, and the stability benefit evaporates.
- KEEL bakes stabilization into the **architecture itself** (the $\alpha$ scaling + inner LN), so it holds throughout training, not just at startup.

The paper's ablation path (Section 4.2) is illustrative: naive scaling → learnable input scaling → decoupled scale/variance → final KEEL formulation. Each step addresses a specific failure mode.

## 5. Key Experimental Results

All experiments use decoder-only LLMs, ~3B params, trained on 190B–1T tokens.

### Stability (Max Tolerable Learning Rate)

| Architecture | 64 layers | 512 layers |
|---|---|---|
| Post-LN | 3.0×10⁻⁴ | 2.8×10⁻⁴ |
| DeepNorm | 3.5×10⁻⁴ | 3.5×10⁻⁴ |
| HybridNorm | 4.9×10⁻⁴ | 3.5×10⁻⁴ |
| Mix-LN | 8.6×10⁻⁴ | 3.5×10⁻⁴ |
| Pre-LN | 7.65×10⁻³ | 4.67×10⁻³ |
| **KEEL** | **1.01×10⁻²** | **6.31×10⁻³** |

KEEL tolerates ~1.3× higher LR than Pre-LN and ~30× higher than vanilla Post-LN at 64 layers.

### Depth Scaling (Average Benchmark Score)

| Depth | Pre-LN | KEEL | Δ |
|---|---|---|---|
| 64L | 37.9 | 39.6 | +1.7 |
| 128L | 45.3 | 46.5 | +1.2 |
| 512L | 54.3 | 58.1 | +3.8 |
| 1024L | 57.9 | 60.9 | +3.0 |

The gap *widens* with depth, especially on reasoning tasks. GSM-8K at 1024L: 58.6 (KEEL) vs 49.8 (Pre-LN), a +8.8 delta.

### Deep vs. Wide (Fixed 3B Params)

| Config | Avg Score |
|---|---|
| Deep Pre-LN (512L, d=1024) | 52.3 |
| Wide Pre-LN (128L, d=2048) | 52.2 |
| **Deep KEEL** (512L, d=1024) | **55.5** |

KEEL makes deep-and-narrow actually beat wide-and-shallow for the first time at this scale. The deep Pre-LN and wide Pre-LN are essentially tied despite the deep model having lower training loss—another finding that training loss and downstream performance decorrelate in deep LLMs.

### Layer Redundancy

Appendix A is arguably the most interesting diagnostic. They remove individual layers and measure ΔPPL. In Pre-LN, middle layers are extremely redundant (removing them barely hurts). KEEL shows significantly less redundancy across all layers, especially in the shallow region—indicating greater effective depth utilization. The same redundancy pattern appears in Qwen2.5-72B and LLaMA-3.3-70B (both Pre-LN).

## 6. Interesting Side Observation: Depth-Wise Test-Time Training

Section 4.3 draws a structural isomorphism between depth-wise residual propagation and sequence-wise recurrence in TTT/Titans-style models. The residual update $x_{l+1} = x_l + G(x_l; W)W_o^\top$ can be viewed as a gradient step on a depth-wise objective. KEEL's $\alpha$ and inner LN then function as optimization stabilizers for this "depth-wise training" process, analogous to gating and state normalization in recurrent TTT. This suggests a duality: techniques for long-context sequence recurrence may transfer to infinite-depth propagation and vice versa.

## 7. Limitations & Gaps

- **Decoder-only only.** No encoder or encoder-decoder experiments. The theoretical analysis generalizes, but empirical validation is absent.
- **Width scaling not addressed.** The paper explicitly notes that as model width increases, Post-LN may need larger $\alpha$ or additional stabilization. This is future work.
- **Low-data regimes.** KEEL's advantages are less pronounced with < ~10B tokens. It needs substantial data to overtake Pre-LN.
- **$\alpha = L$ is a fixed heuristic.** They mention it can be tuned for smaller models, but there's no adaptive or learned gating mechanism. A true Highway gate (learned $T(x)$) might be better but would add parameters and complexity.
- **No FLOP-matched comparisons.** The extra LN per layer in KEEL has compute cost. The paper doesn't report wall-clock time or FLOP budgets. At 1024 layers with two LN ops per sub-layer, this isn't negligible.

---

## 8. Commentary: The Pre-LN Herd Mentality & What This Means for Encoders

The annotated-mpnet project uses Post-LN by default (per the original MPNet recipe), and empirical experience running many training experiments + reading the literature converges on a specific conclusion that this paper now backs with theory:

**The entire field herd-mentality'd into using Pre-LN everywhere because "we need stability!!!!" after GPT-3, when really, unless you're training a huge model (>64 layers) or want to use a very high LR (>>1e-4), Post-LN is fine—and probably better.**

The evidence trail:

1. **The original problem was real but narrow.** Xiong et al. (2020) showed Post-LN's gradient instability. GPT-3 (Brown et al., 2020) adopted Pre-LN. LLaMA followed. Everyone followed LLaMA. The instability is *real* for 96+ layer decoder-only models trained at high learning rates on trillions of tokens. It is **not** a problem for 12–28 layer encoders trained at standard LRs (~1e-4 to 5e-4) on modest token budgets.

2. **The KEEL stability data proves the point.** Look at Table 1: even vanilla Post-LN tolerates LR = 3×10⁻⁴ at 64 layers. For a typical encoder (12–28 layers, peak LR ~2e-4 to 5e-4), vanilla Post-LN is *well within its stable regime*. You don't need Pre-LN. You don't need KEEL. You just need... Post-LN, the original recipe.

3. **Pre-LN's curse of depth is real and it hurts encoders too.** The layer redundancy analysis (Appendix A) shows that Pre-LN models waste depth. For a 28-layer encoder like NeoBERT, this means the middle layers may contribute very little. Post-LN maintains stronger inter-layer coupling and gradient signal to deeper layers. For MTEB/retrieval tasks where representational richness matters, this is not academic—it directly affects embedding quality.

4. **MPNet's two-stream attention already needs Post-LN's coupling.** The masked and permuted pretraining objective in annotated-mpnet is more complex than standard MLM. The query stream and content stream need to develop different but coordinated representations. Post-LN's stronger backward coupling between layers likely helps these two streams co-adapt more effectively than Pre-LN's weakly-coupled identity path.

5. **Recent papers keep rediscovering this.** Mix-LN (Li et al., Dec 2024) shows that adding Post-LN to even just the lower layers improves gradient norms and downstream performance. The "Curse of Depth" paper (Sun et al., Feb 2025) directly measures that Pre-LN deeper layers are underutilized. HybridNorm (Zhuo et al., Mar 2025) tries to get the best of both worlds. GPAS (Chen et al., Jun 2025) tries to fix Pre-LN's activation variance issues. All of these are patches on a decision that shouldn't have been made universally in the first place.

**Bottom line for the NeoBERT and annotated-mpnet projects**: At the encoder scale we operate at (12–28 layers, ≤4096 context, standard LRs), Post-LN is the correct default. If we want to push to deeper encoders (>64 layers for more expressive embeddings), KEEL's Highway-style modification is a clean, low-complexity option that doesn't require changing the optimizer or init scheme. The field's blanket adoption of Pre-LN was a case of solving the right problem at the wrong scope and then cargo-culting the solution everywhere.

For NeoBERT specifically (28 layers, Pre-RMSNorm), this is worth experimenting with: swap to Post-RMSNorm with the KEEL residual scaling ($\alpha = 56$ for 28×2 sub-layers) and compare GLUE/MTEB performance. The prediction: better depth utilization, richer representations, comparable or better downstream scores, with no stability issues at that depth.
