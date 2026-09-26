# CLHR: Closed-Loop Hard Routing

Code for **"Train Soft, Roll Out Hard: Near-Lossless Deployment of Learned
Sparse Attention"** (ICLR 2027 submission, anonymous review).

## Summary

Learned sparse-attention models are trained with a differentiable *soft*
router but deployed with *hard* (discrete, top-$k$) routing decisions. A 31M
model with training-native perplexity 49 reaches perplexity 291 when deployed
with the hard routes it was trained to produce. Because deployed hard routes
are generated layer by layer, this deployment-state distribution is
**closed-loop**: each layer's routing decision depends on the hard decisions
made by every earlier layer, and soft-only training never exposes the model
to that distribution.

We introduce **Closed-Loop Hard Routing (CLHR)**: keep a differentiable soft
path for router learning, but train the backbone on the hard trajectory
induced by its own routing decisions. In an exact reproduction of a public
31M-parameter protocol, CLHR reduces closed-loop deployment excess
($G_{\mathrm{CL}}$) from 1.782 to 0.0035 nats across three seeds — a
hard/native perplexity ratio of $5.94\times \rightarrow 1.0035\times$. With
the method frozen, the same reduction transfers to 300M models on
WikiText-103 ($0.796\to0.049$) and FineWeb-Edu ($0.513\to0.027$), to Native
Sparse Attention (NSA) and MoBA routing, and to MoE soft-to-hard routing. A
matched causal ladder (hard exposure alone, open-loop replay, shuffled
closed-loop routing) shows none of these ablations suffice — the gap only
closes when the backbone is supervised on the useful, self-induced hard
states it will actually see at deployment.

## Repository structure

```
src/
  models/                          shared transformer + gated-attention modules
  data_loading.py                  tokenized-corpus loading (WikiText-103 / FineWeb-Edu)
  flex_block_mask.py               FlexAttention block-mask construction for hard routing

  sparse_attention_rca.py          core RCA/CLHR trainer: standard, hard-exposure,
                                    open-loop-replay, shuffled-closed-loop, and CLHR
                                    conditions (see CONDITIONS list) at 31M scale
  sparse_attention_300m.py         300M-scale DDP training entry point
  sparse_attention_fineweb.py      FineWeb-Edu corpus transfer
  sparse_attention_closed_loop_eval.py,
  sparse_attention_posthoc_kl.py,
  sparse_attention_posthoc_kl_maskagnostic.py,
  sparse_attention_seer_post.py    post-hoc / closed-loop deployment evaluation variants

  sparse_attention_nsa.py          Native Sparse Attention routing family (standard / CLHR)
  sparse_attention_moba.py         MoBA block-routing family (standard / dense_switch / CLHR)
  sparse_attention_block_router.py,
  sparse_attention_dsa.py          additional routing-family baselines
  sparse_attention_ssa_style.py    SSA-style baseline

  qwen3_sparse_rca.py,
  qwen3_multilayer_sparse_rca.py,
  qwen3_gate_utility.py            CLHR applied to a pretrained Qwen3 backbone
  moe_soft_to_hard.py              soft-to-hard routing for MoE

scripts/
  run_lm_eval.py, lm_eval_adapter.py   downstream evaluation (WikiText-2 word
                                        perplexity, LAMBADA) via lm-eval-harness
```

## Setup

```bash
uv sync
```

Requires Python >= 3.10 and a CUDA GPU for training. See `pyproject.toml` for
pinned dependencies (PyTorch, `transformers`, `lm-eval`).

## Usage

Train the 31M CLHR condition (main result):

```bash
uv run python src/sparse_attention_rca.py \
  --condition coherent_closedloop_hard --seed 42 \
  --data-dir /path/to/wikitext103_cache \
  --checkpoint-dir ckpts_rca_clhr_s42 \
  --output results/rca_clhr_s42.json \
  --lambda-rca 0.3
```

Train NSA with CLHR at 31M scale:

```bash
uv run python src/sparse_attention_nsa.py \
  --condition clhr --seed 42 --model-size 31m --corpus wikitext-103 \
  --data-dir /path/to/wikitext103_cache \
  --checkpoint-dir ckpts_nsa_clhr_s42 \
  --output results/nsa_clhr_s42.json \
  --lambda-rca 1.0 --max-steps 50000 --micro-batch 16
```

Train MoBA with CLHR:

```bash
uv run python src/sparse_attention_moba.py \
  --condition clhr --seed 42 \
  --data-dir /path/to/wikitext103_cache \
  --output results/moba_clhr_s42.json \
  --lambda-rca 1.0 --top-k-blocks 2
```

Downstream evaluation (WikiText-2 word perplexity, LAMBADA):

```bash
uv run python scripts/run_lm_eval.py --checkpoint-dir <ckpt_dir> --output <out.json>
```

## Notes on scope

This release includes the code that produced the results reported in the
paper. Exploratory scripts from the development process that did not lead to
reported results are not included. Two evaluation variants
(`sparse_attention_exact_protocol.py`, `sparse_attention_ssa_real.py`) that
depend on an internal loss-wrapper module are also omitted pending a
follow-up cleanup; the numbers they produced are reported in the paper and
are not affected.

## License

MIT — see [LICENSE](LICENSE).
