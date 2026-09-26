"""Canonical, direct BlockMask construction for torch.nn.attention.flex_attention.

Single source of truth for `build_block_mask_direct` / `_ordered_from_boolean`
/ `make_mask_mod`. Originally developed and validated in
scripts/bench_flex_longcontext.py (see tests/test_flex_block_mask.py, 35
tests, all passing against the logic below) and extracted here so that
src/ code (e.g. src/sparse_attention_hierarchical.py) can import it without
importing from scripts/. scripts/bench_flex_longcontext.py now imports these
three names from this module instead of defining its own copies -- there is
exactly one implementation.

Construction route (unchanged from the original): this deliberately does
NOT call `BlockMask.from_kv_blocks` (buggy default `seq_lengths`: uses
`kv_indices.shape[-1] * BLOCK_SIZE[1]`, the padded per-row selection count,
not the true KV block count -- see `build_block_mask_direct`'s docstring
below for the full analysis) and does NOT call `create_block_mask` (which
materializes a dense `(1, H, T, T)` bool internally before block
compression and OOMs at long context). It calls `BlockMask.__init__`
directly with explicit `seq_lengths`, computing both kv-side and q-side
index structures from a boolean block matrix and its transpose via
`_ordered_from_boolean`.

Batch generalization (2026-09-20): the original implementation accepted only
a rank-3 `selection` of shape `(n_heads, num_q_blocks, num_kv_blocks)` --
adequate for a synthetic, batch-independent probe. A real router's block
selection is input-dependent, so the production shape is rank-4:
`(B, n_heads, num_q_blocks, num_kv_blocks)`. Both `_ordered_from_boolean`
and `build_block_mask_direct` below accept either rank:
  - rank-3 input: treated as before -- a single selection broadcast across
    the batch via BlockMask's own leading batch dimension of size 1. Output
    `num_blocks`/`indices` tensors carry a batch dim of size 1, exactly as
    the original code produced (bit-identical construction, not merely
    equivalent).
  - rank-4 input: the leading dim is a REAL per-example batch. Output
    tensors carry that real batch size; each batch element's BlockMask
    structure is derived independently from its own boolean block matrix --
    there is no broadcasting of one example's selection across the batch.
"""
from __future__ import annotations

import math

import torch

from torch.nn.attention.flex_attention import BlockMask


def make_mask_mod(selection, block_size, local_window):
    """Build a flex_attention-compatible mask_mod(b, h, q_idx, kv_idx) -> bool
    implementing: causal AND (local window OR selected block). `selection`
    is a bool tensor (n_heads, num_query_blocks, num_kv_blocks) as produced
    by generate_block_selection. Batch (`b`) is ignored -- selection does
    not vary across batch in this synthetic probe.
    """
    sel = selection

    def mask_mod(b, h, q_idx, kv_idx):
        causal = kv_idx <= q_idx
        local = (q_idx - kv_idx) < local_window
        qb = q_idx // block_size
        kvb = kv_idx // block_size
        selected = sel[h, qb, kvb]
        return causal & (local | selected)

    return mask_mod


def _ordered_from_boolean(admit, device, width=None):
    """Convert a boolean "admits" matrix into the (num_blocks, indices)
    ordered representation flex_attention's BlockMask expects: for each
    (batch, head, row), how many columns are admitted and which column
    indices they are (padded/undefined past that count). Pure tensor ops
    over the block-level (num_blocks x num_blocks) matrix -- never anything
    token-granular. Used for BOTH the kv-side and the q-side (the q-side
    call simply passes admit.transpose(-2, -1)), so there is exactly one
    implementation of "boolean block matrix -> ordered form", and it does
    not go through torch's own _ordered_to_dense/_transpose_ordered at all
    -- see build_block_mask_direct's docstring for why that path is buggy
    (uses the per-row max cardinality as the dense column count instead of
    the true number of columns).

    Accepts either:
      - rank-3 `admit` of shape (n_heads, n_rows, n_cols): treated as a
        single selection broadcast across the batch. Internally promoted to
        a batch dim of size 1 (matching BlockMask's own broadcast
        convention), so the returned tensors carry a leading batch dim of 1
        -- exactly what the original (pre-batching) implementation produced.
      - rank-4 `admit` of shape (batch, n_heads, n_rows, n_cols): a real
        per-example batch. The returned tensors carry that real batch size;
        each batch element's ordering is computed independently (no
        broadcasting of one example onto another).

    `width` (2026-09-20, dynamo-recompile-hazard fix): the padded column
    count of the returned `indices` tensor. THIS is the parameter that
    eliminates the recompilation hazard -- OBSERVED on CUDA (A10G, torch
    2.11.0+cu128): flex_attention recompiled every time this width changed
    between calls (guard failure: "tensor 'block_mask.q_indices' size
    mismatch at index 3"), and past 8 such recompiles (the default
    `torch._dynamo.config.recompile_limit`) dynamo silently falls back to an
    unfused implementation that materializes the full score matrix -- a
    ~180x slowdown (107.8ms vs 0.595ms, 15.76GB vs 1.13GB at T=8192,
    separately measured in this project) that raises no error and still
    returns numerically correct results, so a long run that hit it would
    look like it was working.

    If `width` is None (the pre-fix default, still used by anything that
    has not been updated to pass a geometry-derived width), the old,
    data-dependent behavior is preserved EXACTLY: width = max(observed
    per-row cardinality, 1). This is the actual bug -- `admit` varies with
    the router's input-dependent selection, so this width varies batch to
    batch. `build_block_mask_direct` now always passes an explicit,
    geometry-derived `width`; the None branch remains only so this function
    is not a silent behavior change for any other caller.

    Whatever `width` resolves to, it is REQUIRED to be at least the true
    per-row cardinality for every row (asserted below) -- the whole point
    of a fixed width is a shape that never changes, not a shape that
    sometimes truncates real admitted pairs. It is clamped to `n_cols`
    (indices are column positions in [0, n_cols), so a wider array than
    that could never contain anything real anyway).

    Padding convention (unchanged, now just extended further out): the
    stable descending sort below always pushes every admitted (True) column
    to the front of its row, so any column position >= that row's true
    count -- whether it is inside the OLD max_cols or newly added by a
    larger `width` -- holds an unadmitted column index. Consumers must
    never read past `num_blocks[row]`; that count, not the array width, is
    what determines which entries are live. A larger `width` only adds more
    such inert positions; it cannot make any existing position's meaning
    change.
    """
    if admit.dim() == 3:
        admit = admit.unsqueeze(0)
    elif admit.dim() != 4:
        raise ValueError(
            f"admit must be rank 3 (n_heads, n_rows, n_cols) or rank 4 "
            f"(batch, n_heads, n_rows, n_cols); got shape {tuple(admit.shape)}"
        )

    batch, n_heads, n_rows, n_cols = admit.shape
    num_blocks = admit.sum(dim=-1).to(torch.int32)  # (batch, n_heads, n_rows)
    if width is None:
        pad_width = max(int(num_blocks.max().item()), 1) if num_blocks.numel() else 1
    else:
        pad_width = max(min(int(width), n_cols), 1)
        # Correctness gate, not an optimization: a fixed width is only sound
        # if it is actually an upper bound on every row's true cardinality.
        # If the caller's geometry-derived formula is ever wrong (e.g. a
        # mismatched top_k_blocks argument), truncating here would silently
        # drop real (query, key) pairs -- exactly the kind of bug a "PASS"
        # from the old raw-counter-dump verification script would not catch.
        # Fail loudly instead.
        if num_blocks.numel() and int(num_blocks.max().item()) > pad_width:
            raise ValueError(
                f"_ordered_from_boolean: fixed width {pad_width} is smaller "
                f"than the true max per-row cardinality "
                f"{int(num_blocks.max().item())}; the geometry-derived width "
                f"passed by the caller is unsound for this selection"
            )

    col_idx = torch.arange(n_cols, device=device).view(1, 1, 1, n_cols).expand(
        batch, n_heads, n_rows, n_cols
    )
    # Stable descending sort on the boolean-as-int admit matrix pushes every
    # admitted (True) column index to the front of its row; positions at/
    # past num_blocks[row] are unused padding (from_kv_blocks' own contract
    # for undefined trailing entries).
    order = torch.argsort(admit.to(torch.int32), dim=-1, descending=True, stable=True)
    sorted_cols = torch.gather(col_idx, -1, order)
    indices = sorted_cols[..., :pad_width].to(torch.int32).contiguous()

    num_blocks = num_blocks.contiguous()  # (batch, n_heads, n_rows)
    indices = indices.contiguous()  # (batch, n_heads, n_rows, pad_width)
    return num_blocks, indices


def build_causal_block_mask_direct(seq_len, block_size, device):
    """Direct BlockMask assembly for a PLAIN causal mask -- no routing
    selection, no local-window carve-out at the BlockMask level.

    This is the soft-training-path counterpart to `build_block_mask_direct`
    above: `HierarchicalSparseAttention`'s soft path applies its local
    window AND its routing-derived soft bias entirely inside `score_mod`
    (see `sparse_attention_hierarchical.py`'s soft-path branch), so the
    BlockMask it needs is nothing more than "kv_idx <= q_idx" -- the
    pre-existing `_get_causal_block_mask_flex` built exactly this via
    `create_block_mask(causal_local_mask, None, None, T, T, ...)`, which
    OOMs at long context (see module docstring) because `create_block_mask`
    materializes a dense (1, H, T, T) bool tensor before block-compressing
    it. This function reaches the identical mask_mod / admitted-pair set
    without ever doing that: it builds the (num_blocks, num_blocks) boolean
    lower-triangular block matrix directly (trivial size: e.g. 256x256 at
    T=32768, block_size=128) and feeds it through the same
    `_ordered_from_boolean` helper `build_block_mask_direct` uses, then
    calls `BlockMask.__init__` directly with explicit `seq_lengths` -- same
    construction route, same reasons (see `build_block_mask_direct`'s
    docstring for why `create_block_mask` and `BlockMask.from_kv_blocks`
    are both avoided).

    `block_size` here is the FLEX KERNEL's own block-sparsity granularity
    (the caller passes `_FLEX_KERNEL_BLOCK_SIZE`, e.g. 128) -- unrelated to
    the router's routing block_size (e.g. 32/64), exactly as in the
    pre-existing code's own `BLOCK_SIZE=_FLEX_KERNEL_BLOCK_SIZE` argument to
    `create_block_mask`. The mask_mod returned is pure per-token causality
    (`q_idx >= kv_idx`), not quantized to any block boundary -- the
    kv/q_indices computed here only decide which (query-block, key-block)
    tiles the compiled kernel bothers to visit at all; mask_mod is still
    evaluated per-token within every visited tile, so the result is
    elementwise identical to the plain causal mask regardless of
    `block_size`, provided (as here) no visited tile is a false negative
    (every block with kb <= qb is included, so every true causal pair falls
    inside a visited tile).

    The returned BlockMask has no batch/head dependence (kv_num_blocks /
    kv_indices carry a leading (1, 1) shape), matching the pre-existing
    `create_block_mask(..., B=None, H=None, ...)` call's broadcast
    convention exactly -- it broadcasts across whatever batch/head shape
    q/k/v actually have at call time.
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    qb_idx = torch.arange(num_blocks, device=device).view(num_blocks, 1)
    kb_idx = torch.arange(num_blocks, device=device).view(1, num_blocks)
    causal_block = (kb_idx <= qb_idx)  # (num_blocks, num_blocks) bool

    # (1, 1, num_blocks, num_blocks): batch=1, heads=1 -- broadcast, per the
    # docstring above.
    admit = causal_block.unsqueeze(0).unsqueeze(0)
    kv_num_blocks, kv_indices = _ordered_from_boolean(admit, device)
    q_num_blocks, q_indices = _ordered_from_boolean(
        admit.transpose(-2, -1).contiguous(), device,
    )

    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return BlockMask(
        seq_lengths=(seq_len, seq_len),
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=None,
        full_kv_indices=None,
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        full_q_num_blocks=None,
        full_q_indices=None,
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=mask_mod,
    )


def build_block_mask_direct(selection, block_size, local_window, seq_len, device,
                             top_k_blocks=None):
    """Direct BlockMask assembly straight from block-level data already in
    hand (the router's top-k selection plus the local-window band) --
    WITHOUT calling create_block_mask and WITHOUT evaluating mask_mod over a
    token-granularity index grid. Uses BlockMask.from_kv_blocks, verified via
    verify_blockmask_api()/inspect.signature above (torch 2.13.0, locally
    OBSERVED to accept (kv_num_blocks, kv_indices, full_kv_num_blocks=None,
    full_kv_indices=None, BLOCK_SIZE=128, mask_mod=None, ...)). Whether the
    cluster's torch 2.11.0 exposes the same signature is UNRESOLVED -- see
    module docstring and BLOCKMASK_API_INFO; the hasattr check below makes
    this fail with a clear, guarded (not crashing) message if not.

    Local-window-to-block-granularity rounding convention (EXACT at block
    granularity, not merely conservative): for query block qb spanning
    tokens [qb*block_size, qb*block_size + block_size - 1], a causal local
    pair (j <= i, i - j < local_window) has j in
    [qb*block_size - local_window + 1, qb*block_size + block_size - 1]. The
    block-level footprint of that token range is [kb_min, qb], where
    kb_min = max(0, floor((qb*block_size - local_window + 1) / block_size)).
    The upper bound is always qb itself because local pairs are causal
    (j <= i). Every key block in [kb_min, qb] contains at least one token
    pair that is local for at least one query in this query block; every key
    block outside that range contains none (see
    test_local_window_blocks_included). The union of this local band with
    the router's already-causal-filtered selected blocks is what gets
    expressed at block granularity; EXACT per-token correctness within an
    included block (the causal boundary itself, and the local window's exact
    per-token edge) is still enforced by mask_mod, which IS passed through
    and IS still evaluated by the compiled kernel per-element -- what is
    eliminated is evaluating it over the full T x T token index grid at
    CONSTRUCTION time, which is the diagnosed bug.

    Construction route (2026-09-19 rewrite -- see module docstring /
    predictions/longcontext_systems.json for the three-way bind this
    resolves): this does NOT call BlockMask.from_kv_blocks at all anymore.
    Two independent bugs were found in torch 2.13.0's from_kv_blocks when
    exercised at these sizes, both now bypassed rather than worked around:

    1. compute_q_blocks=True (the default) derives q_num_blocks/q_indices via
       _transpose_ordered -> _ordered_to_dense(..., col_indices.shape[-1]),
       which uses the PER-ROW max cardinality (kv_indices.shape[-1], i.e.
       max_cols) as the dense matrix's column COUNT instead of the true
       number of kv blocks (num_kv_blocks). Whenever a row references a
       key-block index >= max_cols (the common case: max_cols is the
       largest per-row selection size, almost always < num_kv_blocks), this
       raises "IndexError: index X is out of bounds for dimension Y with
       size Z" from inside torch's own vmap'd create_dense_one. OBSERVED
       reproducing with block_size=64, local_window=64, num_kv_blocks=4, an
       all-empty selection (local band only): max_cols=2 but a referenced kv
       index of 3 crashes.

    2. compute_q_blocks=False avoids bug 1 (q_num_blocks/q_indices are just
       None) but from_kv_blocks then falls through to its OWN buggy default
       for seq_lengths (only used when the caller does not pass seq_lengths
       explicitly, which the prior version of this function did not):
       `kv_length = kv_indices.shape[-1] * BLOCK_SIZE[1]` -- this uses
       max_cols (the padded per-row SELECTION count), NOT num_kv_blocks (the
       true number of key blocks), as the multiplicand. Since max_cols is
       almost always << num_kv_blocks for a sparse pattern, the resulting
       block_mask.shape[-1] is far smaller than the real KV_LEN. This is
       EXACTLY the "block_mask shape mismatch vs q_len/kv_len" ValueError
       observed at every config on the v3 GPU run: it is raised at
       torch/nn/attention/flex_attention.py's `block_mask.shape[-2]`/`[-1]`
       check (OBSERVED, torch 2.13.0 source) comparing block_mask.shape
       (derived from the wrong seq_lengths) against query/key's actual
       length, BEFORE any kernel/compile work happens -- i.e. a pure
       BlockMask-construction bug, not evidence that flex_attention's
       compiled kernel structurally requires q_num_blocks/q_indices for a
       forward-only (no .backward()) call. (Confirmed by reading
       torch/_inductor/kernel/flex/flex_attention.py's forward Triton
       lowering: its `input_nodes`/`inputs_for_autotuning` lists pass
       kv_num_blocks/kv_indices/full_kv_num_blocks/full_kv_indices only --
       q_num_blocks/q_indices are read from the BlockMask namedtuple but
       never forwarded into the forward kernel's own input/autotuning
       lists; they are only consumed later for `_compute_dq_write_order_
       from_block_mask`, which is backward-only.)

    Fix: bypass from_kv_blocks entirely and call BlockMask's own __init__
    directly (verified via inspect.signature(BlockMask.__init__), matched
    exactly: seq_lengths, kv_num_blocks, kv_indices, full_kv_num_blocks,
    full_kv_indices, q_num_blocks, q_indices, full_q_num_blocks,
    full_q_indices, BLOCK_SIZE, mask_mod), passing:
      - seq_lengths=(seq_len, seq_len) EXPLICITLY (the true token length,
        sidestepping bug 2 -- __init__ has no "derive it from kv_indices'
        shape" fallback at all, so this class of bug cannot recur here).
      - q_num_blocks/q_indices computed OURSELVES via _ordered_from_boolean
        applied to union.transpose(-2, -1) -- the mechanical transpose of
        the SAME boolean block matrix used for the kv side, done with plain
        tensor ops over the (num_q_blocks x num_kv_blocks) block matrix
        (trivial: 512x512 at T=32768, block_size=64). This never touches
        torch's _transpose_ordered/_ordered_to_dense, so bug 1 cannot recur
        either, and is computed unconditionally (not gated on forward-only
        use) so this BlockMask would also support backward if ever needed.
      - full_kv_*/full_q_*=None (no separate_full_blocks optimization; see
        the unexplored-optimization note below, unchanged from the prior
        version).

    NOTE (unexplored optimization, not a correctness gap): this does not
    implement create_block_mask's separate_full_blocks split (marking blocks
    that need no mask_mod at all so the kernel can skip it -- "about a 15%
    speedup" per BlockMask's own docstring, OBSERVED torch 2.13.0). Every
    included block here is treated as partial. Not implementing that
    optimization cannot make this path slower than the mask_mod path (it is
    strictly less work at construction time); it only potentially leaves
    additional kernel-side speedup on the table, unmeasured here.

    Batch generalization (2026-09-20): `selection` may be rank-3
    (n_heads, num_q_blocks, num_kv_blocks) -- a single selection broadcast
    across the batch, exactly the original contract -- or rank-4
    (B, n_heads, num_q_blocks, num_kv_blocks), a real per-example batch (the
    shape a router's input-dependent top-k selection actually produces).
    `qb_idx`/`kb_idx`/`causal_block`/`local_band` below are pure functions
    of (block_size, local_window, num_q_blocks, num_kv_blocks) -- never of
    `selection` -- so they need no batch handling; they are reshaped to
    broadcast against whichever rank `selection` has. The union, and
    therefore `_ordered_from_boolean`'s output, is computed per batch
    element independently when `selection` is rank-4: nothing here
    broadcasts one example's selection across the batch.

    Fixed-width index padding (2026-09-20, dynamo-recompile-hazard fix):
    `kv_indices`/`q_indices` are (B, H, Qb/Kvb, W) tensors whose last
    dimension W used to be `selection.sum(-1).max()` -- the largest
    per-row cardinality actually OBSERVED in this call's data. Since
    routing is input-dependent, W changed batch to batch, and flex_
    attention's compiled graph is guarded on `block_mask`'s tensor shapes,
    so every new W value OBSERVED on CUDA triggered a dynamo recompile
    ("tensor 'block_mask.q_indices' size mismatch"). `torch._dynamo.config.
    recompile_limit` defaults to 8; past that, flex_attention silently
    falls back to an unfused kernel that materializes the full score
    matrix (~180x slower, separately measured in this project) -- with no
    error and numerically-correct output, so this would not be visible
    except as an unexplained slowdown on a real (many-batch) run.

    W must instead be a pure function of geometry, never of `selection`'s
    data:
      - kv-side width (bounds `kv_indices`, one row per query block): the
        tightest SOUND bound on how many key blocks any query block can
        ever be admitted to is `top_k_blocks + ceil(local_window /
        block_size) + 1` -- the router's own top-k budget, plus the local
        band's block-level span (see the rounding-convention analysis
        above), plus the query block's own diagonal block, which is
        always causally valid and can double-count with the local band's
        span depending on rounding. This is clamped to `num_kv_blocks`
        (a row can never have more columns than exist) and is exact
        PROVIDED `top_k_blocks` is the true budget the caller's router
        used to build `selection` -- `_ordered_from_boolean` asserts this
        at runtime (see its docstring) rather than silently truncating if
        it is ever violated. If `top_k_blocks` is not supplied (kept
        optional for callers migrating incrementally, e.g. scripts/
        bench_flex_longcontext.py's synthetic-probe call sites, which are
        not part of the production training path this fix targets), the
        fallback is `num_kv_blocks` itself -- always sound, since that is
        simply "no row can exceed the true number of columns", just not
        tight.
      - q-side width (bounds `q_indices`, one row per KEY block, since
        the q-side is the transpose of the kv-side boolean matrix): there
        is no equally tight geometry-only bound here. The local band
        contributes at most `ceil(local_window / block_size) + 1` query
        blocks per key block (by the same rounding argument, transposed),
        but the router's top-k selection has NO bound on how many
        DIFFERENT query blocks may all choose to select the SAME key
        block -- worst case, every query block does, independent of
        `top_k_blocks`. So the only sound bound is `num_q_blocks` itself.
        This is deliberately chosen over any tighter-but-unsound
        heuristic, per this task's explicit "prefer correctness and
        shape-stability over tightness" instruction. Note `num_q_blocks`
        is itself already a pure function of geometry (ceil(seq_len /
        block_size)), so this is still shape-stable across calls at
        identical geometry regardless of `selection`'s actual data.
    Both widths depend only on (seq_len, block_size, local_window,
    top_k_blocks) -- never on `selection` -- so two calls at identical
    geometry produce identically-shaped `kv_indices`/`q_indices` even if
    their selections have very different cardinalities (see
    test_index_shapes_invariant_across_selections /
    test_index_width_is_function_of_geometry_only).
    """
    if selection.dim() not in (3, 4):
        raise ValueError(
            f"selection must be rank 3 (n_heads, num_q_blocks, num_kv_blocks) "
            f"or rank 4 (B, n_heads, num_q_blocks, num_kv_blocks); got shape "
            f"{tuple(selection.shape)}"
        )
    n_heads, num_q_blocks, num_kv_blocks = selection.shape[-3:]
    sel = selection.to(device=device, dtype=torch.bool)

    # Fixed, geometry-only pad widths -- see the class-level docstring
    # section above ("Fixed-width index padding") for the derivation and
    # soundness argument for each of these two formulas.
    local_window_blocks = math.ceil(local_window / block_size)
    if top_k_blocks is not None:
        kv_pad_width = min(top_k_blocks + local_window_blocks + 1, num_kv_blocks)
    else:
        kv_pad_width = num_kv_blocks
    q_pad_width = num_q_blocks

    if sel.dim() == 3:
        # make_mask_mod indexes sel[h, qb, kvb] -- correct for the rank-3
        # broadcast-across-batch case (batch `b` is ignored, per its own
        # docstring).
        mask_mod = make_mask_mod(sel, block_size, local_window)
    else:
        # Real per-example selection: index by the actual batch `b` too.
        def mask_mod(b, h, q_idx, kv_idx):
            causal = kv_idx <= q_idx
            local = (q_idx - kv_idx) < local_window
            qb = q_idx // block_size
            kvb = kv_idx // block_size
            selected = sel[b, h, qb, kvb]
            return causal & (local | selected)

    qb_idx = torch.arange(num_q_blocks, device=device).view(num_q_blocks, 1)
    kb_idx = torch.arange(num_kv_blocks, device=device).view(1, num_kv_blocks)
    kb_min = torch.div(
        qb_idx * block_size - local_window + 1, block_size, rounding_mode="floor",
    ).clamp(min=0)
    causal_block = kb_idx <= qb_idx
    local_band = causal_block & (kb_idx >= kb_min)  # (num_q_blocks, num_kv_blocks)

    # Broadcast causal_block/local_band against sel's actual rank (3 or 4):
    # view them with as many leading singleton dims as sel has beyond its
    # trailing (num_q_blocks, num_kv_blocks) pair.
    leading = sel.dim() - 2
    broadcast_shape = (1,) * leading + (num_q_blocks, num_kv_blocks)
    causal_block_b = causal_block.view(broadcast_shape)
    local_band_b = local_band.view(broadcast_shape)

    # Union of selected blocks and the local band, re-intersected with
    # block-level causality defensively (selection is already causal-filtered
    # upstream by generate_block_selection / the router's top-k over a
    # causally-masked score matrix, but this makes that invariant explicit
    # here rather than relying on the caller). Computed per batch element
    # independently when sel is rank-4 -- this is a plain elementwise op,
    # nothing here mixes information across the batch dimension.
    union = (sel | local_band_b) & causal_block_b
    # (n_heads, num_q_blocks, num_kv_blocks) or (B, n_heads, num_q_blocks, num_kv_blocks)

    kv_num_blocks, kv_indices = _ordered_from_boolean(union, device, width=kv_pad_width)
    # The q-side structure is EXACTLY the transpose of the kv-side boolean
    # block matrix -- per key block, which query blocks attend to it -- not
    # an independent computation, so it cannot disagree with kv_indices by
    # construction (see test_q_side_is_transpose_of_kv_side).
    q_num_blocks, q_indices = _ordered_from_boolean(
        union.transpose(-2, -1).contiguous(), device, width=q_pad_width,
    )

    return BlockMask(
        seq_lengths=(seq_len, seq_len),
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=None,
        full_kv_indices=None,
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        full_q_num_blocks=None,
        full_q_indices=None,
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=mask_mod,
    )
