"""Assemble the reconciled lambda=1.0, corrected-masking, 3-seed comparison
bundle for the SSA-vs-CLHR-vs-STE-vs-post-hoc-KL table.

Run with: uv run python scripts/build_reconciled_lam10_bundle.py

Context (see results/RUN_LEDGER.md Section 2(k) and the coordinator report
this script accompanies): four of the five mechanisms below were
investigated by direct read of their training/eval scripts before writing
this file. Two (SSA-style, tuned-STE) already have valid lambda=1.0-
equivalent, corrected-masking, 3-seed results sitting on disk under
ambiguous/scattered filenames -- this script copies+relabels them into
results/reconciled_lam10/ with explicit provenance, rather than re-running
expensive training for mechanisms that don't need it. The other two
(exact-protocol "dense" condition, post-hoc-KL) genuinely have no lambda=1.0
3-seed result anywhere and require the new vast.ai runs launched via
vastai/launch_reconciled_lam10.sh -- this script's DENSE_EVAL / POSTHOC_KL*
sections degrade gracefully (report "pending") until those results are
pulled back via vastai/sync_reconciled_lam10_results.sh.

Landmines this script deliberately avoids (documented so a future editor
doesn't reintroduce them):
  - results/s3/sparse_rca_ste_hard_s42.json is a STALE pre-tuning STE run at
    the wrong learning rate (predates ste_tuning.json). The correct seed-42
    tuned-STE datapoint is ste_tuning.json's own "best_ste" block.
  - results/paper_lam10/README.txt's SSA/CLHR rows were transcribed
    swapped in one place in RUN_LEDGER.md (line ~965) -- SSA-style is
    0.00331+-0.0003, NOT 0.00346 (that's CLHR's own row in the same README).
    This script reads the raw per-seed files directly, not that README, so
    it is not exposed to that transcription error.
  - There are (at least) three different existing "CLHR lambda=1.0, 3-seed"
    G_CL numbers already in the repo (0.00417 ladder_lam10_eval, 0.00346
    paper_lam10/ssa_vs_clhr_full_comparison.json's clhr_lam10 arm, 0.00584
    clhr_lambda_sweep lam10 arm), NOT reconciled with each other pre-task.
    This script anchors ONLY on results/ladder_lam10_eval/ for the CLHR
    number, because its provenance (trained via sparse_attention_exact_
    protocol.py --lambda-rca 1.0 --normalize-loss, re-evaluated via the
    corrected-masking sparse_attention_closed_loop_eval.py) was directly
    confirmed by reading both scripts and the launch yaml that produced it.
    The other two numbers' training/eval provenance was NOT confirmed by
    this investigation and must not be cited in the reconciled table.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "reconciled_lam10"
SEEDS = [42, 123, 456]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def relabel_ssa_style() -> dict:
    """SSA-style has no --lambda-rca flag (structurally a different, dual-
    stream alignment mechanism -- its own analog is --objective-clhr-weight,
    unset in every headline run). The existing 3-seed alignment_weight=1.0
    run already uses correct -inf masking and matches results/paper_lam10/
    README.txt's SSA line -- no new run needed, just relabel + record why.
    """
    per_seed = {}
    g_cls = []
    for seed in SEEDS:
        src = REPO_ROOT / "results" / "ssa_style" / f"ssa_style_aw1_s{seed}.json"
        data = _load(src)
        if data is None:
            per_seed[seed] = {"status": "MISSING", "source": str(src)}
            continue
        g_cl = data["closed_loop_excess_nll"]
        g_cls.append(g_cl)
        per_seed[seed] = {
            "status": "reused_existing_no_new_run_needed",
            "source": str(src.relative_to(REPO_ROOT)),
            "native_nll": data["native_nll"],
            "hard_nll": data["closed_loop_hard_nll"],
            "g_cl": g_cl,
            "note": (
                "SSA-style has no lambda_rca mixing; this is its natural-"
                "default config (alignment_weight=1.0, hard_probability=0.5, "
                "objective_clhr_weight unset), selected via the existing "
                "tune-then-heldout scripts (scripts/run_ssa_dev_queue.sh + "
                "scripts/run_ssa_heldout_queue.sh)."
            ),
        }
        out_path = OUT_DIR / f"reconciled_lam10_ssa_style_s{seed}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({**data, "reconciled_provenance": per_seed[seed]["note"],
                        "reconciled_source": str(src.relative_to(REPO_ROOT))}, f, indent=2)
    mean, sd = _mean_sd(g_cls) if g_cls else (None, None)
    return {"mechanism": "ssa_style", "per_seed": per_seed, "g_cl_mean": mean, "g_cl_sd": sd}


def relabel_tuned_ste() -> dict:
    """ste_tuning.py hardcodes seed=42 (dev-seed tuning only) and searches
    only over `lr` (STE) / `lr x final_temp` (anneal) -- no lambda_rca
    (STE is a hard-forward-pass mechanism, not lambda-mixing). Its winner
    (lr=1e-3) exactly matches what ste_heldout.py hardcodes for seeds
    123/456, so the existing scattered 3-seed set is already the correct
    "tuned STE" result -- no new run needed.
    """
    sources = {
        42: (REPO_ROOT / "results" / "s3" / "ste_tuning.json", "best_ste"),
        123: (REPO_ROOT / "results" / "s3" / "cl_eval_ste_hard_s123.json", None),
        456: (REPO_ROOT / "results" / "s3" / "cl_eval_ste_hard_s456.json", None),
    }
    per_seed = {}
    g_cls = []
    for seed, (src, key) in sources.items():
        data = _load(src)
        if data is None:
            per_seed[seed] = {"status": "MISSING", "source": str(src)}
            continue
        block = data[key] if key else data
        g_cl = block["g_cl"]
        g_cls.append(g_cl)
        per_seed[seed] = {
            "status": "reused_existing_no_new_run_needed",
            "source": str(src.relative_to(REPO_ROOT)) + (f"[{key}]" if key else ""),
            "native_nll": block["native_nll"],
            "cl_nll": block["cl_nll"],
            "g_cl": g_cl,
            "note": (
                "seed 42 is ste_tuning.py's own dev/tuning seed (lr=1e-3 "
                "winner); seeds 123/456 are ste_heldout.py --method ste "
                "runs at that same winning lr, G_CL computed by "
                "sparse_attention_cl_eval_single.py. Do NOT use "
                "results/s3/sparse_rca_ste_hard_s42.json -- that is a stale "
                "pre-tuning run at a different (wrong) learning rate."
            ),
        }
        out_path = OUT_DIR / f"reconciled_lam10_ste_tuned_s{seed}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({**block, "seed": seed, "reconciled_provenance": per_seed[seed]["note"],
                        "reconciled_source": per_seed[seed]["source"]}, f, indent=2)
    mean, sd = _mean_sd(g_cls) if g_cls else (None, None)
    return {"mechanism": "tuned_ste", "per_seed": per_seed, "g_cl_mean": mean, "g_cl_sd": sd}


def anchor_clhr_ladder() -> dict:
    """The ledger-designated headline CLHR arm: results/ladder_lam10_eval/,
    trained via sparse_attention_exact_protocol.py --lambda-rca 1.0
    --normalize-loss, re-evaluated for corrected masking via
    sparse_attention_closed_loop_eval.py. Provenance directly confirmed by
    reading both scripts + the launch yaml. This is the ONLY CLHR lambda=1.0
    number this bundle cites (see module docstring for the other two
    unreconciled candidates that are deliberately NOT cited here).
    """
    per_seed = {}
    g_cls = []
    for seed in SEEDS:
        src = REPO_ROOT / "results" / "ladder_lam10_eval" / f"ladder_lam10_contemporary_closedloop_hard_s{seed}.json"
        data = _load(src)
        if data is None:
            per_seed[seed] = {"status": "MISSING", "source": str(src)}
            continue
        native = data["modes"]["native_soft"]["nll"]
        hard = data["modes"]["closed_loop_gate_hard"]["nll"]
        g_cl = hard - native
        g_cls.append(g_cl)
        per_seed[seed] = {
            "status": "reused_existing_no_new_run_needed",
            "source": str(src.relative_to(REPO_ROOT)),
            "native_nll": native,
            "hard_nll": hard,
            "g_cl": g_cl,
        }
    mean, sd = _mean_sd(g_cls) if g_cls else (None, None)
    return {"mechanism": "clhr_contemporary_closedloop_hard", "per_seed": per_seed,
            "g_cl_mean": mean, "g_cl_sd": sd}


def pending_dense_eval() -> dict:
    """exact_protocol.py's 8-entry CONDITIONS list includes "dense" -- the
    only one of the 8 with NO lambda=1.0, corrected-masking, 3-seed result
    anywhere on disk (confirmed by direct search). Genuinely new run,
    launched via vastai/launch_reconciled_lam10.sh (DENSE_TRAIN_S{42,123,456}
    then DENSE_EVAL). This function reports "pending" until
    vastai/sync_reconciled_lam10_results.sh has pulled
    results/reconciled_lam10/reconciled_lam10_dense_eval.json back.
    """
    src = OUT_DIR / "reconciled_lam10_dense_eval.json"
    data = _load(src)
    if data is None:
        return {"mechanism": "dense", "status": "PENDING_NEW_RUN",
                 "note": "Launch DENSE_TRAIN_S{42,123,456} then DENSE_EVAL "
                         "via vastai/launch_reconciled_lam10.sh, then pull "
                         "with vastai/sync_reconciled_lam10_results.sh."}
    return {"mechanism": "dense", "status": "complete", "raw": data}


def pending_posthoc_kl() -> dict:
    """posthoc_kl.py needs lambda=1.0 checkpoints tagged sparse_rca_
    {learned_gate,contemporary_closedloop_hard,dense}_s{42,123,456} -- the
    existing results/s3/posthoc_kl_comparison.json's checkpoint lambda is
    UNVERIFIED (no launch record found; sparse_attention_rca.py's own
    --lambda-rca defaults to 0.3, so it is likely NOT the headline arm).
    Genuinely new run: 9 posthoc_prereq training jobs + 1 posthoc_kl job +
    1 posthoc_kl_maskagnostic job, via vastai/launch_reconciled_lam10.sh.
    """
    results = {}
    for name in ("posthoc_kl", "posthoc_kl_maskagnostic"):
        src = OUT_DIR / f"reconciled_lam10_{name}.json"
        data = _load(src)
        if data is None:
            results[name] = {"status": "PENDING_NEW_RUN",
                              "note": "Launch POSTHOC_PREREQ_* (9 jobs) then "
                                      f"{name.upper()} via "
                                      "vastai/launch_reconciled_lam10.sh."}
        else:
            results[name] = {"status": "complete", "raw": data}
    return {"mechanism": "post_hoc_kl", **results}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": (
            "Reconciled lambda=1.0, corrected-masking, 3-seed (42/123/456) "
            "comparison bundle across exact-protocol CLHR, SSA-style, "
            "tuned-STE, and post-hoc-KL. See RUN_LEDGER.md Section 2(k) for "
            "the original unreconciled discrepancy this closes."
        ),
        "clhr": anchor_clhr_ladder(),
        "ssa_style": relabel_ssa_style(),
        "tuned_ste": relabel_tuned_ste(),
        "dense": pending_dense_eval(),
        "post_hoc_kl": pending_posthoc_kl(),
    }
    manifest_path = OUT_DIR / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")
    for key in ("clhr", "ssa_style", "tuned_ste"):
        m = manifest[key]
        print(f"  {m['mechanism']}: G_CL = {m['g_cl_mean']:.6f} +/- {m['g_cl_sd']:.6f} (n={len(SEEDS)})")
    print(f"  dense: {manifest['dense']['status']}")
    print(f"  post_hoc_kl.posthoc_kl: {manifest['post_hoc_kl']['posthoc_kl']['status']}")
    print(f"  post_hoc_kl.posthoc_kl_maskagnostic: {manifest['post_hoc_kl']['posthoc_kl_maskagnostic']['status']}")


if __name__ == "__main__":
    main()
