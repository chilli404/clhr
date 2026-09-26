"""Run lm-eval-harness tasks against a SparseTransformer checkpoint, soft vs hard."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lm_eval import simple_evaluate  # noqa: E402
from lm_eval_adapter import SparseTransformerLM  # noqa: E402

TASKS = ["lambada_openai", "hellaswag", "piqa", "arc_easy", "winogrande"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args()

    results = {}
    for mode in ("soft", "hard"):
        lm = SparseTransformerLM(args.checkpoint, mode=mode)
        out = simple_evaluate(model=lm, tasks=TASKS, limit=args.limit, bootstrap_iters=0)
        results[mode] = {
            task: {k: v for k, v in res.items() if isinstance(v, (int, float, str))}
            for task, res in out["results"].items()
        }
        print(f"=== mode={mode} done ===", flush=True)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
