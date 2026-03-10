"""
loop.py — THE ORCHESTRATOR

Autonomous improvement engine. Reads the pipeline, prompts Claude for one
targeted modification, evaluates, commits if improved, reverts if not, loops.

Usage:
    python loop.py                     # run indefinitely
    python loop.py --experiments 50    # run exactly 50 experiments
    python loop.py --model claude-opus-4-6   # use a specific model
    python loop.py --dry-run           # propose changes without committing
"""

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
PIPELINE_PATH = ROOT / "rag_pipeline.py"
PROGRAM_PATH = ROOT / "program.md"
EXPERIMENTS_PATH = ROOT / "results" / "experiments.jsonl"
BEST_CONFIG_PATH = ROOT / "results" / "best_config.json"
README_PATH = ROOT / "README.md"

PROTECTED_FILES = {"corpus_prep.py", "eval.py", "loop.py"}

AGENT_PROMPT = """\
You are an expert RAG systems researcher running an automated optimization experiment.

## Current Pipeline
```python
{pipeline_code}
```

## Research Program
{program_md}

## Recent Experiment History (last 10)
{history}

## Your Task
Propose exactly ONE targeted modification to rag_pipeline.py that you hypothesize
will improve the composite RAGAS score.

Rules:
- Make only ONE change (one parameter, one function, one strategy)
- The change must be testable and reversible
- Do not modify corpus_prep.py or eval.py
- Explain your hypothesis in a single sentence before the code
- Return the COMPLETE modified rag_pipeline.py (not a diff)

Format your response EXACTLY as:
HYPOTHESIS: [one sentence explaining why this change should improve the score]
CHANGE: [one sentence describing what you changed]
---
[complete modified rag_pipeline.py]
"""


def git(cmd: str) -> str:
    result = subprocess.run(
        f"git {cmd}", shell=True, capture_output=True, text=True, cwd=ROOT
    )
    return result.stdout.strip()


def load_history(n: int = 10) -> str:
    if not EXPERIMENTS_PATH.exists():
        return "No experiments yet — this is the first run."
    lines = EXPERIMENTS_PATH.read_text().strip().split("\n")
    recent = lines[-n:]
    bullets = []
    for line in recent:
        try:
            exp = json.loads(line)
            status = "✅ Kept" if exp["kept"] else "❌ Reverted"
            bullets.append(
                f"- Exp #{exp['experiment_id']}: {exp['change_description']} | "
                f"{exp['score_before']:.4f}→{exp['score_after']:.4f} "
                f"(Δ{exp['delta']:+.4f}) {status}"
            )
        except (json.JSONDecodeError, KeyError):
            continue
    return "\n".join(bullets) if bullets else "No valid experiments yet."


def next_experiment_id() -> int:
    if not EXPERIMENTS_PATH.exists():
        return 1
    lines = EXPERIMENTS_PATH.read_text().strip().split("\n")
    for line in reversed(lines):
        try:
            return json.loads(line)["experiment_id"] + 1
        except (json.JSONDecodeError, KeyError):
            continue
    return 1


def parse_agent_response(response: str) -> tuple[str, str, str]:
    """Extract hypothesis, change description, and code from the agent response."""
    hypothesis, change, code = "", "", ""

    for line in response.split("\n"):
        if line.startswith("HYPOTHESIS:"):
            hypothesis = line.replace("HYPOTHESIS:", "").strip()
        elif line.startswith("CHANGE:"):
            change = line.replace("CHANGE:", "").strip()

    if "---" in response:
        after_separator = response.split("---", 1)[1]
        if "```python" in after_separator:
            code = after_separator.split("```python", 1)[1]
            code = code.split("```", 1)[0]
        elif "```" in after_separator:
            code = after_separator.split("```", 1)[1]
            code = code.split("```", 1)[0]
        else:
            code = after_separator
    elif "```python" in response:
        code = response.split("```python", 1)[1].split("```", 1)[0]

    return hypothesis.strip(), change.strip(), code.strip()


def validate_python(code: str) -> bool:
    """Check that the code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def log_experiment(exp: dict):
    EXPERIMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS_PATH, "a") as f:
        f.write(json.dumps(exp) + "\n")


def save_best_config(exp: dict, code: str):
    BEST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_id": exp["experiment_id"],
        "composite_score": exp["score_after"],
        "metrics": exp["metrics"],
        "timestamp": exp["timestamp"],
        "hypothesis": exp["hypothesis"],
        "change_description": exp["change_description"],
    }
    BEST_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def update_readme(experiments: list[dict]):
    """Rewrite the leaderboard section of README.md."""
    kept = [e for e in experiments if e.get("kept")]
    if not kept:
        return

    best = max(kept, key=lambda e: e["score_after"])
    top = sorted(kept, key=lambda e: e["score_after"], reverse=True)[:10]

    rows = []
    for e in top:
        delta = f"+{e['delta']:.4f}" if e["delta"] > 0 else f"{e['delta']:.4f}"
        rows.append(f"| #{e['experiment_id']} | {e['score_after']:.4f} | {delta} | {e['change_description']} |")

    baseline_row = f"| #0 | {kept[0]['score_before']:.4f} | baseline | Initial naive config |"

    leaderboard = "\n".join(rows + [baseline_row])

    readme = f"""# autoRAGresearch 🔬

> Autonomous RAG pipeline optimizer. The agent iterates. You sleep.
> Inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch).

## Current Best Score: {best['score_after']:.4f} (experiment #{best['experiment_id']})

| Experiment | Score | Delta | Change |
|---|---|---|---|
{leaderboard}

## Quick Start

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b && ollama pull nomic-embed-text
echo "ANTHROPIC_API_KEY=your_key" > .env
python setup_data.py
python eval.py          # see baseline score
python loop.py          # start optimizing
```

## How It Works

autoRAGresearch gives an AI agent a real RAG pipeline (`rag_pipeline.py`) and lets it
experiment autonomously. The agent forms a hypothesis, modifies the pipeline,
evaluates it against a fixed query set, keeps improvements, reverts failures,
and loops. The human only writes `program.md` to steer the research direction.

## Files

| File | Who edits | Purpose |
|---|---|---|
| `rag_pipeline.py` | AI agent | The research canvas |
| `program.md` | Human | Research direction |
| `eval.py` + `corpus_prep.py` | Nobody (fixed) | The arena |
| `loop.py` | Nobody (fixed) | The orchestrator |

See `results/experiments.jsonl` for the full experiment log.
"""
    README_PATH.write_text(readme)


def run_loop(args):
    from eval import run_evaluation

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    model = args.model

    # Baseline eval
    print("autoRAGresearch initialized.")
    print("Running baseline evaluation...")
    baseline = run_evaluation()
    if baseline["timed_out"]:
        print(f"Baseline eval failed: {baseline.get('error', 'timeout')}")
        sys.exit(1)

    best_score = baseline["composite_score"]
    print(
        f"Baseline eval: composite_score={best_score:.4f} | "
        f"faithfulness={baseline['faithfulness']:.4f} | "
        f"answer_relevance={baseline['answer_relevance']:.4f} | "
        f"context_precision={baseline['context_precision']:.4f} | "
        f"context_recall={baseline['context_recall']:.4f}\n"
    )

    exp_count = 0
    max_experiments = args.experiments or float("inf")

    while exp_count < max_experiments:
        exp_id = next_experiment_id()
        exp_count += 1

        pipeline_code = PIPELINE_PATH.read_text()
        program_md = PROGRAM_PATH.read_text() if PROGRAM_PATH.exists() else "No program.md found."
        history = load_history()

        prompt = AGENT_PROMPT.format(
            pipeline_code=pipeline_code,
            program_md=program_md,
            history=history,
        )

        # Ask Claude for a modification
        print(f"[Experiment {exp_id}] Asking agent for improvement...")
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            agent_text = response.content[0].text
        except Exception as e:
            print(f"  Agent API error: {e}. Sleeping 30s...")
            time.sleep(30)
            continue

        hypothesis, change_desc, new_code = parse_agent_response(agent_text)
        if not hypothesis:
            hypothesis = "Unnamed hypothesis"
        if not change_desc:
            change_desc = "Undescribed change"

        print(f"  Hypothesis: {hypothesis}")

        if not new_code or not validate_python(new_code):
            print("  ❌ Agent returned invalid Python. Skipping.")
            log_experiment({
                "experiment_id": exp_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hypothesis": hypothesis,
                "change_description": change_desc,
                "score_before": best_score,
                "score_after": 0.0,
                "delta": -best_score,
                "kept": False,
                "metrics": {},
                "avg_latency_ms": 0.0,
                "error": "invalid_python",
            })
            time.sleep(args.sleep)
            continue

        # Write the new pipeline
        backup = pipeline_code
        if not args.dry_run:
            PIPELINE_PATH.write_text(new_code)

        # Evaluate
        print("  Running eval...", end=" ", flush=True)
        t0 = time.time()
        result = run_evaluation()
        elapsed = time.time() - t0
        print(f"({elapsed:.0f}s)")

        new_score = result["composite_score"]
        delta = new_score - best_score

        exp_record = {
            "experiment_id": exp_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hypothesis": hypothesis,
            "change_description": change_desc,
            "score_before": best_score,
            "score_after": new_score,
            "delta": round(delta, 4),
            "kept": False,
            "metrics": {
                "faithfulness": result["faithfulness"],
                "answer_relevance": result["answer_relevance"],
                "context_precision": result["context_precision"],
                "context_recall": result["context_recall"],
            },
            "avg_latency_ms": result["avg_latency_ms"],
        }

        if delta > 0 and not result["timed_out"]:
            exp_record["kept"] = True
            best_score = new_score
            print(f"  Score: {best_score - delta:.4f} → {best_score:.4f} ({delta:+.4f}) ✅ Kept")

            if not args.dry_run:
                git("add rag_pipeline.py")
                msg = f"[autoRAGresearch] {delta:+.4f} | {change_desc} | score: {new_score:.4f}"
                git(f'commit -m "{msg}"')
                save_best_config(exp_record, new_code)
        else:
            print(f"  Score: {best_score:.4f} → {new_score:.4f} ({delta:+.4f}) ❌ Reverted")
            if not args.dry_run:
                PIPELINE_PATH.write_text(backup)

        log_experiment(exp_record)

        # Update README leaderboard
        if EXPERIMENTS_PATH.exists():
            all_exps = []
            for line in EXPERIMENTS_PATH.read_text().strip().split("\n"):
                try:
                    all_exps.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            update_readme(all_exps)

        time.sleep(args.sleep)

    print(f"\nDone. Best score: {best_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="autoRAGresearch — Autonomous RAG Pipeline Optimizer")
    parser.add_argument("--experiments", type=int, default=None, help="Max experiments to run (default: unlimited)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Anthropic model to use")
    parser.add_argument("--dry-run", action="store_true", help="Propose changes without writing or committing")
    parser.add_argument("--sleep", type=int, default=5, help="Seconds to pause between experiments")
    args = parser.parse_args()
    run_loop(args)


if __name__ == "__main__":
    main()
