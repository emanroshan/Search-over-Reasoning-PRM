import argparse
import json
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from src.generation import (
    compute_majority_vote,
    run_greedy,
    run_self_consistency,
)
from src.prm_judge import score_path
from src.search import select_best_path

DATA = Path("data/gsm8k_test_50.jsonl")
OUT = Path("outputs")
ROWS = OUT / "per_problem.jsonl"

GEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"
JUDGE_ID = "Qwen/Qwen2.5-7B-Instruct"

METHODS = [
    "Greedy (N=1)",
    "Self-Consistency (N=5)",
    "Compute-Matched SC (N=15)",
    "PRM Best-of-5 (min)",
    "PRM Best-of-5 (product)",
    "PRM Best-of-5 (step_penalty, lambda=2.0)",
]


def dump_json(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    )


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_candidate(c):
    return {
        "completion": c.get("completion", ""),
        "steps": c.get("steps", []),
        "answer": c.get("answer"),
        "num_tokens": int(c.get("num_tokens", 0)),
    }


def score_selection(candidates, strategy, lam=2.0):
    start = time.perf_counter()
    chosen = select_best_path(
        candidates,
        aggregation_strategy=strategy,
        penalty_lambda=lam,
    )
    return chosen, time.perf_counter() - start


def evaluate_one(problem, index, gen_model, gen_tok,
                 judge_model, judge_tok):
    set_seed(5326 + index)
    question = problem["question"]
    gt = int(problem["ground_truth_answer"])

    greedy = run_greedy(
        model=gen_model,
        tokenizer=gen_tok,
        question=question,
    )

    sc5 = run_self_consistency(
        model=gen_model,
        tokenizer=gen_tok,
        question=question,
        n=5,
        temperature=0.7,
    )

    first_five = [
        normalize_candidate(c) for c in sc5["candidates"]
    ]

    judged = []
    judge_calls = 0
    judge_tokens = 0
    judge_start = time.perf_counter()

    for candidate in first_five:
        assessment = score_path(
            judge_model=judge_model,
            judge_tokenizer=judge_tok,
            question=question,
            candidate=candidate,
        )
        judged.append({
            **candidate,
            "step_scores": assessment["step_scores"],
            "rationales": assessment["rationales"],
        })
        judge_calls += assessment["judge_calls"]
        judge_tokens += assessment["judge_tokens"]

    judge_time = time.perf_counter() - judge_start

    selections = {}
    selection_times = {}

    for key, strategy, lam in [
        ("min", "min", 2.0),
        ("product", "product", 2.0),
        ("step_penalty", "step_penalty", 2.0),
    ]:
        selected, elapsed = score_selection(
            judged, strategy, lam
        )
        selections[key] = selected
        selection_times[key] = elapsed

    ablation = {}

    for key, strategy, lam in [
        ("min", "min", None),
        ("product", "product", None),
        ("penalty_0.0", "step_penalty", 0.0),
        ("penalty_0.5", "step_penalty", 0.5),
        ("penalty_2.0", "step_penalty", 2.0),
    ]:
        if key in selections:
            chosen = selections[key]
        else:
            chosen, _ = score_selection(
                judged, strategy, lam
            )
        ablation[key] = {
            "answer": chosen.get("answer"),
            "num_tied": chosen["num_tied"],
        }

    sc10 = run_self_consistency(
        model=gen_model,
        tokenizer=gen_tok,
        question=question,
        n=10,
        temperature=0.7,
    )

    all_candidates = judged + [
        normalize_candidate(c) for c in sc10["candidates"]
    ]

    if len(all_candidates) != 15:
        raise RuntimeError("Expected 15 candidates")

    scaling = [
        compute_majority_vote([
            c["answer"] for c in all_candidates[:n]
        ])
        for n in range(1, 16)
    ]

    assert scaling[4] == sc5["answer"]

    answers = {
        METHODS[0]: greedy["answer"],
        METHODS[1]: sc5["answer"],
        METHODS[2]: scaling[14],
        METHODS[3]: selections["min"]["answer"],
        METHODS[4]: selections["product"]["answer"],
        METHODS[5]: selections["step_penalty"]["answer"],
    }

    gtime = float(greedy["time_seconds"])
    s5time = float(sc5["time_seconds"])
    s10time = float(sc10["time_seconds"])

    gtokens = int(greedy["generated_tokens"])
    s5tokens = int(sc5["generated_tokens"])
    s10tokens = int(sc10["generated_tokens"])

    costs = {
        METHODS[0]: [1, 0, gtokens, 0, gtime],
        METHODS[1]: [5, 0, s5tokens, 0, s5time],
        METHODS[2]: [
            15, 0, s5tokens + s10tokens, 0,
            s5time + s10time
        ],
        METHODS[3]: [
            5, judge_calls, s5tokens, judge_tokens,
            s5time + judge_time + selection_times["min"]
        ],
        METHODS[4]: [
            5, judge_calls, s5tokens, judge_tokens,
            s5time + judge_time + selection_times["product"]
        ],
        METHODS[5]: [
            5, judge_calls, s5tokens, judge_tokens,
            s5time + judge_time
            + selection_times["step_penalty"]
        ],
    }

    return {
        "id": problem["id"],
        "problem_index": index,
        "seed": 5326 + index,
        "question": question,
        "ground_truth_answer": gt,
        "greedy": {
            "completion": greedy["completion"],
            "steps": greedy["steps"],
            "answer": greedy["answer"],
            "generated_tokens": gtokens,
        },
        "candidates": all_candidates,
        "selected_answers": answers,
        "correct": {
            k: v == gt for k, v in answers.items()
        },
        "sc_scaling_answers": scaling,
        "ablation": ablation,
        "costs": costs,
        "timings": {
            "greedy": gtime,
            "sc5": s5time,
            "judge": judge_time,
            "sc10": s10time,
        },
    }


def write_aggregates(records):
    if not records:
        return

    count = len(records)
    summary = []

    for method in METHODS:
        calls = sum(
            r["costs"][method][0] for r in records
        )
        judge_calls = sum(
            r["costs"][method][1] for r in records
        )
        gen_tokens = sum(
            r["costs"][method][2] for r in records
        )
        judge_tokens = sum(
            r["costs"][method][3] for r in records
        )
        seconds = sum(
            r["costs"][method][4] for r in records
        )

        summary.append({
            "Method": method,
            "Accuracy (%)": round(
                100 * sum(
                    r["correct"][method] for r in records
                ) / count, 2
            ),
            "Generation Calls": calls,
            "Judge Calls": judge_calls,
            "Total LLM Calls": calls + judge_calls,
            "Avg Gen Tokens / Problem": round(
                gen_tokens / count, 2
            ),
            "Avg Judge Tokens / Problem": round(
                judge_tokens / count, 2
            ),
            "Total Time (s)": round(seconds, 3),
        })

    dump_json(OUT / "results_summary.json", summary)

    configs = [
        ("min", "min", None),
        ("product", "Product", None),
        ("penalty_0.0", "Step Penalty", 0.0),
        ("penalty_0.5", "Step Penalty", 0.5),
        ("penalty_2.0", "Step Penalty", 2.0),
    ]

    ablation = []

    for key, strategy, lam in configs:
        ablation.append({
            "Strategy": strategy,
            "Lambda": lam,
            "Accuracy (%)": round(
                100 * sum(
                    r["ablation"][key]["answer"]
                    == r["ground_truth_answer"]
                    for r in records
                ) / count, 2
            ),
            "Tie Rate (%)": round(
                100 * sum(
                    r["ablation"][key]["num_tied"] > 1
                    for r in records
                ) / count, 2
            ),
        })

    dump_json(
        OUT / "aggregation_ablation.json",
        ablation,
    )

    curve = []

    for n in range(1, 16):
        curve.append({
            "N": n,
            "Accuracy (%)": round(
                100 * sum(
                    r["sc_scaling_answers"][n - 1]
                    == r["ground_truth_answer"]
                    for r in records
                ) / count, 2
            ),
            "Avg Gen Tokens / Problem": round(
                sum(
                    sum(
                        c["num_tokens"]
                        for c in r["candidates"][:n]
                    )
                    for r in records
                ) / count, 2
            ),
        })

    dump_json(
        OUT / "sc_scaling_curve.json",
        curve,
    )

    assert (
        curve[4]["Accuracy (%)"]
        == summary[1]["Accuracy (%)"]
    )
    assert (
        curve[14]["Accuracy (%)"]
        == summary[2]["Accuracy (%)"]
    )

    for key in (
        "Generation Calls",
        "Judge Calls",
        "Total LLM Calls",
        "Avg Gen Tokens / Problem",
        "Avg Judge Tokens / Problem",
    ):
        assert (
            summary[3][key]
            == summary[4][key]
            == summary[5][key]
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=50
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete previous per-problem results",
    )
    args = parser.parse_args()

    if not 1 <= args.limit <= 50:
        parser.error(
            "--limit must be between 1 and 50"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required; use Kaggle T4 x2"
        )

    OUT.mkdir(exist_ok=True)

    if args.fresh and ROWS.exists():
        ROWS.unlink()

    problems = read_jsonl(DATA)

    if len(problems) != 50:
        raise ValueError(
            f"Expected 50 problems, found {len(problems)}"
        )

    existing = read_jsonl(ROWS)
    done = {r["id"] for r in existing}

    if len(done) != len(existing):
        raise ValueError(
            "Duplicate IDs in per_problem.jsonl"
        )

    if done - {p["id"] for p in problems}:
        raise ValueError(
            "Saved IDs do not match dataset"
        )

    if all(
        p["id"] in done
        for p in problems[:args.limit]
    ):
        print(
            "Requested problems already saved."
        )
        write_aggregates(existing)
        return

    print("Loading generator on GPU 0...")

    gen_tok = AutoTokenizer.from_pretrained(
        GEN_ID
    )
    gen_model = (
        AutoModelForCausalLM.from_pretrained(
            GEN_ID,
            torch_dtype=torch.float16,
            device_map={"": 0},
        ).eval()
    )

    print("Loading 4-bit NF4 judge on GPU 1...")

    judge_tok = AutoTokenizer.from_pretrained(
        JUDGE_ID
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    judge_model = (
        AutoModelForCausalLM.from_pretrained(
            JUDGE_ID,
            quantization_config=bnb,
            device_map={
                "": 1
                if torch.cuda.device_count() > 1
                else 0
            },
        ).eval()
    )

    for i, p in enumerate(
        problems[:args.limit]
    ):
        if p["id"] in done:
            print(
                f"[{i+1}/{args.limit}] skipped "
                f'{p["id"]} (saved)',
                flush=True,
            )
            continue

        print(
            f'[{i+1}/{args.limit}] running {p["id"]}',
            flush=True,
        )

        record = evaluate_one(
            p, i,
            gen_model, gen_tok,
            judge_model, judge_tok,
        )

        with ROWS.open("a") as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )
            f.flush()

        existing.append(record)
        done.add(p["id"])
        write_aggregates(existing)

        print(
            f"Saved; {len(existing)} problems total",
            flush=True,
        )

    print(
        f"Finished {len(existing)} saved problems. "
        f"Artifacts in {OUT}/"
    )


if __name__ == "__main__":
    main()
