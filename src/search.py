import time
from typing import Any, Callable

from .generation import (
    compute_majority_vote,
    run_self_consistency,
)
from .prm_judge import score_path, build_prm_judge_prompt
from .logprob import continuous_valid_probability

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]


def aggregate_path_score(
    step_scores: list[float],
    strategy: str = "min",
    penalty_lambda: float = 2.0
) -> float:
    """
    Aggregate step-level rewards into one path score.
    """

    if not step_scores:
        return -1.0

    if strategy == "min":
        return float(min(step_scores))

    if strategy == "product":
        result = 1.0

        for reward in step_scores:
            probability = (reward + 1.0) / 2.0
            probability = max(0.01, min(0.99, probability))
            result *= probability

        return float(result)


    if strategy == "product_logprob":
        import math

        if not all(
            math.isfinite(float(p)) and 0 <= float(p) <= 1
            for p in step_scores
        ):
            raise ValueError("Expected probabilities in [0, 1]")

        return float(sum(
            math.log(max(float(p), 1e-12))
            for p in step_scores
        ))

    if strategy == "step_penalty":
        mean_score = sum(step_scores) / len(step_scores)
        invalid_count = sum(
            reward == -1.0 for reward in step_scores
        )

        return float(
            mean_score - penalty_lambda * invalid_count
        )

    raise ValueError(
        f"Unknown aggregation strategy: {strategy}"
    )


def select_best_path(
    candidate_paths: list[dict[str, Any]],
    aggregation_strategy: str = "min",
    penalty_lambda: float = 2.0
) -> dict[str, Any]:
    """
    Select the best eligible candidate using PRM scores.
    Break ties using majority vote.
    """

    if not candidate_paths:
        raise ValueError("candidate_paths cannot be empty")

    scored_candidates = []

    for candidate in candidate_paths:
        scores = (
            candidate["step_probabilities"]
            if aggregation_strategy == "product_logprob"
            else candidate["step_scores"]
        )

        path_score = aggregate_path_score(
            scores,
            strategy=aggregation_strategy,
            penalty_lambda=penalty_lambda
        )

        scored_candidates.append({
            **candidate,
            "path_score": path_score
        })

    has_valid_answer = any(
        candidate.get("answer") is not None
        for candidate in scored_candidates
    )

    if has_valid_answer:
        eligible = [
            candidate
            for candidate in scored_candidates
            if candidate.get("answer") is not None
        ]
    else:
        eligible = scored_candidates

    max_score = max(
        candidate["path_score"]
        for candidate in eligible
    )

    tied = [
        candidate
        for candidate in eligible
        if abs(candidate["path_score"] - max_score) <= 1e-9
    ]

    if len(tied) == 1:
        winner = tied[0]

    else:
        voted_answer = compute_majority_vote(
            [candidate.get("answer") for candidate in tied]
        )

        winner = next(
            (
                candidate
                for candidate in tied
                if candidate.get("answer") == voted_answer
            ),
            tied[0]
        )

    return {
        **winner,
        "num_tied": len(tied)
    }


def run_prm_best_of_n(
    model: Any = None,
    judge_model: Any = None,
    gen_tokenizer: Any = None,
    judge_tokenizer: Any = None,
    question: str = "",
    n: int = 5,
    aggregation_strategy: str = "min",
    penalty_lambda: float = 2.0,
    candidates: list[dict[str, Any]] | None = None,
    generate_fn: GenerateFn | None = None,
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Generate or reuse candidate reasoning paths,
    score each step with the PRM, and select the best.
    """

    start_time = time.perf_counter()

    if candidates is not None:
        candidate_pool = [
            dict(candidate) for candidate in candidates
        ]

        generation_calls = 0
        generated_tokens = 0

    else:
        generation_result = run_self_consistency(
            model=model,
            tokenizer=gen_tokenizer,
            question=question,
            n=n,
            generate_fn=generate_fn
        )

        candidate_pool = [
            dict(candidate)
            for candidate in generation_result["candidates"]
        ]

        generation_calls = generation_result["generation_calls"]
        generated_tokens = generation_result["generated_tokens"]

    judged_candidates = []
    total_judge_calls = 0
    total_judge_tokens = 0

    for candidate in candidate_pool:
        if aggregation_strategy == "product_logprob":
            if judge_model is None or judge_tokenizer is None:
                raise ValueError("product_logprob requires judge model and tokenizer")
            steps = candidate.get("steps", [])
            probabilities = []
            for i, step in enumerate(steps):
                prompt = build_prm_judge_prompt(question, steps[:i], step)
                probabilities.append(
                    continuous_valid_probability(judge_model, judge_tokenizer, prompt)
                )
            scored_candidate = {
                **candidate,
                "step_probabilities": probabilities,
                "rationales": [],
            }
            total_judge_calls += len(steps)
        else:
            result = score_path(
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                question=question,
                candidate=candidate,
                judge_fn=judge_fn,
            )
            scored_candidate = {
                **candidate,
                "step_scores": result["step_scores"],
                "rationales": result["rationales"],
            }
            total_judge_calls += result["judge_calls"]
            total_judge_tokens += result["judge_tokens"]
        judged_candidates.append(scored_candidate)

    selected = select_best_path(
        judged_candidates,
        aggregation_strategy=aggregation_strategy,
        penalty_lambda=penalty_lambda
    )

    return {
        "selected": selected,
        "answer": selected.get("answer"),
        "candidates": judged_candidates,
        "generation_calls": generation_calls,
        "judge_calls": total_judge_calls,
        "generated_tokens": generated_tokens,
        "judge_tokens": total_judge_tokens,
        "time_seconds": float(
            time.perf_counter() - start_time
        )
    }


def run_compute_matched_self_consistency(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    target_calls: int = 15,
    temperature: float = 0.7,
    existing_candidates: list[dict[str, Any]] | None = None,
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Run self-consistency with a fixed generation budget.
    Reuse existing candidates and generate only the deficit.
    """

    start_time = time.perf_counter()

    existing = (
        list(existing_candidates)
        if existing_candidates is not None
        else []
    )

    if target_calls < 0:
        raise ValueError("target_calls cannot be negative")

    if len(existing) > target_calls:
        raise ValueError(
            "Existing candidates exceed target_calls"
        )

    remaining = target_calls - len(existing)

    new_candidates = []
    generation_calls = 0
    generated_tokens = 0

    if remaining > 0:
        result = run_self_consistency(
            model=model,
            tokenizer=tokenizer,
            question=question,
            n=remaining,
            temperature=temperature,
            generate_fn=generate_fn
        )

        new_candidates = result["candidates"]
        generation_calls = result["generation_calls"]
        generated_tokens = result["generated_tokens"]

    all_candidates = existing + new_candidates

    answer = compute_majority_vote(
        [
            candidate.get("answer")
            for candidate in all_candidates
        ]
    )

    return {
        "candidates": all_candidates,
        "answer": answer,
        "generation_calls": generation_calls,
        "total_samples": len(all_candidates),
        "generated_tokens": generated_tokens,
        "time_seconds": float(
            time.perf_counter() - start_time
        )
    }
