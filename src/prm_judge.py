"

import re
from typing import Any, Callable

JudgeFn = Callable[[str], dict]


def build_prm_judge_prompt(
    question: str,
    preceding_steps: list[str],
    current_step: str
) -> str:
    """
    Build a judge prompt using local step validity.
    """

    if preceding_steps:
        history = "\n".join(
            f"Step {i}: {step}"
            for i, step in enumerate(preceding_steps, start=1)
        )
    else:
        history = "(no preceding steps)"

    return (
        "You are a mathematical reasoning verifier.\n"
        "Your task is to judge ONLY the current reasoning step.\n\n"
        "IMPORTANT: assume all preceding steps are correct.\n"
        "Do not penalize the current step for mistakes in "
        "earlier steps.\n"
        "Check whether the current step follows logically "
        "from the given question and preceding steps, "
        "and whether its arithmetic is correct.\n\n"
        f"Question:\n{question}\n\n"
        f"Preceding Steps:\n{history}\n\n"
        f"Current Step:\n{current_step}\n\n"
        "Respond using exactly one of these verdicts:\n"
        "Verdict: VALID\n"
        "Verdict: INVALID\n"
        "Rationale: <one short sentence explaining why>\n"
    )


def parse_judge_decision(
    judge_output: str
) -> tuple[float, str]:
    """
    Convert the judge response into a score and rationale.
    VALID = +1.0, INVALID = -1.0.
    """

    if not isinstance(judge_output, str):
        return -1.0, ""

    explicit_verdict = re.search(
        r"\bverdict\s*:\s*\**\s*(VALID|INVALID)\b",
        judge_output,
        flags=re.IGNORECASE
    )

    if explicit_verdict:
        score = (
            1.0
            if explicit_verdict.group(1).upper() == "VALID"
            else -1.0
        )
    elif re.search(
        r"\bNOT\s+VALID\b|\bINVALID\b",
        judge_output,
        flags=re.IGNORECASE
    ):
        score = -1.0
    elif re.search(
        r"\bVALID\b",
        judge_output,
        flags=re.IGNORECASE
    ):
        score = 1.0
    else:
        score = -1.0

    rationale_match = re.search(
        r"\bRationale\s*:\s*([^\n]*)",
        judge_output,
        flags=re.IGNORECASE
    )

    rationale = ""

    if rationale_match:
        raw = rationale_match.group(1).strip()

        first_sentence = re.search(
            r"^.*?[.!?](?=\s|$)",
            raw
        )

        rationale = (
            first_sentence.group(0).strip()
            if first_sentence
            else raw
        )

    return score, rationale


def score_path(
    judge_model: Any = None,
    judge_tokenizer: Any = None,
    question: str = "",
    candidate: dict[str, Any] | None = None,
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Judge each reasoning step sequentially.
    """

    steps = (candidate or {}).get("steps", [])

    step_scores = []
    rationales = []
    judge_tokens = 0

    for i, current_step in enumerate(steps):

        preceding_steps = steps[:i]

        prompt = build_prm_judge_prompt(
            question,
            preceding_steps,
            current_step
        )

        if judge_fn is not None:
            output = judge_fn(prompt)

        else:
            if judge_model is None or judge_tokenizer is None:
                raise ValueError(
                    "Provide a judge model/tokenizer or judge_fn."
                )

            import torch

            messages = [
                {"role": "user", "content": prompt}
            ]

            formatted_prompt = judge_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = judge_tokenizer(
                formatted_prompt,
                return_tensors="pt"
            )

            device = next(judge_model.parameters()).device

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            input_length = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                output_ids = judge_model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=96,
                    pad_token_id=judge_tokenizer.eos_token_id
                )

            completion_ids = output_ids[0][input_length:]

            output = {
                "text": judge_tokenizer.decode(
                    completion_ids,
                    skip_special_tokens=True
                ),
                "num_tokens": len(completion_ids)
            }

        score, rationale = parse_judge_decision(
            output["text"]
        )

        step_scores.append(score)
        rationales.append(rationale)
        judge_tokens += output["num_tokens"]

    return {
        "step_scores": step_scores,
        "rationales": rationales,
        "judge_calls": len(steps),
        "judge_tokens": judge_tokens
    }
