import time
from typing import Any, Callable

from src.extraction import parse_steps, extract_answer


GenerateFn = Callable[[str, int, dict], list[dict]]



def format_step_prompt(question: str) -> str:
    """
    Create a zero-shot prompt requiring numbered reasoning
    steps and a final integer answer.
    """
    return (
        "Solve the following math word problem carefully.\n"
        "Show your reasoning using numbered steps.\n\n"
        "Required output format:\n"
        "Step 1: First calculation or deduction.\n"
        "Step 2: Second calculation or deduction.\n"
        "Step 3: Continue as needed.\n"
        "Final Answer: <integer>\n\n"
        "Rules:\n"
        "1. Use Step 1:, Step 2:, etc. in order.\n"
        "2. Show all important arithmetic operations.\n"
        "3. Do not use Markdown headings, bold text, or LaTeX.\n"
        "4. Do not use boxed answers.\n"
        "5. Write exactly one Final Answer: line.\n"
        "6. The final answer must be an integer.\n"
        "7. Do not write anything after the final answer.\n\n"
        f"Question: {question}\n"
    )


def compute_majority_vote(
    answers: list[int | None]
) -> int | None:
    """
    Ignore None values and choose the most frequent answer.
    Break ties using the earliest occurrence.
    """
    counts = {}

    for answer in answers:
        if answer is not None:
            counts[answer] = counts.get(answer, 0) + 1

    if not counts:
        return None

    max_count = max(counts.values())

    for answer in answers:
        if answer is not None and counts[answer] == max_count:
            return answer

    return None

def _generate_with_model(
    model: Any,
    tokenizer: Any,
    prompt: str,
    num_samples: int,
    gen_kwargs: dict
) -> list[dict]:
    """
    Generate completions using Hugging Face .generate().

    Return one dictionary per generated sequence:
        {"text": str, "num_tokens": int}
    """
    import torch

    if model is None or tokenizer is None:
        raise ValueError(
            "Provide model/tokenizer or a mock generate_fn."
        )

    messages = [
        {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    device = next(model.parameters()).device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    kwargs = dict(gen_kwargs)
    kwargs["num_return_sequences"] = num_samples

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **kwargs,
            pad_token_id=tokenizer.eos_token_id
        )

    results = []

    for sequence in output_ids:
        completion_ids = sequence[input_length:]

        text = tokenizer.decode(
            completion_ids,
            skip_special_tokens=True
        )

        results.append({
            "text": text,
            "num_tokens": len(completion_ids)
        })

    return results


def run_greedy(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Generate exactly one deterministic reasoning path.
    """
    start_time = time.perf_counter()

    prompt = format_step_prompt(question)

    gen_kwargs = {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512
    }

    if generate_fn is not None:
        outputs = generate_fn(
            prompt, 1, gen_kwargs
        )
    else:
        outputs = _generate_with_model(
            model, tokenizer, prompt, 1, gen_kwargs
        )

    completion = outputs[0]["text"]
    num_tokens = outputs[0]["num_tokens"]

    steps = parse_steps(completion)
    answer = extract_answer(completion)

    elapsed = time.perf_counter() - start_time

    return {
        "prompt": prompt,
        "completion": completion,
        "steps": steps,
        "answer": answer,
        "generation_calls": 1,
        "generated_tokens": num_tokens,
        "time_seconds": elapsed
    }


def run_self_consistency(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    n: int = 5,
    temperature: float = 0.7,
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Generate n sampled reasoning paths and choose
    the majority-voted final answer.
    """
    start_time = time.perf_counter()

    prompt = format_step_prompt(question)

    gen_kwargs = {
        "do_sample": True,
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512
    }

    if generate_fn is not None:
        outputs = generate_fn(
            prompt, n, gen_kwargs
        )
    else:
        outputs = _generate_with_model(
            model, tokenizer, prompt, n, gen_kwargs
        )

    candidates = []

    for output in outputs:
        completion = output["text"]

        candidates.append({
            "completion": completion,
            "steps": parse_steps(completion),
            "answer": extract_answer(completion),
            "num_tokens": output["num_tokens"]
        })

    answers = [
        candidate["answer"]
        for candidate in candidates
    ]

    selected_answer = compute_majority_vote(answers)

    total_tokens = sum(
        candidate["num_tokens"]
        for candidate in candidates
    )

    elapsed = time.perf_counter() - start_time

    return {
        "candidates": candidates,
        "answer": selected_answer,
        "generation_calls": len(candidates),
        "generated_tokens": total_tokens,
        "time_seconds": elapsed
    }
