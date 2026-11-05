%%writefile src/logprob_bonus.py
import math
import torch
import torch.nn.functional as F


@torch.inference_mode()
def continuous_valid_probability(
    model,
    tokenizer,
    prompt
):
    """
    Computes normalized P(VALID) using the judge logits.
    Supports multi-token labels.
    """

    if hasattr(tokenizer, "apply_chat_template"):
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prefix = prompt

    prefix_ids = tokenizer(
        prefix, add_special_tokens=False
    ).input_ids

    if not prefix_ids:
        raise ValueError("Empty prompt")

    device = model.get_input_embeddings().weight.device

    def label_logprob(label):
        # Score the exact continuation after the chat prefix.
        continuation = label

        full_ids = tokenizer(
            prefix + continuation,
            add_special_tokens=False
        ).input_ids

        if full_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "Tokenizer boundary mismatch; use a "
                "prompt ending at a token boundary."
            )

        label_ids = full_ids[len(prefix_ids):]

        if not label_ids:
            raise ValueError("Empty label")

        input_ids = torch.tensor(
            [full_ids], device=device
        )

        logits = model(input_ids=input_ids).logits.float()
        log_probs = F.log_softmax(logits, dim=-1)

        total = 0.0

        for i, token_id in enumerate(label_ids):
            position = len(prefix_ids) + i - 1
            total += log_probs[0, position, token_id].item()

        return total

    log_valid = label_logprob("VALID")
    log_invalid = label_logprob("INVALID")

    normalized = torch.softmax(
        torch.tensor(
            [log_valid, log_invalid],
            dtype=torch.float64
        ),
        dim=0
    )

    return float(normalized[0])


def product_logprob_score(probabilities):
    if not probabilities:
        return float("-inf")

    return sum(
        math.log(max(float(p), 1e-12))
        for p in probabilities
    )