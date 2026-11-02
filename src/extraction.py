
"""
src/extraction.py
Parsing reasoning steps and final integer answers.
"""

import re


def parse_steps(completion_text: str) -> list[str]:
    if not isinstance(completion_text, str):
        return []

    pattern = (
        r"(?:^|\n)\s*"
        r"[#*`\s]*Step\s*\d+\s*[:.]?\s*[*`]*\s*"
        r"(.*?)"
        r"(?=\n\s*[#*`\s]*Step\s*\d+\s*[:.]?"
        r"|\n\s*[#*`\s]*Final\s+Answer\b|$)"
    )

    matches = re.findall(
        pattern,
        completion_text,
        flags=re.IGNORECASE | re.DOTALL
    )

    return [
        match.strip().strip("*#`").strip()
        for match in matches
        if match.strip().strip("*#`").strip()
    ]


def extract_answer(completion_text: str) -> int | None:
    if not isinstance(completion_text, str):
        return None

    pattern = (
        r"(?im)^[ \t]*[#*`]*[ \t]*"
        r"Final\s+Answer[ \t]*\**[ \t]*:?[ \t]*\**[ \t]*"
        r"\$?[ \t]*([+-]?\d[\d,]*(?:\.\d+)?)"
        r"(?=\s|[.!?*`]|$)"
    )

    matches = re.findall(pattern, completion_text)

    if not matches:
        return None

    try:
        value = matches[-1].replace(",", "")

        # Use decimal arithmetic to avoid float precision loss.
        from decimal import Decimal, InvalidOperation
        number = Decimal(value)

        if not number.is_finite():
            return None

        if number != number.to_integral_value():
            return None

        return int(number)

    except (ValueError, InvalidOperation):
        return None
