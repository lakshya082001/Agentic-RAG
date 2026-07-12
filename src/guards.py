from llm_guard import scan_prompt
from llm_guard.input_scanners import PromptInjection, Toxicity

_scanners = None


def _get_scanners():
    global _scanners
    if _scanners is None:
        _scanners = [PromptInjection(), Toxicity()]
    return _scanners


def check_input(text: str) -> tuple[bool, dict]:
    """Scan user input for prompt injection / toxicity.

    Returns (is_safe, scores). is_safe is False if any scanner flags the input.
    """
    _, valid, scores = scan_prompt(_get_scanners(), text)
    return all(valid.values()), scores
