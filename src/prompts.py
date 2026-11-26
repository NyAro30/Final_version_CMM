"""Prompt construction for function and argument selection."""


def chat_prompt(instruction: str, request: str) -> str:
    """Format a prompt for the supplied Qwen chat model."""
    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n</think>\n"
    )


def argument_prompt(
    function_name: str,
    description: str,
    user_request: str,
    parameter_name: str,
    parameter_type: str,
    parameter_names: list[str],
) -> str:
    """Describe the role and expected value of one function argument."""
    position = parameter_names.index(parameter_name) + 1
    prompt = (
        f"Function: {function_name}\n"
        f"Function purpose: {description}\n"
        f"Arguments in definition order: {parameter_names}\n"
        f"Argument to extract: {parameter_name} "
        f"(position {position} of {len(parameter_names)})\n"
        f"Argument type: {parameter_type}\n"
        f"User request: {user_request}\n"
        "Infer the value from the request and the argument's role in "
        "the function. For neutral argument names, use the order in "
        "which the values appear in the request and the definition. "
        "Do not compute the function's result, paraphrase, normalize, or "
        "invent input values. "
        "Return one JSON value with exactly the requested type, "
        "without comments."
    )
    lower_name = parameter_name.lower()
    if "regex" in lower_name or "pattern" in lower_name:
        prompt += (
            " This argument is the pattern used to match text in the "
            "source string. Use Python regular-expression syntax without "
            "slash delimiters. Example: extracting the first word from "
            "a sentence can use the pattern \"[A-Za-z]+\". For matching "
            "words that begin with a capital letter, use a pattern such as "
            "\"\\b[A-Z][a-z]*\\b\"."
        )
    elif "replacement" in lower_name:
        prompt += (
            " This is the exact literal text inserted for each match. "
            "For a request phrased as 'change X to Y', return only Y, "
            "not X, the source text, or the matching pattern."
        )
    elif "source" in lower_name:
        prompt += (
            " This is the original text to transform, not the transformed "
            "result. Copy it exactly, preserving capitalization, spaces, "
            "and punctuation; do not return the pattern or replacement."
        )
    elif "name" in lower_name:
        prompt += (
            " This is the person or entity the request refers to, not "
            "the greeting or other generated output."
        )
    elif parameter_type == "number":
        prompt += (
            " Example: for a function add(left, right) and the request "
            "'add 7 and 8', left is 7 and right is 8. For a function "
            "square_root(value) and the request 'find the square root of "
            "25', value is 25, not the computed result 5."
        )
    elif parameter_type == "string":
        prompt += (
            " Example: for a function reverse(text) and the request "
            "'reverse the word \"planet\"', text is \"planet\". When the "
            "request quotes source text, return that exact text."
        )
    return prompt
